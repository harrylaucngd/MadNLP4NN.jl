#!/usr/bin/env python3
"""Derivative/structure gates for published PFR CNN-NMPC cases 2 and 3."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def relative_error(first, second):
    return float(
        np.linalg.norm(first - second)
        / max(np.linalg.norm(first), np.linalg.norm(second), 1e-14)
    )


def build_case(factory, case, artifact_root):
    if case == 2:
        root = artifact_root / "case2"
        with (root / "published_greybox_usingfunc_output.pkl").open("rb") as handle:
            data = pickle.load(handle)
        current = np.concatenate((data["C"][0], data["T"][0]))
        previous = [data["F"][0], data["Ta"][0], data["C0"][0], data["T0"][0]]
        flow_reference = 14.7 * 1000 / 3600 / (1.86 * 1000) * 2
        evaluator = factory(
            str(root / "2PDEs_Tanh_32hc_2lay.pt"),
            current,
            previous,
            [0, 1],
            [9, 9],
            [570.0, 315.0],
            [1900.0**2, 20.0**2],
            [flow_reference**2, 20.0**2, 1900.0**2, 20.0**2],
            prediction_horizon=40,
            control_horizon=10,
            device="gpu",
        )
    else:
        root = artifact_root / "case3"
        with (root / "published_greybox_usingfunc_output.pkl").open("rb") as handle:
            data = pickle.load(handle)
        components = ["CH4", "H2O", "H2", "CO", "CO2"]
        spatial = sorted(
            key[1]
            for key in data
            if isinstance(key, tuple) and key[0] == "FCH4" and key[1] != 0
        )
        current = np.concatenate(
            [[data[(f"F{component}", location)][0] for location in spatial] for component in components]
        )
        previous = [data["FCH4_in"][0], data["FH2O_in"][0], data["T_in"][0]]
        evaluator = factory(
            str(root / "PFR_Tanh_32hc_3lay.pt"),
            current,
            previous,
            [0, 2, 3],
            [49, 49, 49],
            [2.639, 5.694, 0.085795],
            [2.639, 5.694, 0.085795],
            [100 * 0.112 / 3600 * 1000, 100 * 3 * 0.112 / 3600 * 1000, 848.15],
            prediction_horizon=30,
            control_horizon=10,
            device="gpu",
        )
    return evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=(2, 3), required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    from jax_nn_evaluator import create_pfr_cnn_nmpc_evaluator

    evaluator = build_case(
        create_pfr_cnn_nmpc_evaluator, args.case, args.artifact_root
    )
    point = evaluator.initial_guess()
    rng = np.random.default_rng(args.seed)
    direction = rng.normal(size=evaluator.n)
    direction /= np.linalg.norm(direction)
    multipliers = rng.normal(size=evaluator.m) / np.sqrt(evaluator.m)
    epsilon = 1e-5
    objective_fd = (
        evaluator.evaluate_obj(point + epsilon * direction)
        - evaluator.evaluate_obj(point - epsilon * direction)
    ) / (2 * epsilon)
    objective_ad = evaluator.evaluate_grad(point) @ direction
    constraint_fd = (
        evaluator.evaluate_cons(point + epsilon * direction)
        - evaluator.evaluate_cons(point - epsilon * direction)
    ) / (2 * epsilon)
    constraint_ad = evaluator.evaluate_jprod(point, direction)

    def lagrangian_gradient(value):
        return evaluator.evaluate_grad(value) + evaluator.evaluate_jtprod(
            value, multipliers
        )

    hessian_fd = (
        lagrangian_gradient(point + epsilon * direction)
        - lagrangian_gradient(point - epsilon * direction)
    ) / (2 * epsilon)
    rows, cols = evaluator.hessian_structure()
    values = evaluator.evaluate_hess_coord(point, multipliers, 1.0)
    hessian_product = np.zeros(evaluator.n)
    np.add.at(hessian_product, rows, values * direction[cols])
    off_diagonal = rows != cols
    np.add.at(
        hessian_product,
        cols[off_diagonal],
        values[off_diagonal] * direction[rows[off_diagonal]],
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case": args.case,
        "n": evaluator.n,
        "m": evaluator.m,
        "jacobian_nnz": evaluator.jacobian_nnz,
        "hessian_lower_nnz": evaluator.hessian_nnz,
        "initial_objective": evaluator.evaluate_obj(point),
        "initial_constraint_inf_norm": float(
            np.linalg.norm(evaluator.evaluate_cons(point), ord=np.inf)
        ),
        "objective_directional_relative_error": relative_error(
            np.asarray(objective_ad), np.asarray(objective_fd)
        ),
        "constraint_jvp_relative_error": relative_error(
            constraint_ad, constraint_fd
        ),
        "lagrangian_hessian_vector_relative_error": relative_error(
            hessian_product, hessian_fd
        ),
    }
    print(json.dumps(payload, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
