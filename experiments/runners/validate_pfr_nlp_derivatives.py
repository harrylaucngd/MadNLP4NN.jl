#!/usr/bin/env python3
"""Directional derivative audit for the published PFR gray-box NLP."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def relative_error(candidate, reference):
    return float(
        np.linalg.norm(candidate - reference)
        / max(np.linalg.norm(candidate), np.linalg.norm(reference), 1e-14)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    from jax_nn_evaluator import create_pfr_nmpc_evaluator

    evaluator = create_pfr_nmpc_evaluator(
        str(args.model),
        np.full(10, 0.5),
        np.full(2, 0.5),
        0.4,
        device="gpu",
    )
    point = evaluator.initial_guess()
    rng = np.random.default_rng(args.seed)
    direction = rng.standard_normal(evaluator.n)
    direction /= np.linalg.norm(direction)
    multipliers = rng.standard_normal(evaluator.m) / np.sqrt(evaluator.m)
    epsilon = 1e-5

    objective_fd = (
        evaluator.evaluate_obj(point + epsilon * direction)
        - evaluator.evaluate_obj(point - epsilon * direction)
    ) / (2 * epsilon)
    objective_ad = float(evaluator.evaluate_grad(point) @ direction)
    constraint_fd = (
        evaluator.evaluate_cons(point + epsilon * direction)
        - evaluator.evaluate_cons(point - epsilon * direction)
    ) / (2 * epsilon)
    constraint_ad = evaluator.evaluate_jprod(point, direction)

    def lagrangian_gradient(value):
        return evaluator.evaluate_grad(value) + evaluator.evaluate_jac(value).T @ multipliers

    hessian_fd = (
        lagrangian_gradient(point + epsilon * direction)
        - lagrangian_gradient(point - epsilon * direction)
    ) / (2 * epsilon)
    hessian_ad = evaluator.evaluate_hess(point, multipliers, 1.0) @ direction
    jacobian = evaluator.evaluate_jac(point)
    jacobian_rows, jacobian_cols = evaluator.jacobian_structure()
    jacobian_mask = np.zeros(jacobian.shape, dtype=bool)
    jacobian_mask[jacobian_rows, jacobian_cols] = True
    hessian = evaluator.evaluate_hess(point, multipliers, 1.0)
    hessian_rows, hessian_cols = evaluator.hessian_structure()
    hessian_mask = np.zeros(hessian.shape, dtype=bool)
    hessian_mask[hessian_rows, hessian_cols] = True
    hessian_mask[hessian_cols, hessian_rows] = True
    records = {
        "n": evaluator.n,
        "m": evaluator.m,
        "initial_objective": evaluator.evaluate_obj(point),
        "initial_constraint_inf_norm": float(
            np.linalg.norm(evaluator.evaluate_cons(point), ord=np.inf)
        ),
        "objective_directional_relative_error": relative_error(
            np.asarray(objective_ad), np.asarray(objective_fd)
        ),
        "constraint_jvp_relative_error": relative_error(constraint_ad, constraint_fd),
        "lagrangian_hessian_vector_relative_error": relative_error(
            hessian_ad, hessian_fd
        ),
        "jacobian_declared_nnz": evaluator.jacobian_nnz,
        "hessian_declared_lower_nnz": evaluator.hessian_nnz,
        "max_omitted_jacobian_absolute": float(
            np.max(np.abs(jacobian[~jacobian_mask]))
        ),
        "max_omitted_hessian_absolute": float(
            np.max(np.abs(hessian[~hessian_mask]))
        ),
    }
    print(json.dumps(records, indent=2))
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "seed": args.seed,
        **records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
