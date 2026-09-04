#!/usr/bin/env python3
"""Directional finite-difference audit for the PDEBench JAX NLP oracle."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


def relative_error(actual, expected):
    return float(
        np.linalg.norm(np.asarray(actual) - np.asarray(expected))
        / max(1.0, np.linalg.norm(np.asarray(actual)), np.linalg.norm(np.asarray(expected)))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=9257)
    parser.add_argument("--latent-resolution", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    from jax_nn_evaluator import create_pdebench_darcy_evaluator

    config = json.loads(args.config.read_text())
    with h5py.File(config["data"], "r") as handle:
        target = np.asarray(handle["tensor"][args.sample, 0], dtype=np.float64)
    n = args.latent_resolution**2
    rng = np.random.default_rng(args.seed)
    x = np.full(n, 0.541621900172992) + rng.normal(scale=0.02, size=n)
    direction = rng.normal(size=n)
    direction /= np.linalg.norm(direction)
    multipliers = np.array([0.3, 0.7])
    evaluator = create_pdebench_darcy_evaluator(
        config["model"],
        target.reshape(-1),
        np.full(n, 0.541621900172992),
        full_resolution=128,
        regularization_weight=config["regularization_weight"],
        binary_weight=config.get("binary_weight", 0.0),
        mean_lower=config["mean_lower"],
        mean_upper=config["mean_upper"],
        device="gpu",
    )
    epsilon = args.epsilon
    objective_plus = evaluator.evaluate_obj(x + epsilon * direction)
    objective_minus = evaluator.evaluate_obj(x - epsilon * direction)
    objective_fd = (objective_plus - objective_minus) / (2 * epsilon)
    gradient = evaluator.evaluate_grad(x)
    objective_ad = float(gradient @ direction)

    constraints_plus = evaluator.evaluate_cons(x + epsilon * direction)
    constraints_minus = evaluator.evaluate_cons(x - epsilon * direction)
    constraint_fd = (constraints_plus - constraints_minus) / (2 * epsilon)
    jacobian = evaluator.evaluate_jac(x)
    constraint_ad = jacobian @ direction

    gradient_plus = evaluator.evaluate_grad(x + epsilon * direction)
    gradient_minus = evaluator.evaluate_grad(x - epsilon * direction)
    hessian_fd = (gradient_plus - gradient_minus) / (2 * epsilon)
    hessian = evaluator.evaluate_hess(x, multipliers, 1.0)
    # Constraints are linear means in this configuration, so their Hessians are
    # zero and this is also the objective Hessian-vector product.
    hessian_ad = hessian @ direction

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample": args.sample,
        "latent_resolution": args.latent_resolution,
        "epsilon": epsilon,
        "objective_directional": {
            "finite_difference": objective_fd,
            "automatic_differentiation": objective_ad,
            "relative_error": relative_error(objective_fd, objective_ad),
        },
        "constraint_jvp": {
            "finite_difference": constraint_fd.tolist(),
            "automatic_differentiation": constraint_ad.tolist(),
            "relative_error": relative_error(constraint_fd, constraint_ad),
        },
        "hessian_vector": {
            "relative_error": relative_error(hessian_fd, hessian_ad),
            "finite_difference_norm": float(np.linalg.norm(hessian_fd)),
            "automatic_differentiation_norm": float(np.linalg.norm(hessian_ad)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
