#!/usr/bin/env python3
"""Independent directional gate for the explicit Darcy Gauss--Newton model."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=253)
    parser.add_argument("--latent-resolution", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    from jax_nn_evaluator import create_pdebench_darcy_evaluator

    payload = torch.load(config["data"], map_location="cpu", weights_only=False)
    target = np.asarray(payload["y"][args.sample], dtype=np.float64)
    n = args.latent_resolution**2
    rng = np.random.default_rng(args.seed)
    point = rng.normal(scale=0.2, size=n)
    direction = rng.normal(size=n)
    direction /= np.linalg.norm(direction)
    evaluator = create_pdebench_darcy_evaluator(
        config["model"],
        target.reshape(-1),
        np.zeros(n),
        reference_field=np.zeros(n),
        full_resolution=config["full_resolution"],
        regularization_weight=config["regularization_weight"],
        device="gpu",
    )
    evaluator.set_hessian_mode("gauss_newton")
    rows, cols = evaluator.hessian_structure()
    values = evaluator.evaluate_hess_coord(point, np.zeros(0), 1.0)
    product = np.zeros(n)
    np.add.at(product, rows, values * direction[cols])
    off_diagonal = rows != cols
    np.add.at(product, cols[off_diagonal], values[off_diagonal] * direction[rows[off_diagonal]])
    quadratic_from_coordinates = float(direction @ product)

    plus = evaluator.evaluate_nn(point + args.epsilon * direction)
    minus = evaluator.evaluate_nn(point - args.epsilon * direction)
    pressure_scale = np.sqrt(np.mean(target.reshape(-1) ** 2)) + 1e-12
    residual_direction = (
        (plus - minus)
        / (2 * args.epsilon)
        / pressure_scale
        / np.sqrt(target.size)
    )
    quadratic_from_jvp = float(
        residual_direction @ residual_direction
        + config["regularization_weight"] / n * (direction @ direction)
    )
    relative_error = abs(quadratic_from_coordinates - quadratic_from_jvp) / max(
        abs(quadratic_from_coordinates), abs(quadratic_from_jvp), 1e-14
    )
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample": args.sample,
        "latent_resolution": args.latent_resolution,
        "epsilon": args.epsilon,
        "seed": args.seed,
        "quadratic_from_hessian_coordinates": quadratic_from_coordinates,
        "quadratic_from_finite_difference_jvp": quadratic_from_jvp,
        "relative_error": relative_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
