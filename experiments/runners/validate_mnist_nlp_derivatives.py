#!/usr/bin/env python3
"""Directional derivative and sparsity audit for the adversarial MNIST NLP."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torchvision
from torchvision.transforms import ToTensor


def relative_error(first, second):
    return float(
        np.linalg.norm(first - second)
        / max(np.linalg.norm(first), np.linalg.norm(second), 1e-14)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=6)
    parser.add_argument("--adversarial-label", type=int, default=9)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    from jax_nn_evaluator import create_mnist_adversarial_evaluator

    dataset = torchvision.datasets.MNIST(
        root=args.data_dir, train=False, transform=ToTensor(), download=False
    )
    image, label = dataset[args.sample]
    evaluator = create_mnist_adversarial_evaluator(
        str(args.model), image.reshape(-1).numpy(), args.adversarial_label, device="gpu"
    )
    point = evaluator.initial_guess()
    rng = np.random.default_rng(20260821)
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
    coordinates = evaluator.evaluate_hess_coord(point, multipliers, 1.0)
    hessian_product = np.zeros(evaluator.n)
    np.add.at(hessian_product, rows, coordinates * direction[cols])
    off_diagonal = rows != cols
    np.add.at(
        hessian_product,
        cols[off_diagonal],
        coordinates[off_diagonal] * direction[rows[off_diagonal]],
    )
    jacobian = evaluator.evaluate_jac(point)
    jacobian_rows, jacobian_cols = evaluator.jacobian_structure()
    mask = np.zeros(jacobian.shape, dtype=bool)
    mask[jacobian_rows, jacobian_cols] = True
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "sample": args.sample,
        "true_label": int(label),
        "adversarial_label": args.adversarial_label,
        "n": evaluator.n,
        "m": evaluator.m,
        "jacobian_nnz": evaluator.jacobian_nnz,
        "hessian_lower_nnz": evaluator.hessian_nnz,
        "objective_directional_relative_error": relative_error(
            np.asarray(objective_ad), np.asarray(objective_fd)
        ),
        "constraint_jvp_relative_error": relative_error(
            constraint_ad, constraint_fd
        ),
        "lagrangian_hessian_vector_relative_error": relative_error(
            hessian_product, hessian_fd
        ),
        "max_omitted_jacobian_absolute": float(np.max(np.abs(jacobian[~mask]))),
    }
    print(json.dumps(payload, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
