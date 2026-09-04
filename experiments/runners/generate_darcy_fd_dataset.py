#!/usr/bin/env python3
"""Generate a fully replayable elliptic-Darcy split with fixed provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(count, grid, seed, length_scale, clip):
    from darcy_data import darcy_fd_solve, sample_grf

    rng = np.random.default_rng(seed)
    coefficients = np.empty((count, grid, grid), dtype=np.float32)
    solutions = np.empty_like(coefficients)
    started = time.perf_counter()
    for index in range(count):
        coefficient = np.clip(
            sample_grf(rng, grid, length_scale=length_scale), -clip, clip
        )
        solution = darcy_fd_solve(coefficient)
        coefficients[index] = coefficient
        solutions[index] = solution
        if (index + 1) % max(1, count // 20) == 0:
            print(
                {
                    "completed": index + 1,
                    "count": count,
                    "elapsed_s": time.perf_counter() - started,
                },
                flush=True,
            )
    return coefficients, solutions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--length-scale", type=float, default=0.2)
    parser.add_argument("--coefficient-clip", type=float, default=3.0)
    parser.add_argument("--train-seed", type=int, default=20260823)
    parser.add_argument("--test-seed", type=int, default=20260824)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    started = time.perf_counter()
    train_x, train_y = generate(
        args.train_size,
        args.grid,
        args.train_seed,
        args.length_scale,
        args.coefficient_clip,
    )
    test_x, test_y = generate(
        args.test_size,
        args.grid,
        args.test_seed,
        args.length_scale,
        args.coefficient_clip,
    )
    metadata = {
        "equation": "-div(exp(a) grad u) = 1",
        "boundary_condition": "zero Dirichlet",
        "discretization": "five-point finite difference, harmonic face averaging",
        "grid": args.grid,
        "length_scale": args.length_scale,
        "coefficient_clip": args.coefficient_clip,
    }
    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.test_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"x": torch.from_numpy(train_x), "y": torch.from_numpy(train_y), "metadata": metadata},
        args.train_output,
    )
    torch.save(
        {"x": torch.from_numpy(test_x), "y": torch.from_numpy(test_y), "metadata": metadata},
        args.test_output,
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": "python/darcy_data.py:sample_grf,darcy_fd_solve",
        "train_output": str(args.train_output.resolve()),
        "test_output": str(args.test_output.resolve()),
        "train_sha256": checksum(args.train_output),
        "test_sha256": checksum(args.test_output),
        "train_size": args.train_size,
        "test_size": args.test_size,
        "train_seed": args.train_seed,
        "test_seed": args.test_seed,
        "metadata": metadata,
        "train_coefficient_range": [float(train_x.min()), float(train_x.max())],
        "test_coefficient_range": [float(test_x.min()), float(test_x.max())],
        "train_solution_range": [float(train_y.min()), float(train_y.max())],
        "test_solution_range": [float(test_y.min()), float(test_y.max())],
        "elapsed_s": time.perf_counter() - started,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
