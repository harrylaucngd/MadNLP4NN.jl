#!/usr/bin/env python3
"""Evaluate the released PDEBench Darcy FNO on the official held-out split."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pdebench-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(args.pdebench_repo.resolve()))
    from pdebench.models.fno.fno import FNO2d

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    model = FNO2d(num_channels=1, modes1=12, modes2=12, width=20, initial_step=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    torch.set_float32_matmul_precision("highest")

    with h5py.File(args.data, "r") as handle:
        sample_count = handle["nu"].shape[0]
        test_start = int(0.9 * sample_count)
        coefficient = np.asarray(handle["nu"][test_start:])
        pressure = np.asarray(handle["tensor"][test_start:, 0])
        coordinate_x = np.asarray(handle["x-coordinate"])
        coordinate_y = np.asarray(handle["y-coordinate"])
    grid_x, grid_y = np.meshgrid(coordinate_x, coordinate_y, indexing="ij")
    grid = torch.as_tensor(
        np.stack([grid_x, grid_y], axis=-1)[None], device=device
    )

    errors = []
    with torch.no_grad():
        for start in range(0, coefficient.shape[0], args.batch_size):
            stop = min(start + args.batch_size, coefficient.shape[0])
            batch = torch.as_tensor(
                coefficient[start:stop, ..., None], device=device
            )
            prediction = model(batch, grid.expand(stop - start, -1, -1, -1))
            prediction = prediction.cpu().numpy().reshape(stop - start, 128, 128)
            target = pressure[start:stop]
            numerator = np.linalg.norm((prediction - target).reshape(stop - start, -1), axis=1)
            denominator = np.linalg.norm(target.reshape(stop - start, -1), axis=1)
            errors.extend((numerator / denominator).tolist())
            print(f"{test_start + stop}/{sample_count}", flush=True)

    errors_array = np.asarray(errors)
    quantiles = {str(q): float(np.quantile(errors_array, q)) for q in (0, 0.25, 0.5, 0.75, 1)}
    representatives = {}
    for label, q in (("q25", 0.25), ("median", 0.5), ("q75", 0.75)):
        target = np.quantile(errors_array, q)
        local_index = int(np.argmin(np.abs(errors_array - target)))
        representatives[label] = {
            "sample": test_start + local_index,
            "relative_l2": float(errors_array[local_index]),
        }

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "device": str(device),
        "test_start": test_start,
        "test_count": len(errors),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_loss": checkpoint.get("loss"),
        "quantiles": quantiles,
        "mean": float(np.mean(errors_array)),
        "representatives": representatives,
        "per_sample": [
            {"sample": test_start + index, "relative_l2": float(value)}
            for index, value in enumerate(errors_array)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"quantiles": quantiles, "representatives": representatives}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
