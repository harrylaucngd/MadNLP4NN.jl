#!/usr/bin/env python3
"""Validate the JAX PDEBench FNO against the official PyTorch implementation."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import torch

jax.config.update("jax_default_matmul_precision", "highest")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pdebench-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=9999)
    return parser.parse_args()


def errors(reference, candidate):
    difference = candidate - reference
    return {
        "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
        "max_absolute": float(np.max(np.abs(difference))),
        "reference_norm": float(np.linalg.norm(reference)),
    }


def main():
    args = parse_args()
    project_python = Path(__file__).resolve().parents[2] / "python"
    sys.path.insert(0, str(project_python))
    sys.path.insert(0, str(args.pdebench_repo.resolve()))
    from models.pdebench_fno import build_pdebench_fno2d_jax
    from pdebench.models.fno.fno import FNO2d

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    with h5py.File(args.data, "r") as handle:
        coefficient_full = np.asarray(handle["nu"][args.sample])
        pressure_full = np.asarray(handle["tensor"][args.sample, 0])
        coordinate_x_full = np.asarray(handle["x-coordinate"])
        coordinate_y_full = np.asarray(handle["y-coordinate"])

    records = []
    for resolution in (32, 64, 128):
        stride = coefficient_full.shape[0] // resolution
        coefficient = coefficient_full[::stride, ::stride]
        pressure = pressure_full[::stride, ::stride]
        coordinate_x = coordinate_x_full[::stride]
        coordinate_y = coordinate_y_full[::stride]
        grid_x, grid_y = np.meshgrid(coordinate_x, coordinate_y, indexing="ij")
        grid = np.stack([grid_x, grid_y], axis=-1)[None]

        for precision in ("float32", "float64"):
            dtype_np = np.float32 if precision == "float32" else np.float64
            dtype_jax = jnp.float32 if precision == "float32" else jnp.float64
            torch_model = FNO2d(
                num_channels=1, modes1=12, modes2=12, width=20, initial_step=1
            )
            torch_model.load_state_dict(copy.deepcopy(state))
            if precision == "float32":
                torch_model = torch_model.float()
            else:
                for parameter in torch_model.parameters():
                    parameter.data = parameter.data.to(
                        torch.complex128 if parameter.is_complex() else torch.float64
                    )
            torch_model.eval()
            with torch.no_grad():
                torch_output = torch_model(
                    torch.as_tensor(coefficient[None, ..., None], dtype=getattr(torch, precision)),
                    torch.as_tensor(grid, dtype=getattr(torch, precision)),
                ).cpu().numpy().reshape(-1)

            jax.config.update("jax_enable_x64", precision == "float64")
            forward, params = build_pdebench_fno2d_jax(
                state,
                {
                    "dtype": dtype_jax,
                    "padding": 2,
                    "coordinate_x": coordinate_x.astype(dtype_np),
                    "coordinate_y": coordinate_y.astype(dtype_np),
                },
            )
            jax_output = np.asarray(
                jax.jit(forward)(params, jnp.asarray(coefficient.reshape(-1), dtype=dtype_jax))
            )
            record = {
                "resolution": resolution,
                "precision": precision,
                **errors(torch_output.astype(np.float64), jax_output.astype(np.float64)),
                "surrogate_relative_l2_to_truth": float(
                    np.linalg.norm(jax_output.reshape(pressure.shape) - pressure)
                    / np.linalg.norm(pressure)
                ),
            }
            records.append(record)
            print(record, flush=True)

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "sample": args.sample,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_loss": checkpoint.get("loss"),
        "torch": torch.__version__,
        "jax": jax.__version__,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
