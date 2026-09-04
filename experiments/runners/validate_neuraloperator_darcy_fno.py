#!/usr/bin/env python3
"""Audit Torch/JAX parity for a trained NeuralOperator-Darcy FNO checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

jax.config.update("jax_default_matmul_precision", "highest")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    from models.fno2d_torch import FNO2d
    from models.pdebench_fno import build_pdebench_fno2d_jax

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    state = checkpoint["model_state_dict"]
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    coefficient = data["x"][args.sample].numpy()
    target = data["y"][args.sample].numpy()
    records = []
    for precision in ("float32", "float64"):
        torch_dtype = torch.float32 if precision == "float32" else torch.float64
        jax_dtype = jnp.float32 if precision == "float32" else jnp.float64
        model = FNO2d(
            modes1=config["modes1"],
            modes2=config["modes2"],
            width=config["width"],
            padding=config["padding"],
            layer_count=config["layer_count"],
        )
        model.load_state_dict(copy.deepcopy(state))
        for parameter in model.parameters():
            parameter.data = parameter.data.to(
                (torch.complex128 if precision == "float64" else torch.complex64)
                if parameter.is_complex()
                else torch_dtype
            )
        model.eval()
        with torch.no_grad():
            torch_output = model(torch.as_tensor(coefficient[None], dtype=torch_dtype))[0]

        jax.config.update("jax_enable_x64", precision == "float64")
        forward, params = build_pdebench_fno2d_jax(
            state, config | {"dtype": jax_dtype}
        )
        jax_output = np.asarray(
            jax.jit(forward)(params, jnp.asarray(coefficient.reshape(-1), dtype=jax_dtype))
        ).reshape(target.shape)
        torch_output = torch_output.numpy()
        difference = jax_output.astype(np.float64) - torch_output.astype(np.float64)
        record = {
            "precision": precision,
            "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(torch_output)),
            "max_absolute": float(np.max(np.abs(difference))),
            "surrogate_relative_l2_to_target": float(
                np.linalg.norm(jax_output - target) / np.linalg.norm(target)
            ),
        }
        records.append(record)
        print(record)

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "sample": args.sample,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
