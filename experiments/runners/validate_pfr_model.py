#!/usr/bin/env python3
"""PyTorch/JAX parity audit for the published case-study-1 PFR surrogate."""

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
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    sys.path.insert(0, str(args.upstream_dir.resolve()))
    from models.pfr_pinn import build_pfr_pinn_jax
    from SurPINNs_PDE_FullDiscrete import PIDNN

    state = torch.load(args.model, map_location="cpu", weights_only=False)
    lower = state["lb"].numpy()
    upper = state["ub"].numpy()
    rng = np.random.default_rng(args.seed)
    inputs = rng.uniform(lower, upper, size=(100, lower.size))
    records = []
    for precision in ("float32", "float64"):
        dtype_torch = torch.float32 if precision == "float32" else torch.float64
        dtype_jax = jnp.float32 if precision == "float32" else jnp.float64
        model = PIDNN(12, 10, 24, 6, lower, upper, activ="Tanh")
        model.load_state_dict(copy.deepcopy(state))
        model = model.to(dtype=dtype_torch).eval()
        with torch.no_grad():
            torch_output = model(torch.as_tensor(inputs, dtype=dtype_torch)).numpy()

        jax.config.update("jax_enable_x64", precision == "float64")
        forward, params = build_pfr_pinn_jax(state, {"dtype": dtype_jax})
        jax_output = np.asarray(
            jax.jit(jax.vmap(lambda x: forward(params, x)))(
                jnp.asarray(inputs, dtype=dtype_jax)
            )
        )
        difference = jax_output.astype(np.float64) - torch_output.astype(np.float64)
        records.append(
            {
                "precision": precision,
                "relative_l2": float(
                    np.linalg.norm(difference) / np.linalg.norm(torch_output)
                ),
                "max_absolute": float(np.max(np.abs(difference))),
            }
        )
        print(records[-1])

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
