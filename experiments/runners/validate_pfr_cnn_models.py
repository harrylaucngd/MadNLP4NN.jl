#!/usr/bin/env python3
"""Torch/JAX parity audit for published PFR cases 2 and 3 CNNs."""

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


def infer_config(state):
    count = sum(
        key.startswith("cnn.cnn_layers.") and key.endswith(".weight")
        for key in state
    )
    hidden = int(state["cnn.cnn_layers.0.weight"].shape[0])
    kernel = int(state["cnn.cnn_layers.0.weight"].shape[2])
    final_length = int(state["cnn.linear.weight"].shape[1] // hidden)
    n_fe = final_length + count * (kernel - 1)
    output_channels = int(state["cnn.linear.bias"].numel() // n_fe)
    input_channels = int(state["cnn.cnn_layers.0.weight"].shape[1])
    return {
        "n_fe": n_fe,
        "output_channels": output_channels,
        "control_channels": input_channels - output_channels,
        "layer_count": count,
    }


def torch_forward(state, config, inputs, precision):
    real_dtype = torch.float32 if precision == "float32" else torch.float64
    state = copy.deepcopy(state)
    n_fe = config["n_fe"]
    output_channels = config["output_channels"]
    state_size = n_fe * output_channels
    value = torch.as_tensor(inputs[:, :state_size], dtype=real_dtype).reshape(
        -1, output_channels, n_fe, 1
    )
    controls = torch.as_tensor(inputs[:, state_size:], dtype=real_dtype).reshape(
        -1, config["control_channels"], 1, 1
    ).expand(-1, -1, n_fe, -1)
    value = torch.cat((value, controls), dim=1)
    lower = state["lb"].to(real_dtype)
    upper = state["ub"].to(real_dtype)
    value = 2 * (value - lower) / (upper - lower) - 1
    for index in range(config["layer_count"]):
        value = torch.tanh(
            torch.nn.functional.conv2d(
                value,
                state[f"cnn.cnn_layers.{index}.weight"].to(real_dtype),
                state[f"cnn.cnn_layers.{index}.bias"].to(real_dtype),
            )
        )
    value = torch.nn.functional.linear(
        value.flatten(1),
        state["cnn.linear.weight"].to(real_dtype),
        state["cnn.linear.bias"].to(real_dtype),
    ).reshape(-1, output_channels, n_fe, 1)
    return (
        (upper[:, :output_channels] - lower[:, :output_channels])
        * (value + 1)
        / 2
        + lower[:, :output_channels]
    ).flatten(1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    from backend.device import DeviceManager
    from backend.model_loader import ModelLoader
    from models.pfr_cnn import build_pfr_cnn_jax

    rng = np.random.default_rng(args.seed)
    records = []
    for path in args.models:
        state = torch.load(path, map_location="cpu", weights_only=False)
        config = infer_config(state)
        lower = np.asarray(state["lb"])[0]
        upper = np.asarray(state["ub"])[0]
        output_channels = config["output_channels"]
        n_fe = config["n_fe"]
        state_lower = np.broadcast_to(lower[:output_channels], (output_channels, n_fe, 1))
        state_upper = np.broadcast_to(upper[:output_channels], (output_channels, n_fe, 1))
        control_lower = lower[output_channels:, 0, 0]
        control_upper = upper[output_channels:, 0, 0]
        inputs = np.concatenate(
            (
                rng.uniform(state_lower, state_upper, size=(16, output_channels, n_fe, 1)).reshape(16, -1),
                rng.uniform(control_lower, control_upper, size=(16, config["control_channels"])),
            ),
            axis=1,
        )
        _, _, detected = ModelLoader(DeviceManager("gpu")).load(str(path))
        for precision in ("float32", "float64"):
            reference = torch_forward(state, config, inputs, precision)
            jax.config.update("jax_enable_x64", precision == "float64")
            dtype = jnp.float32 if precision == "float32" else jnp.float64
            forward, params = build_pfr_cnn_jax(
                state, detected | {"dtype": dtype}
            )
            params = jax.device_put(params, jax.devices("gpu")[0])
            candidate = np.asarray(
                jax.jit(jax.vmap(lambda value: forward(params, value)))(
                    jnp.asarray(inputs, dtype=dtype)
                )
            )
            difference = candidate.astype(np.float64) - reference.astype(np.float64)
            record = {
                "model": path.name,
                "precision": precision,
                "config": detected,
                "parameter_count": sum(value.numel() for value in state.values()),
                "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
                "max_absolute": float(np.max(np.abs(difference))),
            }
            records.append(record)
            print(record)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
