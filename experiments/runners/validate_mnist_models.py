#!/usr/bin/env python3
"""Torch/JAX parity audit for the closest-prior MNIST classifiers."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor


def torch_model(checkpoint):
    config = checkpoint["model_config"]
    state = checkpoint["model_state_dict"]
    indices = sorted(
        int(key.split(".")[1])
        for key in state
        if key.startswith("network.") and key.endswith(".weight")
    )
    layers = []
    for position, index in enumerate(indices):
        weight = state[f"network.{index}.weight"]
        layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
        layer.weight.data.copy_(weight)
        layer.bias.data.copy_(state[f"network.{index}.bias"])
        layers.append(layer)
        if position + 1 < len(indices):
            layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers).eval(), config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=6)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    from backend.device import DeviceManager
    from backend.model_loader import ModelLoader

    dataset = torchvision.datasets.MNIST(
        root=args.data_dir, train=False, transform=ToTensor(), download=False
    )
    image, label = dataset[args.sample]
    inputs = image.reshape(-1).numpy()
    records = []
    for path in args.models:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model, config = torch_model(checkpoint)
        for precision in ("float32", "float64"):
            torch_dtype = torch.float32 if precision == "float32" else torch.float64
            jax_dtype = jnp.float32 if precision == "float32" else jnp.float64
            model = model.to(dtype=torch_dtype)
            with torch.no_grad():
                reference = torch.softmax(
                    model(torch.as_tensor(inputs, dtype=torch_dtype)), dim=0
                ).numpy()
            jax.config.update("jax_enable_x64", precision == "float64")
            forward, params, _ = ModelLoader(DeviceManager("gpu")).load(str(path))
            candidate = np.asarray(
                jax.jit(forward)(params, jnp.asarray(inputs, dtype=jax_dtype))
            )
            difference = candidate.astype(np.float64) - reference.astype(np.float64)
            record = {
                "model": path.name,
                "parameters": sum(value.numel() for value in checkpoint["model_state_dict"].values()),
                "precision": precision,
                "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
                "max_absolute": float(np.max(np.abs(difference))),
                "true_label": int(label),
                "predicted_label": int(np.argmax(candidate)),
                "predicted_probability": float(np.max(candidate)),
            }
            records.append(record)
            print(record)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample": args.sample,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
