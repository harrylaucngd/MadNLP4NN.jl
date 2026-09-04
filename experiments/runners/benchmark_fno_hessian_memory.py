#!/usr/bin/env python3
"""Benchmark one exact-Hessian graph in a fresh JAX process."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import jax
import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def block(value):
    jax.block_until_ready(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=124)
    parser.add_argument("--latent-resolution", type=int, default=32)
    parser.add_argument("--full-resolution", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--regularization-weight", type=float, default=1e-4)
    args = parser.parse_args()

    project_python = Path(__file__).resolve().parents[2] / "python"
    sys.path.insert(0, str(project_python))
    from backend.device import DeviceManager
    from backend.model_loader import ModelLoader
    from evaluators.pdebench_darcy import PDEBenchDarcyLatentEvaluator

    if not jax.config.x64_enabled:
        raise RuntimeError("Set JAX_ENABLE_X64=True for the paper experiment")
    devices = jax.devices("gpu")
    if len(devices) != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, found {len(devices)}: {devices}"
        )

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    config = checkpoint.get("model_config") or {}
    parameter_tensor_elements = sum(value.numel() for value in state.values())
    parameter_real_scalars = sum(
        value.numel() * (2 if value.is_complex() else 1)
        for value in state.values()
    )
    layer_count = sum(
        key.startswith("conv") and key.endswith(".weights1") for key in state
    )
    widths = [int(state[f"conv{index}.weights1"].shape[1]) for index in range(layer_count)]
    modes1 = int(state["conv0.weights1"].shape[-2])
    modes2 = int(state["conv0.weights1"].shape[-1])

    if args.data.suffix == ".pt":
        data_payload = torch.load(args.data, map_location="cpu", weights_only=False)
        target = np.asarray(data_payload["y"][args.sample], dtype=np.float64)
    else:
        with h5py.File(args.data, "r") as handle:
            target = np.asarray(handle["tensor"][args.sample, 0], dtype=np.float64)
    if target.shape != (args.full_resolution, args.full_resolution):
        raise ValueError(f"Unexpected target shape {target.shape}")

    latent_size = args.latent_resolution**2
    x0 = np.full(latent_size, 0.4991392195224762, dtype=np.float64)
    manager = DeviceManager("gpu")
    forward_fn, params, loaded_config = ModelLoader(manager).load(str(args.model))
    evaluator = PDEBenchDarcyLatentEvaluator(
        forward_fn,
        params,
        target.reshape(-1),
        x0,
        reference_field=x0,
        full_resolution=args.full_resolution,
        regularization_weight=args.regularization_weight,
        device_manager=manager,
    )
    x = manager.put(x0)
    y = manager.put(np.zeros(0, dtype=np.float64))

    elapsed = []
    checksum = None
    coordinate_count = None
    for _ in range(args.repetitions):
        started = time.perf_counter()
        values = block(evaluator.evaluate_hess_coord_device(x, y, 1.0))
        elapsed.append(time.perf_counter() - started)
        coordinate_count = int(values.size)
        checksum = float(block(jax.numpy.sum(values)))

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "device": str(devices[0]),
        "device_kind": devices[0].device_kind,
        "checkpoint": str(args.model),
        "checkpoint_sha256": sha256(args.model),
        "trained": bool(checkpoint.get("provenance", {}).get("trained", True)),
        "model_config": loaded_config,
        "layer_count": layer_count,
        "widths": widths,
        "modes1": modes1,
        "modes2": modes2,
        "parameter_tensor_elements": int(parameter_tensor_elements),
        "parameter_real_scalars": int(parameter_real_scalars),
        "sample": args.sample,
        "latent_resolution": args.latent_resolution,
        "full_resolution": args.full_resolution,
        "hessian_coordinate_count": coordinate_count,
        "returned_hessian_bytes": coordinate_count * 8,
        "cold_compile_and_evaluate_s": elapsed[0],
        "warm_evaluate_s": elapsed[1:],
        "hessian_checksum": checksum,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
