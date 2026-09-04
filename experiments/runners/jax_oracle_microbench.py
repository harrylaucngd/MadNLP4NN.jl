#!/usr/bin/env python3
"""Measure neural NLP oracle costs without involving an optimization solver.

This benchmark separates JAX compilation, steady device execution, and
device-to-host transfer for the function, gradient, constraint Jacobian, and
Lagrangian Hessian required by a smooth NLP solver.

The network is intentionally synthetic: it is a systems microbenchmark used to
vary input dimension ``n``, constraint/output dimension ``m``, and network
parameter count ``p`` independently. It is not an application benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-hessian",
        action="store_true",
        help="Measure all oracles except the explicit dense Hessian.",
    )
    return parser.parse_args()


def block_until_ready(value: Any) -> Any:
    return jax.block_until_ready(value)


def median_iqr(values: list[float]) -> dict[str, float]:
    values_sorted = sorted(values)
    if len(values_sorted) == 1:
        q1 = q3 = values_sorted[0]
    else:
        quartiles = statistics.quantiles(values_sorted, n=4, method="inclusive")
        q1, q3 = quartiles[0], quartiles[2]
    return {
        "median_s": statistics.median(values_sorted),
        "q1_s": q1,
        "q3_s": q3,
        "min_s": min(values_sorted),
        "max_s": max(values_sorted),
    }


def timed_call(fn: Callable[..., Any], *args: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    value = fn(*args)
    block_until_ready(value)
    return value, time.perf_counter() - start


def benchmark_callable(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    repetitions: int,
) -> dict[str, Any]:
    value, cold_s = timed_call(fn, *args)
    device_times = []
    transfer_times = []
    for _ in range(repetitions):
        value, elapsed = timed_call(fn, *args)
        device_times.append(elapsed)

        # The result is complete on the device at this boundary, so this timing
        # contains only materialization on the host (plus Python dispatch).
        start = time.perf_counter()
        host_value = np.asarray(value)
        # Touch the buffer so that a lazy host view cannot escape the interval.
        _ = host_value.reshape(-1)[0] if host_value.size else None
        transfer_times.append(time.perf_counter() - start)

    return {
        "cold_s": cold_s,
        "device": median_iqr(device_times),
        "device_to_host": median_iqr(transfer_times),
        "shape": list(value.shape),
        "output_bytes": int(value.size * value.dtype.itemsize),
        "dtype": str(value.dtype),
    }


def make_mlp_oracles(
    config: dict[str, Any],
    device: jax.Device,
    seed: int,
) -> tuple[dict[str, Callable[..., Any]], dict[str, Any]]:
    n = int(config["n"])
    m = int(config["m"])
    width = int(config["width"])
    depth = int(config["depth"])
    if min(n, m, width, depth) <= 0:
        raise ValueError(f"All dimensions must be positive: {config}")

    rng = np.random.default_rng(seed)
    dimensions = [n] + [width] * depth + [m]
    params = []
    parameter_count = 0
    for d_in, d_out in zip(dimensions[:-1], dimensions[1:]):
        # Xavier-style scaling keeps the synthetic derivatives away from both
        # saturation and explosion across the tested widths.
        weight = rng.normal(size=(d_out, d_in)) / np.sqrt(d_in)
        bias = rng.normal(scale=0.05, size=(d_out,))
        params.append(
            (
                jax.device_put(jnp.asarray(weight), device),
                jax.device_put(jnp.asarray(bias), device),
            )
        )
        parameter_count += d_out * d_in + d_out

    target = jax.device_put(jnp.linspace(-0.2, 0.2, m), device)
    threshold = jax.device_put(jnp.full((m,), 0.25), device)
    x = jax.device_put(jnp.linspace(-0.5, 0.5, n), device)
    multipliers = jax.device_put(jnp.linspace(0.1, 1.0, m), device)
    objective_weight = jax.device_put(jnp.asarray(1.0), device)

    def network(x_value: jax.Array) -> jax.Array:
        activation = x_value
        for index, (weight, bias) in enumerate(params):
            activation = weight @ activation + bias
            if index < len(params) - 1:
                activation = jnp.tanh(activation)
        return activation

    def objective(x_value: jax.Array) -> jax.Array:
        residual = network(x_value) - target
        return 0.5 * jnp.dot(residual, residual) + 5e-4 * jnp.dot(x_value, x_value)

    def constraints(x_value: jax.Array) -> jax.Array:
        return network(x_value) - threshold

    def lagrangian(
        x_value: jax.Array,
        y_value: jax.Array,
        weight: jax.Array,
    ) -> jax.Array:
        return weight * objective(x_value) + jnp.dot(y_value, constraints(x_value))

    oracles = {
        "network": jax.jit(network, device=device),
        "objective": jax.jit(objective, device=device),
        "gradient": jax.jit(jax.grad(objective), device=device),
        "constraints": jax.jit(constraints, device=device),
        "jacobian": jax.jit(jax.jacrev(constraints), device=device),
        "hessian_lagrangian": jax.jit(
            jax.hessian(lagrangian, argnums=0),
            device=device,
        ),
    }
    metadata = {
        "n": n,
        "m": m,
        "width": width,
        "depth": depth,
        "parameter_count": parameter_count,
        "inputs": {
            "x": x,
            "multipliers": multipliers,
            "objective_weight": objective_weight,
        },
    }
    return oracles, metadata


def nvidia_smi() -> str | None:
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")

    requested_platform = "gpu" if args.device == "gpu" else "cpu"
    devices = jax.devices(requested_platform)
    if not devices:
        raise RuntimeError(f"No JAX {requested_platform} device is available")
    device = devices[0]

    configs = json.loads(args.config.read_text())
    results = []
    for index, config in enumerate(configs):
        print(
            f"[{index + 1}/{len(configs)}] {config['name']}: "
            f"n={config['n']} m={config['m']} width={config['width']} "
            f"depth={config['depth']}",
            flush=True,
        )
        oracles, metadata = make_mlp_oracles(config, device, args.seed + index)
        inputs = metadata.pop("inputs")
        oracle_results = {
            "network": benchmark_callable(
                oracles["network"], (inputs["x"],), args.repetitions
            ),
            "objective": benchmark_callable(
                oracles["objective"], (inputs["x"],), args.repetitions
            ),
            "gradient": benchmark_callable(
                oracles["gradient"], (inputs["x"],), args.repetitions
            ),
            "constraints": benchmark_callable(
                oracles["constraints"], (inputs["x"],), args.repetitions
            ),
            "jacobian": benchmark_callable(
                oracles["jacobian"], (inputs["x"],), args.repetitions
            ),
        }
        if not args.skip_hessian:
            oracle_results["hessian_lagrangian"] = benchmark_callable(
                oracles["hessian_lagrangian"],
                (
                    inputs["x"],
                    inputs["multipliers"],
                    inputs["objective_weight"],
                ),
                args.repetitions,
            )
        results.append(
            {
                "name": config["name"],
                **metadata,
                "oracles": oracle_results,
                "device_memory_stats": device.memory_stats(),
            }
        )

        # Do not carry compiled programs or parameter buffers into the next
        # configuration's memory measurement.
        del oracles, inputs
        jax.clear_caches()

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "config_path": str(args.config.resolve()),
        "config": configs,
        "repetitions": args.repetitions,
        "seed": args.seed,
        "provenance": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "jax": jax.__version__,
            "jaxlib": getattr(jax.lib, "__version__", "unknown"),
            "numpy": np.__version__,
            "device": str(device),
            "jax_devices": [str(item) for item in jax.devices()],
            "jax_enable_x64": bool(jax.config.x64_enabled),
            "xla_python_client_preallocate": os.environ.get(
                "XLA_PYTHON_CLIENT_PREALLOCATE"
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_smi": nvidia_smi(),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
