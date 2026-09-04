#!/usr/bin/env python3
"""Join matched CPU/GPU oracle microbenchmarks and emit a CSV regime table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def median_ms(oracle: dict, section: str = "device") -> float:
    return 1e3 * oracle[section]["median_s"]


def main() -> None:
    args = parse_args()
    cpu = load(args.cpu)
    gpu = load(args.gpu)
    cpu_by_name = {record["name"]: record for record in cpu["results"]}
    gpu_by_name = {record["name"]: record for record in gpu["results"]}
    if cpu_by_name.keys() != gpu_by_name.keys():
        raise ValueError("CPU and GPU runs do not contain the same configurations")

    rows = []
    for name, gpu_record in gpu_by_name.items():
        cpu_record = cpu_by_name[name]
        cpu_oracles = cpu_record["oracles"]
        gpu_oracles = gpu_record["oracles"]
        for oracle_name in sorted(cpu_oracles.keys() & gpu_oracles.keys()):
            cpu_oracle = cpu_oracles[oracle_name]
            gpu_oracle = gpu_oracles[oracle_name]
            cpu_ms = median_ms(cpu_oracle)
            gpu_ms = median_ms(gpu_oracle)
            transfer_ms = median_ms(gpu_oracle, "device_to_host")
            rows.append(
                {
                    "name": name,
                    "n": gpu_record["n"],
                    "m": gpu_record["m"],
                    "parameter_count": gpu_record["parameter_count"],
                    "oracle": oracle_name,
                    "output_bytes": gpu_oracle["output_bytes"],
                    "cpu_device_ms": cpu_ms,
                    "gpu_device_ms": gpu_ms,
                    "gpu_device_to_host_ms": transfer_ms,
                    "gpu_host_staged_ms": gpu_ms + transfer_ms,
                    "gpu_compute_speedup": cpu_ms / gpu_ms,
                    "gpu_host_staged_speedup": cpu_ms / (gpu_ms + transfer_ms),
                    "cpu_cold_ms": 1e3 * cpu_oracle["cold_s"],
                    "gpu_cold_ms": 1e3 * gpu_oracle["cold_s"],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
