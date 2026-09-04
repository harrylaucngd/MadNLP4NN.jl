#!/usr/bin/env python3
"""Compact local monitor for MadNLP4NN experiment processes and artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def command(args):
    return subprocess.run(args, check=False, capture_output=True, text=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("output/runs"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    ps_output = command(
        [
            "ps", "-u", str(os.getuid()), "-o",
            "pid=,etime=,pcpu=,pmem=,rss=,stat=,cmd=", "--sort=-pcpu",
        ]
    )
    processes = [
        line.strip()
        for line in ps_output.splitlines()
        if "experiments/runners/" in line and "monitor.py" not in line
    ]
    gpu = command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader",
        ]
    )
    artifacts = []
    if args.runs.exists():
        for path in sorted(args.runs.glob("*"), key=lambda item: item.stat().st_mtime, reverse=True)[:20]:
            stat = path.stat()
            artifacts.append(
                {
                    "path": str(path),
                    "size": stat.st_size,
                    "age_s": time.time() - stat.st_mtime,
                }
            )
    payload = {"gpu": gpu, "processes": processes, "recent_artifacts": artifacts}
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    print("GPU:", gpu or "unavailable")
    print("\nExperiment processes:")
    print("\n".join(processes) if processes else "none")
    print("\nRecent artifacts:")
    for artifact in artifacts:
        print(
            f"{artifact['age_s']:8.1f}s  {artifact['size']:10d}  {artifact['path']}"
        )


if __name__ == "__main__":
    main()
