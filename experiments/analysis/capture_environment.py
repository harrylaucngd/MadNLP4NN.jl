#!/usr/bin/env python3
"""Capture the exact node and software environment used by the run suite."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as error:
        return f"ERROR: {error}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packages = {}
    for name in (
        "jax",
        "jaxlib",
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "h5py",
    ):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except Exception as error:
            packages[name] = f"ERROR: {error}"
    julia_executable = os.environ.get("JULIA_EXE") or shutil.which("julia")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": packages,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "JULIA_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
            )
        },
        "nvidia_smi": command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "lscpu": command(["lscpu"]),
        "julia_version": (
            command([julia_executable, "--version"])
            if julia_executable
            else "ERROR: Julia not found; set JULIA_EXE"
        ),
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
