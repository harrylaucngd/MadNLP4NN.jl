#!/usr/bin/env python3
"""Generate one shared, deterministic multi-start registry for all solvers."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolutions", default="8,16,32")
    parser.add_argument("--number", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--center", type=float, default=0.4991392195224762)
    parser.add_argument("--amplitude", type=float, default=0.22)
    parser.add_argument("--lower", type=float, default=0.0)
    parser.add_argument("--upper", type=float, default=1.0)
    args = parser.parse_args()
    if args.number < 1:
        raise ValueError("number must be positive")

    rng = np.random.default_rng(args.seed)
    starts = {}
    for resolution in map(int, args.resolutions.split(",")):
        entries = []
        for index in range(args.number):
            if index == 0:
                values = np.full((resolution, resolution), args.center)
                identifier = "constant"
            else:
                noise = rng.standard_normal((resolution, resolution))
                smooth = gaussian_filter(
                    noise, sigma=max(0.75, resolution / 8), mode="reflect"
                )
                smooth = (smooth - smooth.mean()) / max(smooth.std(), 1e-12)
                values = np.clip(
                    args.center + args.amplitude * smooth,
                    args.lower + 0.02 * (args.upper - args.lower),
                    args.upper - 0.02 * (args.upper - args.lower),
                )
                identifier = f"smooth_random_{index}"
            flat = np.asarray(values, dtype=np.float64).reshape(-1)
            entries.append(
                {
                    "id": identifier,
                    "sha256": hashlib.sha256(flat.tobytes()).hexdigest(),
                    "mean": float(np.mean(flat)),
                    "std": float(np.std(flat)),
                    "min": float(np.min(flat)),
                    "max": float(np.max(flat)),
                    "values": flat.tolist(),
                }
            )
        starts[str(resolution)] = entries

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": "numpy.PCG64 + scipy.ndimage.gaussian_filter(reflect)",
        "seed": args.seed,
        "number": args.number,
        "center": args.center,
        "amplitude": args.amplitude,
        "lower": args.lower,
        "upper": args.upper,
        "starts": starts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
