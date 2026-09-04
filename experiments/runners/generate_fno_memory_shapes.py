#!/usr/bin/env python3
"""Generate parameter-matched FNO shapes for an AD-memory experiment.

The checkpoints are deliberately untrained: only the static computation graph
is used to measure exact-Hessian memory.  Prediction quality and optimizer
outcomes are outside the scope of this mechanistic ablation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


PROFILES = {
    # Matches the 1,188,353-parameter replay FNO behind the 33.54-GiB result.
    "replay": {
        "target": 1188353,
        "resolution": 64,
        "coordinate_scheme": "left_endpoint",
        "shapes": (
            {"layer_count": 1, "width": 64},
            {"layer_count": 2, "width": 45},
            {"layer_count": 3, "width": 37},
            {"layer_count": 4, "width": 32},
            {"layer_count": 6, "width": 26},
        ),
    },
    # Matches the 465,377-parameter public PDEBench checkpoint.
    "public": {
        "target": 465377,
        "resolution": 128,
        "coordinate_scheme": "cell_center",
        "shapes": (
            {"layer_count": 1, "width": 40},
            {"layer_count": 2, "width": 28},
            {"layer_count": 3, "width": 23},
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_counts(model):
    tensor_elements = sum(parameter.numel() for parameter in model.parameters())
    real_scalars = sum(
        parameter.numel() * (2 if parameter.is_complex() else 1)
        for parameter in model.parameters()
    )
    return int(tensor_elements), int(real_scalars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="replay")
    parser.add_argument("--target-parameters", type=int)
    args = parser.parse_args()

    project_python = Path(__file__).resolve().parents[2] / "python"
    sys.path.insert(0, str(project_python))
    from models.fno2d_torch import FNO2d

    profile = PROFILES[args.profile]
    target_parameters = args.target_parameters or profile["target"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, shape in enumerate(profile["shapes"]):
        torch.manual_seed(args.seed + index)
        model = FNO2d(
            modes1=args.modes,
            modes2=args.modes,
            width=shape["width"],
            layer_count=shape["layer_count"],
        )
        tensor_elements, real_scalars = parameter_counts(model)
        name = f"fno_l{shape['layer_count']}_w{shape['width']}_m{args.modes}.pt"
        path = args.output_dir / name
        config = {
            "model_type": "pdebench_fno2d",
            "modes1": args.modes,
            "modes2": args.modes,
            "width": shape["width"],
            "padding": 2,
            "layer_count": shape["layer_count"],
            "coordinate_scheme": profile["coordinate_scheme"],
            "training_resolution": profile["resolution"],
        }
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": config,
                "provenance": {
                    "purpose": "static-graph exact-Hessian memory ablation",
                    "trained": False,
                    "seed": args.seed + index,
                },
            },
            path,
        )
        records.append(
            {
                **shape,
                "modes1": args.modes,
                "modes2": args.modes,
                "checkpoint": str(path),
                "sha256": sha256(path),
                "parameter_tensor_elements": tensor_elements,
                "parameter_real_scalars": real_scalars,
                "target_parameter_tensor_elements": target_parameters,
                "relative_parameter_mismatch": (
                    tensor_elements - target_parameters
                )
                / target_parameters,
            }
        )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "parameter-matched FNO topology ablation for AD memory",
        "trained": False,
        "profile": args.profile,
        "seed": args.seed,
        "fixed_modes": args.modes,
        "target_parameter_tensor_elements": target_parameters,
        "shapes": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
