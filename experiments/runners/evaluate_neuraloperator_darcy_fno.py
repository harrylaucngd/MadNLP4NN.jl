#!/usr/bin/env python3
"""Evaluate every held-out sample and select fixed FNO-quality strata."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    from models.fno2d_torch import FNO2d

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    model = FNO2d(
        modes1=config["modes1"],
        modes2=config["modes2"],
        width=config["width"],
        padding=config["padding"],
        layer_count=config["layer_count"],
    ).cuda()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    payload = torch.load(args.data, map_location="cpu", weights_only=False)
    loader = DataLoader(
        TensorDataset(payload["x"].float(), payload["y"].float()),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    records = []
    offset = 0
    with torch.no_grad():
        for coefficient, target in loader:
            coefficient = coefficient.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            prediction = model(coefficient)
            errors = torch.linalg.vector_norm(
                (prediction - target).flatten(1), dim=1
            ) / torch.linalg.vector_norm(target.flatten(1), dim=1)
            means = coefficient.mean(dim=(1, 2))
            for local, (error, mean) in enumerate(zip(errors.cpu(), means.cpu())):
                records.append(
                    {
                        "sample": offset + local,
                        "relative_l2": float(error),
                        "coefficient_mean": float(mean),
                    }
                )
            offset += coefficient.shape[0]

    values = np.asarray([record["relative_l2"] for record in records])
    strata = {}
    for name, quantile in (("q25", 0.25), ("median", 0.5), ("q75", 0.75)):
        value = float(np.quantile(values, quantile))
        sample = int(np.argmin(np.abs(values - value)))
        strata[name] = {"quantile": quantile, "value": value, **records[sample]}
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "summary": {
            "mean": float(np.mean(values)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "max": float(np.max(values)),
        },
        "strata": strata,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"summary": result["summary"], "strata": strata}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
