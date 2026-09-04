#!/usr/bin/env python3
"""Train a self-contained FNO on the public NeuralOperator Darcy split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def checksum(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_l2(prediction, target):
    residual = torch.linalg.vector_norm((prediction - target).flatten(1), dim=1)
    scale = torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(1e-12)
    return torch.mean(residual / scale)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    for coefficient, solution in loader:
        coefficient = coefficient.to(device, non_blocking=True)
        solution = solution.to(device, non_blocking=True)
        loss = torch.linalg.vector_norm(
            (model(coefficient) - solution).flatten(1), dim=1
        ) / torch.linalg.vector_norm(solution.flatten(1), dim=1).clamp_min(1e-12)
        losses.append(loss.cpu())
    values = torch.cat(losses).numpy()
    return {
        "mean_relative_l2": float(np.mean(values)),
        "median_relative_l2": float(np.median(values)),
        "q25_relative_l2": float(np.quantile(values, 0.25)),
        "q75_relative_l2": float(np.quantile(values, 0.75)),
        "max_relative_l2": float(np.max(values)),
    }


def main():
    args = parse_args()
    project_python = Path(__file__).resolve().parents[2] / "python"
    sys.path.insert(0, str(project_python))
    from models.fno2d_torch import FNO2d

    if not torch.cuda.is_available():
        raise RuntimeError("This registered training runner requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")

    train_payload = torch.load(args.train_data, map_location="cpu", weights_only=False)
    test_payload = torch.load(args.test_data, map_location="cpu", weights_only=False)
    coefficient = train_payload["x"].float()
    solution = train_payload["y"].float()
    split = coefficient.shape[0] - args.validation_size
    if split <= 0:
        raise ValueError("validation-size must leave at least one training sample")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(coefficient[:split], solution[:split]),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        TensorDataset(coefficient[split:], solution[split:]),
        batch_size=2 * args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_payload["x"].float(), test_payload["y"].float()),
        batch_size=2 * args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = FNO2d(modes1=args.modes, modes2=args.modes, width=args.width).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    best = float("inf")
    history = []
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        training_losses = []
        for coefficient_batch, solution_batch in train_loader:
            coefficient_batch = coefficient_batch.to(device, non_blocking=True)
            solution_batch = solution_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = relative_l2(model(coefficient_batch), solution_batch)
            loss.backward()
            optimizer.step()
            training_losses.append(float(loss.detach()))
        scheduler.step()
        validation = evaluate(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "train_mean_relative_l2": float(np.mean(training_losses)),
            "validation_mean_relative_l2": validation["mean_relative_l2"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_s": time.perf_counter() - epoch_started,
        }
        history.append(record)
        print(record, flush=True)
        if record["validation_mean_relative_l2"] < best:
            best = record["validation_mean_relative_l2"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "model_type": "pdebench_fno2d",
                        "modes1": args.modes,
                        "modes2": args.modes,
                        "width": args.width,
                        "padding": 2,
                        "layer_count": 4,
                        "coordinate_scheme": "left_endpoint",
                        "training_resolution": int(coefficient.shape[-1]),
                    },
                    "training": {
                        "epoch": epoch,
                        "seed": args.seed,
                        "validation_mean_relative_l2": best,
                    },
                },
                args.checkpoint,
            )

    selected = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state_dict"])
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_data": str(args.train_data.resolve()),
        "test_data": str(args.test_data.resolve()),
        "train_data_sha256": checksum(args.train_data),
        "test_data_sha256": checksum(args.test_data),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checksum(args.checkpoint),
        "config": vars(args) | {"train_data": str(args.train_data), "test_data": str(args.test_data), "checkpoint": str(args.checkpoint), "output": str(args.output)},
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "selected_epoch": selected["training"]["epoch"],
        "validation": evaluate(model, validation_loader, device),
        "test": evaluate(model, test_loader, device),
        "total_elapsed_s": time.perf_counter() - started,
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
