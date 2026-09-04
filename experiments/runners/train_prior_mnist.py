#!/usr/bin/env python3
"""Train the exact smooth MNIST architecture from Parker et al.'s repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data = torchvision.datasets.MNIST(
        root=args.data_dir, train=True, transform=ToTensor(), download=True
    )
    test_data = torchvision.datasets.MNIST(
        root=args.data_dir, train=False, transform=ToTensor(), download=True
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    layers = [torch.nn.Linear(784, args.nodes)]
    for _ in range(args.layers):
        layers.extend((torch.nn.Tanh(), torch.nn.Linear(args.nodes, args.nodes)))
    layers.extend((torch.nn.Tanh(), torch.nn.Linear(args.nodes, 10)))
    network = torch.nn.Sequential(*layers).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        network.train()
        losses = []
        for images, labels in loader:
            images = images.to(device, non_blocking=True).reshape(-1, 784)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(network(images), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch, "mean_loss": float(np.mean(losses))})
        print(history[-1], flush=True)

    @torch.no_grad()
    def accuracy(dataset):
        evaluation_loader = DataLoader(dataset, batch_size=512, shuffle=False)
        correct = total = 0
        network.eval()
        for images, labels in evaluation_loader:
            prediction = network(images.to(device).reshape(-1, 784)).argmax(dim=1)
            labels = labels.to(device)
            correct += int((prediction == labels).sum())
            total += labels.numel()
        return correct / total

    train_accuracy = accuracy(train_data)
    test_accuracy = accuracy(test_data)
    state = {
        f"network.{key}": value.detach().cpu()
        for key, value in network.state_dict().items()
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": state,
            "model_config": {
                "model_type": "mlp",
                "input_dim": 784,
                "output_dim": 10,
                "hidden_width": args.nodes,
                "hidden_affine_layers": args.layers + 1,
                "activation": "tanh",
                "output_activation": "softmax",
                "source": "Robbybp/moai-examples train-mnist.py",
            },
        },
        args.checkpoint,
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in network.parameters()),
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "checkpoint_sha256": sha256(args.checkpoint),
        "elapsed_s": time.perf_counter() - started,
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("parameter_count", "train_accuracy", "test_accuracy", "elapsed_s")}, indent=2))


if __name__ == "__main__":
    main()
