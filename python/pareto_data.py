"""
pareto_data.py

Synthetic data generation and surrogate training for Case Study II:
constrained multi-network Pareto tracing.

Problem setup
-------------
We create a synthetic multi-objective design problem in R^n:

  f1(x) = ‖x - a₁‖²   (distance to target a₁, measured through NN₁)
  f2(x) = ‖x - a₂‖²   (distance to target a₂, measured through NN₂)
  h(x)  = g(x) - 0    (feasibility: some nonlinear condition < 0)

Each of f1, f2, h is represented by a small MLP trained on samples.
The true Pareto front is a curve in (f1, f2) space connecting the
minimisers of f1 and f2 subject to the feasibility constraint.

Usage
-----
    python pareto_data.py --n 20 --output_dir output/pareto --seed 42
    python pareto_data.py --train --n 20 --output_dir output/pareto --epochs 200
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

log = logging.getLogger(__name__)


# ============================================================================
# Synthetic problem definition
# ============================================================================

def make_pareto_problem(n: int = 20, seed: int = 42) -> dict:
    """Define the synthetic Pareto problem parameters.

    Returns
    -------
    dict with keys:
        a1, a2 : (n,) target vectors
        A_feas  : (n, n) PSD matrix for feasibility: x'Ax <= threshold
        threshold : float
    """
    rng = np.random.default_rng(seed)
    a1 = rng.uniform(-0.5, 0.5, n)
    a2 = rng.uniform(-0.5, 0.5, n)

    # Feasibility: x' A x <= threshold  (quadratic constraint for non-trivial front)
    B = rng.standard_normal((n, n // 2))
    A_feas = B @ B.T / n  # PSD
    # threshold: roughly half of max over the box [-1,1]^n
    threshold = np.trace(A_feas) * 0.4

    return {"a1": a1, "a2": a2, "A_feas": A_feas, "threshold": threshold}


def f1_true(x: np.ndarray, a1: np.ndarray) -> float:
    return float(np.sum((x - a1) ** 2))


def f2_true(x: np.ndarray, a2: np.ndarray) -> float:
    return float(np.sum((x - a2) ** 2))


def h_true(x: np.ndarray, A_feas: np.ndarray, threshold: float) -> float:
    return float(x @ A_feas @ x - threshold)


# ============================================================================
# Training data generation
# ============================================================================

def generate_pareto_data(
    n: int = 20,
    n_samples: int = 5000,
    seed: int = 42,
    output_dir: str = "output/pareto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample random x ∈ [-1,1]^n and compute (f1, f2, h) labels."""
    problem = make_pareto_problem(n=n, seed=seed)
    a1, a2 = problem["a1"], problem["a2"]
    A_feas, threshold = problem["A_feas"], problem["threshold"]

    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n_samples, n))

    f1 = np.array([f1_true(x, a1) for x in X], dtype=np.float32)
    f2 = np.array([f2_true(x, a2) for x in X], dtype=np.float32)
    h  = np.array([h_true(x, A_feas, threshold) for x in X], dtype=np.float32)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"pareto_data_n{n}_s{n_samples}_seed{seed}.npz")
    np.savez_compressed(save_path,
                        X=X, f1=f1, f2=f2, h=h,
                        a1=a1, a2=a2, A_feas=A_feas, threshold=threshold)
    log.info("Saved Pareto data: %s", save_path)

    return X, f1, f2, h


# ============================================================================
# Surrogate network and training
# ============================================================================

class SurrogateNet(nn.Module):
    """Small MLP surrogate for f1, f2, or h (scalar output)."""

    def __init__(self, n_in: int, hidden: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(n_in, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_surrogate(
    X: np.ndarray,
    y: np.ndarray,
    n_in: int,
    label: str = "surrogate",
    hidden: int = 64,
    n_layers: int = 3,
    lr: float = 1e-3,
    epochs: int = 200,
    batch_size: int = 128,
    seed: int = 42,
    output_dir: str = "output/models/pareto",
    device: str = "cpu",
) -> str:
    """Train a surrogate MLP and save the checkpoint."""
    torch.manual_seed(seed)
    torch_device = torch.device("cuda" if device == "gpu" and torch.cuda.is_available() else "cpu")

    model = SurrogateNet(n_in=n_in, hidden=hidden, n_layers=n_layers).to(torch_device)
    optimizer_cls = optim.Adam
    opt = optimizer_cls(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.MSELoss()

    X_t = torch.tensor(X, dtype=torch.float32).to(torch_device)
    y_t = torch.tensor(y, dtype=torch.float32).to(torch_device)

    n_samples = len(X)
    rng = np.random.default_rng(seed)

    log.info("Training %s surrogate: n=%d, epochs=%d", label, n_in, epochs)
    for epoch in range(1, epochs + 1):
        idx = rng.permutation(n_samples)
        epoch_loss = 0.0
        for start in range(0, n_samples, batch_size):
            b_idx = idx[start:start + batch_size]
            Xb, yb = X_t[b_idx], y_t[b_idx]
            pred = model(Xb)
            loss = criterion(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss)
        scheduler.step()
        if epoch % max(1, epochs // 5) == 0:
            log.info("  %s epoch %d/%d loss=%.6f", label, epoch, epochs, epoch_loss)

    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"{label}_n{n_in}_seed{seed}.pt")

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "model_type": "mlp",
            "input_dim": n_in,
            "output_dim": 1,
            "hidden_dims": [hidden] * n_layers,
            "activation": "relu",
            "dropout_rate": 0.0,
        },
        "train_args": {"lr": lr, "epochs": epochs},
        "history": {},
    }, ckpt_path)
    log.info("Saved %s checkpoint: %s", label, ckpt_path)

    return ckpt_path


def train_all_surrogates(
    n: int = 20,
    n_samples: int = 5000,
    seed: int = 42,
    data_dir: str = "output/pareto",
    output_dir: str = "output/models/pareto",
    epochs: int = 200,
    hidden: int = 64,
    device: str = "cpu",
) -> dict:
    """Generate data and train f1, f2, h surrogates.

    Returns dict with keys 'f1_path', 'f2_path', 'h_path'.
    """
    X, f1, f2, h = generate_pareto_data(n=n, n_samples=n_samples, seed=seed,
                                         output_dir=data_dir)
    trained = {}
    for label, y in [("f1_net", f1), ("f2_net", f2), ("h_net", h)]:
        trained[label] = train_surrogate(
            X, y, n_in=n, label=label, hidden=hidden,
            epochs=epochs, seed=seed, output_dir=output_dir, device=device,
        )
    return {
        "f1_path": trained["f1_net"],
        "f2_path": trained["f2_net"],
        "h_path": trained["h_net"],
    }


# ============================================================================
# CLI
# ============================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="Pareto surrogate data / training")
    p.add_argument("--n", type=int, default=20, help="Decision variable dimension")
    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="output/pareto")
    p.add_argument("--model_dir", type=str, default="output/models/pareto")
    p.add_argument("--train", action="store_true")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parse_args()

    if args.train:
        paths = train_all_surrogates(
            n=args.n, n_samples=args.n_samples, seed=args.seed,
            data_dir=args.output_dir, output_dir=args.model_dir,
            epochs=args.epochs, hidden=args.hidden, device=args.device,
        )
        print("Trained surrogates:")
        for k, v in paths.items():
            print(f"  {k}: {v}")
    else:
        generate_pareto_data(
            n=args.n, n_samples=args.n_samples, seed=args.seed,
            output_dir=args.output_dir,
        )
