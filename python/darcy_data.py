"""
darcy_data.py

Data generation and FNO training for the Darcy flow case study.

Generates pairs (a, u) where:
- a(x,y) : log-permeability field on an N×N grid (input to FNO)
- u(x,y) : pressure field satisfying −∇·(exp(a)∇u) = f  (target)

with zero Dirichlet BCs and unit source f = 1.

Usage (standalone)
------------------
    python darcy_data.py --grid 64 --n_samples 1000 \
        --output_dir output/darcy --seed 42

    python darcy_data.py --train_fno --grid 64 \
        --data_dir output/darcy --epochs 200
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Tuple

import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve

log = logging.getLogger(__name__)


# ============================================================================
# Gaussian random field permeability samples
# ============================================================================

def sample_grf(
    rng: np.random.Generator,
    n: int,
    length_scale: float = 0.2,
    variance: float = 1.0,
) -> np.ndarray:
    """Sample a log-permeability field from a Gaussian random field.

    Uses a simplified spectral method: white noise in Fourier space,
    filtered by a Matérn-like power spectrum decay.

    Returns
    -------
    a : (n, n) array  (log-permeability; use exp(a) for conductivity)
    """
    kx = np.fft.rfftfreq(n, d=1.0 / n)
    ky = np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")  # rfft shape: (n, n//2+1)

    freq_sq = KX ** 2 + KY ** 2
    power = (1.0 + (2 * np.pi * length_scale) ** 2 * freq_sq) ** (-2.0)
    power[0, 0] = 0.0  # remove DC (zero mean)

    noise = rng.standard_normal((n, n // 2 + 1)) + 1j * rng.standard_normal((n, n // 2 + 1))
    a_rfft = noise * np.sqrt(power)
    a = np.fft.irfft2(a_rfft, s=(n, n))
    a = a / (np.std(a) + 1e-8) * np.sqrt(variance)
    return a  # log-permeability


# ============================================================================
# Finite-difference Darcy solver
# ============================================================================

def darcy_fd_solve(a: np.ndarray, f: np.ndarray = None) -> np.ndarray:
    """Solve the 2-D Darcy equation on an N×N grid with zero Dirichlet BCs.

    −∇·(κ(x,y)∇u(x,y)) = f(x,y),  u|∂Ω = 0

    where κ = exp(a).

    Uses finite differences with harmonic averaging of κ at cell faces.

    Parameters
    ----------
    a : (N, N) log-permeability field
    f : (N, N) source term (default: ones)

    Returns
    -------
    u : (N, N) pressure field
    """
    N = a.shape[0]
    assert a.shape == (N, N)
    h = 1.0 / (N - 1)
    kappa = np.exp(a)  # conductivity

    if f is None:
        f = np.ones((N, N))

    # Harmonic averages at cell faces
    # kappa_{i+1/2, j} = harmonic mean of kappa[i,j] and kappa[i+1,j]
    def harmonic(k1, k2):
        return 2.0 * k1 * k2 / (k1 + k2 + 1e-30)

    kE = harmonic(kappa[:-1, :], kappa[1:, :])   # (N-1, N)
    kW = kE                                         # symmetric
    kN = harmonic(kappa[:, :-1], kappa[:, 1:])   # (N, N-1)
    kS = kN

    n_int = (N - 2) ** 2  # interior points only (Dirichlet BCs)

    def idx(i, j):  # interior: i,j in [1, N-2]
        return (i - 1) * (N - 2) + (j - 1)

    rows, cols, vals = [], [], []
    rhs = np.zeros(n_int)

    for i in range(1, N - 1):
        for j in range(1, N - 1):
            k = idx(i, j)
            diag = 0.0

            # East face: kE[i, j]
            ke = kE[i, j] / h ** 2
            if i + 1 <= N - 2:
                rows.append(k); cols.append(idx(i + 1, j)); vals.append(-ke)
            diag += ke

            # West face: kW[i-1, j]
            kw = kE[i - 1, j] / h ** 2
            if i - 1 >= 1:
                rows.append(k); cols.append(idx(i - 1, j)); vals.append(-kw)
            diag += kw

            # North face: kN[i, j]
            kn = kN[i, j] / h ** 2
            if j + 1 <= N - 2:
                rows.append(k); cols.append(idx(i, j + 1)); vals.append(-kn)
            diag += kn

            # South face: kS[i, j-1]
            ks = kN[i, j - 1] / h ** 2
            if j - 1 >= 1:
                rows.append(k); cols.append(idx(i, j - 1)); vals.append(-ks)
            diag += ks

            rows.append(k); cols.append(k); vals.append(diag)
            rhs[k] = f[i, j]

    from scipy.sparse import csr_matrix
    A = csr_matrix((vals, (rows, cols)), shape=(n_int, n_int))
    u_int = spsolve(A, rhs)

    u = np.zeros((N, N))
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            u[i, j] = u_int[idx(i, j)]

    return u


# ============================================================================
# Dataset generation
# ============================================================================

def generate_darcy_dataset(
    n_samples: int,
    grid_n: int = 64,
    length_scale: float = 0.2,
    seed: int = 42,
    output_dir: str = "output/darcy",
    split: str = "train",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (a, u) pairs and save to disk.

    Returns
    -------
    a_data : (n_samples, grid_n, grid_n) log-permeability fields
    u_data : (n_samples, grid_n, grid_n) pressure fields
    """
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    a_data = np.zeros((n_samples, grid_n, grid_n), dtype=np.float32)
    u_data = np.zeros((n_samples, grid_n, grid_n), dtype=np.float32)

    t0 = time.time()
    for i in range(n_samples):
        a = sample_grf(rng, grid_n, length_scale=length_scale)
        u = darcy_fd_solve(a)
        a_data[i] = a.astype(np.float32)
        u_data[i] = u.astype(np.float32)
        if (i + 1) % max(1, n_samples // 10) == 0:
            elapsed = time.time() - t0
            log.info("  %d/%d samples (%.1fs)", i + 1, n_samples, elapsed)

    out_path = os.path.join(output_dir, f"darcy_{split}_n{n_samples}_g{grid_n}_s{seed}.npz")
    np.savez_compressed(out_path, a=a_data, u=u_data,
                        grid_n=grid_n, length_scale=length_scale, seed=seed)
    log.info("Saved dataset: %s", out_path)

    return a_data, u_data


# ============================================================================
# FNO training (JAX / Optax)
# ============================================================================

def train_fno(
    data_dir: str,
    grid_n: int = 64,
    d_v: int = 32,
    n_layers: int = 4,
    k_max: int = 16,
    lr: float = 1e-3,
    epochs: int = 200,
    batch_size: int = 32,
    seed: int = 42,
    output_dir: str = "output/models/darcy",
    device: str = "cpu",
) -> str:
    """Train an FNO2D surrogate on Darcy flow data.

    Returns path to saved checkpoint (.npz).
    """
    import jax
    import jax.numpy as jnp

    try:
        import optax
    except ImportError:
        raise ImportError("optax is required for FNO training: pip install optax")

    from models.fno import fno2d_forward, make_random_fno2d_params

    jax.config.update("jax_enable_x64", True)

    # Load data
    import glob
    train_files = sorted(glob.glob(os.path.join(data_dir, f"darcy_train_*_g{grid_n}_*.npz")))
    if not train_files:
        raise FileNotFoundError(f"No training data found in {data_dir}. "
                                "Run generate_darcy_dataset first.")

    data = np.load(train_files[0])
    a_all = data["a"].astype(np.float64)   # (N_samples, grid_n, grid_n)
    u_all = data["u"].astype(np.float64)

    n_samples = len(a_all)
    rng = np.random.default_rng(seed)

    # Initialise params
    key = jax.random.PRNGKey(seed)
    params = make_random_fno2d_params(
        key, n_in=1, n_out=1, d_v=d_v, n_layers=n_layers,
        k_max_x=k_max, k_max_y=k_max, grid_nx=grid_n, grid_ny=grid_n,
    )

    # Optimizer
    schedule = optax.cosine_decay_schedule(lr, epochs * (n_samples // batch_size))
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    def loss_fn(params, a_batch, u_batch):
        preds = jax.vmap(lambda a: fno2d_forward(params, a))(a_batch)
        return jnp.mean((preds - u_batch) ** 2)

    @jax.jit
    def step(params, opt_state, a_batch, u_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, a_batch, u_batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    log.info("Training FNO2D: grid=%d, d_v=%d, k_max=%d, epochs=%d", grid_n, d_v, k_max, epochs)

    for epoch in range(1, epochs + 1):
        idx = rng.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            batch_idx = idx[start:start + batch_size]
            a_b = a_all[batch_idx].reshape(len(batch_idx), -1)  # (B, N*N)
            u_b = u_all[batch_idx].reshape(len(batch_idx), -1)  # (B, N*N)
            params, opt_state, loss = step(params, opt_state, a_b, u_b)
            epoch_loss += float(loss)
            n_batches += 1

        if epoch % max(1, epochs // 10) == 0:
            log.info("  Epoch %d/%d: loss=%.6f", epoch, epochs, epoch_loss / n_batches)

    # Save checkpoint
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(
        output_dir, f"fno_darcy_{grid_n}x{grid_n}_dv{d_v}_seed{seed}.npz"
    )
    _save_fno_params(params, ckpt_path, grid_n, k_max, n_layers, d_v)
    log.info("Saved FNO checkpoint: %s", ckpt_path)

    return ckpt_path


def _save_fno_params(params: dict, path: str, grid_n: int, k_max: int,
                     n_layers: int, d_v: int) -> None:
    """Save FNO params as a .npz file."""
    import numpy as np
    save_dict = {
        "grid_nx": grid_n,
        "grid_ny": grid_n,
        "k_max_x": k_max,
        "k_max_y": k_max,
        "n_layers": n_layers,
        "d_v": d_v,
        "lift_W": np.array(params["lift_W"]),
        "lift_b": np.array(params["lift_b"]),
        "proj1_W": np.array(params["proj1_W"]),
        "proj1_b": np.array(params["proj1_b"]),
        "proj2_W": np.array(params["proj2_W"]),
        "proj2_b": np.array(params["proj2_b"]),
    }
    for i, lp in enumerate(params["fourier_layers"]):
        save_dict[f"fourier_{i}_spec_re"] = np.array(lp["spectral"]["weight_re"])
        save_dict[f"fourier_{i}_spec_im"] = np.array(lp["spectral"]["weight_im"])
        save_dict[f"fourier_{i}_bypass_W"] = np.array(lp["bypass_W"])
        save_dict[f"fourier_{i}_bypass_b"] = np.array(lp["bypass_b"])

    np.savez_compressed(path, **save_dict)


# ============================================================================
# CLI
# ============================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="Darcy flow data generation and FNO training")
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="output/darcy")
    p.add_argument("--length_scale", type=float, default=0.2)
    p.add_argument("--train_fno", action="store_true")
    p.add_argument("--data_dir", type=str, default="output/darcy")
    p.add_argument("--d_v", type=int, default=32)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--k_max", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()

    if args.train_fno:
        train_fno(
            data_dir=args.data_dir,
            grid_n=args.grid,
            d_v=args.d_v,
            n_layers=args.n_layers,
            k_max=args.k_max,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            output_dir=os.path.join(args.output_dir, "models"),
            device=args.device,
        )
    else:
        generate_darcy_dataset(
            n_samples=args.n_samples,
            grid_n=args.grid,
            length_scale=args.length_scale,
            seed=args.seed,
            output_dir=args.output_dir,
            split="train",
        )
        generate_darcy_dataset(
            n_samples=args.n_samples // 5,
            grid_n=args.grid,
            length_scale=args.length_scale,
            seed=args.seed + 1,
            output_dir=args.output_dir,
            split="test",
        )
