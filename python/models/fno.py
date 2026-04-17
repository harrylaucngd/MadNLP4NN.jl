"""
fno.py

Fourier Neural Operator (FNO2D) in JAX.

Architecture (Li et al. 2021, https://arxiv.org/abs/2010.08895):
  1. Lifting layer:        (N,M,1)  → (N,M,d_v)
  2. k Fourier layers:     (N,M,d_v) → (N,M,d_v), each consisting of
       - SpectralConv2d: FFT → keep top k_max×k_max modes → learned
         complex multiplication → IFFT  (Fourier path)
       - Local linear bypass (W·x + b)  (bypass path)
       - Sum + smooth activation
  3. Projection:           (N,M,d_v) → (N,M,128) → (N,M,1) → flatten

All functions follow the parameter-dict / pure-function convention used
throughout this codebase.

Public API
----------
build_fno2d_jax(state_dict, config) → (forward_fn, params)
    Load FNO2D from a PyTorch/npz checkpoint.

make_random_fno2d_params(n_modes, d_v, n_layers, key) → params
    Initialise random FNO2D parameters for training from scratch.

fno2d_forward(params, a_flat) → u_flat
    Pure JAX forward pass.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .mlp import smooth_relu

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spectral convolution (Fourier path)
# ---------------------------------------------------------------------------

def _spectral_conv2d(
    params: Dict,
    x: jnp.ndarray,
    k_max_x: int,
    k_max_y: int,
) -> jnp.ndarray:
    """Apply the Fourier-space linear transform for one FNO layer.

    Parameters
    ----------
    params : dict with keys 'weight_re', 'weight_im'
        Shape: (k_max_x, k_max_y, d_v, d_v)
    x : (N, M, d_v)
    k_max_x, k_max_y : number of Fourier modes retained in each direction

    Returns
    -------
    out : (N, M, d_v)
    """
    N, M, d_v = x.shape

    # FFT along spatial axes; rfft2 returns (N, M//2+1, d_v)
    x_ft = jnp.fft.rfft2(x, axes=(0, 1))

    # Complex weight matrix
    W = params["weight_re"] + 1j * params["weight_im"]  # (kx, ky, d_v, d_v)

    # Select and multiply top modes
    x_top = x_ft[:k_max_x, :k_max_y, :]  # (kx, ky, d_v)
    # einsum: 'xyi, xyio -> xyo' – multiply each (d_v,) vector by the (d_v,d_v) weight matrix
    x_top_out = jnp.einsum("xyi,xyio->xyo", x_top, W)

    # Pad back to full spectral size
    M_rfft = M // 2 + 1
    x_ft_out = jnp.zeros((N, M_rfft, d_v), dtype=jnp.complex128)
    x_ft_out = x_ft_out.at[:k_max_x, :k_max_y, :].set(x_top_out)

    # IFFT
    out = jnp.fft.irfft2(x_ft_out, s=(N, M), axes=(0, 1))  # (N, M, d_v)
    return out


# ---------------------------------------------------------------------------
# Single FNO2D layer
# ---------------------------------------------------------------------------

def _fno2d_layer(
    params: Dict,
    x: jnp.ndarray,
    k_max_x: int,
    k_max_y: int,
) -> jnp.ndarray:
    """One FNO layer: Fourier path + bypass + activation.

    params keys: 'spectral', 'bypass_W', 'bypass_b'
    """
    # Fourier / spectral path
    x_spec = _spectral_conv2d(params["spectral"], x, k_max_x, k_max_y)

    # Local bypass: (N, M, d_v) @ (d_v, d_v) + (d_v,)
    x_bypass = jnp.einsum("...i,io->...o", x, params["bypass_W"]) + params["bypass_b"]

    return smooth_relu(x_spec + x_bypass)


# ---------------------------------------------------------------------------
# Full FNO2D forward
# ---------------------------------------------------------------------------

def fno2d_forward(params: Dict, a_flat: jnp.ndarray) -> jnp.ndarray:
    """FNO2D forward pass.

    Parameters
    ----------
    params : dict
        Must contain:
        - 'grid_nx', 'grid_ny' : int (stored as 0-d arrays or scalars)
        - 'k_max_x', 'k_max_y' : int
        - 'lift_W' : (1, d_v) or (n_in, d_v)
        - 'lift_b' : (d_v,)
        - 'fourier_layers' : list of layer param dicts
        - 'proj1_W' : (d_v, 128), 'proj1_b' : (128,)
        - 'proj2_W' : (128, 1),  'proj2_b' : (1,)
    a_flat : (N*M,) or (N*M*c,) flat permeability field

    Returns
    -------
    u_flat : (N*M,) flat pressure field
    """
    grid_nx = int(params["grid_nx"])
    grid_ny = int(params["grid_ny"])
    k_max_x = int(params["k_max_x"])
    k_max_y = int(params["k_max_y"])

    n_spatial = grid_nx * grid_ny
    n_in = len(a_flat) // n_spatial
    a = a_flat.reshape(grid_nx, grid_ny, n_in)  # (N, M, n_in)

    # Lifting
    x = jnp.einsum("...i,io->...o", a, params["lift_W"]) + params["lift_b"]
    x = smooth_relu(x)  # (N, M, d_v)

    # Fourier layers
    for layer_params in params["fourier_layers"]:
        x = _fno2d_layer(layer_params, x, k_max_x, k_max_y)

    # Projection to output
    x = jnp.einsum("...i,io->...o", x, params["proj1_W"]) + params["proj1_b"]
    x = smooth_relu(x)
    x = jnp.einsum("...i,io->...o", x, params["proj2_W"]) + params["proj2_b"]

    return x.reshape(-1)  # (N*M,)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def build_fno2d_jax(
    state_dict: Dict,
    config: Dict,
) -> Tuple[Callable, Dict]:
    """Build a JAX FNO2D from a PyTorch/npz checkpoint.

    The checkpoint is expected to follow one of:
    1. Standard dict with keys like 'lift_W', 'fourier_layers.0.spectral.weight_re', …
    2. Our custom .npz format produced by python/darcy_data.py training.

    Returns
    -------
    forward_fn : callable (params, x) -> y
    params : dict of JAX arrays
    """
    # Detect format
    if any(isinstance(v, np.ndarray) for v in state_dict.values()):
        # npz-style: numpy arrays
        params = _load_fno_from_npz_dict(state_dict, config)
    else:
        # torch-style tensors
        params = _load_fno_from_torch_dict(state_dict, config)

    return fno2d_forward, params


def _load_fno_from_npz_dict(npz: Dict, config: Dict) -> Dict:
    """Load FNO2D parameters from a numpy dict (saved with np.savez)."""

    def _arr(key):
        return jnp.array(npz[key], dtype=jnp.float64)

    grid_nx = int(config.get("grid_nx", npz.get("grid_nx", 64)))
    grid_ny = int(config.get("grid_ny", npz.get("grid_ny", grid_nx)))
    k_max_x = int(config.get("k_max_x", npz.get("k_max_x", 16)))
    k_max_y = int(config.get("k_max_y", npz.get("k_max_y", k_max_x)))
    n_layers = int(config.get("n_layers", npz.get("n_layers", 4)))
    d_v = int(config.get("d_v", npz.get("d_v", 32)))

    params = {
        "grid_nx": grid_nx,
        "grid_ny": grid_ny,
        "k_max_x": k_max_x,
        "k_max_y": k_max_y,
        "lift_W": _arr("lift_W"),
        "lift_b": _arr("lift_b"),
        "fourier_layers": [],
        "proj1_W": _arr("proj1_W"),
        "proj1_b": _arr("proj1_b"),
        "proj2_W": _arr("proj2_W"),
        "proj2_b": _arr("proj2_b"),
    }

    for i in range(n_layers):
        layer_params = {
            "spectral": {
                "weight_re": _arr(f"fourier_{i}_spec_re"),
                "weight_im": _arr(f"fourier_{i}_spec_im"),
            },
            "bypass_W": _arr(f"fourier_{i}_bypass_W"),
            "bypass_b": _arr(f"fourier_{i}_bypass_b"),
        }
        params["fourier_layers"].append(layer_params)

    return params


def _load_fno_from_torch_dict(state_dict: Dict, config: Dict) -> Dict:
    """Load FNO2D parameters from a PyTorch state dict."""

    def _arr(key):
        t = state_dict[key]
        return jnp.array(t.detach().cpu().numpy(), dtype=jnp.float64)

    grid_nx = int(config.get("grid_nx", 64))
    grid_ny = int(config.get("grid_ny", grid_nx))
    k_max_x = int(config.get("k_max_x", 16))
    k_max_y = int(config.get("k_max_y", k_max_x))
    d_v = int(config.get("d_v", 32))

    # Count Fourier layers
    n_layers = 0
    while f"fourier_layers.{n_layers}.spectral.weight_re" in state_dict:
        n_layers += 1

    # Fall back to neuraloperator-style keys
    if n_layers == 0:
        return _load_fno_neuraloperator_style(state_dict, config)

    params = {
        "grid_nx": grid_nx,
        "grid_ny": grid_ny,
        "k_max_x": k_max_x,
        "k_max_y": k_max_y,
        "lift_W": _arr("lift_W"),
        "lift_b": _arr("lift_b"),
        "fourier_layers": [],
        "proj1_W": _arr("proj1_W"),
        "proj1_b": _arr("proj1_b"),
        "proj2_W": _arr("proj2_W"),
        "proj2_b": _arr("proj2_b"),
    }

    for i in range(n_layers):
        pfx = f"fourier_layers.{i}"
        layer_params = {
            "spectral": {
                "weight_re": _arr(f"{pfx}.spectral.weight_re"),
                "weight_im": _arr(f"{pfx}.spectral.weight_im"),
            },
            "bypass_W": _arr(f"{pfx}.bypass_W"),
            "bypass_b": _arr(f"{pfx}.bypass_b"),
        }
        params["fourier_layers"].append(layer_params)

    return params


def _load_fno_neuraloperator_style(state_dict: Dict, config: Dict) -> Dict:
    """Best-effort loader for neuraloperator library FNO checkpoints."""
    log.warning(
        "Attempting neuraloperator-style FNO loading.  "
        "Key mapping may need adjustment."
    )

    def _arr(key):
        return jnp.array(state_dict[key].detach().cpu().numpy(), dtype=jnp.float64)

    grid_nx = int(config.get("grid_nx", 64))
    grid_ny = int(config.get("grid_ny", grid_nx))
    k_max_x = int(config.get("k_max_x", 16))
    k_max_y = int(config.get("k_max_y", k_max_x))

    # Lifting = fc0
    lift_W = _arr("fc0.weight")   # (d_v, 1)
    lift_W = jnp.transpose(lift_W, (1, 0))  # (1, d_v)
    lift_b = _arr("fc0.bias")

    # Count SpectralConv layers: convN.weights1, convN.weights2
    n_layers = 0
    while f"conv{n_layers}.weights1" in state_dict:
        n_layers += 1

    fourier_layers = []
    for i in range(n_layers):
        # weights1 shape: (d_v, d_v, k_max_x, k_max_y//2+1) – complex stored as float
        # neuraloperator stores them as torch.cfloat; we handle both
        w1 = state_dict[f"conv{i}.weights1"].detach().cpu()
        w2 = state_dict[f"conv{i}.weights2"].detach().cpu()

        if w1.is_complex():
            w1_re = w1.real.numpy().transpose(2, 3, 0, 1)  # (kx, ky, d_v_in, d_v_out)
            w1_im = w1.imag.numpy().transpose(2, 3, 0, 1)
        else:
            # real representation: shape (d_v, d_v, kx, ky, 2)
            w1_re = w1[..., 0].numpy().transpose(2, 3, 0, 1)
            w1_im = w1[..., 1].numpy().transpose(2, 3, 0, 1)

        bypass_W = _arr(f"w{i}.weight")  # (d_v, d_v)
        bypass_W = jnp.transpose(bypass_W, (1, 0))  # (d_v_in, d_v_out)
        bypass_b = _arr(f"w{i}.bias")

        fourier_layers.append({
            "spectral": {
                "weight_re": jnp.array(w1_re, dtype=jnp.float64),
                "weight_im": jnp.array(w1_im, dtype=jnp.float64),
            },
            "bypass_W": bypass_W,
            "bypass_b": bypass_b,
        })

    # Projection = fc1, fc2
    proj1_W = jnp.transpose(_arr("fc1.weight"), (1, 0))
    proj1_b = _arr("fc1.bias")
    proj2_W = jnp.transpose(_arr("fc2.weight"), (1, 0))
    proj2_b = _arr("fc2.bias")

    return {
        "grid_nx": grid_nx,
        "grid_ny": grid_ny,
        "k_max_x": k_max_x,
        "k_max_y": k_max_y,
        "lift_W": lift_W,
        "lift_b": lift_b,
        "fourier_layers": fourier_layers,
        "proj1_W": proj1_W,
        "proj1_b": proj1_b,
        "proj2_W": proj2_W,
        "proj2_b": proj2_b,
    }


# ---------------------------------------------------------------------------
# Random initialisation (for training from scratch)
# ---------------------------------------------------------------------------

def make_random_fno2d_params(
    key: jnp.ndarray,
    n_in: int = 1,
    n_out: int = 1,
    d_v: int = 32,
    n_layers: int = 4,
    k_max_x: int = 16,
    k_max_y: int = 16,
    grid_nx: int = 64,
    grid_ny: int = 64,
    proj_hidden: int = 128,
) -> Dict:
    """Randomly initialised FNO2D parameter dict (for training)."""

    def _normal(key, shape, scale=0.02):
        return jax.random.normal(key, shape, dtype=jnp.float64) * scale

    keys = jax.random.split(key, 4 + n_layers * 4)
    ki = iter(keys)

    params = {
        "grid_nx": grid_nx,
        "grid_ny": grid_ny,
        "k_max_x": k_max_x,
        "k_max_y": k_max_y,
        "lift_W": _normal(next(ki), (n_in, d_v)),
        "lift_b": jnp.zeros(d_v),
        "fourier_layers": [],
        "proj1_W": _normal(next(ki), (d_v, proj_hidden)),
        "proj1_b": jnp.zeros(proj_hidden),
        "proj2_W": _normal(next(ki), (proj_hidden, n_out)),
        "proj2_b": jnp.zeros(n_out),
    }

    for _ in range(n_layers):
        k_max_y_rfft = k_max_y  # use full k_max_y for the rfft modes
        layer_params = {
            "spectral": {
                "weight_re": _normal(next(ki), (k_max_x, k_max_y_rfft, d_v, d_v)),
                "weight_im": _normal(next(ki), (k_max_x, k_max_y_rfft, d_v, d_v)),
            },
            "bypass_W": _normal(next(ki), (d_v, d_v)),
            "bypass_b": jnp.zeros(d_v),
        }
        # consume the bypass_b key slot
        next(ki)
        params["fourier_layers"].append(layer_params)

    return params
