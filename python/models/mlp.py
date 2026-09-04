"""
mlp.py

JAX builders for MLP and ResMLP architectures.

These functions were extracted from the original jax_nn_evaluator.py.
All forward functions have the signature: ``(params, x) -> output``.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Shared activation
# ---------------------------------------------------------------------------

def smooth_relu(x: jnp.ndarray, alpha: float = 1e-3) -> jnp.ndarray:
    """Smooth ReLU (softplus): α · log(1 + exp(x/α)).

    C∞ approximation to ReLU; required for second-order methods.
    """
    return alpha * jax.nn.softplus(x / alpha)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

def build_mlp_jax(
    state_dict: Dict,
    config: Dict,
) -> Tuple[Callable, Dict]:
    """Build a JAX MLP from a PyTorch state dict.

    Returns
    -------
    forward : callable
        ``(params, x) -> logits``
    params : dict
        JAX parameter dict.
    """
    keys = list(state_dict.keys())
    if any(k.startswith("network.") for k in keys):
        prefix = "network"
    elif any(k.startswith("layers.") for k in keys):
        prefix = "layers"
    elif any(k.startswith("net.") for k in keys):
        prefix = "net"
    else:
        prefix = "layers"

    layer_indices = sorted(set(
        int(k.split(".")[1])
        for k in keys
        if "weight" in k and k.startswith(f"{prefix}.")
    ))

    if not layer_indices:
        raise ValueError("No linear layers found in state_dict")

    params = {}
    for i, idx in enumerate(layer_indices):
        W = state_dict[f"{prefix}.{idx}.weight"].detach().cpu().numpy()
        b = state_dict[f"{prefix}.{idx}.bias"].detach().cpu().numpy()
        params[f"layer_{i}"] = {
            "W": jnp.array(W, dtype=jnp.float64),
            "b": jnp.array(b, dtype=jnp.float64),
        }

    n_layers = len(params)
    activation_name = config.get("activation", "smooth_relu").lower()
    output_activation = config.get("output_activation", "identity").lower()

    def activate(value):
        if activation_name == "tanh":
            return jnp.tanh(value)
        if activation_name == "sigmoid":
            return jax.nn.sigmoid(value)
        if activation_name == "softplus":
            return jax.nn.softplus(value)
        return smooth_relu(value)

    def forward(params_dict, x):
        out = x
        for i in range(n_layers):
            W = params_dict[f"layer_{i}"]["W"]
            b = params_dict[f"layer_{i}"]["b"]
            out = jnp.dot(W, out) + b
            if i < n_layers - 1:
                out = activate(out)
        if output_activation == "softmax":
            out = jax.nn.softmax(out)
        return out

    return forward, params


# ---------------------------------------------------------------------------
# ResMLP
# ---------------------------------------------------------------------------

def build_resmlp_jax(
    state_dict: Dict,
    config: Dict,
) -> Tuple[Callable, Dict]:
    """Build a JAX Residual MLP from a PyTorch state dict."""
    # Support both "init_proj.weight" and "input_proj.0.weight" key styles
    if "init_proj.weight" in state_dict:
        ip_w = state_dict["init_proj.weight"]
        ip_b = state_dict["init_proj.bias"]
    elif "input_proj.0.weight" in state_dict:
        ip_w = state_dict["input_proj.0.weight"]
        ip_b = state_dict["input_proj.0.bias"]
    else:
        raise KeyError("Cannot find initial projection weights in state_dict")

    blocks = []
    params = {
        "init_proj": {
            "W": jnp.array(ip_w.detach().cpu().numpy(), dtype=jnp.float64),
            "b": jnp.array(ip_b.detach().cpu().numpy(), dtype=jnp.float64),
        },
        "blocks": blocks,
        "output": {
            "W": jnp.array(state_dict["output_layer.weight"].detach().cpu().numpy(), dtype=jnp.float64),
            "b": jnp.array(state_dict["output_layer.bias"].detach().cpu().numpy(), dtype=jnp.float64),
        },
    }

    n_blocks = config.get("num_blocks", config.get("n_blocks", 0))
    for i in range(n_blocks):
        bp = {
            "fc1": {
                "W": jnp.array(state_dict[f"blocks.{i}.fc1.weight"].detach().cpu().numpy(), dtype=jnp.float64),
                "b": jnp.array(state_dict[f"blocks.{i}.fc1.bias"].detach().cpu().numpy(), dtype=jnp.float64),
            },
            "fc2": {
                "W": jnp.array(state_dict[f"blocks.{i}.fc2.weight"].detach().cpu().numpy(), dtype=jnp.float64),
                "b": jnp.array(state_dict[f"blocks.{i}.fc2.bias"].detach().cpu().numpy(), dtype=jnp.float64),
            },
        }
        blocks.append(bp)

    def forward(params_dict, x):
        out = jnp.dot(params_dict["init_proj"]["W"], x) + params_dict["init_proj"]["b"]
        out = smooth_relu(out)
        for block in params_dict["blocks"]:
            identity = out
            res = jnp.dot(block["fc1"]["W"], out) + block["fc1"]["b"]
            res = smooth_relu(res)
            res = jnp.dot(block["fc2"]["W"], res) + block["fc2"]["b"]
            out = smooth_relu(res + identity)
        out = jnp.dot(params_dict["output"]["W"], out) + params_dict["output"]["b"]
        return out

    return forward, params
