"""JAX loader for the published PFR-NMPC tanh surrogate checkpoints."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import jax.numpy as jnp


def build_pfr_pinn_jax(state_dict: Dict, config: Dict) -> Tuple[Callable, Dict]:
    dtype = config.get("dtype", jnp.float64)
    layer_indices = sorted(
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("dnn.dense_layers.") and key.endswith(".weight")
    )
    params = {
        "lower": jnp.asarray(state_dict["lb"].detach().cpu().numpy(), dtype=dtype),
        "upper": jnp.asarray(state_dict["ub"].detach().cpu().numpy(), dtype=dtype),
        "layers": [
            {
                "weight": jnp.asarray(
                    state_dict[f"dnn.dense_layers.{index}.weight"].detach().cpu().numpy(),
                    dtype=dtype,
                ),
                "bias": jnp.asarray(
                    state_dict[f"dnn.dense_layers.{index}.bias"].detach().cpu().numpy(),
                    dtype=dtype,
                ),
            }
            for index in layer_indices
        ],
    }

    def forward(p, x):
        value = 2.0 * (x - p["lower"]) / (p["upper"] - p["lower"]) - 1.0
        for index, layer in enumerate(p["layers"]):
            value = layer["weight"] @ value + layer["bias"]
            if index + 1 < len(p["layers"]):
                value = jnp.tanh(value)
        return value

    return forward, params
