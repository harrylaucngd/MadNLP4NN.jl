"""JAX loader for the published no-padding PFR CNN surrogate checkpoints."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np


def build_pfr_cnn_jax(state_dict: Dict, config: Dict) -> Tuple[Callable, Dict]:
    dtype = config.get("dtype", jnp.float64)
    layer_indices = sorted(
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("cnn.cnn_layers.") and key.endswith(".weight")
    )
    hidden_channels = int(state_dict["cnn.cnn_layers.0.weight"].shape[0])
    kernel_height = int(state_dict["cnn.cnn_layers.0.weight"].shape[2])
    final_length = int(state_dict["cnn.linear.weight"].shape[1] // hidden_channels)
    n_fe = final_length + len(layer_indices) * (kernel_height - 1)
    out_channels = int(state_dict["cnn.linear.bias"].numel() // n_fe)
    in_channels = int(state_dict["cnn.cnn_layers.0.weight"].shape[1])
    control_channels = in_channels - out_channels
    params = {
        "lower": jnp.asarray(state_dict["lb"].detach().cpu().numpy(), dtype=dtype),
        "upper": jnp.asarray(state_dict["ub"].detach().cpu().numpy(), dtype=dtype),
        "convolutions": [
            {
                "weight": jnp.asarray(
                    state_dict[f"cnn.cnn_layers.{index}.weight"]
                    .detach()
                    .cpu()
                    .numpy(),
                    dtype=dtype,
                ),
                "bias": jnp.asarray(
                    state_dict[f"cnn.cnn_layers.{index}.bias"]
                    .detach()
                    .cpu()
                    .numpy(),
                    dtype=dtype,
                ),
            }
            for index in layer_indices
        ],
        "linear_weight": jnp.asarray(
            state_dict["cnn.linear.weight"].detach().cpu().numpy(), dtype=dtype
        ),
        "linear_bias": jnp.asarray(
            state_dict["cnn.linear.bias"].detach().cpu().numpy(), dtype=dtype
        ),
    }

    def forward(p, x_flat):
        state_size = out_channels * n_fe
        state = x_flat[:state_size].reshape(out_channels, n_fe, 1)
        controls = x_flat[state_size:].reshape(control_channels, 1, 1)
        controls = jnp.broadcast_to(
            controls, (control_channels, n_fe, 1)
        )
        value = jnp.concatenate((state, controls), axis=0)[None, ...]
        value = 2.0 * (value - p["lower"]) / (p["upper"] - p["lower"]) - 1.0
        for layer in p["convolutions"]:
            value = jax.lax.conv_general_dilated(
                value,
                layer["weight"],
                window_strides=(1, 1),
                padding="VALID",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )
            value = jnp.tanh(value + layer["bias"][None, :, None, None])
        value = value.reshape(-1)
        value = p["linear_weight"] @ value + p["linear_bias"]
        value = value.reshape(out_channels, n_fe, 1)
        lower_output = p["lower"][0, :out_channels]
        upper_output = p["upper"][0, :out_channels]
        value = (upper_output - lower_output) * (value + 1.0) / 2.0 + lower_output
        return value.reshape(-1)

    return forward, params
