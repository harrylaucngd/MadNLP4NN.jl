"""JAX implementation of the PDEBench 2-D Darcy FNO architecture.

The implementation follows the public PDEBench FNO2d definition and supports
its released checkpoints. It keeps both positive and negative first-axis
Fourier weights and appends cell-center coordinates to the coefficient field.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np


def _torch_array(tensor, dtype):
    value = tensor.detach().cpu().numpy()
    return jnp.asarray(value, dtype=dtype)


def build_pdebench_fno2d_jax(
    state_dict: Dict,
    config: Dict,
) -> Tuple[Callable, Dict]:
    """Build the released PDEBench FNO2d as ``forward(params, x_flat)``."""
    real_dtype = config.get("dtype", jnp.float64)
    complex_dtype = jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64
    layer_count = 0
    while f"conv{layer_count}.weights1" in state_dict:
        layer_count += 1
    if layer_count == 0:
        raise ValueError("Checkpoint does not contain PDEBench Fourier layers")

    padding = int(config.get("padding", 2))
    modes1 = int(state_dict["conv0.weights1"].shape[-2])
    modes2 = int(state_dict["conv0.weights1"].shape[-1])
    params = {
        "fc0_weight": _torch_array(state_dict["fc0.weight"], real_dtype),
        "fc0_bias": _torch_array(state_dict["fc0.bias"], real_dtype),
        "fc1_weight": _torch_array(state_dict["fc1.weight"], real_dtype),
        "fc1_bias": _torch_array(state_dict["fc1.bias"], real_dtype),
        "fc2_weight": _torch_array(state_dict["fc2.weight"], real_dtype),
        "fc2_bias": _torch_array(state_dict["fc2.bias"], real_dtype),
        "layers": [],
        "coordinate_x": (
            None
            if config.get("coordinate_x") is None
            else jnp.asarray(config["coordinate_x"], dtype=real_dtype)
        ),
        "coordinate_y": (
            None
            if config.get("coordinate_y") is None
            else jnp.asarray(config["coordinate_y"], dtype=real_dtype)
        ),
    }
    for index in range(layer_count):
        params["layers"].append(
            {
                "weights1": _torch_array(
                    state_dict[f"conv{index}.weights1"], complex_dtype
                ),
                "weights2": _torch_array(
                    state_dict[f"conv{index}.weights2"], complex_dtype
                ),
                "pointwise_weight": _torch_array(
                    state_dict[f"w{index}.weight"][:, :, 0, 0], real_dtype
                ),
                "pointwise_bias": _torch_array(
                    state_dict[f"w{index}.bias"], real_dtype
                ),
            }
        )

    def spectral_convolution(layer, value):
        # value: (channels, nx, ny), identical to the no-batch PDEBench path.
        nx, ny = value.shape[-2:]
        transformed = jnp.fft.rfft2(value, axes=(-2, -1))
        output = jnp.zeros(
            (value.shape[0], nx, ny // 2 + 1), dtype=complex_dtype
        )
        positive = jnp.einsum(
            "ixy,ioxy->oxy",
            transformed[:, :modes1, :modes2],
            layer["weights1"],
        )
        negative = jnp.einsum(
            "ixy,ioxy->oxy",
            transformed[:, -modes1:, :modes2],
            layer["weights2"],
        )
        output = output.at[:, :modes1, :modes2].set(positive)
        output = output.at[:, -modes1:, :modes2].set(negative)
        return jnp.fft.irfft2(output, s=(nx, ny), axes=(-2, -1))

    def forward(p, x_flat):
        spatial_size = int(round(np.sqrt(x_flat.shape[0])))
        if spatial_size * spatial_size != x_flat.shape[0]:
            raise ValueError("PDEBench FNO input length must be a square")
        if p["coordinate_x"] is None:
            coordinate_offset = (
                0.0 if config.get("coordinate_scheme") == "left_endpoint" else 0.5
            )
            coordinate_x = (
                jnp.arange(spatial_size, dtype=real_dtype) + coordinate_offset
            ) / spatial_size
        else:
            coordinate_x = p["coordinate_x"]
        if p["coordinate_y"] is None:
            coordinate_offset = (
                0.0 if config.get("coordinate_scheme") == "left_endpoint" else 0.5
            )
            coordinate_y = (
                jnp.arange(spatial_size, dtype=real_dtype) + coordinate_offset
            ) / spatial_size
        else:
            coordinate_y = p["coordinate_y"]
        if coordinate_x.shape[0] != spatial_size or coordinate_y.shape[0] != spatial_size:
            raise ValueError("Configured PDEBench coordinates do not match input resolution")
        grid_x, grid_y = jnp.meshgrid(coordinate_x, coordinate_y, indexing="ij")
        coefficient = jnp.reshape(x_flat, (spatial_size, spatial_size, 1))
        value = jnp.concatenate(
            [coefficient, grid_x[..., None], grid_y[..., None]], axis=-1
        )
        value = jnp.einsum("...i,oi->...o", value, p["fc0_weight"]) + p["fc0_bias"]
        value = jnp.transpose(value, (2, 0, 1))
        value = jnp.pad(value, ((0, 0), (0, padding), (0, padding)))

        for index, layer in enumerate(p["layers"]):
            spectral = spectral_convolution(layer, value)
            pointwise = jnp.einsum(
                "oi,ihw->ohw", layer["pointwise_weight"], value
            ) + layer["pointwise_bias"][:, None, None]
            value = spectral + pointwise
            if index + 1 < len(p["layers"]):
                value = jax.nn.gelu(value, approximate=False)

        value = value[:, :-padding, :-padding]
        value = jnp.transpose(value, (1, 2, 0))
        value = jnp.einsum("...i,oi->...o", value, p["fc1_weight"]) + p["fc1_bias"]
        value = jax.nn.gelu(value, approximate=False)
        value = jnp.einsum("...i,oi->...o", value, p["fc2_weight"]) + p["fc2_bias"]
        return jnp.reshape(value, (-1,))

    return forward, params
