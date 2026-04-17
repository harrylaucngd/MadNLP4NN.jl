"""
resnet.py

JAX builder for CIFAR-style ResNets.

Extracted from jax_nn_evaluator.py and placed in its own module.
All JAX primitive ops (conv, batchnorm, pool) are implemented here.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp

from .mlp import smooth_relu


# ---------------------------------------------------------------------------
# JAX primitives
# ---------------------------------------------------------------------------

def _conv2d(x, weight, bias=None, stride: int = 1, padding: int = 0):
    """2-D convolution: x (C,H,W), weight (C_out,C_in,KH,KW) → (C_out,H',W')."""
    if x.ndim == 3:
        x_fmt = jnp.expand_dims(jnp.transpose(x, (1, 2, 0)), 0)  # (1,H,W,C)
    else:
        x_fmt = x
    w_fmt = jnp.transpose(weight, (2, 3, 1, 0))  # (KH,KW,C_in,C_out)
    out = jax.lax.conv_general_dilated(
        x_fmt, w_fmt,
        window_strides=(stride, stride),
        padding=((padding, padding), (padding, padding)),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )  # (1,H',W',C_out)
    if bias is not None:
        out = out + bias.reshape(1, 1, 1, -1)
    return jnp.transpose(out[0], (2, 0, 1))  # (C_out,H',W')


def _batchnorm(x, weight, bias, running_mean, running_var, eps: float = 1e-5):
    """Inference-mode batch normalisation: x (C,H,W)."""
    mean = running_mean.reshape(-1, 1, 1)
    var = running_var.reshape(-1, 1, 1)
    gamma = weight.reshape(-1, 1, 1)
    beta = bias.reshape(-1, 1, 1)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta


def _avg_pool2d(x, kernel_size: int, stride: int = None):
    if stride is None:
        stride = kernel_size
    x_fmt = jnp.expand_dims(jnp.transpose(x, (1, 2, 0)), 0)
    out = jax.lax.reduce_window(
        x_fmt, 0.0, jax.lax.add,
        window_dimensions=(1, kernel_size, kernel_size, 1),
        window_strides=(1, stride, stride, 1),
        padding="VALID",
    ) / (kernel_size * kernel_size)
    return jnp.transpose(out[0], (2, 0, 1))


# ---------------------------------------------------------------------------
# ResNet builder
# ---------------------------------------------------------------------------

def build_resnet_jax(
    state_dict: Dict,
    config: Dict,
) -> Tuple[Callable, Dict]:
    """Build a JAX CIFAR-style ResNet from a PyTorch state dict."""
    def _arr(key):
        return jnp.array(state_dict[key].detach().cpu().numpy(), dtype=jnp.float64)

    params = {
        "conv1": {"weight": _arr("conv1.weight")},
        "bn1": {
            "weight": _arr("bn1.weight"),
            "bias": _arr("bn1.bias"),
            "running_mean": _arr("bn1.running_mean"),
            "running_var": _arr("bn1.running_var"),
        },
        "layers": {},
        "fc": {
            "weight": _arr("fc.weight"),
            "bias": _arr("fc.bias"),
        },
    }

    for layer_name in ("layer1", "layer2", "layer3"):
        layer_blocks = []
        block_idx = 0
        while f"{layer_name}.{block_idx}.conv1.weight" in state_dict:
            pfx = f"{layer_name}.{block_idx}"
            bp = {
                "conv1": {"weight": _arr(f"{pfx}.conv1.weight")},
                "bn1": {
                    "weight": _arr(f"{pfx}.bn1.weight"),
                    "bias": _arr(f"{pfx}.bn1.bias"),
                    "running_mean": _arr(f"{pfx}.bn1.running_mean"),
                    "running_var": _arr(f"{pfx}.bn1.running_var"),
                },
                "conv2": {"weight": _arr(f"{pfx}.conv2.weight")},
                "bn2": {
                    "weight": _arr(f"{pfx}.bn2.weight"),
                    "bias": _arr(f"{pfx}.bn2.bias"),
                    "running_mean": _arr(f"{pfx}.bn2.running_mean"),
                    "running_var": _arr(f"{pfx}.bn2.running_var"),
                },
            }
            if f"{pfx}.shortcut.0.weight" in state_dict:
                bp["shortcut"] = {
                    "conv": {"weight": _arr(f"{pfx}.shortcut.0.weight")},
                    "bn": {
                        "weight": _arr(f"{pfx}.shortcut.1.weight"),
                        "bias": _arr(f"{pfx}.shortcut.1.bias"),
                        "running_mean": _arr(f"{pfx}.shortcut.1.running_mean"),
                        "running_var": _arr(f"{pfx}.shortcut.1.running_var"),
                    },
                }
            layer_blocks.append(bp)
            block_idx += 1
        params["layers"][layer_name] = layer_blocks

    layer_strides = {"layer1": (1, 1), "layer2": (2, 1), "layer3": (2, 1)}
    input_dim = config.get("input_dim", 3072)
    in_ch = config.get("input_channels", 3)
    img_h = img_w = int(round((input_dim // in_ch) ** 0.5))

    def forward(p, x):
        if x.ndim == 1:
            x = x.reshape(in_ch, img_h, img_w)
        out = _conv2d(x, p["conv1"]["weight"], bias=None, stride=1, padding=1)
        out = _batchnorm(out, p["bn1"]["weight"], p["bn1"]["bias"],
                         p["bn1"]["running_mean"], p["bn1"]["running_var"])
        out = smooth_relu(out)
        for lname in ("layer1", "layer2", "layer3"):
            fs, os_ = layer_strides[lname]
            for bi, block in enumerate(p["layers"][lname]):
                identity = out
                stride = fs if bi == 0 else os_
                res = _conv2d(out, block["conv1"]["weight"], stride=stride, padding=1)
                res = _batchnorm(res, block["bn1"]["weight"], block["bn1"]["bias"],
                                 block["bn1"]["running_mean"], block["bn1"]["running_var"])
                res = smooth_relu(res)
                res = _conv2d(res, block["conv2"]["weight"], stride=1, padding=1)
                res = _batchnorm(res, block["bn2"]["weight"], block["bn2"]["bias"],
                                 block["bn2"]["running_mean"], block["bn2"]["running_var"])
                if "shortcut" in block:
                    identity = _conv2d(identity, block["shortcut"]["conv"]["weight"],
                                       stride=stride, padding=0)
                    identity = _batchnorm(identity,
                                          block["shortcut"]["bn"]["weight"],
                                          block["shortcut"]["bn"]["bias"],
                                          block["shortcut"]["bn"]["running_mean"],
                                          block["shortcut"]["bn"]["running_var"])
                elif stride > 1:
                    identity = _avg_pool2d(identity, kernel_size=stride, stride=stride)
                    pad_ch = res.shape[0] - identity.shape[0]
                    if pad_ch > 0:
                        identity = jnp.concatenate(
                            [identity,
                             jnp.zeros((pad_ch, identity.shape[1], identity.shape[2]))],
                            axis=0,
                        )
                out = smooth_relu(res + identity)
        out = jnp.mean(out, axis=(1, 2))
        out = jnp.dot(p["fc"]["weight"], out) + p["fc"]["bias"]
        return out

    return forward, params
