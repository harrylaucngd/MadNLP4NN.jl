"""
model_loader.py

Unified model loading from PyTorch checkpoints to JAX parameter pytrees.

Delegates to architecture-specific builders in python/models/.  Supports:
- MLP and ResMLP (from project's own training pipeline)
- CIFAR-style ResNet (acquired pretrained checkpoints)
- FNO2D (Fourier Neural Operator for Darcy flow)
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Tuple

import torch

log = logging.getLogger(__name__)


class ModelLoader:
    """Load PyTorch checkpoints and convert them to JAX parameter dicts.

    Parameters
    ----------
    device_manager : DeviceManager, optional
        If supplied, parameters are placed on the managed device after
        loading.  If None, parameters stay on the default JAX device.
    """

    def __init__(self, device_manager=None) -> None:
        self.dm = device_manager

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(
        self,
        model_path: str,
    ) -> Tuple[Callable, Dict, Dict]:
        """Load a checkpoint and return ``(forward_fn, params, config)``.

        The returned ``forward_fn`` has signature ``(params, x) -> output``.
        """
        log.info("Loading checkpoint: %s", model_path)

        checkpoint = torch.load(model_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            config = checkpoint.get("model_config", {})
        else:
            state_dict = checkpoint
            config = _infer_config(state_dict)

        model_type = config.get("model_type", "unknown").lower()
        log.info("  model_type=%s", model_type)

        forward_fn, params = self._build(model_type, state_dict, config)

        if self.dm is not None:
            params = self.dm.put(params)

        return forward_fn, params, config

    # ------------------------------------------------------------------
    # Dispatch to architecture builders
    # ------------------------------------------------------------------

    def _build(
        self,
        model_type: str,
        state_dict: Dict,
        config: Dict,
    ) -> Tuple[Callable, Dict]:
        if model_type in ("mlp", "mlpclassifier"):
            from models.mlp import build_mlp_jax
            return build_mlp_jax(state_dict, config)
        elif model_type in ("resmlp", "resmlpclassifier"):
            from models.mlp import build_resmlp_jax
            return build_resmlp_jax(state_dict, config)
        elif model_type in ("resnet",):
            from models.resnet import build_resnet_jax
            return build_resnet_jax(state_dict, config)
        elif model_type in ("fno", "fno2d"):
            from models.fno import build_fno2d_jax
            return build_fno2d_jax(state_dict, config)
        else:
            raise ValueError(
                f"Unknown model type: {model_type!r}. "
                f"Supported: mlp, resmlp, resnet, fno2d."
            )


# ============================================================================
# Config inference (fallback when no model_config in checkpoint)
# ============================================================================

def _infer_config(state_dict: Dict) -> Dict:
    """Best-effort architecture detection from weight key names."""
    keys = list(state_dict.keys())
    has_conv = any(
        "conv" in k.lower() and len(state_dict[k].shape) == 4
        for k in keys
        if "weight" in k
    )
    is_resnet = has_conv and any(
        k.startswith(("layer1.", "layer2.", "layer3.")) for k in keys
    )
    is_fno = any("fourier_layers" in k or "spectral_conv" in k for k in keys)

    if is_fno:
        return {"model_type": "fno2d"}
    if is_resnet:
        ic = state_dict["conv1.weight"].shape[1] if "conv1.weight" in state_dict else 3
        od = state_dict["fc.weight"].shape[0] if "fc.weight" in state_dict else 10
        return {
            "model_type": "resnet",
            "input_dim": ic * 32 * 32,
            "output_dim": int(od),
            "input_channels": int(ic),
        }

    is_resmlp = any("blocks." in k for k in keys)
    is_mlp = any(
        k.startswith(("network.", "layers.")) for k in keys
    )

    if is_resmlp:
        model_type = "resmlp"
    elif is_mlp:
        model_type = "mlp"
    else:
        raise ValueError(
            f"Cannot infer model type from state_dict keys: {keys[:5]} ..."
        )

    layer_shapes = []
    for k in sorted(keys):
        if "weight" in k and ("network." in k or "layers." in k):
            parts = k.split(".")
            try:
                idx = int(parts[1])
                s = state_dict[k].shape
                if len(s) == 2:
                    layer_shapes.append((idx, s))
            except (ValueError, IndexError):
                continue

    if len(layer_shapes) < 2:
        raise ValueError("Could not determine layer structure from state_dict.")

    layer_shapes.sort()
    input_dim = int(layer_shapes[0][1][1])
    hidden_dims = [int(s[1][0]) for s in layer_shapes[:-1]]
    output_dim = int(layer_shapes[-1][1][0])

    return {
        "model_type": model_type,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dims": hidden_dims,
        "activation": "relu",
        "dropout_rate": 0.0,
    }
