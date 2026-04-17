"""
device.py

Centralized JAX device management.

Supports explicit CPU or NVIDIA GPU placement so that both evaluation
paths (Julia/Flux and Python/JAX) can be directed to the same hardware
resource.

Usage
-----
    dm = DeviceManager("gpu")          # NVIDIA GPU
    x_jax = dm.array(x_numpy)         # array on GPU
    model_params = dm.put(model_params)  # move param pytree to GPU
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

log = logging.getLogger(__name__)

SUPPORTED_DEVICES = ("cpu", "gpu", "auto")


class DeviceManager:
    """Manage JAX device selection and array placement.

    Parameters
    ----------
    device : str
        One of "cpu", "gpu", or "auto".  With "auto" the manager picks
        a GPU if any is available, otherwise falls back to CPU.
    """

    def __init__(self, device: str = "cpu") -> None:
        if device not in SUPPORTED_DEVICES:
            raise ValueError(
                f"device must be one of {SUPPORTED_DEVICES}, got {device!r}"
            )
        self.device_str = device
        self._jax_device = None
        self._setup()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """Apply global JAX config and select the concrete device."""
        # Float64 and deterministic matmul are required for MadNLP compatibility
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")

        if self.device_str == "cpu":
            # Hint to JAX – only effective before the backend is initialised
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
            try:
                self._jax_device = jax.devices("cpu")[0]
            except Exception:
                self._jax_device = jax.devices()[0]

        elif self.device_str == "gpu":
            try:
                self._jax_device = jax.devices("gpu")[0]
                log.info("DeviceManager: using GPU device %s", self._jax_device)
            except Exception as exc:
                raise RuntimeError(
                    "GPU device requested but no JAX GPU backend is available. "
                    "Install jaxlib with CUDA support (e.g. `pip install jax[cuda12]`)."
                ) from exc

        else:  # "auto"
            all_devs = jax.devices()
            gpu_devs = [
                d for d in all_devs
                if d.platform.lower() in ("gpu", "cuda")
            ]
            self._jax_device = gpu_devs[0] if gpu_devs else all_devs[0]
            log.info(
                "DeviceManager (auto): selected %s",
                self._jax_device,
            )

    # ------------------------------------------------------------------
    # Array helpers
    # ------------------------------------------------------------------

    @property
    def jax_device(self):
        return self._jax_device

    def put(self, pytree: Any) -> Any:
        """Place a JAX array or a pytree of arrays on the managed device."""
        return jax.device_put(pytree, self._jax_device)

    def array(self, data, dtype=jnp.float64) -> jnp.ndarray:
        """Create a JAX array on the managed device."""
        return self.put(jnp.array(data, dtype=dtype))

    def to_numpy(self, x: jnp.ndarray) -> np.ndarray:
        """Convert a JAX array to a CPU NumPy array."""
        return np.array(x)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def is_gpu(self) -> bool:
        return self._jax_device.platform.lower() in ("gpu", "cuda")

    def describe(self) -> Dict[str, Any]:
        return {
            "requested": self.device_str,
            "actual": str(self._jax_device),
            "platform": self._jax_device.platform,
            "is_gpu": self.is_gpu,
        }

    def __repr__(self) -> str:
        d = self.describe()
        return f"DeviceManager(requested={d['requested']!r}, actual={d['actual']!r})"
