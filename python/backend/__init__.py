"""backend: JAX device management and model loading infrastructure."""

from .device import DeviceManager
from .model_loader import ModelLoader

__all__ = ["DeviceManager", "ModelLoader"]
