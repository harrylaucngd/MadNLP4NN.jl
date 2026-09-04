"""evaluators: composable JAX evaluator classes."""

from .base import BaseEvaluator
from .standard import StandardEvaluator
from .multi_network import MultiNetworkEvaluator
from .darcy import DarcyInversionEvaluator
from .pdebench_darcy import PDEBenchDarcyLatentEvaluator
from .pfr_nmpc import PFRNMPCEvaluator
from .mnist_adversarial import MNISTAdversarialEvaluator
from .pfr_cnn_nmpc import PFRCNNNMPCEvaluator

__all__ = [
    "BaseEvaluator",
    "StandardEvaluator",
    "MultiNetworkEvaluator",
    "DarcyInversionEvaluator",
    "PDEBenchDarcyLatentEvaluator",
    "PFRNMPCEvaluator",
    "MNISTAdversarialEvaluator",
    "PFRCNNNMPCEvaluator",
]
