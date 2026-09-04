"""
jax_nn_evaluator.py

Backward-compatible facade for the refactored JAX evaluator stack.

Public API (unchanged for compatibility with Julia NeuralNetworkNLPModel):
    create_evaluator(model_path, target, x0, **kwargs)
        → StandardEvaluator (single network, generic target matching)

New factory functions for proposal case studies:
    create_darcy_evaluator(fno_path, target_pressure, x0, **kwargs)
        → DarcyInversionEvaluator

    create_pareto_evaluator(f1_path, f2_path, x0, **kwargs)
        → MultiNetworkEvaluator

The module-level smooth_relu and load_pytorch_model_to_jax are re-exported
for any downstream code that imports them directly from here.
"""

from __future__ import annotations

import sys
import os
import logging
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

# ------------------------------------------------------------------
# Configure JAX globally on import (required before any backend init)
# ------------------------------------------------------------------
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

# ------------------------------------------------------------------
# Add this directory to sys.path so sub-packages are importable
# ------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ------------------------------------------------------------------
# Import from refactored sub-packages
# ------------------------------------------------------------------
from backend.device import DeviceManager
from backend.model_loader import ModelLoader
from models.mlp import smooth_relu, build_mlp_jax, build_resmlp_jax
from models.resnet import build_resnet_jax
from models.fno import build_fno2d_jax, fno2d_forward
from evaluators.standard import StandardEvaluator
from evaluators.multi_network import MultiNetworkEvaluator
from evaluators.darcy import DarcyInversionEvaluator
from evaluators.pdebench_darcy import PDEBenchDarcyLatentEvaluator
from evaluators.pfr_nmpc import PFRNMPCEvaluator
from evaluators.mnist_adversarial import MNISTAdversarialEvaluator
from evaluators.pfr_cnn_nmpc import PFRCNNNMPCEvaluator

log = logging.getLogger(__name__)


# ============================================================================
# Backward-compatible class alias
# ============================================================================

# JAXNeuralNetworkEvaluator is now an alias for StandardEvaluator so existing
# code that isinstance-checks against the old name continues to work.
JAXNeuralNetworkEvaluator = StandardEvaluator


# ============================================================================
# Backward-compatible model loader
# ============================================================================

def load_pytorch_model_to_jax(model_path: str, device: str = "cpu"):
    """Load a PyTorch checkpoint and return (forward_fn, params, config).

    This is the original public function, retained for backward compatibility.
    Now delegates to ModelLoader.
    """
    dm = DeviceManager(device)
    loader = ModelLoader(dm)
    return loader.load(model_path)


# ============================================================================
# Factory functions (called from Julia)
# ============================================================================

def create_evaluator(
    model_path: str,
    target,
    x0,
    *,
    objective_weight: float = 1.0,
    regularization_weight: float = 0.0,
    bounds=None,   # kept for API compatibility; box bounds are now lvar/uvar
    radius: Optional[float] = None,
    device: str = "cpu",
    **_ignored,
) -> StandardEvaluator:
    """Create a standard single-network JAX evaluator.

    Compatible with the original create_evaluator API.  Box bounds are now
    handled by Julia as lvar/uvar and should not be passed here.
    """
    dm = DeviceManager(device)
    loader = ModelLoader(dm)
    forward_fn, params, config = loader.load(model_path)

    target_arr = np.array(target, dtype=np.float64)
    x0_arr = np.array(x0, dtype=np.float64)

    return StandardEvaluator(
        forward_fn=forward_fn,
        params=params,
        target=target_arr,
        x0=x0_arr,
        objective_weight=objective_weight,
        regularization_weight=regularization_weight,
        radius=radius,
        device_manager=dm,
    )


def create_darcy_evaluator(
    fno_path: str,
    target_pressure,
    x0,
    *,
    regularization_weight: float = 1e-2,
    budget: Optional[float] = None,
    tau: Optional[float] = None,
    L_flat=None,
    L_n: int = 0,
    device: str = "cpu",
    **_ignored,
) -> DarcyInversionEvaluator:
    """Create a JAX evaluator for Darcy FNO inversion (Case Study I).

    Parameters
    ----------
    fno_path : str
        Path to the FNO checkpoint (.pt or .npz).
    target_pressure : array-like
        Target pressure field y_tar.
    x0 : array-like
        Initial permeability field.
    regularization_weight : float
        Tikhonov λ.
    budget : float or None
        Budget bound B; activates linear constraint 1ᵀx ≤ B.
    tau : float or None
        Smoothness bound τ; activates quadratic constraint xᵀLx ≤ τ.
    L_flat : flat array or None
        Flattened regularity operator (L_n × L_n) → length L_n².
    L_n : int
        Side length of L (must match when L_flat is supplied).
    device : str
        "cpu" or "gpu".
    """
    dm = DeviceManager(device)
    loader = ModelLoader(dm)
    forward_fn, params, config = loader.load(fno_path)

    L_matrix = None
    if tau is not None and L_flat is not None and L_n > 0:
        L_matrix = np.array(L_flat, dtype=np.float64).reshape(L_n, L_n)

    return DarcyInversionEvaluator(
        forward_fn=forward_fn,
        params=params,
        target_pressure=np.array(target_pressure, dtype=np.float64),
        x0=np.array(x0, dtype=np.float64),
        objective_weight=0.5,
        regularization_weight=regularization_weight,
        budget=budget,
        tau=tau,
        L_matrix=L_matrix,
        device_manager=dm,
    )


def create_pdebench_darcy_evaluator(
    fno_path: str,
    target_pressure,
    x0,
    *,
    reference_field=None,
    full_resolution: int = 128,
    observation_indices=None,
    regularization_weight: float = 1e-4,
    binary_weight: float = 0.0,
    mean_lower=None,
    mean_upper=None,
    tv_upper=None,
    device: str = "gpu",
):
    """Create the latent-field PDEBench Darcy inversion evaluator."""
    dm = DeviceManager(device)
    loader = ModelLoader(dm)
    forward_fn, params, _ = loader.load(fno_path)
    return PDEBenchDarcyLatentEvaluator(
        forward_fn,
        params,
        np.asarray(target_pressure, dtype=np.float64),
        np.asarray(x0, dtype=np.float64),
        reference_field=(
            None
            if reference_field is None
            else np.asarray(reference_field, dtype=np.float64)
        ),
        full_resolution=full_resolution,
        observation_indices=observation_indices,
        regularization_weight=regularization_weight,
        binary_weight=binary_weight,
        mean_lower=mean_lower,
        mean_upper=mean_upper,
        tv_upper=tv_upper,
        device_manager=dm,
    )


def create_pareto_evaluator(
    f1_path: str,
    f2_path: str,
    x0,
    *,
    f3_path: Optional[str] = None,
    alpha: float = 0.5,
    device: str = "cpu",
    **_ignored,
) -> MultiNetworkEvaluator:
    """Create a JAX evaluator for constrained Pareto tracing (Case Study II).

    Parameters
    ----------
    f1_path, f2_path : str
        Checkpoint paths for the two competing objectives.
    x0 : array-like
        Initial point.
    f3_path : str or None
        Checkpoint path for the learned feasibility surrogate h (optional).
    alpha : float
        Scalarization weight α ∈ [0,1].
    device : str
        "cpu" or "gpu".
    """
    dm = DeviceManager(device)
    loader = ModelLoader(dm)

    paths = [f1_path, f2_path]
    if f3_path is not None:
        paths.append(f3_path)

    forward_fns, params_list = [], []
    for path in paths:
        fn, p, _ = loader.load(path)
        forward_fns.append(fn)
        params_list.append(p)

    return MultiNetworkEvaluator(
        forward_fns=forward_fns,
        params_list=params_list,
        alpha=alpha,
        x0=np.array(x0, dtype=np.float64),
        device_manager=dm,
    )


def create_pfr_nmpc_evaluator(
    model_path: str,
    current_state,
    previous_control,
    setpoint: float,
    *,
    prediction_horizon: int = 40,
    control_horizon: int = 10,
    parameters_as_variables: bool = False,
    device: str = "gpu",
) -> PFRNMPCEvaluator:
    """Create the published case-study-1 PFR gray-box NMPC oracle."""
    dm = DeviceManager(device)
    loader = ModelLoader(dm)
    forward_fn, params, _ = loader.load(model_path)
    return PFRNMPCEvaluator(
        forward_fn,
        params,
        np.asarray(current_state, dtype=np.float64),
        np.asarray(previous_control, dtype=np.float64),
        setpoint,
        prediction_horizon=prediction_horizon,
        control_horizon=control_horizon,
        parameters_as_variables=parameters_as_variables,
        device_manager=dm,
    )


def create_mnist_adversarial_evaluator(
    model_path: str,
    reference_image,
    adversarial_label: int,
    *,
    threshold: float = 0.6,
    device: str = "gpu",
) -> MNISTAdversarialEvaluator:
    """Create the closest-prior L1 adversarial MNIST problem oracle."""
    dm = DeviceManager(device)
    loader = ModelLoader(dm)
    forward_fn, params, _ = loader.load(model_path)
    return MNISTAdversarialEvaluator(
        forward_fn,
        params,
        np.asarray(reference_image, dtype=np.float64),
        adversarial_label,
        threshold=threshold,
        device_manager=dm,
    )


def create_pfr_cnn_nmpc_evaluator(
    model_path: str,
    current_state,
    previous_control,
    tracking_channels,
    tracking_indices,
    tracking_targets,
    tracking_scales,
    control_scales,
    *,
    prediction_horizon: int,
    control_horizon: int,
    device: str = "gpu",
) -> PFRCNNNMPCEvaluator:
    dm = DeviceManager(device)
    loader = ModelLoader(dm)
    forward_fn, params, config = loader.load(model_path)
    return PFRCNNNMPCEvaluator(
        forward_fn,
        params,
        config,
        np.asarray(current_state, dtype=np.float64),
        np.asarray(previous_control, dtype=np.float64),
        tracking_channels,
        tracking_indices,
        tracking_targets,
        tracking_scales,
        control_scales,
        prediction_horizon=prediction_horizon,
        control_horizon=control_horizon,
        device_manager=dm,
    )


def load_pfr_cnn_published_case(output_path: str, case: int):
    """Load first-step state/control data from the companion PFR artifacts."""
    import pickle

    with open(output_path, "rb") as handle:
        data = pickle.load(handle)
    if case == 2:
        current = np.concatenate((data["C"][0], data["T"][0]))
        previous = np.asarray(
            [data["F"][0], data["Ta"][0], data["C0"][0], data["T0"][0]],
            dtype=np.float64,
        )
        first = np.asarray(
            [data["F"][1], data["Ta"][1], data["C0"][1], data["T0"][1]],
            dtype=np.float64,
        )
    elif case == 3:
        components = ["CH4", "H2O", "H2", "CO", "CO2"]
        spatial = sorted(
            key[1]
            for key in data
            if isinstance(key, tuple) and key[0] == "FCH4" and key[1] != 0
        )
        current = np.concatenate(
            [
                [data[(f"F{component}", location)][0] for location in spatial]
                for component in components
            ]
        )
        previous = np.asarray(
            [data["FCH4_in"][0], data["FH2O_in"][0], data["T_in"][0]],
            dtype=np.float64,
        )
        first = np.asarray(
            [data["FCH4_in"][1], data["FH2O_in"][1], data["T_in"][1]],
            dtype=np.float64,
        )
    else:
        raise ValueError("PFR CNN case must be 2 or 3")
    return {
        "current_state": np.asarray(current, dtype=np.float64),
        "previous_control": previous,
        "published_first_control": first,
    }


# ============================================================================
# Standalone test
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        forward_fn, params, config = load_pytorch_model_to_jax(model_path)
        n_in = config.get("input_dim", 784)
        n_out = config.get("output_dim", 10)
        x_test = np.random.randn(n_in)
        target = np.zeros(n_out)
        target[0] = 1.0

        ev = create_evaluator(model_path, target, x_test)
        print(f"Objective: {ev.evaluate_obj(x_test):.4f}")
        print(f"Gradient norm: {np.linalg.norm(ev.evaluate_grad(x_test)):.4f}")
        print("Evaluator test passed!")
    else:
        print("Usage: python jax_nn_evaluator.py <model_path>")
