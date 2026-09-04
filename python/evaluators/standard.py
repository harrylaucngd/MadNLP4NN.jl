"""
standard.py

StandardEvaluator: single-network JAX evaluator for the canonical
target-matching NLP.

This is a clean rewrite of the original JAXNeuralNetworkEvaluator,
re-using the shared backend infrastructure.  It replaces the monolithic
class while exposing the same public interface.

Objective:
    f(x) = obj_weight * ||N_θ(x) - target||² + reg_weight * ||x - x0||²

Constraints (optional):
    - Spherical: ||x - x0||² - radius² ≤ 0
    (box bounds are handled by Julia as lvar/uvar)
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Optional, Union, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .base import BaseEvaluator

log = logging.getLogger(__name__)


class StandardEvaluator(BaseEvaluator):
    """Single-network JAX evaluator for target-matching optimization.

    Parameters
    ----------
    forward_fn : callable
        JAX forward function ``(params, x) -> output``.
    params : pytree
        Model parameters (JAX arrays).
    target : array-like, shape (m_out,)
        Target network output.
    x0 : array-like, shape (n,)
        Initial point (used as regularization center).
    objective_weight : float
        Weight on the squared misfit term.
    regularization_weight : float
        Weight on the Tikhonov regularization term.
    radius : float or None
        Spherical constraint radius (None = no spherical constraint).
    device_manager : DeviceManager, optional
        If provided, arrays are placed on the managed device.
    """

    def __init__(
        self,
        forward_fn,
        params,
        target: np.ndarray,
        x0: np.ndarray,
        objective_weight: float = 1.0,
        regularization_weight: float = 0.0,
        radius: Optional[float] = None,
        device_manager=None,
    ) -> None:
        self.dm = device_manager
        self._put = (lambda a: a) if device_manager is None else device_manager.put

        self.forward_fn = forward_fn
        self.params = params
        self.target = self._put(jnp.array(target, dtype=jnp.float64))
        self.x0 = self._put(jnp.array(x0, dtype=jnp.float64))
        self.obj_weight = float(objective_weight)
        self.reg_weight = float(regularization_weight)
        self.radius = float(radius) if radius is not None else None

        self.n = int(len(x0))
        self.m = 1 if self.radius is not None else 0

        log.info(
            "StandardEvaluator: n=%d, m=%d, radius=%s",
            self.n, self.m, self.radius,
        )

        self._compile()

    # ------------------------------------------------------------------
    # JIT compilation
    # ------------------------------------------------------------------

    def _compile(self) -> None:
        """Build and JIT-compile all derivative functions."""
        params = self.params
        forward_fn = self.forward_fn
        target = self.target
        x0 = self.x0
        obj_w = self.obj_weight
        reg_w = self.reg_weight
        radius = self.radius

        def _obj(x):
            out = forward_fn(params, x)
            f = obj_w * jnp.sum((out - target) ** 2)
            if reg_w > 0.0:
                f += reg_w * jnp.sum((x - x0) ** 2)
            return f

        def _cons(x):
            if radius is None:
                return jnp.zeros(0, dtype=jnp.float64)
            diff = x - x0
            return jnp.array([jnp.sum(diff ** 2) - radius ** 2])

        def _lagrangian(x, y, w):
            L = w * _obj(x)
            if radius is not None:
                c = _cons(x)
                L += jnp.dot(y, c)
            return L

        self._obj_jit = jax.jit(_obj)
        self._grad_jit = jax.jit(jax.grad(_obj))
        self._forward_jit = jax.jit(partial(forward_fn, params))

        if self.m > 0:
            self._cons_jit = jax.jit(_cons)
            self._jac_jit = jax.jit(jax.jacobian(_cons))
            self._hess_lag_jit = jax.jit(
                jax.hessian(_lagrangian, argnums=0)
            )
            self._hess_lag_fn = self._hess_lag_jit
        else:
            self._cons_jit = lambda x: jnp.zeros(0, dtype=jnp.float64)
            self._jac_jit = lambda x: jnp.zeros((0, self.n), dtype=jnp.float64)

        self._hess_obj_jit = jax.jit(jax.hessian(_obj))

    # ------------------------------------------------------------------
    # BaseEvaluator interface
    # ------------------------------------------------------------------

    def evaluate_nn(self, x: np.ndarray) -> np.ndarray:
        x_jax = self._put(jnp.array(x, dtype=jnp.float64))
        return np.array(self._forward_jit(x_jax))

    def evaluate_obj(self, x: np.ndarray) -> float:
        x_jax = self._put(jnp.array(x, dtype=jnp.float64))
        return float(self._obj_jit(x_jax))

    def evaluate_grad(self, x: np.ndarray) -> np.ndarray:
        x_jax = self._put(jnp.array(x, dtype=jnp.float64))
        return np.array(self._grad_jit(x_jax))

    def evaluate_cons(self, x: np.ndarray) -> np.ndarray:
        x_jax = self._put(jnp.array(x, dtype=jnp.float64))
        return np.array(self._cons_jit(x_jax))

    def evaluate_jac(self, x: np.ndarray) -> np.ndarray:
        x_jax = self._put(jnp.array(x, dtype=jnp.float64))
        return np.array(self._jac_jit(x_jax))

    def evaluate_hess(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray],
        obj_weight: float = 1.0,
    ) -> np.ndarray:
        x_jax = self._put(jnp.array(x, dtype=jnp.float64))

        if self.m > 0 and y is not None and not np.allclose(y, 0):
            y_jax = self._put(jnp.array(y, dtype=jnp.float64))
            H = self._hess_lag_fn(x_jax, y_jax, float(obj_weight))
        else:
            H = float(obj_weight) * self._hess_obj_jit(x_jax)

        return np.array(H)
