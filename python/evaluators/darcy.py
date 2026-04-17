"""
darcy.py

DarcyInversionEvaluator: JAX evaluator for Case Study I — neural operator
inversion of 2-D Darcy flow.

Objective:
    f(x) = (w/2) * ||G_θ(x) - y_tar||² + (λ/2) * ||x - x0||²

Constraints (all optional, independently activatable):
    budget:     1ᵀx ≤ B             → c1(x) = sum(x) - B ≤ 0
    smoothness: xᵀ L x ≤ τ          → c2(x) = x'Lx - τ ≤ 0

Box bounds are handled on the Julia side as lvar/uvar.
"""

from __future__ import annotations

import logging
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .base import BaseEvaluator

log = logging.getLogger(__name__)


class DarcyInversionEvaluator(BaseEvaluator):
    """JAX evaluator for constrained FNO-based Darcy inversion.

    Parameters
    ----------
    forward_fn : callable
        JAX forward function for the FNO, ``(params, x) -> y``.
    params : pytree
        FNO parameter pytree.
    target_pressure : array-like, shape (n_out,)
        Target pressure field y_tar.
    x0 : array-like, shape (n,)
        Initial permeability field (regularization center).
    objective_weight : float
        Weight w on the squared misfit term (default 0.5).
    regularization_weight : float
        Tikhonov weight λ (default 1e-2).
    budget : float or None
        Budget bound B for the linear constraint.
    tau : float or None
        Smoothness bound τ for the quadratic constraint.
    L_matrix : np.ndarray or None
        Regularity operator L, shape (n, n).  Required when tau is given.
    device_manager : DeviceManager, optional
    """

    def __init__(
        self,
        forward_fn,
        params,
        target_pressure: np.ndarray,
        x0: np.ndarray,
        objective_weight: float = 0.5,
        regularization_weight: float = 1e-2,
        budget: Optional[float] = None,
        tau: Optional[float] = None,
        L_matrix: Optional[np.ndarray] = None,
        device_manager=None,
    ) -> None:
        self.dm = device_manager
        self._put = (lambda a: a) if device_manager is None else device_manager.put

        self.forward_fn = forward_fn
        self.params = self._put(params)
        self.target = self._put(jnp.array(target_pressure, dtype=jnp.float64))
        self.x0 = self._put(jnp.array(x0, dtype=jnp.float64))
        self.obj_w = float(objective_weight)
        self.reg_w = float(regularization_weight)

        self.budget = float(budget) if budget is not None else None
        self.tau = float(tau) if tau is not None else None
        self.L = None
        if tau is not None and L_matrix is not None:
            self.L = self._put(jnp.array(L_matrix, dtype=jnp.float64))

        self.n = int(len(x0))
        self.m = (1 if self.budget is not None else 0) + (1 if self.tau is not None else 0)

        log.info(
            "DarcyInversionEvaluator: n=%d, m=%d, budget=%s, tau=%s",
            self.n, self.m, self.budget, self.tau,
        )

        self._compile()

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def _compile(self) -> None:
        params = self.params
        fwd = self.forward_fn
        target = self.target
        x0 = self.x0
        obj_w = self.obj_w
        reg_w = self.reg_w
        budget = self.budget
        tau = self.tau
        L = self.L

        def _fwd(x):
            return fwd(params, x)

        def _obj(x):
            out = _fwd(x)
            f = obj_w * jnp.sum((out - target) ** 2)
            if reg_w > 0.0:
                f += reg_w * jnp.sum((x - x0) ** 2)
            return f

        def _cons(x):
            parts = []
            if budget is not None:
                parts.append(jnp.array([jnp.sum(x) - budget]))
            if tau is not None and L is not None:
                parts.append(jnp.array([jnp.dot(x, jnp.dot(L, x)) - tau]))
            return jnp.concatenate(parts) if parts else jnp.zeros(0, dtype=jnp.float64)

        def _lagrangian(x, y, w):
            Lval = w * _obj(x)
            if self.m > 0:
                c = _cons(x)
                Lval += jnp.dot(y, c)
            return Lval

        self._fwd_jit = jax.jit(_fwd)
        self._obj_jit = jax.jit(_obj)
        self._grad_jit = jax.jit(jax.grad(_obj))
        self._cons_jit = jax.jit(_cons)
        self._jac_jit = jax.jit(jax.jacobian(_cons))
        self._hess_obj_jit = jax.jit(jax.hessian(_obj))

        if self.m > 0:
            self._hess_lag_fn = lambda x, y, w: jax.hessian(
                lambda x_: _lagrangian(x_, y, w)
            )(x)
        else:
            self._hess_lag_fn = None

    # ------------------------------------------------------------------
    # BaseEvaluator interface
    # ------------------------------------------------------------------

    def evaluate_nn(self, x: np.ndarray) -> np.ndarray:
        x_jax = self._put(jnp.array(x, dtype=jnp.float64))
        return np.array(self._fwd_jit(x_jax))

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
