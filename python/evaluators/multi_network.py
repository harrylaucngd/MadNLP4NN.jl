"""
multi_network.py

MultiNetworkEvaluator: JAX evaluator for constrained Pareto tracing
(Case Study II).

Wraps two objective networks (f1, f2) and an optional feasibility surrogate
(h) into a single NLP:

    min_x   α · f1(x) + (1-α) · f2(x)
    s.t.    h(x) ≤ 0              [when h is provided]

All three networks share the same decision variable x.

This evaluator also exposes evaluate_individuals(x) which returns
[f1(x), f2(x)] (and optionally h(x)) so that the Julia orchestrator can
record individual objectives for Pareto quality metrics.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Optional, List

import jax
import jax.numpy as jnp
import numpy as np

from .base import BaseEvaluator

log = logging.getLogger(__name__)


class MultiNetworkEvaluator(BaseEvaluator):
    """JAX evaluator for constrained multi-network Pareto tracing.

    Parameters
    ----------
    forward_fns : list of callable
        List of JAX forward functions ``[(params1, x) -> y1, ...]``.
        Minimum 2 (f1, f2); optional third is h.
    params_list : list of pytree
        Parameter pytrees for each network.
    alpha : float
        Scalarization weight α ∈ [0,1].
    x0 : array-like, shape (n,)
        Initial point.
    device_manager : DeviceManager, optional
    """

    def __init__(
        self,
        forward_fns: List,
        params_list: List,
        alpha: float,
        x0: np.ndarray,
        device_manager=None,
    ) -> None:
        assert len(forward_fns) >= 2, "Need at least 2 networks (f1, f2)"
        assert len(forward_fns) == len(params_list)

        self.dm = device_manager
        self._put = (lambda a: a) if device_manager is None else device_manager.put

        self.forward_fns = forward_fns
        self.params_list = [self._put(p) for p in params_list]
        self.alpha = float(alpha)
        self.x0 = self._put(jnp.array(x0, dtype=jnp.float64))
        self.has_h = len(forward_fns) >= 3

        self.n = int(len(x0))
        self.m = 1 if self.has_h else 0

        log.info(
            "MultiNetworkEvaluator: n=%d, alpha=%.4f, has_h=%s",
            self.n, self.alpha, self.has_h,
        )

        self._compile()

    # ------------------------------------------------------------------
    # JIT compilation
    # ------------------------------------------------------------------

    def _compile(self) -> None:
        fns = self.forward_fns
        ps = self.params_list
        alpha = self.alpha
        has_h = self.has_h

        def _f1(x):
            return fns[0](ps[0], x)

        def _f2(x):
            return fns[1](ps[1], x)

        def _h(x):
            return fns[2](ps[2], x)

        def _obj(x):
            # Both f1 and f2 return scalar (single-output networks)
            v1 = jnp.squeeze(_f1(x))
            v2 = jnp.squeeze(_f2(x))
            return alpha * v1 + (1.0 - alpha) * v2

        def _cons(x):
            if not has_h:
                return jnp.zeros(0, dtype=jnp.float64)
            return jnp.reshape(_h(x), (1,))

        def _lagrangian(x, y, w):
            L = w * _obj(x)
            if has_h:
                L += jnp.dot(y, _cons(x))
            return L

        self._obj_jit = jax.jit(_obj)
        self._grad_jit = jax.jit(jax.grad(_obj))
        self._f1_jit = jax.jit(_f1)
        self._f2_jit = jax.jit(_f2)

        if has_h:
            self._h_jit = jax.jit(_h)
            self._cons_jit = jax.jit(_cons)
            self._jac_jit = jax.jit(jax.jacobian(_cons))
            self._hess_lag_fn = lambda x, y, w: jax.hessian(
                lambda x_: _lagrangian(x_, y, w)
            )(x)
        else:
            self._cons_jit = lambda x: jnp.zeros(0, dtype=jnp.float64)
            self._jac_jit = lambda x: jnp.zeros((0, self.n), dtype=jnp.float64)

        self._hess_obj_jit = jax.jit(jax.hessian(_obj))

    # ------------------------------------------------------------------
    # BaseEvaluator interface
    # ------------------------------------------------------------------

    def evaluate_nn(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate_individuals(x)

    def evaluate_individuals(self, x: np.ndarray) -> np.ndarray:
        """Return [f1(x), f2(x)] or [f1(x), f2(x), h(x)]."""
        x_jax = self._put(jnp.array(x, dtype=jnp.float64))
        f1 = float(jnp.squeeze(self._f1_jit(x_jax)))
        f2 = float(jnp.squeeze(self._f2_jit(x_jax)))
        if self.has_h:
            h = float(jnp.squeeze(self._h_jit(x_jax)))
            return np.array([f1, f2, h])
        return np.array([f1, f2])

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
