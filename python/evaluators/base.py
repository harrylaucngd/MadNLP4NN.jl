"""
base.py

Abstract base class for all JAX-based NLP evaluators.

Every evaluator exposes the same surface expected by the Julia
NeuralNetworkNLPModel callbacks:

    evaluate_obj(x)               → float
    evaluate_grad(x)              → np.ndarray (n,)
    evaluate_cons(x)              → np.ndarray (m,)   [empty if m==0]
    evaluate_jac(x)               → np.ndarray (m, n) [empty if m==0]
    evaluate_hess(x, y, w)        → np.ndarray (n, n)
    evaluate_nn(x)                → np.ndarray         [raw network output]

plus the dimension attributes:
    n  — number of decision variables
    m  — number of general inequality constraints (not counting box bounds)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


class BaseEvaluator(ABC):
    """Abstract NLP evaluator that wraps one or more JAX forward models."""

    # Subclasses must set these in __init__
    n: int   # decision variable dimension
    m: int   # number of general inequality constraints

    # ------------------------------------------------------------------
    # Required methods
    # ------------------------------------------------------------------

    @abstractmethod
    def evaluate_obj(self, x: np.ndarray) -> float:
        """Objective value f(x)."""

    @abstractmethod
    def evaluate_grad(self, x: np.ndarray) -> np.ndarray:
        """Gradient ∇f(x), shape (n,)."""

    @abstractmethod
    def evaluate_cons(self, x: np.ndarray) -> np.ndarray:
        """Constraint values c(x) ≤ 0, shape (m,).  Empty array when m=0."""

    @abstractmethod
    def evaluate_jac(self, x: np.ndarray) -> np.ndarray:
        """Constraint Jacobian ∂c/∂x, shape (m, n).  Empty when m=0."""

    @abstractmethod
    def evaluate_hess(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray],
        obj_weight: float,
    ) -> np.ndarray:
        """Hessian of the Lagrangian L = w·f(x) + yᵀ c(x), shape (n, n)."""

    @abstractmethod
    def evaluate_nn(self, x: np.ndarray) -> np.ndarray:
        """Raw network output for diagnostics, shape determined by network."""

    # ------------------------------------------------------------------
    # Optional: multi-network diagnostics (Pareto case)
    # ------------------------------------------------------------------

    def evaluate_individuals(self, x: np.ndarray) -> np.ndarray:
        """Return individual network outputs stacked into one array.

        Default implementation returns the same value as evaluate_nn.
        Overridden by MultiNetworkEvaluator.
        """
        return self.evaluate_nn(x)

    # ------------------------------------------------------------------
    # Device-resident interface
    # ------------------------------------------------------------------

    def _as_device_array(self, value):
        """Return ``value`` as float64 on this evaluator's JAX device.

        When ``value`` was imported from a CUDA.jl buffer through DLPack,
        ``jnp.asarray`` and same-device ``device_put`` preserve that buffer.
        """
        array = jnp.asarray(value, dtype=jnp.float64)
        dm = getattr(self, "dm", None)
        return array if dm is None else dm.put(array)

    def evaluate_obj_device(self, x):
        """Return the objective as a scalar JAX array without NumPy staging."""
        return self._obj_jit(self._as_device_array(x))

    def evaluate_grad_device(self, x):
        """Return the gradient as a JAX array on the evaluator device."""
        return self._grad_jit(self._as_device_array(x))

    def evaluate_cons_device(self, x):
        """Return constraints as a JAX array on the evaluator device."""
        return self._cons_jit(self._as_device_array(x))

    def evaluate_jac_coord_device(self, x):
        """Return declared Jacobian coordinates on the evaluator device."""
        if not hasattr(self, "_jac_coord_jit"):
            rows, cols = self.jacobian_structure()
            rows = self._put(jnp.asarray(rows, dtype=jnp.int32))
            cols = self._put(jnp.asarray(cols, dtype=jnp.int32))
            self._jac_coord_jit = jax.jit(
                lambda value: self._jac_jit(value)[rows, cols]
            )
        return self._jac_coord_jit(self._as_device_array(x))

    def evaluate_jac_dense_device(self, x):
        """Return the two-dimensional constraint Jacobian on device."""
        return self._jac_jit(self._as_device_array(x))

    def _build_jacobian_product_functions(self):
        if self.m > 0:
            self._jprod_jit = jax.jit(
                lambda x, vector: jax.jvp(
                    self._cons_jit, (x,), (vector,)
                )[1]
            )

            def _transpose_product(x, vector):
                _, pullback = jax.vjp(self._cons_jit, x)
                return pullback(vector)[0]

            self._jtprod_jit = jax.jit(_transpose_product)
        else:
            self._jprod_jit = jax.jit(
                lambda x, vector: jnp.zeros((0,), dtype=jnp.float64)
            )
            self._jtprod_jit = jax.jit(
                lambda x, vector: jnp.zeros((self.n,), dtype=jnp.float64)
            )

    def evaluate_jprod_device(self, x, vector):
        """Return ``J(x) @ vector`` without materializing the Jacobian."""
        if not hasattr(self, "_jprod_jit"):
            self._build_jacobian_product_functions()
        return self._jprod_jit(
            self._as_device_array(x), self._as_device_array(vector)
        )

    def evaluate_jtprod_device(self, x, vector):
        """Return ``J(x).T @ vector`` using a JAX VJP."""
        if not hasattr(self, "_jtprod_jit"):
            self._build_jacobian_product_functions()
        return self._jtprod_jit(
            self._as_device_array(x), self._as_device_array(vector)
        )

    def evaluate_jprod(self, x, vector) -> np.ndarray:
        return np.asarray(self.evaluate_jprod_device(x, vector))

    def evaluate_jtprod(self, x, vector) -> np.ndarray:
        return np.asarray(self.evaluate_jtprod_device(x, vector))

    def _build_hessian_coordinate_function(self):
        row_indices, col_indices = self.hessian_structure()
        rows = self._as_device_array(row_indices).astype(jnp.int32)
        cols = self._as_device_array(col_indices).astype(jnp.int32)

        if self.m > 0:
            def _coordinates(x, y, weight):
                hessian = self._hess_lag_jit(x, y, weight)
                return hessian[rows, cols]
        else:
            def _coordinates(x, y, weight):
                del y
                hessian = weight * self._hess_obj_jit(x)
                return hessian[rows, cols]
        self._hess_coord_jit = jax.jit(_coordinates)

    def jacobian_structure(self):
        """Zero-based row/column indices for declared Jacobian nonzeros."""
        rows = np.repeat(np.arange(self.m, dtype=np.int32), self.n)
        cols = np.tile(np.arange(self.n, dtype=np.int32), self.m)
        return rows, cols

    def hessian_structure(self):
        """Zero-based lower-triangle structure, column-major by default."""
        rows = np.concatenate(
            [np.arange(column, self.n, dtype=np.int32) for column in range(self.n)]
        )
        cols = np.concatenate(
            [
                np.full(self.n - column, column, dtype=np.int32)
                for column in range(self.n)
            ]
        )
        return rows, cols

    @property
    def jacobian_nnz(self):
        return int(len(self.jacobian_structure()[0]))

    @property
    def hessian_nnz(self):
        return int(len(self.hessian_structure()[0]))

    def evaluate_jac_coord(self, x):
        return np.asarray(self.evaluate_jac_coord_device(x))

    def evaluate_hess_coord(self, x, y, obj_weight=1.0):
        return np.asarray(self.evaluate_hess_coord_device(x, y, obj_weight))

    def evaluate_hess_coord_device(self, x, y, obj_weight=1.0):
        """Return lower-triangular Lagrangian-Hessian coordinates on device."""
        if not hasattr(self, "_hess_coord_jit"):
            self._build_hessian_coordinate_function()
        x_device = self._as_device_array(x)
        if self.m > 0:
            y_device = self._as_device_array(y)
        else:
            y_device = jnp.zeros((0,), dtype=jnp.float64)
        return self._hess_coord_jit(x_device, y_device, float(obj_weight))

    def evaluate_hess_dense_device(self, x, y, obj_weight=1.0):
        """Return the full Lagrangian Hessian as a JAX device array."""
        x_device = self._as_device_array(x)
        if self.m > 0:
            y_device = self._as_device_array(y)
            return self._hess_lag_jit(x_device, y_device, float(obj_weight))
        return float(obj_weight) * self._hess_obj_jit(x_device)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"n={self.n}, m={self.m})"
        )
