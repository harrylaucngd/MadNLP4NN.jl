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
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"n={self.n}, m={self.m})"
        )
