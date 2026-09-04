"""Latent-field inversion evaluator for the public PDEBench Darcy benchmark."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .base import BaseEvaluator


class PDEBenchDarcyLatentEvaluator(BaseEvaluator):
    """Optimize a coarse latent coefficient field through a full-resolution FNO.

    The latent field is bilinearly interpolated to the PDEBench resolution,
    matching the parameterization used by PDEBench's gradient-based inverse
    baseline. The observation target always comes from the held-out simulator
    data, never from the surrogate itself.
    """

    def __init__(
        self,
        forward_fn,
        params,
        target_pressure,
        x0,
        *,
        reference_field=None,
        full_resolution: int = 128,
        observation_indices=None,
        regularization_weight: float = 1e-4,
        binary_weight: float = 0.0,
        mean_lower: Optional[float] = None,
        mean_upper: Optional[float] = None,
        tv_upper: Optional[float] = None,
        tv_epsilon: float = 1e-4,
        device_manager=None,
    ):
        self.dm = device_manager
        self._put = (lambda value: value) if device_manager is None else device_manager.put
        self.forward_fn = forward_fn
        self.params = self._put(params)
        self.target = self._put(jnp.asarray(target_pressure, dtype=jnp.float64).reshape(-1))
        self.x0 = self._put(jnp.asarray(x0, dtype=jnp.float64).reshape(-1))
        if reference_field is None:
            reference_field = x0
        self.reference_field = self._put(
            jnp.asarray(reference_field, dtype=jnp.float64).reshape(-1)
        )
        self.full_resolution = int(full_resolution)
        self.latent_resolution = int(round(np.sqrt(len(x0))))
        if self.latent_resolution**2 != len(x0):
            raise ValueError("Latent coefficient length must be a square")
        if self.reference_field.size != len(x0):
            raise ValueError("Regularization reference must match latent coefficient length")
        if self.target.size != self.full_resolution**2:
            raise ValueError("Target pressure does not match full resolution")
        if observation_indices is None:
            observation_indices = np.arange(self.target.size, dtype=np.int32)
        self.observation_indices = self._put(
            jnp.asarray(observation_indices, dtype=jnp.int32)
        )
        self.regularization_weight = float(regularization_weight)
        self.binary_weight = float(binary_weight)
        self.mean_lower = None if mean_lower is None else float(mean_lower)
        self.mean_upper = None if mean_upper is None else float(mean_upper)
        self.tv_upper = None if tv_upper is None else float(tv_upper)
        self.tv_epsilon = float(tv_epsilon)
        self.hessian_mode = "exact"
        self.n = int(len(x0))
        self.m = sum(
            value is not None
            for value in (self.mean_upper, self.mean_lower, self.tv_upper)
        )
        self._compile()

    def _compile(self):
        forward_fn = self.forward_fn
        params = self.params
        target = self.target
        reference_field = self.reference_field
        indices = self.observation_indices
        latent_resolution = self.latent_resolution
        full_resolution = self.full_resolution
        regularization_weight = self.regularization_weight
        binary_weight = self.binary_weight
        mean_lower = self.mean_lower
        mean_upper = self.mean_upper
        tv_upper = self.tv_upper
        tv_epsilon = self.tv_epsilon
        target_observed = target[indices]
        pressure_scale = jnp.sqrt(jnp.mean(target_observed**2)) + 1e-12

        def _field(latent):
            coarse = latent.reshape(latent_resolution, latent_resolution)
            return jax.image.resize(
                coarse,
                (full_resolution, full_resolution),
                method="linear",
                antialias=False,
            )

        def _prediction(latent):
            return forward_fn(params, _field(latent).reshape(-1))

        def _obj(latent):
            prediction = _prediction(latent)
            residual = (prediction[indices] - target_observed) / pressure_scale
            value = 0.5 * jnp.mean(residual**2)
            if regularization_weight > 0:
                value += 0.5 * regularization_weight * jnp.mean(
                    (latent - reference_field) ** 2
                )
            if binary_weight > 0:
                field = _field(latent)
                double_well = ((field - 0.1) * (field - 1.0)) ** 2
                value += binary_weight * jnp.mean(double_well)
            return value

        def _cons(latent):
            field = _field(latent)
            constraints = []
            if mean_upper is not None:
                constraints.append(jnp.mean(field) - mean_upper)
            if mean_lower is not None:
                constraints.append(mean_lower - jnp.mean(field))
            if tv_upper is not None:
                dx = jnp.diff(field, axis=0)
                dy = jnp.diff(field, axis=1)
                tv = jnp.mean(jnp.sqrt(dx**2 + tv_epsilon**2))
                tv += jnp.mean(jnp.sqrt(dy**2 + tv_epsilon**2))
                constraints.append(tv - tv_upper)
            return (
                jnp.stack(constraints)
                if constraints
                else jnp.zeros((0,), dtype=jnp.float64)
            )

        def _lagrangian(latent, multipliers, objective_weight):
            return objective_weight * _obj(latent) + jnp.dot(multipliers, _cons(latent))

        self._field_jit = jax.jit(_field)
        self._prediction_jit = jax.jit(_prediction)
        self._obj_jit = jax.jit(_obj)
        self._grad_jit = jax.jit(jax.grad(_obj))
        self._cons_jit = jax.jit(_cons)
        self._jac_jit = jax.jit(jax.jacrev(_cons))
        self._hess_obj_jit = jax.jit(jax.hessian(_obj))
        if self.m > 0:
            self._hess_lag_jit = jax.jit(jax.hessian(_lagrangian, argnums=0))
            self._hess_lag_fn = self._hess_lag_jit
        else:
            self._hess_lag_fn = None

        hessian_rows, hessian_cols = self.hessian_structure()
        hessian_rows = self._put(jnp.asarray(hessian_rows, dtype=jnp.int32))
        hessian_cols = self._put(jnp.asarray(hessian_cols, dtype=jnp.int32))

        def _exact_hessian_coordinates(latent, multipliers, objective_weight):
            if self.m > 0:
                hessian = self._hess_lag_jit(
                    latent, multipliers, objective_weight
                )
            else:
                hessian = objective_weight * self._hess_obj_jit(latent)
            return hessian[hessian_rows, hessian_cols]

        def _gauss_newton_hessian_coordinates(
            latent, multipliers, objective_weight
        ):
            del multipliers

            def normalized_residual(value):
                prediction = _prediction(value)
                return (
                    (prediction[indices] - target_observed)
                    / pressure_scale
                    / jnp.sqrt(target_observed.size)
                )

            # The latent dimension is smaller than the 4,096 pressure outputs;
            # forward-mode avoids batching one reverse sweep per output.
            residual_jacobian = jax.jacfwd(normalized_residual)(latent)
            hessian = objective_weight * (
                residual_jacobian.T @ residual_jacobian
            )
            if regularization_weight > 0:
                hessian += (
                    objective_weight
                    * regularization_weight
                    / latent.size
                    * jnp.eye(latent.size, dtype=jnp.float64)
                )
            return hessian[hessian_rows, hessian_cols]

        self._exact_hess_coord_jit = jax.jit(_exact_hessian_coordinates)
        self._gauss_newton_hess_coord_jit = jax.jit(
            _gauss_newton_hessian_coordinates
        )

    def _host_input(self, value):
        return self._as_device_array(value)

    def evaluate_nn(self, x):
        return np.asarray(self._prediction_jit(self._host_input(x)))

    def evaluate_nn_full_field(self, field):
        """Evaluate the surrogate without passing through the latent resize."""
        field_device = self._put(
            jnp.asarray(field, dtype=jnp.float64).reshape(-1)
        )
        return np.asarray(self.forward_fn(self.params, field_device))

    def evaluate_field(self, x):
        return np.asarray(self._field_jit(self._host_input(x)))

    def projection_metrics(self, truth_field, lower_bound, upper_bound):
        """Compute the box-constrained best latent representation of a truth field."""
        from scipy.optimize import minimize

        truth = self._put(
            jnp.asarray(truth_field, dtype=jnp.float64).reshape(
                self.full_resolution, self.full_resolution
            )
        )

        def projection_objective(latent):
            residual = self._field_jit(latent) - truth
            return 0.5 * jnp.mean(residual**2)

        objective = jax.jit(projection_objective)
        gradient = jax.jit(jax.grad(projection_objective))
        initial = np.asarray(
            jax.image.resize(
                truth,
                (self.latent_resolution, self.latent_resolution),
                method="linear",
                antialias=False,
            )
        ).reshape(-1)
        initial = np.clip(initial, lower_bound, upper_bound)
        result = minimize(
            lambda value: float(objective(self._host_input(value))),
            initial,
            jac=lambda value: np.asarray(gradient(self._host_input(value))),
            method="L-BFGS-B",
            bounds=[(lower_bound, upper_bound)] * self.n,
            options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 1000},
        )
        projected = np.asarray(self._field_jit(self._host_input(result.x)))
        truth_host = np.asarray(truth)
        return {
            "success": bool(result.success),
            "iterations": int(result.nit),
            "coefficient_relative_l2_floor": float(
                np.linalg.norm(projected - truth_host) / np.linalg.norm(truth_host)
            ),
            "coefficient_rmse_floor": float(
                np.sqrt(np.mean((projected - truth_host) ** 2))
            ),
        }

    def evaluate_obj(self, x):
        return float(self._obj_jit(self._host_input(x)))

    def evaluate_grad(self, x):
        return np.asarray(self._grad_jit(self._host_input(x)))

    def evaluate_cons(self, x):
        return np.asarray(self._cons_jit(self._host_input(x)))

    def evaluate_jac(self, x):
        return np.asarray(self._jac_jit(self._host_input(x)))

    def evaluate_hess(self, x, y, obj_weight=1.0):
        x_device = self._host_input(x)
        if self.m > 0:
            y_device = self._host_input(y)
            hessian = self._hess_lag_jit(x_device, y_device, float(obj_weight))
        else:
            hessian = float(obj_weight) * self._hess_obj_jit(x_device)
        return np.asarray(hessian)

    def set_hessian_mode(self, mode):
        if mode not in ("exact", "gauss_newton"):
            raise ValueError(f"Unknown Hessian mode: {mode}")
        if mode == "gauss_newton" and (
            self.binary_weight != 0.0 or self.m != 0
        ):
            raise ValueError(
                "Gauss-Newton is currently defined only for the unconstrained "
                "least-squares/Tikhonov Darcy formulation"
            )
        self.hessian_mode = mode

    def evaluate_hess_coord_device(self, x, y, obj_weight=1.0):
        x_device = self._as_device_array(x)
        y_device = self._as_device_array(y)
        if self.hessian_mode == "gauss_newton":
            return self._gauss_newton_hess_coord_jit(
                x_device, y_device, float(obj_weight)
            )
        return self._exact_hess_coord_jit(
            x_device, y_device, float(obj_weight)
        )
