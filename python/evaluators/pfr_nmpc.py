"""Gray-box PFR-NMPC evaluator matching Elorza Casas et al. case study 1."""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .base import BaseEvaluator


class PFRNMPCEvaluator(BaseEvaluator):
    """Published 430-variable, 400-equality gray-box NMPC problem."""

    def __init__(
        self,
        forward_fn,
        params,
        current_state,
        previous_control,
        setpoint,
        *,
        prediction_horizon: int = 40,
        control_horizon: int = 10,
        parameters_as_variables: bool = False,
        device_manager=None,
    ):
        self.dm = device_manager
        self._put = (lambda value: value) if device_manager is None else device_manager.put
        self.forward_fn = forward_fn
        self.params = self._put(params)
        self.current_state = self._put(
            jnp.asarray(current_state, dtype=jnp.float64).reshape(-1)
        )
        self.previous_control = self._put(
            jnp.asarray(previous_control, dtype=jnp.float64).reshape(-1)
        )
        self.setpoint = float(setpoint)
        self.prediction_horizon = int(prediction_horizon)
        self.control_horizon = int(control_horizon)
        self.parameters_as_variables = bool(parameters_as_variables)
        self.state_dimension = int(self.current_state.size)
        self.control_dimension = int(self.previous_control.size)
        self.state_size = (self.prediction_horizon + 1) * self.state_dimension
        self.control_size = self.control_horizon * self.control_dimension
        self.base_decision_size = self.state_size + self.control_size
        self.parameter_size = 3 if self.parameters_as_variables else 0
        self.n = self.base_decision_size + self.parameter_size
        self.m = self.prediction_horizon * self.state_dimension
        self._build_sparsity()
        self._compile()

    def _build_sparsity(self):
        jacobian_rows = []
        jacobian_cols = []
        hessian_support = set()

        def add_hessian_block(indices):
            for first in indices:
                for second in indices:
                    row, col = max(first, second), min(first, second)
                    hessian_support.add((row, col))

        for step in range(self.prediction_horizon):
            state_inputs = list(
                range(
                    step * self.state_dimension,
                    (step + 1) * self.state_dimension,
                )
            )
            control_step = min(step, self.control_horizon - 1)
            control_inputs = list(
                range(
                    self.state_size + control_step * self.control_dimension,
                    self.state_size + (control_step + 1) * self.control_dimension,
                )
            )
            network_inputs = state_inputs + control_inputs
            add_hessian_block(network_inputs)
            for state in range(self.state_dimension):
                row = step * self.state_dimension + state
                jacobian_rows.extend([row] * (len(network_inputs) + 1))
                jacobian_cols.extend(
                    network_inputs
                    + [(step + 1) * self.state_dimension + state]
                )

        # Tracking objective and control-movement objective.
        outlet_indices = [
            step * self.state_dimension + self.state_dimension - 1
            for step in range(self.prediction_horizon + 1)
        ]
        for index in outlet_indices:
            hessian_support.add((index, index))
        for control in range(self.control_dimension):
            control_indices = [
                self.state_size + step * self.control_dimension + control
                for step in range(self.control_horizon)
            ]
            for index in control_indices:
                hessian_support.add((index, index))
            for first, second in zip(control_indices[1:], control_indices[:-1]):
                hessian_support.add((max(first, second), min(first, second)))

        if self.parameters_as_variables:
            setpoint_index = self.base_decision_size
            previous_indices = [setpoint_index + 1, setpoint_index + 2]
            hessian_support.add((setpoint_index, setpoint_index))
            for outlet in outlet_indices:
                hessian_support.add((max(setpoint_index, outlet), min(setpoint_index, outlet)))
            for control, previous in enumerate(previous_indices):
                first_control = self.state_size + control
                hessian_support.add((previous, previous))
                hessian_support.add(
                    (max(previous, first_control), min(previous, first_control))
                )

        ordered_hessian = sorted(hessian_support, key=lambda pair: (pair[1], pair[0]))
        self._jacobian_rows = np.asarray(jacobian_rows, dtype=np.int32)
        self._jacobian_cols = np.asarray(jacobian_cols, dtype=np.int32)
        self._hessian_rows = np.asarray(
            [pair[0] for pair in ordered_hessian], dtype=np.int32
        )
        self._hessian_cols = np.asarray(
            [pair[1] for pair in ordered_hessian], dtype=np.int32
        )

    def jacobian_structure(self):
        return self._jacobian_rows, self._jacobian_cols

    def hessian_structure(self):
        return self._hessian_rows, self._hessian_cols

    def _compile(self):
        forward_fn = self.forward_fn
        params = self.params
        prediction_horizon = self.prediction_horizon
        control_horizon = self.control_horizon
        state_dimension = self.state_dimension
        control_dimension = self.control_dimension
        state_size = self.state_size
        base_decision_size = self.base_decision_size
        fixed_previous_control = self.previous_control
        fixed_setpoint = self.setpoint
        parameters_as_variables = self.parameters_as_variables

        def _unpack(decision):
            states = decision[:state_size].reshape(
                prediction_horizon + 1, state_dimension
            )
            controls = decision[state_size:base_decision_size].reshape(
                control_horizon, control_dimension
            )
            held_controls = jnp.repeat(
                controls[-1:, :], prediction_horizon - control_horizon, axis=0
            )
            full_controls = jnp.concatenate((controls, held_controls), axis=0)
            return states, controls, full_controls

        def _problem_parameters(decision):
            if parameters_as_variables:
                problem = decision[base_decision_size:]
                return problem[0], problem[1:]
            return fixed_setpoint, fixed_previous_control

        def _predict(states, full_controls):
            inputs = jnp.concatenate((states[:-1], full_controls), axis=1)
            return jax.vmap(lambda value: forward_fn(params, value))(inputs)

        def _obj(decision):
            states, controls, _ = _unpack(decision)
            setpoint, previous_control = _problem_parameters(decision)
            tracking = jnp.sum((setpoint - states[:, -1]) ** 2)
            first_move = controls[0] - previous_control
            later_moves = controls[1:] - controls[:-1]
            movement = jnp.sum(first_move**2) + jnp.sum(later_moves**2)
            return tracking + movement

        def _cons(decision):
            states, _, full_controls = _unpack(decision)
            return (states[1:] - _predict(states, full_controls)).reshape(-1)

        def _lagrangian(decision, multipliers, objective_weight):
            return objective_weight * _obj(decision) + jnp.dot(
                multipliers, _cons(decision)
            )

        def _initial_guess_for(runtime_state, runtime_previous_control, runtime_setpoint):
            controls = jnp.full(
                (control_horizon, control_dimension), 0.5, dtype=jnp.float64
            )
            held_controls = jnp.repeat(
                controls[-1:, :], prediction_horizon - control_horizon, axis=0
            )
            full_controls = jnp.concatenate((controls, held_controls), axis=0)

            def step(state, control):
                next_state = forward_fn(
                    params, jnp.concatenate((state, control), axis=0)
                )
                return next_state, next_state

            _, future = jax.lax.scan(step, runtime_state, full_controls)
            states = jnp.concatenate((runtime_state[None, :], future), axis=0)
            decision = jnp.concatenate((states.reshape(-1), controls.reshape(-1)))
            if parameters_as_variables:
                decision = jnp.concatenate(
                    (
                        decision,
                        jnp.asarray([runtime_setpoint], dtype=jnp.float64),
                        runtime_previous_control,
                    )
                )
            return decision

        def _initial_guess():
            return _initial_guess_for(
                self.current_state,
                fixed_previous_control,
                fixed_setpoint,
            )

        self._unpack_jit = jax.jit(_unpack)
        self._prediction_jit = jax.jit(lambda value: _predict(*(_unpack(value)[::2])))
        self._obj_jit = jax.jit(_obj)
        self._grad_jit = jax.jit(jax.grad(_obj))
        self._cons_jit = jax.jit(_cons)
        self._jac_jit = jax.jit(jax.jacrev(_cons))
        self._hess_obj_jit = jax.jit(jax.hessian(_obj))
        self._hess_lag_jit = jax.jit(jax.hessian(_lagrangian, argnums=0))
        self._hess_lag_fn = self._hess_lag_jit
        self._initial_guess_jit = jax.jit(_initial_guess)
        self._initial_guess_for_jit = jax.jit(_initial_guess_for)

    def initial_guess(self):
        return np.asarray(self._initial_guess_jit())

    def initial_guess_for(self, current_state, previous_control, setpoint):
        """Build a feasible rollout guess for updated closed-loop parameters."""
        return np.asarray(
            self._initial_guess_for_jit(
                self._as_device_array(current_state),
                self._as_device_array(previous_control),
                float(setpoint),
            )
        )

    def unpack(self, decision):
        states, controls, full_controls = self._unpack_jit(self._as_device_array(decision))
        return np.asarray(states), np.asarray(controls), np.asarray(full_controls)

    @property
    def problem_parameter_slice(self):
        return (self.base_decision_size, self.n)

    @staticmethod
    def mechanistic_step(state, control, sample_time=0.1):
        """Advance the published finite-volume PFR model by one control step."""
        from scipy.integrate import solve_ivp

        state = np.asarray(state, dtype=np.float64)
        flow, inlet = np.asarray(control, dtype=np.float64)
        spatial_step = 1.0 / state.size

        def derivative(_time, value):
            upstream = np.concatenate(([inlet], value[:-1]))
            spatial_derivative = (value - upstream) / spatial_step
            return -flow * spatial_derivative - value**2

        solution = solve_ivp(derivative, [0.0, sample_time], state)
        return np.asarray(solution.y[:, -1])

    def evaluate_nn(self, decision):
        return np.asarray(self._prediction_jit(self._as_device_array(decision)))

    def evaluate_obj(self, decision):
        return float(self._obj_jit(self._as_device_array(decision)))

    def evaluate_grad(self, decision):
        return np.asarray(self._grad_jit(self._as_device_array(decision)))

    def evaluate_cons(self, decision):
        return np.asarray(self._cons_jit(self._as_device_array(decision)))

    def evaluate_jac(self, decision):
        return np.asarray(self._jac_jit(self._as_device_array(decision)))

    def evaluate_hess(
        self,
        decision,
        multipliers: Optional[np.ndarray],
        obj_weight: float = 1.0,
    ):
        decision = self._as_device_array(decision)
        multipliers = self._as_device_array(multipliers)
        return np.asarray(
            self._hess_lag_jit(decision, multipliers, float(obj_weight))
        )
