"""Structured gray-box NMPC evaluator for published PFR CNN cases 2 and 3."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .base import BaseEvaluator


class PFRCNNNMPCEvaluator(BaseEvaluator):
    def __init__(
        self,
        forward_fn,
        params,
        model_config,
        current_state,
        previous_control,
        tracking_channels,
        tracking_indices,
        tracking_targets,
        tracking_scales,
        control_scales,
        *,
        prediction_horizon,
        control_horizon,
        device_manager=None,
    ):
        self.dm = device_manager
        self._put = (lambda value: value) if device_manager is None else device_manager.put
        self.forward_fn = forward_fn
        self.params = self._put(params)
        self.n_fe = int(model_config["n_fe"])
        self.state_channels = int(model_config["output_channels"])
        self.control_dimension = int(model_config["control_channels"])
        self.state_dimension = self.state_channels * self.n_fe
        self.prediction_horizon = int(prediction_horizon)
        self.control_horizon = int(control_horizon)
        self.state_size = (self.prediction_horizon + 1) * self.state_dimension
        self.control_size = self.control_horizon * self.control_dimension
        self.n = self.state_size + self.control_size
        self.m = self.prediction_horizon * self.state_dimension
        self.current_state = self._put(
            jnp.asarray(current_state, dtype=jnp.float64).reshape(self.state_dimension)
        )
        self.previous_control = self._put(
            jnp.asarray(previous_control, dtype=jnp.float64).reshape(
                self.control_dimension
            )
        )
        self.tracking_channels = tuple(map(int, tracking_channels))
        self.tracking_indices = tuple(map(int, tracking_indices))
        self.tracking_targets = tuple(map(float, tracking_targets))
        self.tracking_scales = tuple(map(float, tracking_scales))
        self.control_scales = tuple(map(float, control_scales))
        if not (
            len(self.tracking_channels)
            == len(self.tracking_indices)
            == len(self.tracking_targets)
            == len(self.tracking_scales)
        ):
            raise ValueError("Tracking term arrays must have equal lengths")
        if len(self.control_scales) != self.control_dimension:
            raise ValueError("One movement scale is required per control")
        lower = np.asarray(params["lower"])
        upper = np.asarray(params["upper"])
        state_lower = np.broadcast_to(
            lower[0, : self.state_channels],
            (self.state_channels, self.n_fe, 1),
        ).reshape(-1)
        state_upper = np.broadcast_to(
            upper[0, : self.state_channels],
            (self.state_channels, self.n_fe, 1),
        ).reshape(-1)
        control_lower = lower[0, self.state_channels :, 0, 0]
        control_upper = upper[0, self.state_channels :, 0, 0]
        self.variable_lower = np.concatenate(
            (
                np.tile(state_lower, self.prediction_horizon + 1),
                np.tile(control_lower, self.control_horizon),
            )
        )
        self.variable_upper = np.concatenate(
            (
                np.tile(state_upper, self.prediction_horizon + 1),
                np.tile(control_upper, self.control_horizon),
            )
        )
        self.variable_lower[: self.state_dimension] = np.asarray(self.current_state)
        self.variable_upper[: self.state_dimension] = np.asarray(self.current_state)
        self._build_sparsity()
        self._compile()

    def _build_sparsity(self):
        state_dim = self.state_dimension
        control_dim = self.control_dimension
        local_dim = state_dim + control_dim
        jac_rows = []
        jac_cols = []
        hessian_support = set()
        local_rows, local_cols = np.tril_indices(local_dim)
        network_maps = []

        for step in range(self.prediction_horizon):
            state_indices = list(range(step * state_dim, (step + 1) * state_dim))
            control_step = min(step, self.control_horizon - 1)
            control_indices = list(
                range(
                    self.state_size + control_step * control_dim,
                    self.state_size + (control_step + 1) * control_dim,
                )
            )
            inputs = state_indices + control_indices
            for output in range(state_dim):
                row = step * state_dim + output
                jac_rows.extend([row] * (local_dim + 1))
                jac_cols.extend(inputs + [(step + 1) * state_dim + output])
            for row, col in zip(local_rows, local_cols):
                hessian_support.add((max(inputs[row], inputs[col]), min(inputs[row], inputs[col])))

        objective_terms = []
        for channel, spatial, _target, scale in zip(
            self.tracking_channels,
            self.tracking_indices,
            self.tracking_targets,
            self.tracking_scales,
        ):
            for step in range(self.prediction_horizon + 1):
                index = step * state_dim + channel * self.n_fe + spatial
                hessian_support.add((index, index))
                objective_terms.append((index, index, 2.0 / scale))
        for control, scale in enumerate(self.control_scales):
            indices = [
                self.state_size + step * control_dim + control
                for step in range(self.control_horizon)
            ]
            hessian_support.add((indices[0], indices[0]))
            objective_terms.append((indices[0], indices[0], 2.0 / scale))
            for current, previous in zip(indices[1:], indices[:-1]):
                hessian_support.add((current, current))
                hessian_support.add((previous, previous))
                hessian_support.add((max(current, previous), min(current, previous)))
                objective_terms.extend(
                    (
                        (current, current, 2.0 / scale),
                        (previous, previous, 2.0 / scale),
                        (max(current, previous), min(current, previous), -2.0 / scale),
                    )
                )

        ordered = sorted(hessian_support, key=lambda pair: (pair[1], pair[0]))
        support_index = {pair: index for index, pair in enumerate(ordered)}
        for step in range(self.prediction_horizon):
            state_indices = list(range(step * state_dim, (step + 1) * state_dim))
            control_step = min(step, self.control_horizon - 1)
            control_indices = list(
                range(
                    self.state_size + control_step * control_dim,
                    self.state_size + (control_step + 1) * control_dim,
                )
            )
            inputs = state_indices + control_indices
            network_maps.append(
                [
                    support_index[(max(inputs[row], inputs[col]), min(inputs[row], inputs[col]))]
                    for row, col in zip(local_rows, local_cols)
                ]
            )
        self._jacobian_rows = np.asarray(jac_rows, dtype=np.int32)
        self._jacobian_cols = np.asarray(jac_cols, dtype=np.int32)
        self._hessian_rows = np.asarray([pair[0] for pair in ordered], dtype=np.int32)
        self._hessian_cols = np.asarray([pair[1] for pair in ordered], dtype=np.int32)
        self._local_hessian_rows = np.asarray(local_rows, dtype=np.int32)
        self._local_hessian_cols = np.asarray(local_cols, dtype=np.int32)
        self._network_hessian_maps = np.asarray(network_maps, dtype=np.int32)
        self._objective_hessian_maps = np.asarray(
            [support_index[(row, col)] for row, col, _ in objective_terms],
            dtype=np.int32,
        )
        self._objective_hessian_values = np.asarray(
            [value for _, _, value in objective_terms], dtype=np.float64
        )

    def jacobian_structure(self):
        return self._jacobian_rows, self._jacobian_cols

    def hessian_structure(self):
        return self._hessian_rows, self._hessian_cols

    def _compile(self):
        forward_fn = self.forward_fn
        params = self.params
        P = self.prediction_horizon
        M = self.control_horizon
        state_dim = self.state_dimension
        control_dim = self.control_dimension
        state_size = self.state_size
        previous_control = self.previous_control
        tracking_channels = self.tracking_channels
        tracking_indices = self.tracking_indices
        tracking_targets = self.tracking_targets
        tracking_scales = self.tracking_scales
        control_scales = self.control_scales

        def _unpack(decision):
            states = decision[:state_size].reshape(P + 1, state_dim)
            controls = decision[state_size:].reshape(M, control_dim)
            full_controls = jnp.concatenate(
                (controls, jnp.repeat(controls[-1:], P - M, axis=0)), axis=0
            )
            return states, controls, full_controls

        def _inputs(decision):
            states, controls, full_controls = _unpack(decision)
            return states, controls, jnp.concatenate((states[:-1], full_controls), axis=1)

        def _predictions(local_inputs):
            return jax.vmap(lambda value: forward_fn(params, value))(local_inputs)

        def _obj(decision):
            states, controls, _ = _inputs(decision)
            value = 0.0
            states_reshaped = states.reshape(P + 1, self.state_channels, self.n_fe)
            for channel, spatial, target, scale in zip(
                tracking_channels,
                tracking_indices,
                tracking_targets,
                tracking_scales,
            ):
                value += jnp.sum(
                    (states_reshaped[:, channel, spatial] - target) ** 2
                ) / scale
            for control, scale in enumerate(control_scales):
                value += (controls[0, control] - previous_control[control]) ** 2 / scale
                value += jnp.sum(
                    (controls[1:, control] - controls[:-1, control]) ** 2
                ) / scale
            return value

        def _cons(decision):
            states, _, local_inputs = _inputs(decision)
            return (states[1:] - _predictions(local_inputs)).reshape(-1)

        self._unpack_jit = jax.jit(_unpack)
        self._obj_jit = jax.jit(_obj)
        self._grad_jit = jax.jit(jax.grad(_obj))
        self._cons_jit = jax.jit(_cons)

        def _jacobian_coordinates(decision):
            states, _, local_inputs = _inputs(decision)
            del states
            local_jacobians = jax.vmap(jax.jacrev(lambda value: forward_fn(params, value)))(
                local_inputs
            )
            blocks = jnp.concatenate(
                (
                    -local_jacobians.reshape(P, state_dim, -1),
                    jnp.ones((P, state_dim, 1), dtype=jnp.float64),
                ),
                axis=2,
            )
            return blocks.reshape(-1)

        self._structured_jac_coord_jit = jax.jit(_jacobian_coordinates)
        local_rows = self._put(jnp.asarray(self._local_hessian_rows, dtype=jnp.int32))
        local_cols = self._put(jnp.asarray(self._local_hessian_cols, dtype=jnp.int32))
        network_maps = self._put(jnp.asarray(self._network_hessian_maps, dtype=jnp.int32))
        objective_maps = self._put(jnp.asarray(self._objective_hessian_maps, dtype=jnp.int32))
        objective_values = self._put(
            jnp.asarray(self._objective_hessian_values, dtype=jnp.float64)
        )
        hessian_nnz = self.hessian_nnz

        def _hessian_coordinates(decision, multipliers, objective_weight):
            _, _, local_inputs = _inputs(decision)
            local_multipliers = multipliers.reshape(P, state_dim)

            def local_hessian(value, multiplier):
                return jax.hessian(
                    lambda local: -jnp.dot(multiplier, forward_fn(params, local))
                )(value)

            local = jax.vmap(local_hessian)(local_inputs, local_multipliers)
            coordinates = jnp.zeros(hessian_nnz, dtype=jnp.float64)
            coordinates = coordinates.at[network_maps.reshape(-1)].add(
                local[:, local_rows, local_cols].reshape(-1)
            )
            coordinates = coordinates.at[objective_maps].add(
                objective_weight * objective_values
            )
            return coordinates

        self._structured_hess_coord_jit = jax.jit(_hessian_coordinates)

        def _initial_guess():
            controls = jnp.repeat(previous_control[None, :], M, axis=0)
            full_controls = jnp.concatenate(
                (controls, jnp.repeat(controls[-1:], P - M, axis=0)), axis=0
            )

            def step(state, control):
                next_state = forward_fn(params, jnp.concatenate((state, control)))
                return next_state, next_state

            _, future = jax.lax.scan(step, self.current_state, full_controls)
            states = jnp.concatenate((self.current_state[None, :], future), axis=0)
            return jnp.concatenate((states.reshape(-1), controls.reshape(-1)))

        self._initial_guess_jit = jax.jit(_initial_guess)

    def initial_guess(self):
        return np.asarray(self._initial_guess_jit())

    def unpack(self, decision):
        return tuple(
            np.asarray(value)
            for value in self._unpack_jit(self._as_device_array(decision))
        )

    def evaluate_nn(self, decision):
        states, _, controls = self.unpack(decision)
        inputs = np.concatenate((states[:-1], controls), axis=1)
        return np.asarray(
            jax.vmap(lambda value: self.forward_fn(self.params, value))(
                self._as_device_array(inputs)
            )
        )

    def evaluate_obj(self, decision):
        return float(self._obj_jit(self._as_device_array(decision)))

    def evaluate_grad(self, decision):
        return np.asarray(self._grad_jit(self._as_device_array(decision)))

    def evaluate_cons(self, decision):
        return np.asarray(self._cons_jit(self._as_device_array(decision)))

    def evaluate_jac_coord_device(self, decision):
        return self._structured_jac_coord_jit(self._as_device_array(decision))

    def evaluate_hess_coord_device(self, decision, multipliers, obj_weight=1.0):
        return self._structured_hess_coord_jit(
            self._as_device_array(decision),
            self._as_device_array(multipliers),
            float(obj_weight),
        )

    def evaluate_jac(self, decision):
        matrix = np.zeros((self.m, self.n))
        matrix[self._jacobian_rows, self._jacobian_cols] = self.evaluate_jac_coord(
            decision
        )
        return matrix

    def evaluate_hess(self, decision, multipliers, obj_weight=1.0):
        matrix = np.zeros((self.n, self.n))
        values = self.evaluate_hess_coord(decision, multipliers, obj_weight)
        matrix[self._hessian_rows, self._hessian_cols] = values
        matrix[self._hessian_cols, self._hessian_rows] = values
        return matrix
