"""Closest-prior reduced-space MNIST adversarial-generation NLP."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .base import BaseEvaluator


class MNISTAdversarialEvaluator(BaseEvaluator):
    def __init__(
        self,
        forward_fn,
        params,
        reference_image,
        adversarial_label,
        *,
        threshold=0.6,
        device_manager=None,
    ):
        self.dm = device_manager
        self._put = (lambda value: value) if device_manager is None else device_manager.put
        self.forward_fn = forward_fn
        self.params = self._put(params)
        self.parameter_count = int(
            sum(np.prod(value.shape) for value in jax.tree.leaves(params))
        )
        self.reference = self._put(
            jnp.asarray(reference_image, dtype=jnp.float64).reshape(784)
        )
        self.adversarial_label = int(adversarial_label)
        self.threshold = float(threshold)
        self.pixel_size = 784
        self.output_size = 10
        self.output_offset = self.pixel_size
        self.positive_offset = self.output_offset + self.output_size
        self.negative_offset = self.positive_offset + self.pixel_size
        self.n = self.negative_offset + self.pixel_size
        self.m = 10 + 784 + 1 + 10
        self._build_sparsity()
        self._compile()

    def _build_sparsity(self):
        rows = []
        cols = []
        pixels = list(range(self.pixel_size))
        for output in range(self.output_size):
            rows.extend([output] * (self.pixel_size + 1))
            cols.extend(pixels + [self.output_offset + output])
        for pixel in range(self.pixel_size):
            row = self.output_size + pixel
            rows.extend((row, row, row))
            cols.extend(
                (
                    pixel,
                    self.positive_offset + pixel,
                    self.negative_offset + pixel,
                )
            )
        target_row = self.output_size + self.pixel_size
        rows.append(target_row)
        cols.append(self.output_offset + self.adversarial_label)
        for output in range(self.output_size):
            rows.append(target_row + 1 + output)
            cols.append(self.output_offset + output)
        self._jacobian_rows = np.asarray(rows, dtype=np.int32)
        self._jacobian_cols = np.asarray(cols, dtype=np.int32)
        self._hessian_rows = np.concatenate(
            [
                np.arange(column, self.pixel_size, dtype=np.int32)
                for column in range(self.pixel_size)
            ]
        )
        self._hessian_cols = np.concatenate(
            [
                np.full(self.pixel_size - column, column, dtype=np.int32)
                for column in range(self.pixel_size)
            ]
        )

    def jacobian_structure(self):
        return self._jacobian_rows, self._jacobian_cols

    def hessian_structure(self):
        return self._hessian_rows, self._hessian_cols

    def _compile(self):
        forward_fn = self.forward_fn
        params = self.params
        reference = self.reference
        pixel_size = self.pixel_size
        output_offset = self.output_offset
        positive_offset = self.positive_offset
        negative_offset = self.negative_offset
        adversarial_label = self.adversarial_label

        def _unpack(decision):
            pixels = decision[:pixel_size]
            outputs = decision[output_offset:positive_offset]
            positive = decision[positive_offset:negative_offset]
            negative = decision[negative_offset:]
            return pixels, outputs, positive, negative

        def _obj(decision):
            _, _, positive, negative = _unpack(decision)
            return jnp.sum(positive + negative)

        def _cons(decision):
            pixels, outputs, positive, negative = _unpack(decision)
            prediction = forward_fn(params, pixels)
            network_equalities = outputs - prediction
            slack_equalities = pixels - reference - positive + negative
            target = outputs[adversarial_label : adversarial_label + 1]
            return jnp.concatenate(
                (network_equalities, slack_equalities, target, outputs)
            )

        def _lagrangian(decision, multipliers, objective_weight):
            return objective_weight * _obj(decision) + jnp.dot(
                multipliers, _cons(decision)
            )

        self._unpack_jit = jax.jit(_unpack)
        self._forward_jit = jax.jit(lambda pixels: forward_fn(params, pixels))
        self._obj_jit = jax.jit(_obj)
        self._grad_jit = jax.jit(jax.grad(_obj))
        self._cons_jit = jax.jit(_cons)
        self._jac_jit = jax.jit(jax.jacrev(_cons))
        self._hess_obj_jit = jax.jit(jax.hessian(_obj))
        self._hess_lag_jit = jax.jit(jax.hessian(_lagrangian, argnums=0))
        self._hess_lag_fn = self._hess_lag_jit

        rows = self._put(jnp.asarray(self._hessian_rows, dtype=jnp.int32))
        cols = self._put(jnp.asarray(self._hessian_cols, dtype=jnp.int32))

        def _network_hessian_coordinates(decision, multipliers, _objective_weight):
            pixels = decision[:pixel_size]
            network_multipliers = multipliers[: self.output_size]

            def weighted_network(pixel_values):
                return -jnp.dot(
                    network_multipliers, forward_fn(params, pixel_values)
                )

            hessian = jax.hessian(weighted_network)(pixels)
            return hessian[rows, cols]

        self._sparse_hess_coord_jit = jax.jit(_network_hessian_coordinates)

    def initial_guess(self):
        outputs = np.asarray(self._forward_jit(self.reference))
        return np.concatenate(
            (np.asarray(self.reference), outputs, np.zeros(2 * self.pixel_size))
        )

    def initial_guess_from_pixels(self, pixels):
        """Build a network/slack-consistent start from a perturbed image."""
        pixels = np.asarray(pixels, dtype=np.float64).reshape(self.pixel_size)
        reference = np.asarray(self.reference)
        outputs = np.asarray(self._forward_jit(self._as_device_array(pixels)))
        delta = pixels - reference
        positive = np.maximum(delta, 0.0)
        negative = np.maximum(-delta, 0.0)
        return np.concatenate((pixels, outputs, positive, negative))

    def unpack(self, decision):
        return tuple(
            np.asarray(value)
            for value in self._unpack_jit(self._as_device_array(decision))
        )

    def evaluate_nn(self, decision):
        return np.asarray(
            self._forward_jit(self._as_device_array(decision)[: self.pixel_size])
        )

    def evaluate_obj(self, decision):
        return float(self._obj_jit(self._as_device_array(decision)))

    def evaluate_grad(self, decision):
        return np.asarray(self._grad_jit(self._as_device_array(decision)))

    def evaluate_cons(self, decision):
        return np.asarray(self._cons_jit(self._as_device_array(decision)))

    def evaluate_jac(self, decision):
        return np.asarray(self._jac_jit(self._as_device_array(decision)))

    def evaluate_hess(self, decision, multipliers, obj_weight=1.0):
        return np.asarray(
            self._hess_lag_jit(
                self._as_device_array(decision),
                self._as_device_array(multipliers),
                float(obj_weight),
            )
        )

    def evaluate_hess_coord_device(self, decision, multipliers, obj_weight=1.0):
        return self._sparse_hess_coord_jit(
            self._as_device_array(decision),
            self._as_device_array(multipliers),
            float(obj_weight),
        )
