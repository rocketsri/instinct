"""Coordinate/type feedback-basis plasticity for P5.2 recovery cycle 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from instinct.p5_development.plasticity import FloatArray, NetworkState, forward


@dataclass(frozen=True, slots=True)
class FeedbackBasisProgram:
    """Fixed Fourier feedback basis with time-modulated local plasticity."""

    coefficients: FloatArray
    step_size: float = 0.04

    def __post_init__(self) -> None:
        values = np.asarray(self.coefficients, dtype=np.float64)
        if values.shape != (12,) or not np.isfinite(values).all():
            raise ValueError("feedback-basis program requires twelve finite coefficients")
        if not 0.0 < self.step_size <= 0.5:
            raise ValueError("step_size must lie in (0, 0.5]")

    def serialization_manifest(self) -> dict[str, int | float | str]:
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        return {
            "coefficient_count": 12,
            "coefficient_bytes": int(coefficients.nbytes),
            "step_size_count": 1,
            "step_size_bytes": 8,
            "fixed_basis_count": 6,
            "coordinate_embedding_bytes": 0,
            "architecture_token_bytes": 0,
            "task_conditioning_bytes": 0,
            "broadcast_module_bytes": 0,
            "initializer_bytes": 0,
            "persistent_synapse_state_bytes": 0,
            "total_bytes": int(coefficients.nbytes + 8),
            "dtype": "float64",
        }

    def feedback(
        self,
        post_coordinates: FloatArray,
        *,
        layer_coordinate: float,
        is_output: bool,
        step: int,
    ) -> FloatArray:
        coordinates = np.asarray(post_coordinates, dtype=np.float64)
        if coordinates.ndim != 1:
            raise ValueError("post coordinates must be one-dimensional")
        if is_output:
            return np.ones_like(coordinates)
        c = np.asarray(self.coefficients, dtype=np.float64)
        time = step / (step + 5.0)
        logits = (
            c[0]
            + c[1] * np.cos(2 * np.pi * coordinates)
            + c[2] * np.sin(2 * np.pi * coordinates)
            + c[3] * np.cos(4 * np.pi * coordinates)
            + c[4] * np.cos(2 * np.pi * layer_coordinate)
            + c[5] * np.sin(2 * np.pi * layer_coordinate)
            + time * (c[6] + c[7] * np.sin(2 * np.pi * coordinates))
        )
        return np.tanh(logits)

    def local_delta_layer(
        self,
        *,
        pre_activity: FloatArray,
        post_activity: FloatArray,
        broadcast_error: FloatArray,
        weights: FloatArray,
        post_coordinates: FloatArray,
        layer_coordinate: float,
        is_output: bool,
        step: int,
    ) -> FloatArray:
        pre = np.asarray(pre_activity, dtype=np.float64)
        post = np.asarray(post_activity, dtype=np.float64)
        error = np.asarray(broadcast_error, dtype=np.float64)
        if pre.ndim != 2 or post.ndim != 2 or error.shape != (pre.shape[0],):
            raise ValueError("activities/error have incompatible batch shapes")
        if weights.shape != (post.shape[1], pre.shape[1]):
            raise ValueError("weight shape does not match endpoint activities")
        feedback = self.feedback(
            post_coordinates,
            layer_coordinate=layer_coordinate,
            is_output=is_output,
            step=step,
        )
        error_pre = np.einsum("n,ni->i", error, pre) / pre.shape[0]
        error_post_pre = np.einsum("n,nj,ni->ji", error, post, pre) / pre.shape[0]
        c = np.asarray(self.coefficients, dtype=np.float64)
        time = step / (step + 5.0)
        rate = np.exp(np.clip(c[8] + time * c[9], -2.0, 2.0))
        signal = feedback[:, None] * error_pre[None, :] + c[10] * error_post_pre
        signal += c[11] * weights
        return -self.step_size * rate * signal / np.sqrt(max(pre.shape[1], 1))

    def update(
        self,
        state: NetworkState,
        x: FloatArray,
        y: FloatArray,
        *,
        step: int,
    ) -> NetworkState:
        prediction, activations = forward(state, x)
        broadcast_error = prediction - y
        n_layers = len(state.weights)
        updated = []
        for layer, weight in enumerate(state.weights):
            post_width = weight.shape[0]
            coordinates = (np.arange(post_width, dtype=np.float64) + 0.5) / post_width
            layer_coordinate = (layer + 0.5) / n_layers
            delta = self.local_delta_layer(
                pre_activity=activations[layer],
                post_activity=activations[layer + 1],
                broadcast_error=broadcast_error,
                weights=weight,
                post_coordinates=coordinates,
                layer_coordinate=layer_coordinate,
                is_output=layer == n_layers - 1,
                step=step,
            )
            updated.append(weight + delta)
        return NetworkState(tuple(updated))
