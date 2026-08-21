"""Plasticity-only components for the bounded P5.2 regression pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Topology:
    width: int
    depth: int

    def __post_init__(self) -> None:
        if self.width < 2 or self.depth < 1:
            raise ValueError("width must be >= 2 and depth must be >= 1")

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        return (3, *(self.width for _ in range(self.depth)), 1)

    @property
    def parameter_count(self) -> int:
        return sum(a * b for a, b in zip(self.layer_sizes[:-1], self.layer_sizes[1:]))


@dataclass(frozen=True, slots=True)
class RegressionTask:
    family: str
    coefficients: FloatArray

    def targets(self, x: FloatArray) -> FloatArray:
        a, b, c, d = self.coefficients
        if self.family == "sinusoid":
            return a * np.sin(b * x + c) + d
        if self.family == "polynomial":
            return a * x**3 + b * x**2 + c * x + d
        if self.family == "composition":
            return a * np.sin(b * x + c) + d * x**2
        if self.family == "warped":
            return a * np.sin(b * x + c) + d * x**3
        if self.family == "chirp":
            return a * np.sin(b * x**2 + c) + d * x
        raise ValueError(f"unknown task family {self.family!r}")


def make_task(seed: int, family: str) -> RegressionTask:
    rng = np.random.default_rng(seed)
    if family == "sinusoid":
        coefficients = np.array(
            [
                rng.uniform(0.4, 1.2),
                rng.uniform(0.7, 2.0),
                rng.uniform(-1, 1),
                rng.uniform(-0.2, 0.2),
            ]
        )
    elif family == "polynomial":
        coefficients = rng.uniform(-0.6, 0.6, size=4)
    elif family in {"composition", "warped", "chirp"}:
        coefficients = np.array(
            [
                rng.uniform(0.4, 1.0),
                rng.uniform(0.8, 1.8),
                rng.uniform(-0.8, 0.8),
                rng.uniform(-0.35, 0.35),
            ]
        )
    else:
        raise ValueError(f"unknown task family {family!r}")
    return RegressionTask(family, coefficients.astype(np.float64))


def features(x: FloatArray) -> FloatArray:
    values = np.asarray(x, dtype=np.float64)
    return np.column_stack((values, values**2, np.ones_like(values)))


@dataclass
class NetworkState:
    weights: tuple[FloatArray, ...]

    def copy(self) -> NetworkState:
        return NetworkState(tuple(weight.copy() for weight in self.weights))


class PlasticityProgram(Protocol):
    def update(
        self,
        state: NetworkState,
        x: FloatArray,
        y: FloatArray,
        *,
        step: int,
    ) -> NetworkState: ...

    def serialization_manifest(self) -> dict[str, int | float | str]: ...


def conventional_initialization(topology: Topology, *, seed: int) -> NetworkState:
    """Fixed standard fan-in initialization, independent of the plasticity program."""
    rng = np.random.default_rng(seed + 1009 * topology.width + 9176 * topology.depth)
    weights = tuple(
        rng.normal(0.0, 0.45 / np.sqrt(fan_in), size=(fan_out, fan_in)).astype(np.float64)
        for fan_in, fan_out in zip(topology.layer_sizes[:-1], topology.layer_sizes[1:])
    )
    return NetworkState(weights)


def forward(state: NetworkState, x: FloatArray) -> tuple[FloatArray, tuple[FloatArray, ...]]:
    activation = features(x)
    activations = [activation]
    for index, weight in enumerate(state.weights):
        activation = activation @ weight.T
        if index < len(state.weights) - 1:
            activation = np.tanh(activation)
        activations.append(activation)
    return activation[:, 0], tuple(activations)


def mse(state: NetworkState, x: FloatArray, y: FloatArray) -> float:
    prediction, _ = forward(state, x)
    return float(np.mean((prediction - y) ** 2))


@dataclass(frozen=True, slots=True)
class LocalPlasticityProgram:
    """One fixed rule shared by every synapse, width, depth, and task."""

    coefficients: FloatArray
    step_size: float = 0.04

    def __post_init__(self) -> None:
        values = np.asarray(self.coefficients, dtype=np.float64)
        if values.shape not in {(4,), (8,)} or not np.isfinite(values).all():
            raise ValueError("plasticity program requires four or eight finite coefficients")
        if not 0.0 < self.step_size <= 0.5:
            raise ValueError("step_size must lie in (0, 0.5]")

    def serialization_manifest(self) -> dict[str, int | float | str]:
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        return {
            "coefficient_count": int(coefficients.size),
            "coefficient_bytes": int(coefficients.nbytes),
            "step_size_count": 1,
            "step_size_bytes": 8,
            "coordinate_embedding_bytes": 0,
            "architecture_token_bytes": 0,
            "task_conditioning_bytes": 0,
            "broadcast_module_bytes": 0,
            "initializer_bytes": 0,
            "total_bytes": int(coefficients.nbytes + 8),
            "dtype": "float64",
        }

    def local_delta_layer(
        self,
        *,
        pre_activity: FloatArray,
        post_activity: FloatArray,
        broadcast_error: FloatArray,
        weights: FloatArray,
        post_coordinates: FloatArray,
        is_output: bool,
        step: int,
    ) -> FloatArray:
        """Compute synapse updates from only its endpoints, weight, error, and type."""
        del step  # time is declared but this four-term pilot does not exploit it.
        pre = np.asarray(pre_activity, dtype=np.float64)
        post = np.asarray(post_activity, dtype=np.float64)
        error = np.asarray(broadcast_error, dtype=np.float64)
        if pre.ndim != 2 or post.ndim != 2 or error.shape != (pre.shape[0],):
            raise ValueError("activities/error have incompatible batch shapes")
        if weights.shape != (post.shape[1], pre.shape[1]):
            raise ValueError("weight shape does not match endpoint activities")
        coordinates = np.asarray(post_coordinates, dtype=np.float64)
        if coordinates.shape != (post.shape[1],):
            raise ValueError("one coordinate is required per postsynaptic neuron")
        feedback = np.ones(post.shape[1]) if is_output else np.cos(2 * np.pi * coordinates)
        error_pre = np.einsum("n,ni->i", error, pre) / pre.shape[0]
        error_post_pre = np.einsum("n,nj,ni->ji", error, post, pre) / pre.shape[0]
        post_pre = np.einsum("nj,ni->ji", post, pre) / pre.shape[0]
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        a, b, c, d = coefficients[:4]
        signal = (
            a * feedback[:, None] * error_pre[None, :]
            + b * error_post_pre
            + c * post_pre
            + d * weights
        )
        if coefficients.size == 8:
            e, f, g, h = coefficients[4:]
            robust_error_pre = np.einsum("n,ni->i", np.tanh(error), pre) / pre.shape[0]
            derivative_term = np.einsum(
                "n,nj,nj,ni->ji", error, 1.0 - post**2, post, pre
            ) / pre.shape[0]
            coordinate_gate = np.sin(2 * np.pi * coordinates)
            activity_scale = np.mean(np.abs(post), axis=0)
            signal += (
                e * feedback[:, None] * robust_error_pre[None, :]
                + f * derivative_term
                + g * coordinate_gate[:, None] * error_pre[None, :]
                + h * activity_scale[:, None] * weights
            )
        return -self.step_size * signal / np.sqrt(max(pre.shape[1], 1))

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
        updated = []
        for layer, weight in enumerate(state.weights):
            post_width = weight.shape[0]
            coordinates = (np.arange(post_width, dtype=np.float64) + 0.5) / post_width
            delta = self.local_delta_layer(
                pre_activity=activations[layer],
                post_activity=activations[layer + 1],
                broadcast_error=broadcast_error,
                weights=weight,
                post_coordinates=coordinates,
                is_output=layer == len(state.weights) - 1,
                step=step,
            )
            updated.append(weight + delta)
        return NetworkState(tuple(updated))


def backprop_update(
    state: NetworkState,
    x: FloatArray,
    y: FloatArray,
    *,
    step_size: float,
) -> NetworkState:
    """Direct learned ceiling using the same examples and one update per step."""
    prediction, activations = forward(state, x)
    delta = (2.0 / len(x)) * (prediction - y)[:, None]
    gradients: list[FloatArray] = []
    for layer in range(len(state.weights) - 1, -1, -1):
        gradients.append(delta.T @ activations[layer])
        if layer:
            delta = (delta @ state.weights[layer]) * (1.0 - activations[layer] ** 2)
    gradients.reverse()
    return NetworkState(
        tuple(weight - step_size * gradient for weight, gradient in zip(state.weights, gradients))
    )


def adaptation_curve(
    initial: NetworkState,
    task: RegressionTask,
    *,
    program: PlasticityProgram | None,
    steps: int,
    direct_backprop: bool = False,
) -> FloatArray:
    support_x = np.linspace(-1.0, 1.0, 24)
    query_x = np.linspace(-1.1, 1.1, 48)
    support_y, query_y = task.targets(support_x), task.targets(query_x)
    state = initial.copy()
    losses = [mse(state, query_x, query_y)]
    for step in range(steps):
        if direct_backprop:
            state = backprop_update(state, support_x, support_y, step_size=0.04)
        elif program is not None:
            state = program.update(state, support_x, support_y, step=step)
        losses.append(mse(state, query_x, query_y))
    return np.asarray(losses, dtype=np.float64)
