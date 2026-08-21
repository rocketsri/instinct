from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CoordinateGenerator:
    """Fixed-length program; architecture size changes outputs, never parameters."""

    coefficients: FloatArray = field(
        default_factory=lambda: np.array([0.7, -0.3, 0.2, 0.05], dtype=np.float64)
    )

    def __post_init__(self) -> None:
        if np.asarray(self.coefficients).shape != (4,):
            raise ValueError("generator has exactly four fixed coefficients")

    @property
    def code_bytes(self) -> int:
        return int(np.asarray(self.coefficients).nbytes)

    def serialization_manifest(self) -> dict[str, int | str]:
        """Everything persistent that belongs in the program-length claim."""
        coefficients = np.asarray(self.coefficients)
        return {
            "coefficient_count": int(coefficients.size),
            "coefficient_bytes": int(coefficients.nbytes),
            "coordinate_embedding_bytes": 0,
            "architecture_token_bytes": 0,
            "task_token_bytes": 0,
            "broadcast_module_bytes": 0,
            "total_bytes": int(coefficients.nbytes),
            "dtype": str(coefficients.dtype),
        }

    def weights(self, pre_coordinates: FloatArray, post_coordinates: FloatArray) -> FloatArray:
        x = np.asarray(pre_coordinates, dtype=np.float64)[None, :]
        y = np.asarray(post_coordinates, dtype=np.float64)[:, None]
        a, b, c, _ = self.coefficients
        raw = a * np.sin(np.pi * (x + y)) + b * np.cos(np.pi * (x - y)) + c * x * y
        return raw / np.sqrt(max(x.shape[1], 1))

    def local_delta(
        self,
        pre_activity: FloatArray,
        post_activity: FloatArray,
        broadcast_error: FloatArray,
        pre_coordinates: FloatArray,
        post_coordinates: FloatArray,
        *,
        step: int,
    ) -> FloatArray:
        pre = np.asarray(pre_activity, dtype=np.float64)[None, :]
        post = np.asarray(post_activity, dtype=np.float64)[:, None]
        error = np.asarray(broadcast_error, dtype=np.float64)
        if error.shape != (len(post_coordinates),):
            raise ValueError("broadcast_error must have one scalar per postsynaptic neuron")
        eta = abs(float(self.coefficients[3])) / np.sqrt(max(pre.shape[1], 1))
        coordinate_gate = 1.0 + 0.1 * self.weights(pre_coordinates, post_coordinates)
        return eta * coordinate_gate * np.tanh(pre * post + error[:, None] + 0.01 * step)

    def response(self, width: int, post_coordinates: FloatArray) -> FloatArray:
        pre_coordinates = (np.arange(width, dtype=np.float64) + 0.5) / width
        signal = np.sin(2 * np.pi * pre_coordinates)
        return self.weights(pre_coordinates, post_coordinates) @ signal / np.sqrt(width)

    def function_output(
        self,
        pre_activity: FloatArray,
        pre_coordinates: FloatArray,
        post_coordinates: FloatArray,
    ) -> FloatArray:
        """Width-normalized generated layer output for equivariance tests."""
        activity = np.asarray(pre_activity, dtype=np.float64)
        if activity.shape != np.asarray(pre_coordinates).shape:
            raise ValueError("pre_activity and pre_coordinates must align")
        return (
            self.weights(pre_coordinates, post_coordinates)
            @ activity
            / np.sqrt(max(activity.size, 1))
        )
