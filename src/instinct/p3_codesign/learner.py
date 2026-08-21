"""Small auditable time-conditioned softmax reflex for P3.1."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from instinct.core.rng import SeedScope

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass(frozen=True, slots=True)
class ReflexSnapshot:
    """Immutable full schedule produced by a frozen learner or baseline."""

    name: str
    logits: FloatArray
    fingerprint: str

    def probabilities(self, remaining: int) -> FloatArray:
        if not 1 <= remaining < self.logits.shape[0]:
            raise ValueError("remaining outside snapshot schedule")
        return _softmax(self.logits[remaining])

    def actions(self, remaining: int) -> IntArray:
        return np.argmax(self.logits[remaining], axis=1).astype(np.int64)


def snapshot_from_logits(name: str, logits: FloatArray) -> ReflexSnapshot:
    values = np.asarray(logits, dtype=np.float64).copy()
    digest = hashlib.sha256(name.encode() + values.tobytes()).hexdigest()[:16]
    values.setflags(write=False)
    return ReflexSnapshot(name=name, logits=values, fingerprint=digest)


def fixed_snapshot(
    name: str, actions: IntArray, max_delay: int, n_actions: int
) -> ReflexSnapshot:
    action_map = np.asarray(actions, dtype=np.int64)
    n_states = action_map.size
    logits = np.full((max_delay + 1, n_states, n_actions), -30.0)
    for remaining in range(1, max_delay + 1):
        logits[remaining, np.arange(n_states), action_map] = 30.0
    return snapshot_from_logits(name, logits)


def uniform_snapshot(n_states: int, n_actions: int, max_delay: int) -> ReflexSnapshot:
    return snapshot_from_logits(
        "random", np.zeros((max_delay + 1, n_states, n_actions), dtype=np.float64)
    )


@dataclass
class TimeConditionedSoftmaxReflex:
    n_states: int
    n_actions: int
    max_delay: int
    learning_rate: float = 0.15
    weights: FloatArray | None = None

    def __post_init__(self) -> None:
        if self.n_states < 1 or self.n_actions < 2 or self.max_delay < 1:
            raise ValueError("invalid learner dimensions")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must lie in (0, 1]")
        if self.weights is None:
            self.weights = np.zeros((self.n_states, self.n_actions, 2), dtype=np.float64)

    def _logits(self, state: int, remaining: int) -> FloatArray:
        assert self.weights is not None
        time = remaining / self.max_delay
        return self.weights[state, :, 0] + time * self.weights[state, :, 1]

    def fit(
        self,
        states: IntArray,
        remaining: IntArray,
        targets: IntArray,
        *,
        seed: int,
        epochs: int,
    ) -> None:
        assert self.weights is not None
        if not (states.shape == remaining.shape == targets.shape):
            raise ValueError("training arrays must have equal shapes")
        lanes = np.arange(states.size, dtype=np.int64)
        stream = SeedScope(seed).child("p3", "learner").stream("order")
        for epoch in range(epochs):
            order = np.argsort(stream.uniform(lanes, 1, tick=epoch)[:, 0])
            for index in order:
                state = int(states[index])
                left = int(remaining[index])
                target = int(targets[index])
                logits = self._logits(state, left)[None, :]
                probs = _softmax(logits)[0]
                gradient = probs
                gradient[target] -= 1.0
                time = left / self.max_delay
                self.weights[state, :, 0] -= self.learning_rate * gradient
                self.weights[state, :, 1] -= self.learning_rate * time * gradient

    def snapshot(self, name: str = "planner_aware") -> ReflexSnapshot:
        logits = np.zeros((self.max_delay + 1, self.n_states, self.n_actions))
        for remaining in range(1, self.max_delay + 1):
            for state in range(self.n_states):
                logits[remaining, state] = self._logits(state, remaining)
        return snapshot_from_logits(name, logits)
