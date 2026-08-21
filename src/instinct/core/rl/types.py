"""Typed interfaces and immutable records for lightweight RL experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from instinct.core.env import EnvState
from instinct.core.rng import SeedScope

__all__ = [
    "EpisodeBatch",
    "EpisodeConfig",
    "Learner",
    "Policy",
    "TransitionBatch",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@runtime_checkable
class Policy(Protocol):
    """A policy whose stochasticity is addressed through :class:`SeedScope`."""

    def act(
        self,
        state: EnvState,
        *,
        scope: SeedScope,
        episode: int,
        tick: int,
        explore: bool,
    ) -> IntArray: ...


@runtime_checkable
class Learner(Policy, Protocol):
    """An online learner with canonical checkpoint state."""

    def observe(self, transition: TransitionBatch) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True, slots=True)
class TransitionBatch:
    """One synchronous environment step over a set of episode lanes."""

    state: EnvState
    action: IntArray
    reward: FloatArray
    next_state: EnvState
    done: BoolArray
    truncated: BoolArray
    active: BoolArray
    tick: int

    @property
    def n_active(self) -> int:
        return int(self.active.sum())


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    """Collection semantics shared by training and evaluation."""

    max_steps: int = 100
    return_gamma: float = 1.0
    explore: bool = False

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not 0.0 <= self.return_gamma <= 1.0:
            raise ValueError("return_gamma must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class EpisodeBatch:
    """Per-episode outcomes in the same order as the requested IDs."""

    episode_ids: IntArray
    returns: FloatArray
    lengths: IntArray
    terminated: BoolArray
    truncated: BoolArray

    def __post_init__(self) -> None:
        n = self.episode_ids.shape[0]
        for name, value in (
            ("returns", self.returns),
            ("lengths", self.lengths),
            ("terminated", self.terminated),
            ("truncated", self.truncated),
        ):
            if value.shape != (n,):
                raise ValueError(f"{name} has shape {value.shape}, expected {(n,)}")

    @property
    def mean_return(self) -> float:
        return float(self.returns.mean()) if self.returns.size else float("nan")
