"""Minimal deterministic batched Q-learning reference implementation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from instinct.core.env import EnvState
from instinct.core.rl.types import TransitionBatch
from instinct.core.rng import SeedScope

__all__ = ["TabularQLearner"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass
class TabularQLearner:
    """Batch Q-learning for environments exposing integer field ``state_key``.

    Updates sharing a state/action pair are averaged against one frozen Q table
    per environment tick. This makes a training step invariant to lane order;
    sequentially mutating Q once per lane would make batching itself a hidden
    optimizer choice.
    """

    n_states: int
    n_actions: int
    alpha: float = 0.2
    gamma: float = 0.95
    epsilon: float = 0.1
    state_key: str = "s"
    q: FloatArray | None = field(default=None, repr=False)
    updates: int = 0

    def __post_init__(self) -> None:
        if self.n_states < 1 or self.n_actions < 1:
            raise ValueError("n_states and n_actions must be positive")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must lie in (0, 1]")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0, 1]")
        if self.q is None:
            self.q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        else:
            self.q = np.asarray(self.q, dtype=np.float64).copy()
            if self.q.shape != (self.n_states, self.n_actions):
                raise ValueError(
                    f"q has shape {self.q.shape}, expected {(self.n_states, self.n_actions)}"
                )

    def _states(self, state: EnvState) -> IntArray:
        if self.state_key not in state.fields:
            raise KeyError(f"environment state has no field {self.state_key!r}")
        values = np.asarray(state[self.state_key], dtype=np.int64)
        if values.shape != (state.n_lanes,):
            raise ValueError(f"state field {self.state_key!r} must be one-dimensional")
        if np.any((values < 0) | (values >= self.n_states)):
            raise ValueError("tabular state index outside learner state space")
        return values

    def act(
        self,
        state: EnvState,
        *,
        scope: SeedScope,
        episode: int,
        tick: int,
        explore: bool,
    ) -> IntArray:
        assert self.q is not None
        states = self._states(state)
        greedy = np.argmax(self.q[states], axis=1).astype(np.int64)
        if not explore or self.epsilon == 0.0:
            return greedy
        choose_random = scope.stream("epsilon").uniform(
            state.lane_ids, 1, episode=episode, tick=tick
        )[:, 0] < self.epsilon
        random_action = scope.stream("random-action").integers(
            state.lane_ids, self.n_actions, 1, episode=episode, tick=tick
        )[:, 0]
        return np.where(choose_random, random_action, greedy).astype(np.int64)

    def observe(self, transition: TransitionBatch) -> None:
        assert self.q is not None
        active = np.flatnonzero(transition.active)
        if active.size == 0:
            return
        state = self._states(transition.state)[active]
        nxt = self._states(transition.next_state)[active]
        action = np.asarray(transition.action, dtype=np.int64)[active]
        if np.any((action < 0) | (action >= self.n_actions)):
            raise ValueError("transition action outside learner action space")
        reward = np.asarray(transition.reward, dtype=np.float64)[active]
        done = np.asarray(transition.done, dtype=bool)[active]

        frozen = self.q.copy()
        target = reward + self.gamma * (~done) * frozen[nxt].max(axis=1)
        pair = state * self.n_actions + action
        for encoded in np.unique(pair):
            members = pair == encoded
            s, a = divmod(int(encoded), self.n_actions)
            mean_target = float(target[members].mean())
            self.q[s, a] += self.alpha * (mean_target - self.q[s, a])
        self.updates += int(active.size)

    def state_dict(self) -> dict[str, Any]:
        assert self.q is not None
        return {
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "state_key": self.state_key,
            "q": self.q.tolist(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "state_key": self.state_key,
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError(
                    f"checkpoint {name}={state.get(name)!r} does not match learner {value!r}"
                )
        q = np.asarray(state.get("q"), dtype=np.float64)
        if q.shape != (self.n_states, self.n_actions) or not np.isfinite(q).all():
            raise ValueError("checkpoint Q table has an invalid shape or nonfinite values")
        updates = int(state.get("updates", -1))
        if updates < 0:
            raise ValueError("checkpoint update count must be nonnegative")
        self.q = q.copy()
        self.updates = updates
