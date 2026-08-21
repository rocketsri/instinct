"""Low-overhead step, call, update, and wall-clock accounting hooks."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Literal

__all__ = ["AccountingHook", "AccountingSnapshot", "TimerKind"]

TimerKind = Literal["environment", "policy", "learner"]


@dataclass(frozen=True, slots=True)
class AccountingSnapshot:
    env_calls: int = 0
    env_steps: int = 0
    policy_calls: int = 0
    action_decisions: int = 0
    learner_calls: int = 0
    learner_updates: int = 0
    wall_clock_s: float = 0.0
    environment_s: float = 0.0
    policy_s: float = 0.0
    learner_s: float = 0.0

    def as_record(self) -> dict[str, int | float]:
        return asdict(self)


class AccountingHook:
    """Mutable accounting sink with an injectable monotonic clock."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._values: dict[str, int | float] = AccountingSnapshot().as_record()

    def record(
        self,
        *,
        env_calls: int = 0,
        env_steps: int = 0,
        policy_calls: int = 0,
        action_decisions: int = 0,
        learner_calls: int = 0,
        learner_updates: int = 0,
    ) -> None:
        increments = {
            "env_calls": env_calls,
            "env_steps": env_steps,
            "policy_calls": policy_calls,
            "action_decisions": action_decisions,
            "learner_calls": learner_calls,
            "learner_updates": learner_updates,
        }
        if any(value < 0 for value in increments.values()):
            raise ValueError("accounting increments must be nonnegative")
        for name, value in increments.items():
            self._values[name] = int(self._values[name]) + value

    @contextmanager
    def timed(self, kind: TimerKind) -> Iterator[None]:
        start = self._clock()
        try:
            yield
        finally:
            elapsed = max(0.0, self._clock() - start)
            key = f"{kind}_s"
            self._values[key] = float(self._values[key]) + elapsed
            self._values["wall_clock_s"] = float(self._values["wall_clock_s"]) + elapsed

    def snapshot(self) -> AccountingSnapshot:
        return AccountingSnapshot(
            env_calls=int(self._values["env_calls"]),
            env_steps=int(self._values["env_steps"]),
            policy_calls=int(self._values["policy_calls"]),
            action_decisions=int(self._values["action_decisions"]),
            learner_calls=int(self._values["learner_calls"]),
            learner_updates=int(self._values["learner_updates"]),
            wall_clock_s=float(self._values["wall_clock_s"]),
            environment_s=float(self._values["environment_s"]),
            policy_s=float(self._values["policy_s"]),
            learner_s=float(self._values["learner_s"]),
        )

    def state_dict(self) -> dict[str, int | float]:
        return self.snapshot().as_record()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        allowed = set(AccountingSnapshot().as_record())
        if set(state) != allowed:
            raise ValueError(f"accounting state fields must be exactly {sorted(allowed)}")
        restored = AccountingSnapshot(
            env_calls=int(state["env_calls"]),
            env_steps=int(state["env_steps"]),
            policy_calls=int(state["policy_calls"]),
            action_decisions=int(state["action_decisions"]),
            learner_calls=int(state["learner_calls"]),
            learner_updates=int(state["learner_updates"]),
            wall_clock_s=float(state["wall_clock_s"]),
            environment_s=float(state["environment_s"]),
            policy_s=float(state["policy_s"]),
            learner_s=float(state["learner_s"]),
        )
        if any(value < 0 for value in restored.as_record().values()):
            raise ValueError("accounting state cannot contain negative values")
        self._values = restored.as_record()
