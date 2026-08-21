"""Episode ownership and checkpoint-safe accounting controls."""

from __future__ import annotations

import pytest

from instinct.core.holdout import OverlapError
from instinct.core.rl import AccountingHook, EpisodeRegistry


class TickClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


def test_episode_registry_rejects_cross_role_reuse_and_unknown_collection_ids() -> None:
    registry = EpisodeRegistry.from_splits(train=[1, 2], validation=[3], eval=[4, 5])
    with pytest.raises(OverlapError):
        registry.register("eval", [2])
    with pytest.raises(ValueError, match="not registered"):
        registry.require("train", [1, 99])
    with pytest.raises(ValueError, match="not registered"):
        registry.require("train", [4])


def test_registry_state_round_trips_without_weakening_split_ownership() -> None:
    source = EpisodeRegistry.from_splits(train=[1, 2], validation=[3], eval=[4, 5])
    restored = EpisodeRegistry()
    restored.load_state_dict(source.state_dict())
    assert restored.state_dict() == source.state_dict()
    with pytest.raises(OverlapError):
        restored.register("validation", [4])


def test_accounting_hook_counts_and_round_trips() -> None:
    hook = AccountingHook(clock=TickClock())
    with hook.timed("environment"):
        pass
    with hook.timed("policy"):
        pass
    hook.record(env_calls=2, env_steps=7, policy_calls=2, action_decisions=7)
    state = hook.state_dict()
    assert state["wall_clock_s"] == pytest.approx(0.002)
    assert state["environment_s"] == pytest.approx(0.001)
    assert state["policy_s"] == pytest.approx(0.001)

    restored = AccountingHook(clock=TickClock())
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    with pytest.raises(ValueError, match="nonnegative"):
        restored.record(env_steps=-1)
