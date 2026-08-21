"""Tabular learner known answers and interrupted/resumed training equivalence."""

from __future__ import annotations

import numpy as np
import pytest

from instinct.core.checkpoint import CheckpointStore
from instinct.core.env import EnvState
from instinct.core.envs.tabular import TabularEnv, corridor_with_pit
from instinct.core.rl import (
    AccountingHook,
    EpisodeConfig,
    EpisodeRegistry,
    TabularQLearner,
    TrainingCheckpoint,
    TransitionBatch,
    train_episodes,
)
from instinct.core.rng import SeedScope


class TickClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


def _registry() -> EpisodeRegistry:
    return EpisodeRegistry.from_splits(
        train=range(40), validation=range(100, 110), eval=range(200, 210)
    )


def _learner(n_states: int, n_actions: int) -> TabularQLearner:
    learner = TabularQLearner(
        n_states=n_states,
        n_actions=n_actions,
        alpha=0.25,
        gamma=0.9,
        epsilon=0.25,
    )
    assert learner.q is not None
    learner.q[:, 1] = 0.1  # begin with a deliberately poor preference for waiting
    return learner


def _train_batches(env, learner, registry, accounting, batches) -> None:  # type: ignore[no-untyped-def]
    for batch in batches:
        train_episodes(
            env,
            learner,
            batch,
            registry=registry,
            scope=SeedScope(71),
            config=EpisodeConfig(max_steps=20, explore=True),
            accounting=accounting,
        )


def test_q_update_averages_duplicate_state_action_targets_independent_of_lane_order() -> None:
    state = EnvState(np.array([1, 2]), {"s": np.array([0, 0])})
    nxt = EnvState(np.array([1, 2]), {"s": np.array([1, 1])})
    transition = TransitionBatch(
        state=state,
        action=np.array([0, 0]),
        reward=np.array([1.0, 3.0]),
        next_state=nxt,
        done=np.array([True, True]),
        truncated=np.array([False, False]),
        active=np.array([True, True]),
        tick=0,
    )
    a = TabularQLearner(2, 2, alpha=0.5)
    b = TabularQLearner(2, 2, alpha=0.5)
    a.observe(transition)
    reverse = np.array([1, 0])
    b.observe(
        TransitionBatch(
            state=state.take(reverse),
            action=transition.action[reverse],
            reward=transition.reward[reverse],
            next_state=nxt.take(reverse),
            done=transition.done[reverse],
            truncated=transition.truncated[reverse],
            active=transition.active[reverse],
            tick=0,
        )
    )
    assert a.q is not None and b.q is not None
    assert a.q[0, 0] == 1.0  # alpha * mean([1, 3])
    assert np.array_equal(a.q, b.q)


def test_checkpointed_training_matches_uninterrupted_training(tmp_path) -> None:
    mdp = corridor_with_pit(length=5, gamma=0.9, slip=0.15)
    env = TabularEnv(mdp, start_state=0)
    batches = [list(range(start, start + 5)) for start in range(0, 40, 5)]

    clean_learner = _learner(mdp.n_states, mdp.n_actions)
    clean_registry = _registry()
    clean_accounting = AccountingHook(clock=TickClock())
    _train_batches(env, clean_learner, clean_registry, clean_accounting, batches)

    partial_learner = _learner(mdp.n_states, mdp.n_actions)
    partial_registry = _registry()
    partial_accounting = AccountingHook(clock=TickClock())
    _train_batches(env, partial_learner, partial_registry, partial_accounting, batches[:3])
    training_scope = SeedScope(71).child("resume-test")
    checkpoint = TrainingCheckpoint.capture(
        3,
        partial_learner,
        partial_registry,
        partial_accounting,
        training_scope,
    )
    store = CheckpointStore(tmp_path)
    store.save(checkpoint.state_dict())

    resumed_learner = _learner(mdp.n_states, mdp.n_actions)
    resumed_registry = EpisodeRegistry()
    resumed_accounting = AccountingHook(clock=TickClock())
    loaded = store.load()
    assert loaded is not None
    restored = TrainingCheckpoint.from_state_dict(loaded)
    next_batch = restored.restore(
        resumed_learner, resumed_registry, resumed_accounting
    )
    assert restored.restored_scope() == training_scope
    _train_batches(
        env,
        resumed_learner,
        resumed_registry,
        resumed_accounting,
        batches[next_batch:],
    )

    assert clean_learner.q is not None and resumed_learner.q is not None
    assert np.array_equal(clean_learner.q, resumed_learner.q)
    assert clean_learner.updates == resumed_learner.updates
    assert clean_registry.state_dict() == resumed_registry.state_dict()
    clean_counts = clean_accounting.state_dict()
    resumed_counts = resumed_accounting.state_dict()
    for key in (
        "env_calls",
        "env_steps",
        "policy_calls",
        "action_decisions",
        "learner_calls",
        "learner_updates",
    ):
        assert clean_counts[key] == resumed_counts[key]
    for key in ("wall_clock_s", "environment_s", "policy_s", "learner_s"):
        assert clean_counts[key] == pytest.approx(resumed_counts[key])
    assert not np.allclose(clean_learner.q, 0.0)
