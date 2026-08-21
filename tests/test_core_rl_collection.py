"""Deterministic collection, reference equivalence, pairing, and accounting."""

from __future__ import annotations

import numpy as np

from instinct.core.env import EnvState
from instinct.core.envs.tabular import TabularEnv, chase_chain
from instinct.core.rl import (
    EpisodeConfig,
    EpisodeRegistry,
    collect_episodes,
    collect_episodes_naive,
    evaluate_paired,
)
from instinct.core.rng import SeedScope


class FixedPolicy:
    def __init__(self, action: int) -> None:
        self.action = action

    def act(
        self,
        state: EnvState,
        *,
        scope: SeedScope,
        episode: int,
        tick: int,
        explore: bool,
    ) -> np.ndarray:
        del scope, episode, tick, explore
        return np.full(state.n_lanes, self.action, dtype=np.int64)


def _registry() -> EpisodeRegistry:
    return EpisodeRegistry.from_splits(
        train=range(0, 16), validation=range(100, 108), eval=range(200, 216)
    )


def test_batched_collection_matches_one_episode_at_a_time_reference() -> None:
    env = TabularEnv(chase_chain(n_positions=5, drift=0.4), start_state=0)
    ids = list(range(200, 212))
    config = EpisodeConfig(max_steps=25, return_gamma=0.95)
    kwargs = dict(role="eval", registry=_registry(), scope=SeedScope(41), config=config)
    batched = collect_episodes(env, FixedPolicy(2), ids, **kwargs)
    naive = collect_episodes_naive(env, FixedPolicy(2), ids, **kwargs)
    assert np.array_equal(batched.episode_ids, naive.episode_ids)
    assert np.array_equal(batched.returns, naive.returns)
    assert np.array_equal(batched.lengths, naive.lengths)
    assert np.array_equal(batched.terminated, naive.terminated)
    assert np.array_equal(batched.truncated, naive.truncated)


def test_horizon_truncation_is_recorded_separately_from_termination() -> None:
    env = TabularEnv(chase_chain(n_positions=5, drift=0.0), start_state=0)
    batch = collect_episodes(
        env,
        FixedPolicy(0),
        [200, 201],
        role="eval",
        registry=_registry(),
        scope=SeedScope(45),
        config=EpisodeConfig(max_steps=1),
    )
    assert np.array_equal(batch.terminated, [False, False])
    assert np.array_equal(batch.truncated, [True, True])


def test_collection_is_invariant_to_lane_order() -> None:
    env = TabularEnv(chase_chain(n_positions=4, drift=0.5), start_state=0)
    ids = np.arange(200, 210, dtype=np.int64)
    config = EpisodeConfig(max_steps=15)
    first = collect_episodes(
        env, FixedPolicy(0), ids, role="eval", registry=_registry(), scope=SeedScope(42), config=config
    )
    order = np.array([7, 2, 9, 0, 4, 1, 8, 5, 3, 6])
    shuffled = collect_episodes(
        env,
        FixedPolicy(0),
        ids[order],
        role="eval",
        registry=_registry(),
        scope=SeedScope(42),
        config=config,
    )
    inverse = np.argsort(order)
    assert np.array_equal(first.returns, shuffled.returns[inverse])
    assert np.array_equal(first.lengths, shuffled.lengths[inverse])


def test_paired_evaluation_reuses_environment_noise_and_reports_differences() -> None:
    env = TabularEnv(chase_chain(n_positions=5, drift=0.6), start_state=0)
    paired = evaluate_paired(
        env,
        {"a": FixedPolicy(2), "b": FixedPolicy(2)},
        range(200, 216),
        registry=_registry(),
        scope=SeedScope(43),
        config=EpisodeConfig(max_steps=20),
    )
    assert paired.common_random_numbers
    assert np.array_equal(paired.paired_difference("a", "b"), np.zeros(16))
    for cost in paired.accounting.values():
        assert cost.env_steps == int(paired.episodes["a"].lengths.sum())
        assert cost.action_decisions == cost.env_steps
        assert cost.env_calls == 20
        assert cost.policy_calls == 20
        assert cost.wall_clock_s >= 0.0


def test_policy_action_shape_and_range_are_checked_before_environment_step() -> None:
    env = TabularEnv(chase_chain(n_positions=3), start_state=0)

    class BadPolicy(FixedPolicy):
        def act(self, state: EnvState, **kwargs) -> np.ndarray:  # type: ignore[no-untyped-def]
            del kwargs
            return np.full(state.n_lanes, 99, dtype=np.int64)

    import pytest

    with pytest.raises(ValueError, match="action outside"):
        collect_episodes(
            env,
            BadPolicy(99),
            [200],
            role="eval",
            registry=_registry(),
            scope=SeedScope(44),
        )
