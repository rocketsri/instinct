"""Deterministic batched collection, naive reference, and paired evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from instinct.core.env import BatchedEnv
from instinct.core.rl.accounting import AccountingHook, AccountingSnapshot
from instinct.core.rl.registry import EpisodeRegistry, EpisodeRole
from instinct.core.rl.types import (
    EpisodeBatch,
    EpisodeConfig,
    Learner,
    Policy,
    TransitionBatch,
)
from instinct.core.rng import SeedScope

__all__ = [
    "PairedEvaluation",
    "collect_episodes",
    "collect_episodes_naive",
    "evaluate_paired",
    "train_episodes",
]

FloatArray = npt.NDArray[np.float64]


def _episode_array(episode_ids: Sequence[int]) -> npt.NDArray[np.int64]:
    ids = np.asarray(episode_ids, dtype=np.int64)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("episode_ids must be a nonempty 1-D sequence")
    if np.unique(ids).size != ids.size:
        raise ValueError("episode_ids must be unique within one collection call")
    if ids.min() < 0 or ids.max() >= 2**32:
        raise ValueError("episode IDs must lie in [0, 2**32)")
    return ids


def collect_episodes(
    env: BatchedEnv,
    policy: Policy,
    episode_ids: Sequence[int],
    *,
    role: EpisodeRole,
    registry: EpisodeRegistry,
    scope: SeedScope,
    config: EpisodeConfig | None = None,
    learner: Learner | None = None,
    accounting: AccountingHook | None = None,
    environment_scope: SeedScope | None = None,
    policy_scope: SeedScope | None = None,
) -> EpisodeBatch:
    """Collect whole episodes as lanes, with episode IDs serving as lane IDs.

    The environment API accepts one scalar ``episode`` counter for a batch.
    This harness fixes that counter at zero and places the experiment's episode
    ID in the lane counter instead. The pair ``(tick, lane=episode_id)`` remains
    globally addressable, so batching, lane order, and process resume cannot
    change any environment draw.
    """

    config = config or EpisodeConfig()
    ids = _episode_array(episode_ids)
    registry.require(role, [int(value) for value in ids])
    hook = accounting or AccountingHook()
    env_scope = environment_scope or scope.child("environment")
    act_scope = policy_scope or scope.child("policy")

    with hook.timed("environment"):
        state = env.reset(ids, scope=env_scope, episode=0)
    returns = np.zeros(ids.size, dtype=np.float64)
    lengths = np.zeros(ids.size, dtype=np.int64)
    active = np.ones(ids.size, dtype=bool)
    discount = 1.0

    for tick in range(config.max_steps):
        if not active.any():
            break
        with hook.timed("policy"):
            actions = np.asarray(
                policy.act(
                    state,
                    scope=act_scope,
                    episode=0,
                    tick=tick,
                    explore=config.explore,
                ),
                dtype=np.int64,
            )
        if actions.shape != (ids.size,):
            raise ValueError(
                f"policy returned actions with shape {actions.shape}, expected {ids.shape}"
            )
        if np.any((actions < 0) | (actions >= env.n_actions)):
            raise ValueError("policy returned an action outside the environment action space")
        hook.record(policy_calls=1, action_decisions=int(active.sum()))

        before = state.copy()
        with hook.timed("environment"):
            result = env.step(state, actions, scope=env_scope, episode=0, tick=tick)
        hook.record(env_calls=1, env_steps=int(active.sum()))
        reward = np.where(active, result.reward, 0.0).astype(np.float64)
        done = np.asarray(result.done, dtype=bool)
        truncated = active & ~done & (tick == config.max_steps - 1)
        transition = TransitionBatch(
            state=before,
            action=actions.copy(),
            reward=reward.copy(),
            next_state=result.state.copy(),
            done=done.copy(),
            truncated=truncated.copy(),
            active=active.copy(),
            tick=tick,
        )
        if learner is not None:
            with hook.timed("learner"):
                learner.observe(transition)
            hook.record(learner_calls=1, learner_updates=transition.n_active)

        returns += discount * reward
        lengths += active.astype(np.int64)
        active &= ~done
        state = result.state
        discount *= config.return_gamma

    return EpisodeBatch(
        episode_ids=ids.copy(),
        returns=returns,
        lengths=lengths,
        terminated=~active,
        truncated=active.copy(),
    )


def collect_episodes_naive(
    env: BatchedEnv,
    policy: Policy,
    episode_ids: Sequence[int],
    *,
    role: EpisodeRole,
    registry: EpisodeRegistry,
    scope: SeedScope,
    config: EpisodeConfig | None = None,
    accounting: AccountingHook | None = None,
) -> EpisodeBatch:
    """Literal one-episode-at-a-time reference for the batched collector."""

    ids = _episode_array(episode_ids)
    batches = [
        collect_episodes(
            env,
            policy,
            [int(episode_id)],
            role=role,
            registry=registry,
            scope=scope,
            config=config,
            accounting=accounting,
        )
        for episode_id in ids
    ]
    return EpisodeBatch(
        episode_ids=ids.copy(),
        returns=np.concatenate([batch.returns for batch in batches]),
        lengths=np.concatenate([batch.lengths for batch in batches]),
        terminated=np.concatenate([batch.terminated for batch in batches]),
        truncated=np.concatenate([batch.truncated for batch in batches]),
    )


def train_episodes(
    env: BatchedEnv,
    learner: Learner,
    episode_ids: Sequence[int],
    *,
    registry: EpisodeRegistry,
    scope: SeedScope,
    config: EpisodeConfig | None = None,
    accounting: AccountingHook | None = None,
) -> EpisodeBatch:
    """Train only on episodes explicitly owned by the train split."""

    config = config or EpisodeConfig(explore=True)
    if not config.explore:
        config = replace(config, explore=True)
    return collect_episodes(
        env,
        learner,
        episode_ids,
        role="train",
        registry=registry,
        scope=scope,
        config=config,
        learner=learner,
        accounting=accounting,
    )


@dataclass(frozen=True, slots=True)
class PairedEvaluation:
    episodes: Mapping[str, EpisodeBatch]
    accounting: Mapping[str, AccountingSnapshot]
    common_random_numbers: bool

    def paired_difference(self, left: str, right: str) -> FloatArray:
        a, b = self.episodes[left], self.episodes[right]
        if not np.array_equal(a.episode_ids, b.episode_ids):
            raise ValueError("paired arms do not contain the same episode IDs")
        return a.returns - b.returns


def evaluate_paired(
    env: BatchedEnv,
    policies: Mapping[str, Policy],
    episode_ids: Sequence[int],
    *,
    registry: EpisodeRegistry,
    scope: SeedScope,
    config: EpisodeConfig | None = None,
    common_random_numbers: bool = True,
) -> PairedEvaluation:
    """Evaluate named arms on whole eval episodes, optionally sharing noise."""

    config = config or EpisodeConfig()
    if not policies:
        raise ValueError("at least one policy is required")
    outcomes: dict[str, EpisodeBatch] = {}
    costs: dict[str, AccountingSnapshot] = {}
    shared_env = scope.child("paired-environment")
    for name, policy in policies.items():
        hook = AccountingHook()
        env_scope = shared_env if common_random_numbers else scope.child("environment", name)
        outcomes[name] = collect_episodes(
            env,
            policy,
            episode_ids,
            role="eval",
            registry=registry,
            scope=scope.child("arm", name),
            config=replace(config, explore=False),
            accounting=hook,
            environment_scope=env_scope,
            policy_scope=scope.child("policy", name),
        )
        costs[name] = hook.snapshot()
    return PairedEvaluation(outcomes, costs, common_random_numbers)
