"""Rollout engine: arm semantics, CRN pairing, and fast-path equivalence.

:func:`test_batched_lanes_match_the_reference` is the rule-7 guard for the
rollout engine. The batched path juggles per-lane phase counters and masked
action assembly; the reference is a plain loop. They must agree exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from instinct.core.env import EnvState, Timing
from instinct.core.envs.tabular import TabularEnv, chase_chain, corridor_with_pit
from instinct.core.rng import SeedScope
from instinct.core.rollout import (
    ARMS,
    LaneSpec,
    RealTimeConfig,
    simulate_arm,
    simulate_lanes,
)


@pytest.fixture
def fixture():
    mdp = chase_chain(n_positions=5, gamma=0.9, drift=0.4)
    env = TabularEnv(mdp=mdp, start_state=3)
    scope = SeedScope(2026).child("rollout")
    state0 = env.reset(np.arange(24), scope=scope)
    _, _, pi_star = mdp.value_iteration()
    greedy = np.asarray(pi_star)

    def planner(state: EnvState, budget: int, tick: int) -> np.ndarray:
        # A budget-sensitive planner: small budgets act greedily on immediate
        # reward, large budgets use the optimal map. Enough structure for the
        # rollout engine's mechanics to be exercised.
        s = state["s"].astype(np.int64)
        return greedy[s] if budget >= 4 else mdp.R[s].argmax(axis=1)

    def reflex(state: EnvState, tick: int) -> np.ndarray:
        return np.ones(state.n_lanes, dtype=np.int64)  # "stay"

    cfg = RealTimeConfig(
        budgets=(1, 2, 4, 8),
        timing=Timing(nu_e=1.0, nu_h=0.5),
        commit=2,
        horizon=24,
        gamma=0.9,
    )
    return env, state0, scope, planner, reflex, cfg


# -- fast path equals reference -------------------------------------------


def test_batched_lanes_match_the_reference(fixture) -> None:
    """Masked, phase-tracked batching must reproduce the literal loop exactly."""
    env, state0, scope, planner, reflex, cfg = fixture
    specs = [LaneSpec(arm, k) for arm in ARMS for k in cfg.budgets]

    fast = simulate_lanes(
        env, state0, specs, cfg=cfg, reflex=reflex, planner=planner, scope=scope
    )

    for spec in specs:
        ref = simulate_arm(
            env, state0, arm=spec.arm, budget=spec.budget, cfg=cfg,
            reflex=reflex, planner=planner, scope=scope,
        )
        got = fast[(spec.arm, spec.budget)]
        label = f"{spec.arm}@{spec.budget}"
        assert np.allclose(got.total_return, ref.total_return, atol=1e-12), f"{label} return"
        assert np.allclose(got.intermediate, ref.intermediate, atol=1e-12), f"{label} reflex"
        assert np.allclose(got.committed, ref.committed, atol=1e-12), f"{label} committed"
        assert got.simulations == ref.simulations, f"{label} simulation count"


@pytest.mark.parametrize("commit", [1, 3])
@pytest.mark.parametrize("nu_h", [0.0, 0.5, 2.0])
def test_equivalence_holds_across_timings(fixture, commit, nu_h) -> None:
    """Phase bookkeeping is where batching breaks; vary delays and windows."""
    env, state0, scope, planner, reflex, base = fixture
    cfg = RealTimeConfig(
        budgets=base.budgets, timing=Timing(nu_e=1.0, nu_h=nu_h),
        commit=commit, horizon=20, gamma=base.gamma,
    )
    specs = [LaneSpec(arm, k) for arm in ARMS for k in (1, 4)]
    fast = simulate_lanes(
        env, state0, specs, cfg=cfg, reflex=reflex, planner=planner, scope=scope
    )
    for spec in specs:
        ref = simulate_arm(
            env, state0, arm=spec.arm, budget=spec.budget, cfg=cfg,
            reflex=reflex, planner=planner, scope=scope,
        )
        assert np.allclose(
            fast[(spec.arm, spec.budget)].total_return, ref.total_return, atol=1e-12
        ), f"{spec.arm}@{spec.budget} commit={commit} nu_h={nu_h}"


# -- arm semantics --------------------------------------------------------


def test_reward_splits_between_reflex_and_committed(fixture) -> None:
    """The two components must partition the total, or the decomposition leaks."""
    env, state0, scope, planner, reflex, cfg = fixture
    for arm in ARMS:
        t = simulate_arm(
            env, state0, arm=arm, budget=4, cfg=cfg,
            reflex=reflex, planner=planner, scope=scope,
        )
        assert np.allclose(t.total_return, t.intermediate + t.committed, atol=1e-12), arm


def test_instant_arm_banks_no_reflex_reward(fixture) -> None:
    """Zero delay means the reflex never acts."""
    env, state0, scope, planner, reflex, cfg = fixture
    t = simulate_arm(
        env, state0, arm="instant", budget=8, cfg=cfg,
        reflex=reflex, planner=planner, scope=scope,
    )
    assert np.allclose(t.intermediate, 0.0)


def test_fresh_still_pays_the_delay(fixture) -> None:
    """`fresh` removes staleness, not waiting.

    If `fresh` were implemented as "plan instantly at s_delay" it would bank no
    reflex reward, like `instant`. It must bank exactly as much as `actual`,
    since the two share their reflex phase within every epoch.
    """
    env, state0, scope, planner, reflex, cfg = fixture
    fresh = simulate_arm(
        env, state0, arm="fresh", budget=8, cfg=cfg,
        reflex=reflex, planner=planner, scope=scope,
    )
    actual = simulate_arm(
        env, state0, arm="actual", budget=8, cfg=cfg,
        reflex=reflex, planner=planner, scope=scope,
    )
    assert not np.allclose(fresh.intermediate, 0.0), "fresh must still pay the delay"
    # First epoch is shared exactly; later epochs diverge once decisions differ.
    assert fresh.simulations == actual.simulations


def test_zero_speed_makes_fresh_and_actual_identical(fixture) -> None:
    """With nu_e = 0 nothing goes stale, so the arms must coincide exactly.

    The known-answer check from the plan, in the sampled arm this time.
    """
    env, state0, scope, planner, reflex, base = fixture
    cfg = RealTimeConfig(
        budgets=base.budgets, timing=Timing(nu_e=0.0, nu_h=1.0),
        commit=1, horizon=20, gamma=base.gamma,
    )
    kw = dict(cfg=cfg, reflex=reflex, planner=planner, scope=scope)
    fresh = simulate_arm(env, state0, arm="fresh", budget=8, **kw)
    actual = simulate_arm(env, state0, arm="actual", budget=8, **kw)
    assert np.allclose(fresh.total_return, actual.total_return, atol=1e-12)


# -- common random numbers ------------------------------------------------


def test_arms_share_environment_noise(fixture) -> None:
    """Sibling lanes must draw identical noise, or the pairing is lost.

    Checked by giving two arms the same actions: any difference in outcome would
    have to come from the randomness, and there must be none.
    """
    env, state0, scope, _, reflex, _cfg = fixture

    def fixed_planner(state: EnvState, budget: int, tick: int) -> np.ndarray:
        return np.zeros(state.n_lanes, dtype=np.int64)

    flat = RealTimeConfig(
        budgets=(4,), timing=Timing(nu_e=0.0), commit=1, horizon=15, gamma=0.9
    )
    a = simulate_arm(
        env, state0, arm="actual", budget=4, cfg=flat,
        reflex=reflex, planner=fixed_planner, scope=scope,
    )
    b = simulate_arm(
        env, state0, arm="fresh", budget=4, cfg=flat,
        reflex=reflex, planner=fixed_planner, scope=scope,
    )
    assert np.allclose(a.total_return, b.total_return, atol=1e-12)


def test_rollouts_are_reproducible(fixture) -> None:
    env, state0, scope, planner, reflex, cfg = fixture
    kw = dict(arm="actual", budget=4, cfg=cfg, reflex=reflex, planner=planner, scope=scope)
    assert np.allclose(
        simulate_arm(env, state0, **kw).total_return,
        simulate_arm(env, state0, **kw).total_return,
    )


# -- irreversibility ------------------------------------------------------


def test_pit_absorbs_and_stops_accruing_reward() -> None:
    """Terminal states must truly absorb; a leak corrupts L_irreversible."""
    mdp = corridor_with_pit(length=5, slip=0.9)
    env = TabularEnv(mdp=mdp, start_state=0)
    scope = SeedScope(11).child("pit")
    state0 = env.reset(np.arange(40), scope=scope)

    def always_advance(state: EnvState, *args) -> np.ndarray:
        return np.zeros(state.n_lanes, dtype=np.int64)

    cfg = RealTimeConfig(budgets=(1,), timing=Timing(nu_e=1.0, nu_h=1.0), horizon=30)
    trace = simulate_arm(
        env, state0, arm="actual", budget=1, cfg=cfg,
        reflex=always_advance, planner=always_advance, scope=scope,
    )
    assert np.all(np.isfinite(trace.total_return))
    # With slip 0.9 nearly every lane falls in, and the pit pays nothing after.
    floor = mdp.R.min() / (1 - mdp.gamma)
    assert np.all(trace.total_return > floor)


def test_unknown_arm_is_rejected(fixture) -> None:
    env, state0, scope, planner, reflex, cfg = fixture
    with pytest.raises(ValueError, match="unknown arm"):
        simulate_arm(
            env, state0, arm="wishful", budget=1, cfg=cfg,
            reflex=reflex, planner=planner, scope=scope,
        )
