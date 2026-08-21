from __future__ import annotations

import numpy as np

from instinct.core.env import EnvState
from instinct.core.envs.tabular import TabularEnv, chase_chain, corridor_with_pit
from instinct.core.rng import SeedScope
from instinct.p1_atlas.sampled import (
    decompose_handoff,
    event_delays,
    matched_failure_delta,
    simulate_handoff,
)


def test_observed_latency_uses_random_event_phase() -> None:
    latency = np.full(4, 0.6)
    phase = np.array([0.0, 0.39, 0.4, 0.99])
    assert np.array_equal(
        event_delays(latency, event_rate_hz=1.0, phase=phase),
        np.array([0, 0, 1, 1]),
    )


def test_sampled_handoff_identity_closes_episodewise() -> None:
    mdp = chase_chain(n_positions=5, gamma=0.9, drift=0.45)
    env = TabularEnv(mdp=mdp, start_state=3)
    scope = SeedScope(321).child("sampled_identity")
    lanes = np.arange(96, dtype=np.int64)
    state0 = env.reset(lanes, scope=scope)
    _, _, optimal = mdp.value_iteration()

    def planner(state: EnvState, budget: int, tick: int) -> np.ndarray:
        s = state["s"].astype(np.int64)
        return optimal[s] if budget >= 4 else mdp.R[s].argmax(axis=1)

    def hold(state: EnvState, tick: int) -> np.ndarray:
        return np.ones(state.n_lanes, dtype=np.int64)

    phase = np.linspace(0.0, 0.99, state0.n_lanes)
    result = decompose_handoff(
        env,
        state0,
        budget=8,
        base_budget=1,
        latency_s=np.linspace(0.8, 1.3, state0.n_lanes),
        base_latency_s=np.linspace(0.1, 0.25, state0.n_lanes),
        event_rate_hz=2.0,
        phase=phase,
        horizon=18,
        gamma=mdp.gamma,
        planner=planner,
        reflex=hold,
        scope=scope,
        hardware_price_per_s=0.03,
    )
    assert np.max(np.abs(result.epsilon_id)) < 1e-12
    assert np.array_equal(result.actual.intermediate_return, result.fresh.intermediate_return)
    assert np.unique(result.actual.delay_events).size > 1


def test_pending_handoff_still_pays_observed_hardware_cost() -> None:
    mdp = chase_chain(n_positions=4, gamma=0.9, drift=0.2)
    env = TabularEnv(mdp=mdp, start_state=0)
    scope = SeedScope(12).child("pending")
    state0 = env.reset(np.arange(20), scope=scope)

    def stay(state: EnvState, *args: int) -> np.ndarray:
        return np.ones(state.n_lanes, dtype=np.int64)

    trace = simulate_handoff(
        env,
        state0,
        budget=16,
        latency_s=np.full(state0.n_lanes, 10.0),
        event_rate_hz=1.0,
        phase=np.zeros(state0.n_lanes),
        arm="actual",
        horizon=3,
        gamma=mdp.gamma,
        planner=stay,
        reflex=stay,
        scope=scope,
        hardware_price_per_s=0.2,
    )
    assert np.all(trace.pending_at_horizon)
    assert np.all(trace.hardware_cost == 2.0)


def test_irreversibility_is_matched_at_the_longer_arrival() -> None:
    mdp = corridor_with_pit(length=7, gamma=0.95, slip=0.35)
    env = TabularEnv(mdp=mdp, start_state=0)
    scope = SeedScope(811).child("matched_failure")
    state0 = env.reset(np.arange(1000), scope=scope)

    def advance(state: EnvState, tick: int) -> np.ndarray:
        return np.zeros(state.n_lanes, dtype=np.int64)

    def safe(state: EnvState, tick: int) -> np.ndarray:
        return np.ones(state.n_lanes, dtype=np.int64)

    def failure(state: EnvState) -> np.ndarray:
        return mdp.failure[state["s"].astype(np.int64)]

    delta = matched_failure_delta(
        env,
        state0,
        latency_s=np.full(state0.n_lanes, 4.0),
        base_latency_s=np.full(state0.n_lanes, 1.0),
        event_rate_hz=1.0,
        phase=np.zeros(state0.n_lanes),
        reflex=advance,
        safe=safe,
        failure=failure,
        scope=scope,
    )
    assert np.mean(delta) > 0.1
    assert set(np.unique(delta)) <= {0.0, 1.0}
