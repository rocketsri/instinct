"""Matched one-handoff instrumentation for P1.1 sampled environments.

The recurring rollout remains the operational policy-value measurement.  This
module answers the narrower causal question needed by the named decomposition:
from one shared root, what changes when a single plan is instant, stale at its
observed arrival, or refreshed at that exact same arrival state and time?

Observed latency is supplied per episode.  Environment events occur on a unit
grid with a randomized phase, so delay is ``floor(phase + rate * latency)``;
there is no assumed linear per-simulation latency and no deterministic phase
lock.  Hardware cost is an additional utility price for the observed runtime,
not a second subtraction of the delay already expressed by world advancement.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from instinct.core.env import BatchedEnv, EnvState
from instinct.core.rng import SeedScope
from instinct.core.rollout import PlannerFn, ReflexFn

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
FailureFn = Callable[[EnvState], BoolArray]


@dataclass(frozen=True, slots=True)
class HandoffTrace:
    total_return: FloatArray
    intermediate_return: FloatArray
    delay_events: IntArray
    latency_s: FloatArray
    hardware_cost: FloatArray
    failed: BoolArray
    pending_at_horizon: BoolArray


@dataclass(frozen=True, slots=True)
class SampledDecomposition:
    sigma: FloatArray
    G_plan: FloatArray
    R_intermediate: FloatArray
    L_arrival: FloatArray
    L_wait: FloatArray
    C_hw: FloatArray
    L_base_delay: FloatArray
    epsilon_id: FloatArray
    actual: HandoffTrace
    fresh: HandoffTrace
    instant: HandoffTrace
    base: HandoffTrace
    instant_base: HandoffTrace


def event_delays(
    latency_s: FloatArray,
    *,
    event_rate_hz: float,
    phase: FloatArray,
) -> IntArray:
    """Map observed latency to crossed event boundaries.

    ``phase`` is the fraction of the current event interval already elapsed at
    planner launch.  Randomizing it avoids deterministic delay staircases and
    boundary resonance while retaining exact, preregistered event ordering:
    a completion exactly on a boundary lands after that event.
    """
    latency = np.asarray(latency_s, dtype=np.float64)
    phase_array = np.asarray(phase, dtype=np.float64)
    if latency.ndim != 1 or phase_array.shape != latency.shape:
        raise ValueError("latency and phase must be aligned one-dimensional arrays")
    if np.any(latency < 0) or not np.isfinite(latency).all():
        raise ValueError("latency must be finite and nonnegative")
    if event_rate_hz < 0 or not np.isfinite(event_rate_hz):
        raise ValueError("event_rate_hz must be finite and nonnegative")
    if np.any((phase_array < 0) | (phase_array >= 1)):
        raise ValueError("phase must lie in [0, 1)")
    return np.floor(phase_array + event_rate_hz * latency).astype(np.int64)


def _zeros_failure(state: EnvState) -> BoolArray:
    return np.zeros(state.n_lanes, dtype=bool)


def simulate_handoff(
    env: BatchedEnv,
    state0: EnvState,
    *,
    budget: int,
    latency_s: FloatArray,
    event_rate_hz: float,
    phase: FloatArray,
    arm: str,
    horizon: int,
    gamma: float,
    planner: PlannerFn,
    reflex: ReflexFn,
    scope: SeedScope,
    failure: FailureFn = _zeros_failure,
    hardware_price_per_s: float = 0.0,
    episode: int = 0,
) -> HandoffTrace:
    """Evaluate one launched plan and then return control to the reflex.

    ``fresh`` is evaluation-only: it waits through the same prefix as
    ``actual`` and substitutes a recommendation computed from that shared
    arrival state without adding another delay or hardware charge.
    """
    if arm not in {"actual", "fresh", "instant"}:
        raise ValueError("arm must be actual, fresh, or instant")
    if budget < 1 or horizon < 1 or not 0 < gamma <= 1:
        raise ValueError("budget/horizon must be positive and gamma must lie in (0, 1]")
    latency = np.asarray(latency_s, dtype=np.float64)
    if latency.shape != (state0.n_lanes,):
        raise ValueError("one observed latency is required per lane")
    delays = event_delays(latency, event_rate_hz=event_rate_hz, phase=phase)
    if arm == "instant":
        delays = np.zeros_like(delays)

    # Computation launches for every live episode at the root and is paid even
    # when its result does not arrive inside the finite evaluation horizon.
    root_action = planner(state0, budget, 0)
    landed_action = root_action.copy()
    state = state0.copy()
    alive = np.ones(state0.n_lanes, dtype=bool)
    failed = np.asarray(failure(state), dtype=bool).copy()
    total = np.zeros(state0.n_lanes)
    intermediate = np.zeros(state0.n_lanes)
    applied = np.zeros(state0.n_lanes, dtype=bool)
    discount = 1.0

    for tick in range(horizon):
        arriving = alive & ~applied & (delays == tick)
        if arm == "fresh" and arriving.any():
            indices = np.flatnonzero(arriving)
            landed_action[indices] = planner(state.take(indices), budget, 0)

        reflex_action = reflex(state, tick)
        actions = np.where(arriving, landed_action, reflex_action)
        waiting = alive & ~applied & (tick < delays)
        result = env.step(state, actions, scope=scope, episode=episode, tick=tick)
        credited = discount * result.reward * alive
        total += credited
        intermediate += credited * waiting
        state = result.state
        failed |= np.asarray(failure(state), dtype=bool)
        alive &= ~result.done
        applied |= arriving
        discount *= gamma

    cost = hardware_price_per_s * latency
    return HandoffTrace(
        total_return=total,
        intermediate_return=intermediate,
        delay_events=delays,
        latency_s=latency,
        hardware_cost=cost,
        failed=failed,
        pending_at_horizon=~applied & alive,
    )


def decompose_handoff(
    env: BatchedEnv,
    state0: EnvState,
    *,
    budget: int,
    base_budget: int,
    latency_s: FloatArray,
    base_latency_s: FloatArray,
    event_rate_hz: float,
    phase: FloatArray,
    horizon: int,
    gamma: float,
    planner: PlannerFn,
    reflex: ReflexFn,
    scope: SeedScope,
    failure: FailureFn = _zeros_failure,
    hardware_price_per_s: float = 0.0,
    episode: int = 0,
) -> SampledDecomposition:
    """Construct the frozen identity episode by episode from matched arms."""

    def trace(k: int, latency: FloatArray, arm: str) -> HandoffTrace:
        return simulate_handoff(
            env,
            state0,
            budget=k,
            latency_s=latency,
            event_rate_hz=event_rate_hz,
            phase=phase,
            arm=arm,
            horizon=horizon,
            gamma=gamma,
            planner=planner,
            reflex=reflex,
            scope=scope,
            failure=failure,
            hardware_price_per_s=hardware_price_per_s,
            episode=episode,
        )

    actual = trace(budget, latency_s, "actual")
    fresh = trace(budget, latency_s, "fresh")
    instant = trace(budget, latency_s, "instant")
    base = trace(base_budget, base_latency_s, "actual")
    instant_base = trace(base_budget, base_latency_s, "instant")

    G_plan = instant.total_return - instant_base.total_return
    R_intermediate = actual.intermediate_return - base.intermediate_return
    L_arrival = fresh.total_return - actual.total_return
    L_wait = R_intermediate - (fresh.total_return - instant.total_return)
    C_hw = actual.hardware_cost - base.hardware_cost
    L_base_delay = instant_base.total_return - base.total_return
    sigma = actual.total_return - actual.hardware_cost - base.total_return + base.hardware_cost
    explained = G_plan + R_intermediate - L_arrival - L_wait - C_hw + L_base_delay
    epsilon_id = sigma - explained
    return SampledDecomposition(
        sigma=sigma,
        G_plan=G_plan,
        R_intermediate=R_intermediate,
        L_arrival=L_arrival,
        L_wait=L_wait,
        C_hw=C_hw,
        L_base_delay=L_base_delay,
        epsilon_id=epsilon_id,
        actual=actual,
        fresh=fresh,
        instant=instant,
        base=base,
        instant_base=instant_base,
    )


def matched_failure_delta(
    env: BatchedEnv,
    state0: EnvState,
    *,
    latency_s: FloatArray,
    base_latency_s: FloatArray,
    event_rate_hz: float,
    phase: FloatArray,
    reflex: ReflexFn,
    safe: ReflexFn,
    failure: FailureFn,
    scope: SeedScope,
    episode: int = 0,
) -> FloatArray:
    """Paired failure difference at the common longer arrival time.

    The counterfactual follows the same reflex through the base arrival and a
    frozen safe policy thereafter.  It is outside the return-unit identity.
    """
    delay = event_delays(latency_s, event_rate_hz=event_rate_hz, phase=phase)
    base_delay = event_delays(base_latency_s, event_rate_hz=event_rate_hz, phase=phase)
    target = np.maximum(delay, base_delay)

    def rollout(*, recover: bool) -> BoolArray:
        state = state0.copy()
        alive = np.ones(state.n_lanes, dtype=bool)
        failed = np.asarray(failure(state), dtype=bool).copy()
        captured = target == 0
        outcome = np.where(captured, failed, False)
        for tick in range(int(target.max(initial=0))):
            acting = alive & ~captured
            reflex_actions = reflex(state, tick)
            if recover:
                safe_actions = safe(state, tick)
                actions = np.where(tick >= base_delay, safe_actions, reflex_actions)
            else:
                actions = reflex_actions
            result = env.step(state, actions, scope=scope, episode=episode, tick=tick)
            state = result.state
            failed |= np.asarray(failure(state), dtype=bool) & acting
            alive &= ~result.done
            now = ~captured & (target == tick + 1)
            outcome = np.where(now, failed, outcome)
            captured |= now
        return outcome

    return rollout(recover=False).astype(np.float64) - rollout(recover=True).astype(np.float64)
