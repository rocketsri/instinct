"""Exact and common-random-number labels for P3.1 reflex actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from instinct.core.mdp import AffineMap, TabularMDP
from instinct.core.rng import SeedScope

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


class ReflexSchedule(Protocol):
    """A reflex distribution indexed by ticks remaining before plan arrival."""

    def probabilities(self, remaining: int) -> FloatArray: ...


def scheduled_phase(mdp: TabularMDP, reflex: ReflexSchedule, delay: int) -> AffineMap:
    """Compose the time-conditioned reflex for exactly ``delay`` ticks."""
    if delay < 0:
        raise ValueError("delay must be nonnegative")
    phase = AffineMap.identity(mdp.n_states)
    for remaining in range(delay, 0, -1):
        phase = phase.then(mdp.policy_phase(reflex.probabilities(remaining)))
    return phase


def planner_actions_for_reflex(
    mdp: TabularMDP,
    *,
    planner_q: FloatArray,
    simulated_reflex: ReflexSchedule,
    delay: int,
) -> IntArray:
    """Plan against the same reflex-induced arrival distribution that will execute."""
    if planner_q.shape != (mdp.n_states, mdp.n_actions):
        raise ValueError("planner_q shape does not match MDP")
    kernel = scheduled_phase(mdp, simulated_reflex, delay).kernel
    scores = np.column_stack([kernel @ planner_q[:, a] for a in range(mdp.n_actions)])
    return np.argmax(scores, axis=1).astype(np.int64)


def option_values(
    mdp: TabularMDP,
    *,
    execution_reflex: ReflexSchedule,
    planner_model_reflex: ReflexSchedule,
    planner_q: FloatArray,
    delay: int,
) -> tuple[FloatArray, IntArray, AffineMap]:
    """Exact one-interval value, allowing explicit planner/reflex mismatch."""
    actions = planner_actions_for_reflex(
        mdp,
        planner_q=planner_q,
        simulated_reflex=planner_model_reflex,
        delay=delay,
    )
    phase = scheduled_phase(mdp, execution_reflex, delay)
    continuation = np.array(
        [phase.kernel[s] @ planner_q[:, actions[s]] for s in range(mdp.n_states)]
    )
    return phase.offset + continuation, actions, phase


def exact_first_action_labels(
    mdp: TabularMDP,
    *,
    start_state: int,
    pending_action: int,
    planner_q: FloatArray,
    suffix_reflex: ReflexSchedule,
    remaining: int,
) -> FloatArray:
    """Expected option return for every possible first reflex action."""
    if not 0 <= start_state < mdp.n_states:
        raise ValueError("start_state outside MDP")
    if not 0 <= pending_action < mdp.n_actions:
        raise ValueError("pending_action outside MDP")
    if remaining < 1:
        raise ValueError("remaining must be positive")
    tail = scheduled_phase(mdp, suffix_reflex, remaining - 1)
    continuation = tail.offset + tail.kernel @ planner_q[:, pending_action]
    return mdp.R[start_state] + mdp.gamma * (mdp.P[start_state] @ continuation)


@dataclass(frozen=True, slots=True)
class SampledLabels:
    returns: FloatArray
    mean_returns: FloatArray
    arrival_states: IntArray
    terminated: BoolArray
    truncated: BoolArray


def sampled_first_action_labels(
    mdp: TabularMDP,
    *,
    start_state: int,
    pending_action: int,
    planner_q: FloatArray,
    suffix_reflex: ReflexSchedule,
    remaining: int,
    n_rollouts: int,
    scope: SeedScope,
) -> SampledLabels:
    """Monte Carlo branch labels sharing transition draws across action arms.

    ``truncated`` means the fixed reflex-prefix horizon expired without a task
    terminal. It is deliberately distinct from ``terminated``.
    """
    if n_rollouts < 1:
        raise ValueError("n_rollouts must be positive")
    if remaining < 1:
        raise ValueError("remaining must be positive")
    lanes = np.arange(n_rollouts, dtype=np.int64)
    returns = np.zeros((mdp.n_actions, n_rollouts), dtype=np.float64)
    arrivals = np.zeros((mdp.n_actions, n_rollouts), dtype=np.int64)
    terminated = np.zeros((mdp.n_actions, n_rollouts), dtype=bool)
    cdf = np.cumsum(mdp.P, axis=2)
    stream = scope.stream("p3-counterfactual-transition")

    for branch_action in range(mdp.n_actions):
        states = np.full(n_rollouts, start_state, dtype=np.int64)
        active = ~mdp.terminal[states]
        discount = 1.0
        for tick in range(remaining):
            if tick == 0:
                actions = np.full(n_rollouts, branch_action, dtype=np.int64)
            else:
                left = remaining - tick
                actions = np.argmax(suffix_reflex.probabilities(left)[states], axis=1)
            returns[branch_action] += discount * np.where(
                active, mdp.R[states, actions], 0.0
            )
            uniforms = stream.uniform(lanes, 1, episode=0, tick=tick)[:, 0]
            rows = cdf[states, actions]
            nxt = (uniforms[:, None] >= rows).sum(axis=1).astype(np.int64)
            nxt = np.minimum(nxt, mdp.n_states - 1)
            states = np.where(active, nxt, states)
            active &= ~mdp.terminal[states]
            discount *= mdp.gamma
        returns[branch_action] += discount * np.where(
            active, planner_q[states, pending_action], 0.0
        )
        arrivals[branch_action] = states
        terminated[branch_action] = ~active

    return SampledLabels(
        returns=returns,
        mean_returns=returns.mean(axis=1),
        arrival_states=arrivals,
        terminated=terminated,
        truncated=~terminated,
    )
