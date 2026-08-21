"""Exact implementation of P3's frozen one-interval objective."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import numpy.typing as npt

from instinct.core.mdp import TabularMDP

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def option_objective(
    mdp: TabularMDP,
    *,
    reflex: IntArray,
    planner_action: IntArray,
    planner_q: FloatArray,
    delay: int,
) -> FloatArray:
    """Return J(mu;k) for every start state, exactly as frozen in section 6.2."""
    if planner_q.shape != (mdp.n_states, mdp.n_actions):
        raise ValueError("planner_q shape does not match MDP")
    actions = np.asarray(planner_action, dtype=np.int64)
    if actions.shape != (mdp.n_states,):
        raise ValueError("planner_action must contain one stale action per start state")
    phase = mdp.reflex_phase(reflex, delay)
    continuation = np.empty(mdp.n_states)
    for s in range(mdp.n_states):
        continuation[s] = phase.kernel[s] @ planner_q[:, actions[s]]
    return phase.offset + continuation


def policy_fingerprint(policy: IntArray) -> str:
    import hashlib

    return hashlib.sha256(np.asarray(policy, dtype=np.int64).tobytes()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class ReflexMapRow:
    policy: tuple[int, ...]
    value: float


def enumerate_reflexes(
    mdp: TabularMDP,
    *,
    planner_action: IntArray,
    planner_q: FloatArray,
    delay: int,
    start_state: int = 0,
) -> tuple[ReflexMapRow, ...]:
    rows = []
    for actions in product(range(mdp.n_actions), repeat=mdp.n_states):
        policy = np.asarray(actions, dtype=np.int64)
        value = option_objective(
            mdp,
            reflex=policy,
            planner_action=planner_action,
            planner_q=planner_q,
            delay=delay,
        )[start_state]
        rows.append(ReflexMapRow(tuple(actions), float(value)))
    return tuple(rows)


def crossplay_matrix(
    mdp: TabularMDP,
    *,
    execution_reflexes: tuple[IntArray, ...],
    planned_actions: tuple[IntArray, ...],
    planner_q: FloatArray,
    delay: int,
    start_state: int = 0,
) -> FloatArray:
    """Planner-model columns crossed with deployed-reflex rows."""
    if not execution_reflexes or len(execution_reflexes) != len(planned_actions):
        raise ValueError("cross-play needs equally many nonempty reflex/action variants")
    matrix = np.empty((len(execution_reflexes), len(planned_actions)))
    for i, reflex in enumerate(execution_reflexes):
        for j, actions in enumerate(planned_actions):
            matrix[i, j] = option_objective(
                mdp,
                reflex=reflex,
                planner_action=actions,
                planner_q=planner_q,
                delay=delay,
            )[start_state]
    return matrix


def constructed_case(kind: str, gamma: float = 0.8) -> tuple[TabularMDP, IntArray, FloatArray, int]:
    """Known-answer MDPs whose optimal first reflex action is strict."""
    if kind not in {"hold", "greedy", "interior", "destructive"}:
        raise ValueError(f"unknown constructed case {kind!r}")
    S, A = 4, 3
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    for a in range(A):
        P[0, a, a + 1] = 1.0
    for s in range(1, S):
        P[s, :, s] = 1.0
    q_arrival = {
        "hold": ([0.0, 0.2, 1.0], [2.0, 1.0, -5.0], 0),
        "greedy": ([0.0, 2.0, 0.2], [0.0, 0.0, 0.0], 1),
        "interior": ([0.0, 1.0, 0.25], [0.0, 0.0, 2.0], 2),
        "destructive": ([0.0, 0.1, 3.0], [1.0, 0.8, -10.0], 0),
    }
    rewards, q_values, expected = q_arrival[kind]
    R[0] = rewards
    mdp = TabularMDP(P=P, R=R, gamma=gamma, terminal=np.zeros(S, dtype=bool), name=f"p3-{kind}")
    mdp.validate()
    planner_q = np.zeros((S, A))
    planner_q[:, 0] = np.asarray([0.0, *q_values])
    planner_action = np.zeros(S, dtype=np.int64)
    return mdp, planner_action, planner_q, expected
