"""Shared fixtures and reference implementations.

Naive simulators live here on purpose. They are the ``--impl=naive`` references
that rule 7 requires: slow, obviously correct, and never used outside tests.
When an optimized path and one of these disagree, the optimized path is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from instinct.core.mdp import FRESH, STALE, ArrivalMode, TabularMDP


def random_mdp(
    n_states: int = 12,
    n_actions: int = 3,
    gamma: float = 0.9,
    seed: int = 0,
    n_terminal: int = 0,
    sparsity: int = 4,
) -> TabularMDP:
    """A random but well-formed MDP, with optional absorbing terminal states."""
    rng = np.random.default_rng(seed)
    P = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            succ = rng.choice(n_states, size=min(sparsity, n_states), replace=False)
            w = rng.dirichlet(np.ones(succ.size))
            P[s, a, succ] = w
    R = rng.normal(size=(n_states, n_actions))

    terminal = np.zeros(n_states, dtype=bool)
    if n_terminal:
        term_idx = rng.choice(n_states, size=n_terminal, replace=False)
        terminal[term_idx] = True
        P[term_idx] = 0.0
        for t in term_idx:
            P[t, :, t] = 1.0
        R[term_idx] = 0.0

    mdp = TabularMDP(P=P, R=R, gamma=gamma, terminal=terminal, name=f"random-{seed}")
    mdp.validate()
    return mdp


def simulate_realtime(
    mdp: TabularMDP,
    *,
    reflex: np.ndarray,
    planner_action: np.ndarray,
    delay: int,
    commit: int,
    arrival: ArrivalMode,
    start_state: int,
    n_episodes: int,
    horizon: int,
    seed: int,
) -> tuple[float, float]:
    """Monte Carlo reference for :meth:`TabularMDP.realtime_value`.

    The per-tick logic is a literal transcription of the process description:
    launch the planner, let the reflex act for ``delay`` ticks, land the
    planner's action, commit it, start a new epoch. Only the "for each episode"
    wrapper is vectorized — episodes step in lockstep across a leading axis —
    because a pure Python loop over 20k episodes takes minutes and this runs in
    CI. The tick-level structure, which is the part under test, is untouched.

    Returns ``(mean, standard error)``.
    """
    rng = np.random.default_rng(seed)
    reflex_pi = mdp.as_stochastic(reflex)
    reflex_cdf = np.cumsum(reflex_pi, axis=1)
    trans_cdf = np.cumsum(mdp.P, axis=2)

    def sample(cdf_rows: np.ndarray) -> np.ndarray:
        """Inverse-CDF draw, one per row."""
        u = rng.random(cdf_rows.shape[0])[:, None]
        return (u >= cdf_rows).sum(axis=1)

    s = np.full(n_episodes, start_state, dtype=np.int64)
    total = np.zeros(n_episodes)
    discount = 1.0
    t = 0
    while t < horizon:
        epoch_start = s.copy()  # the state the planner sees
        for _ in range(delay):
            if t >= horizon:
                break
            a = sample(reflex_cdf[s])
            total += discount * mdp.R[s, a]
            s = sample(trans_cdf[s, a])
            discount *= mdp.gamma
            t += 1
        # The arriving action: decided from the stale snapshot, or refreshed.
        landed = planner_action[epoch_start] if arrival == STALE else planner_action[s]
        for _ in range(commit):
            if t >= horizon:
                break
            total += discount * mdp.R[s, landed]
            s = sample(trans_cdf[s, landed])
            discount *= mdp.gamma
            t += 1

    return float(total.mean()), float(total.std(ddof=1) / np.sqrt(n_episodes))


@pytest.fixture
def small_mdp() -> TabularMDP:
    return random_mdp(n_states=8, n_actions=3, gamma=0.85, seed=7)


__all__ = ["FRESH", "STALE", "random_mdp", "simulate_realtime", "small_mdp"]
