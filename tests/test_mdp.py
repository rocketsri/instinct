"""Exact MDP machinery, checked against closed forms and against Monte Carlo.

The headline test is :func:`test_realtime_value_matches_monte_carlo`. The claim
that the real-time process is a finite Markov chain is what lets P1's tabular
arm skip sampling entirely, so that claim gets checked against a simulator that
implements the process description literally.
"""

from __future__ import annotations

import numpy as np
import pytest

from instinct.core.mdp import FRESH, STALE, AffineMap, TabularMDP, bellman_trace

from .conftest import random_mdp, simulate_realtime

# -- affine monoid --------------------------------------------------------


def test_affine_composition_is_associative() -> None:
    rng = np.random.default_rng(0)
    maps = [
        AffineMap(offset=rng.normal(size=4), kernel=0.3 * rng.random((4, 4)), steps=1)
        for _ in range(3)
    ]
    left = maps[0].then(maps[1]).then(maps[2])
    right = maps[0].then(maps[1].then(maps[2]))
    assert np.allclose(left.offset, right.offset)
    assert np.allclose(left.kernel, right.kernel)


def test_identity_is_the_unit() -> None:
    rng = np.random.default_rng(1)
    m = AffineMap(offset=rng.normal(size=5), kernel=0.2 * rng.random((5, 5)), steps=1)
    ident = AffineMap.identity(5)
    for composed in (ident.then(m), m.then(ident)):
        assert np.allclose(composed.offset, m.offset)
        assert np.allclose(composed.kernel, m.kernel)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 16, 31, 64])
def test_binary_exponentiation_matches_repeated_composition(n: int) -> None:
    """O(log n) powers must equal the O(n) chain they replace."""
    mdp = random_mdp(n_states=6, n_actions=2, seed=3)
    step = mdp.policy_phase(np.zeros(mdp.n_states, dtype=np.int64))

    naive = AffineMap.identity(mdp.n_states)
    for _ in range(n):
        naive = naive.then(step)

    fast = step.repeated(n)
    assert fast.steps == n
    assert np.allclose(fast.offset, naive.offset, atol=1e-12)
    assert np.allclose(fast.kernel, naive.kernel, atol=1e-12)


# -- exact solutions ------------------------------------------------------


def test_policy_value_satisfies_bellman_equation() -> None:
    mdp = random_mdp(n_states=10, n_actions=3, gamma=0.9, seed=5)
    pi = np.array([s % mdp.n_actions for s in range(mdp.n_states)], dtype=np.int64)
    V = mdp.policy_value(pi)
    r_pi = mdp.R[np.arange(mdp.n_states), pi]
    P_pi = mdp.P[np.arange(mdp.n_states), pi]
    assert np.allclose(V, r_pi + mdp.gamma * P_pi @ V, atol=1e-10)


def test_value_iteration_matches_policy_value_of_its_own_policy() -> None:
    mdp = random_mdp(n_states=12, n_actions=4, gamma=0.95, seed=11)
    V, Q, pi = mdp.value_iteration()
    assert np.allclose(V, mdp.policy_value(pi), atol=1e-8)
    # Optimality: no single-step deviation improves on V*.
    assert np.all(Q.max(axis=1) <= V + 1e-9)


def test_two_state_chain_matches_hand_computed_value() -> None:
    """A closed form worked out by hand, so the solver is not just self-consistent."""
    gamma = 0.5
    # State 0 -> state 1 with reward 1; state 1 absorbs with reward 0.
    P = np.array([[[0.0, 1.0]], [[0.0, 1.0]]])
    R = np.array([[1.0], [0.0]])
    mdp = TabularMDP(P=P, R=R, gamma=gamma, terminal=np.array([False, True]))
    mdp.validate()
    # V(0) = 1 + gamma * V(1) = 1; V(1) = 0.
    assert np.allclose(mdp.policy_value(np.zeros(2, dtype=np.int64)), [1.0, 0.0])


def test_validate_rejects_leaky_terminal_states() -> None:
    """A terminal state that leaks would silently corrupt irreversibility measures."""
    mdp = random_mdp(n_states=5, n_actions=2, seed=2, n_terminal=1)
    t = int(np.flatnonzero(mdp.terminal)[0])
    leaky = mdp.P.copy()
    leaky[t, 0] = np.full(5, 1.0 / 5)
    with pytest.raises(ValueError, match="absorbing"):
        TabularMDP(P=leaky, R=mdp.R, gamma=mdp.gamma, terminal=mdp.terminal).validate()


# -- one value-iteration run yields every budget --------------------------


def test_bellman_trace_matches_independent_runs_per_depth() -> None:
    """The prefix-sharing shortcut must be exact, not approximate."""
    mdp = random_mdp(n_states=14, n_actions=4, gamma=0.9, seed=13)
    depths = [1, 2, 3, 5, 8, 13]
    shared = bellman_trace(mdp, depths)

    for k in depths:
        V = np.zeros(mdp.n_states)
        Q = mdp.q_from_v(V)
        for _ in range(k):
            Q = mdp.q_from_v(V)
            V = Q.max(axis=1)
        assert np.array_equal(shared[k], Q.argmax(axis=1)), f"depth {k} disagrees"


def test_bellman_trace_converges_to_optimal_policy() -> None:
    mdp = random_mdp(n_states=10, n_actions=3, gamma=0.8, seed=17)
    _, _, pi_star = mdp.value_iteration()
    deep = bellman_trace(mdp, [400])[400]
    assert np.array_equal(deep, pi_star)


# -- the real-time process ------------------------------------------------


def test_zero_delay_stale_and_fresh_coincide() -> None:
    """With no delay there is no staleness, so arrival regret must vanish exactly.

    This is the ``nu_e = 0`` known-answer check from the plan, in its purest
    form: if these two arms ever differ at zero delay, the estimator is broken.
    """
    mdp = random_mdp(n_states=9, n_actions=3, gamma=0.9, seed=19)
    reflex = np.zeros(mdp.n_states, dtype=np.int64)
    _, _, planner = mdp.value_iteration()
    kw = dict(reflex=reflex, planner_action=planner, delay=0, commit=2)
    assert np.allclose(
        mdp.realtime_value(**kw, arrival=STALE),
        mdp.realtime_value(**kw, arrival=FRESH),
        atol=1e-12,
    )


def test_arrival_regret_is_non_negative_for_the_optimal_planner() -> None:
    """Acting on a stale optimal decision cannot beat acting on a fresh one."""
    mdp = random_mdp(n_states=16, n_actions=4, gamma=0.9, seed=23)
    reflex = np.zeros(mdp.n_states, dtype=np.int64)
    _, _, planner = mdp.value_iteration()
    for delay in (1, 2, 5):
        kw = dict(reflex=reflex, planner_action=planner, delay=delay, commit=1)
        fresh = mdp.realtime_value(**kw, arrival=FRESH)
        stale = mdp.realtime_value(**kw, arrival=STALE)
        assert np.all(fresh - stale > -1e-9), f"negative arrival regret at delay {delay}"


def test_fresh_arm_equals_a_plain_policy_when_reflex_is_the_planner() -> None:
    """Sanity anchor: if reflex and planner agree, delay cannot matter."""
    mdp = random_mdp(n_states=8, n_actions=3, gamma=0.85, seed=29)
    _, _, planner = mdp.value_iteration()
    baseline = mdp.policy_value(planner)
    for delay in (0, 1, 4):
        got = mdp.realtime_value(
            reflex=planner, planner_action=planner, delay=delay, commit=1, arrival=FRESH
        )
        assert np.allclose(got, baseline, atol=1e-10)


@pytest.mark.parametrize("arrival", [STALE, FRESH])
@pytest.mark.parametrize(("delay", "commit"), [(0, 1), (1, 1), (2, 3), (3, 1)])
def test_realtime_value_matches_monte_carlo(arrival: str, delay: int, commit: int) -> None:
    """The load-bearing test: the closed form must agree with a literal simulator.

    The whole tabular arm rests on the real-time process being a finite Markov
    chain. Here that is checked against a nested-loop simulator written straight
    from the process description, at several delays and commit windows and in
    both arrival modes. Agreement is asserted inside the Monte Carlo confidence
    interval, since the reference is itself an estimate.
    """
    mdp = random_mdp(n_states=6, n_actions=3, gamma=0.8, seed=31)
    reflex = np.array([s % mdp.n_actions for s in range(mdp.n_states)], dtype=np.int64)
    _, _, planner = mdp.value_iteration()

    exact = mdp.realtime_value(
        reflex=reflex, planner_action=planner, delay=delay, commit=commit, arrival=arrival
    )

    for start in (0, mdp.n_states // 2):
        mean, sem = simulate_realtime(
            mdp,
            reflex=reflex,
            planner_action=planner,
            delay=delay,
            commit=commit,
            arrival=arrival,
            start_state=start,
            n_episodes=20_000,
            # gamma**80 < 1e-7, so truncation is far below the sampling error.
            horizon=80,
            seed=1234 + start,
        )
        assert abs(exact[start] - mean) < 4.0 * sem + 1e-3, (
            f"{arrival} d={delay} c={commit} s={start}: exact {exact[start]:.5f} "
            f"vs MC {mean:.5f} +- {sem:.5f}"
        )


def test_reflex_phase_cache_returns_equal_results() -> None:
    """Caching must not change any number it returns."""
    mdp = random_mdp(n_states=7, n_actions=2, seed=37)
    reflex = np.ones(mdp.n_states, dtype=np.int64)
    first = mdp.reflex_phase(reflex, 5)
    second = mdp.reflex_phase(reflex, 5)
    assert np.array_equal(first.kernel, second.kernel)
    assert np.array_equal(first.offset, second.offset)
    fresh_mdp = random_mdp(n_states=7, n_actions=2, seed=37)
    uncached = fresh_mdp.policy_phase(reflex).repeated(5)
    assert np.allclose(first.kernel, uncached.kernel, atol=1e-12)
