"""Planner correctness, and the exactness of budget-trace sharing.

:func:`test_budget_trace_matches_independent_searches` is the one that licenses
the ``O(sum_k k)`` to ``O(k_max)`` optimization. If it fails, every budget sweep
in P1 is reading actions no independent search would have produced.
"""

from __future__ import annotations

import numpy as np
import pytest

from instinct.core.planner import MCTS, ExactLookaheadPlanner, TabularModel, _Node
from instinct.core.rng import SeedScope

from .conftest import random_mdp


@pytest.fixture
def setup():
    mdp = random_mdp(n_states=10, n_actions=3, gamma=0.9, seed=41)
    model = TabularModel(mdp)
    mcts = MCTS(model=model, gamma=mdp.gamma, rollout_depth=8)
    stream = SeedScope(99).child("planner").stream("mcts")
    return mdp, mcts, stream


# -- the prefix-sharing guarantee -----------------------------------------


def test_budget_trace_matches_independent_searches(setup) -> None:
    """One search to k_max must agree with a separate search at every budget.

    This is exactness, not similarity: UCT is incremental, and because each
    simulation draws randomness addressed by its own index, simulation i is
    identical whether the total budget is 4 or 64.
    """
    _, mcts, stream = setup
    budgets = [1, 2, 3, 5, 8, 16, 32, 64]

    for root in (0, 3, 7):
        shared = mcts.recommend_trace(root, budgets, stream=stream, lane=root)
        for k in budgets:
            alone = mcts.recommend_trace(root, [k], stream=stream, lane=root)
            assert shared[k] == alone[k], (
                f"root {root} budget {k}: shared trace says {shared[k]}, "
                f"independent search says {alone[k]}"
            )


def test_budget_trace_is_deterministic(setup) -> None:
    _, mcts, stream = setup
    budgets = [1, 4, 16]
    first = mcts.recommend_trace(2, budgets, stream=stream, lane=0)
    second = mcts.recommend_trace(2, budgets, stream=stream, lane=0)
    assert first == second


def test_different_lanes_search_differently(setup) -> None:
    """Lanes must be independent, or paired arms would share search noise."""
    _, mcts, stream = setup
    traces = [
        tuple(mcts.recommend_trace(2, [1, 2, 3, 4], stream=stream, lane=i).values())
        for i in range(12)
    ]
    assert len({t for t in traces}) > 1


def test_trace_covers_exactly_the_requested_budgets(setup) -> None:
    _, mcts, stream = setup
    trace = mcts.recommend_trace(1, [3, 9, 3], stream=stream, lane=0)
    assert sorted(trace) == [3, 9]


def test_actions_are_always_legal(setup) -> None:
    mdp, mcts, stream = setup
    trace = mcts.recommend_trace(0, [1, 5, 20], stream=stream, lane=0)
    assert all(0 <= a < mdp.n_actions for a in trace.values())


def test_zero_budget_is_rejected(setup) -> None:
    _, mcts, stream = setup
    with pytest.raises(ValueError, match="budgets"):
        mcts.recommend_trace(0, [0], stream=stream, lane=0)


# -- search quality -------------------------------------------------------


def test_more_search_approaches_the_optimal_action() -> None:
    """A planner whose budget does not buy anything would make P1 vacuous.

    Deliberately a weak check on a deliberately easy MDP: the point is to detect
    a broken search, not to claim UCT is near-optimal at these budgets.
    """
    mdp = random_mdp(n_states=6, n_actions=3, gamma=0.9, seed=5)
    _, _, pi_star = mdp.value_iteration()
    mcts = MCTS(model=TabularModel(mdp), gamma=mdp.gamma, rollout_depth=15)
    stream = SeedScope(3).child("q").stream("mcts")

    def agreement(budget: int) -> float:
        hits = [
            mcts.recommend_trace(s, [budget], stream=stream, lane=s * 37 + trial)[budget]
            == pi_star[s]
            for s in range(mdp.n_states)
            for trial in range(12)
        ]
        return float(np.mean(hits))

    assert agreement(200) >= agreement(1)


def test_backup_credits_the_whole_path_not_just_the_root() -> None:
    """Every node on the selected path must be updated, and counts stay consistent.

    Note this search keys nodes by state, so a state reachable more than once in
    a single descent is credited once per visit — a transposition table rather
    than a strict tree. That is intended (it shares statistics across paths that
    reach the same state), which is why the root's total is ``>= n_sims`` rather
    than ``== n_sims``.
    """
    n_sims = 30
    mdp = random_mdp(n_states=6, n_actions=3, gamma=0.9, seed=8)
    mcts = MCTS(model=TabularModel(mdp), gamma=mdp.gamma, rollout_depth=6)
    stream = SeedScope(4).child("v").stream("mcts")
    tree: dict[int, _Node] = {0: _Node.make(3)}
    draws = stream.uniform(np.array([0]), count=2 * (6 + 8) * n_sims)[0].reshape(n_sims, -1)
    for sim in range(n_sims):
        mcts._simulate(0, tree, draws[sim])

    assert tree[0].total >= n_sims, "root is on every path, so it is credited every simulation"
    assert len(tree) > 1, "search should have expanded beyond the root"
    for state, node in tree.items():
        assert node.total == int(node.visits.sum()), f"state {state} has inconsistent counts"
    assert sum(n.total for n in tree.values()) > tree[0].total, "deeper nodes must be credited too"


# -- exact lookahead ------------------------------------------------------


def test_exact_lookahead_shares_one_value_iteration_run() -> None:
    mdp = random_mdp(n_states=12, n_actions=3, gamma=0.9, seed=61)
    planner = ExactLookaheadPlanner(mdp)
    maps = planner.action_maps([1, 4, 9])
    assert set(maps) == {1, 4, 9}
    for k, m in maps.items():
        assert m.shape == (mdp.n_states,)
        assert planner.recommend_trace(5, [k])[k] == int(m[5])


def test_deep_exact_lookahead_is_optimal() -> None:
    mdp = random_mdp(n_states=8, n_actions=3, gamma=0.8, seed=67)
    _, _, pi_star = mdp.value_iteration()
    deep = ExactLookaheadPlanner(mdp).action_maps([300])[300]
    assert np.array_equal(deep, pi_star)
