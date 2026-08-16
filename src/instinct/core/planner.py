"""Planners that expose their whole budget curve from a single search.

A search with budget ``k`` is a strict prefix of a search with budget ``k' > k``:
UCT is incremental, so simulation ``i`` does exactly the same thing whether the
total budget is 16 or 512. P1 sweeps a grid of budgets at every state, which
naively costs ``O(sum_k k)`` simulations. Reading the recommendation off a single
search at ``k_max`` instead costs ``O(k_max)`` — roughly 10x on a 12-point grid
to 512 — and it is **exact**, not an approximation: the actions returned are the
same ones independent searches would have produced.

That exactness has one requirement, and it is the reason this module takes a
:class:`~instinct.core.rng.Stream` rather than a plain generator. Simulation
``i`` must consume randomness addressed by ``i``, not randomness pulled
sequentially from a shared generator. With a sequential generator, running 512
simulations would draw a different sequence than running 16, and the prefix
property would quietly fail. Counter-based addressing makes it hold by
construction, and ``test_planner.py`` checks it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from instinct.core.mdp import TabularMDP, bellman_trace
from instinct.core.rng import Stream

__all__ = [
    "MCTS",
    "BudgetTrace",
    "ExactLookaheadPlanner",
    "PlanningModel",
    "TabularModel",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# budget -> recommended action, for one root state
BudgetTrace = dict[int, int]


@runtime_checkable
class PlanningModel(Protocol):
    """The generative model a planner searches over.

    Deliberately separate from :class:`~instinct.core.env.BatchedEnv`: keeping
    them apart lets the sweep hand the planner a *deliberately imperfect* model
    later, which is how model error becomes a variable rather than a confound.
    """

    n_actions: int

    def step(self, state: int, action: int, u: float) -> tuple[int, float, bool]:
        """Advance one step. ``u`` is a uniform draw, supplied by the caller.

        Taking the random number as an argument rather than drawing it keeps the
        model pure, which is what makes a search replayable.
        """
        ...

    def reward(self, state: int, action: int) -> float: ...


@dataclass(frozen=True, slots=True)
class TabularModel:
    """A :class:`PlanningModel` backed by an exact :class:`TabularMDP`."""

    mdp: TabularMDP
    _cdf: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_cdf", np.cumsum(self.mdp.P, axis=2))

    @property
    def n_actions(self) -> int:
        return self.mdp.n_actions

    @property
    def gamma(self) -> float:
        return self.mdp.gamma

    def step(self, state: int, action: int, u: float) -> tuple[int, float, bool]:
        nxt = int(np.searchsorted(self._cdf[state, action], u, side="right"))
        nxt = min(nxt, self.mdp.n_states - 1)  # guard the u == 1.0 edge
        return nxt, float(self.mdp.R[state, action]), bool(self.mdp.terminal[nxt])

    def reward(self, state: int, action: int) -> float:
        return float(self.mdp.R[state, action])


@dataclass
class _Node:
    """Statistics for one state.

    Nodes are keyed by state rather than by path, so this is a transposition
    table rather than a strict tree: paths that reach the same state share
    statistics, and a state recurring within a single descent is credited once
    per visit. That is the right choice for the small recurrent MDPs P1 uses,
    where a strict tree would relearn the same state many times over.
    """

    visits: IntArray
    values: FloatArray  # running mean return per action
    total: int = 0

    @staticmethod
    def make(n_actions: int) -> _Node:
        return _Node(visits=np.zeros(n_actions, dtype=np.int64), values=np.zeros(n_actions))


@dataclass
class MCTS:
    """UCT search that reports its recommendation at every budget on the grid.

    ``rollout_depth`` truncates the random playout; ``c_uct`` is the exploration
    constant. Both are configuration, not tuning targets — P1 measures how the
    *value* of computation varies with the world's speed, so the planner is held
    fixed across a sweep and its quality is a controlled variable.
    """

    model: PlanningModel
    gamma: float = 0.95
    c_uct: float = 1.41421356
    rollout_depth: int = 20

    def recommend_trace(
        self,
        root: int,
        budgets: Sequence[int],
        *,
        stream: Stream,
        lane: int,
        episode: int = 0,
        tick: int = 0,
    ) -> BudgetTrace:
        """Search once to ``max(budgets)``, snapshotting the choice at each budget.

        Returns ``{budget: action}``. Equivalent to running an independent search
        per budget, at a fraction of the cost.
        """
        if not budgets:
            return {}
        wanted = sorted(set(int(b) for b in budgets))
        if wanted[0] < 1:
            raise ValueError("budgets must be >= 1")
        k_max = wanted[-1]
        want = set(wanted)

        tree: dict[int, _Node] = {root: _Node.make(self.model.n_actions)}
        trace: BudgetTrace = {}

        # One flat block of uniforms per simulation, addressed by simulation
        # index. This is what makes simulation i independent of the total budget.
        per_sim = 2 * (self.rollout_depth + 8)
        draws = stream.uniform(
            np.array([lane]), count=per_sim * k_max, episode=episode, tick=tick
        )[0].reshape(k_max, per_sim)

        for sim in range(k_max):
            self._simulate(root, tree, draws[sim])
            if (sim + 1) in want:
                trace[sim + 1] = self._recommend(tree, root)

        return trace

    def _recommend(self, tree: dict[int, _Node], root: int) -> int:
        node = tree.get(root)
        if node is None or node.total == 0:
            return 0
        # Robust child: most-visited, which is the standard choice and less
        # jumpy than argmax-value at small budgets.
        return int(np.argmax(node.visits))

    def _select(self, node: _Node, u: float) -> int:
        """UCT, visiting each action once before trading off value against novelty."""
        unvisited = np.flatnonzero(node.visits == 0)
        if unvisited.size:
            return int(unvisited[min(int(u * unvisited.size), unvisited.size - 1)])
        exploration = self.c_uct * np.sqrt(np.log(max(node.total, 1)) / node.visits)
        return int(np.argmax(node.values + exploration))

    def _simulate(self, root: int, tree: dict[int, _Node], draws: FloatArray) -> float:
        """One UCT iteration: select, expand, roll out, back up."""
        path: list[tuple[int, int]] = []  # (state acted in, action taken)
        rewards: list[float] = []
        state = root
        cursor = 0
        done = False

        # -- selection: descend while we are still inside the tree --
        while not done and state in tree and cursor + 2 <= draws.size:
            action = self._select(tree[state], float(draws[cursor]))
            cursor += 1
            path.append((state, action))
            state, reward, done = self.model.step(state, action, float(draws[cursor]))
            cursor += 1
            rewards.append(reward)
            if len(path) >= self.rollout_depth:
                break

        # -- expansion: one new node per simulation --
        if not done and state not in tree:
            tree[state] = _Node.make(self.model.n_actions)

        # -- rollout: uniform-random playout from the leaf --
        leaf_return = 0.0
        discount = 1.0
        depth = 0
        while not done and depth < self.rollout_depth and cursor + 2 <= draws.size:
            action = min(int(draws[cursor] * self.model.n_actions), self.model.n_actions - 1)
            cursor += 1
            state, reward, done = self.model.step(state, action, float(draws[cursor]))
            cursor += 1
            leaf_return += discount * reward
            discount *= self.gamma
            depth += 1

        # -- backup: discounted return from each visited node, walked backwards --
        g = leaf_return
        for i in reversed(range(len(path))):
            g = rewards[i] + self.gamma * g
            st, act = path[i]
            node = tree[st]
            node.visits[act] += 1
            node.total += 1
            # Incremental mean, so a node's value is the average return through it.
            node.values[act] += (g - node.values[act]) / node.visits[act]
        return g


@dataclass(frozen=True, slots=True)
class ExactLookaheadPlanner:
    """Depth-limited exact lookahead, for the tabular arm.

    Removes planner approximation error from the picture entirely, so that a
    measured change in the compute-freshness surface is attributable to the
    environment and the delay rather than to search noise. Its budget trace comes
    from :func:`~instinct.core.mdp.bellman_trace` — one value-iteration run for
    the whole budget axis, the exact-arm counterpart of MCTS prefix sharing.
    """

    mdp: TabularMDP

    def action_maps(self, budgets: Sequence[int]) -> dict[int, IntArray]:
        """``{budget: action map over all states}``, from a single VI run."""
        return bellman_trace(self.mdp, [int(b) for b in budgets])

    def recommend_trace(self, root: int, budgets: Sequence[int]) -> BudgetTrace:
        return {k: int(m[root]) for k, m in self.action_maps(budgets).items()}
