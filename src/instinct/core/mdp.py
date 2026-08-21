"""Exact finite MDPs, and the closed-form solve for the real-time process.

The central observation, and the highest-leverage decision in the codebase:

    For a fixed reflex policy and a fixed planning delay, the real-time
    "plan while the world moves, then act on a stale decision" process is
    *itself* a finite Markov chain.

So its value is a linear solve, not a sampled average. That has three
consequences that shape everything downstream.

1. **The start-state axis becomes free.** One solve returns the value of every
   state simultaneously, where Monte Carlo would need a separate batch of
   rollouts per state.
2. **The decomposition becomes exact.** `G_plan`, `L_arrival`, `R_intermediate`
   and `L_irreversible` are differences of exactly-computed quantities, so the
   interaction residual is at machine epsilon and the "decomposition noise
   swamps the budget effect" kill condition is cleared by construction rather
   than by spending seeds on it.
3. **The sampled arms get ground truth.** Anything the gridworld and arcade
   estimators produce can be validated against a setting where the right answer
   is known, which is what makes their Monte Carlo trustworthy.

Two structural tricks keep this cheap:

*Affine maps compose.* A phase of the process maps values as ``V |-> g + D V``
with ``D`` a discounted kernel. Composing two phases gives
``(g1 + D1 g2, D1 D2)``, a monoid — so ``d`` steps of reflex behaviour cost
``O(log d)`` matrix products by binary exponentiation, not ``O(d)``, and the
reward accumulator comes along for free in the same recursion.

*One value-iteration run yields every budget.* A depth-``k`` lookahead planner is
the ``k``-th iterate of the Bellman operator, so the sweep over planning budgets
reads off snapshots of a single run. This is the exact-arm counterpart of the
prefix-sharing trick used for MCTS in :mod:`instinct.core.planner`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

__all__ = [
    "AffineMap",
    "ArrivalMode",
    "TabularMDP",
    "bellman_trace",
    "reflex_phase",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

# How the action that arrives at the end of the planning delay was chosen.
# These name the counterfactual arms of the P1 decomposition.
ArrivalMode = str
STALE: ArrivalMode = "stale"  # computed from s_0, applied at s_d  -> `actual`
FRESH: ArrivalMode = "fresh"  # computed from s_d, applied at s_d  -> `fresh`


@dataclass(frozen=True, slots=True)
class AffineMap:
    """A phase of the process, as the value map ``V |-> offset + kernel @ V``.

    ``kernel`` carries the discount, so it is ``gamma**steps`` times a stochastic
    matrix and its spectral radius is below one. Keeping discount inside the
    kernel is what makes composition a clean monoid with no bookkeeping of step
    counts.
    """

    offset: FloatArray  # (S,) discounted reward accumulated during the phase
    kernel: FloatArray  # (S, S) discounted state-to-state kernel
    steps: int

    def then(self, other: AffineMap) -> AffineMap:
        """Run ``self``, then ``other``. Associative, with :meth:`identity` as unit."""
        return AffineMap(
            offset=self.offset + self.kernel @ other.offset,
            kernel=self.kernel @ other.kernel,
            steps=self.steps + other.steps,
        )

    @staticmethod
    def identity(n_states: int) -> AffineMap:
        return AffineMap(
            offset=np.zeros(n_states),
            kernel=np.eye(n_states),
            steps=0,
        )

    def repeated(self, n: int) -> AffineMap:
        """``self`` applied ``n`` times, in ``O(log n)`` matrix products.

        Binary exponentiation over the affine monoid. The reward accumulator
        rides along in ``offset``, so the discounted sum of rewards over ``n``
        steps costs nothing beyond the kernel powers we already need.
        """
        if n < 0:
            raise ValueError(f"n={n} must be non-negative")
        result = AffineMap.identity(self.kernel.shape[0])
        base = self
        while n:
            if n & 1:
                result = result.then(base)
            n >>= 1
            if n:
                base = base.then(base)
        return result

    def solve_fixed_point(self) -> FloatArray:
        """Solve ``V = offset + kernel @ V`` for every state at once.

        This is the payoff: the whole start-state axis of a sweep in a single
        dense solve.
        """
        n = self.kernel.shape[0]
        return np.linalg.solve(np.eye(n) - self.kernel, self.offset).astype(np.float64)


@dataclass(frozen=True)
class TabularMDP:
    """A finite discounted MDP with exact solution methods.

    ``P`` is ``(S, A, S)`` and row-stochastic over the last axis; ``R`` is
    ``(S, A)`` expected immediate reward. Terminal states must be absorbing with
    zero reward — :meth:`validate` checks this, because a leaky terminal state
    silently corrupts every irreversibility measurement built on top.
    """

    P: FloatArray
    R: FloatArray
    gamma: float
    terminal: BoolArray
    name: str = "mdp"
    # Terminal does not imply failure (goals are terminal too). ``None`` means
    # that failure semantics were not supplied.
    failure: BoolArray | None = None
    # Cache for reflex phases, keyed by (policy fingerprint, delay).
    _phase_cache: dict[tuple[int, int], AffineMap] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def n_states(self) -> int:
        return int(self.P.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.P.shape[1])

    def validate(self) -> None:
        S, A = self.n_states, self.n_actions
        if self.P.shape != (S, A, S):
            raise ValueError(f"P has shape {self.P.shape}, expected {(S, A, S)}")
        if self.R.shape != (S, A):
            raise ValueError(f"R has shape {self.R.shape}, expected {(S, A)}")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError(f"gamma={self.gamma} must lie in [0, 1)")
        row_sums = self.P.sum(axis=2)
        if not np.allclose(row_sums, 1.0, atol=1e-9):
            worst = np.unravel_index(np.abs(row_sums - 1.0).argmax(), row_sums.shape)
            raise ValueError(f"P rows must sum to 1; worst at state/action {worst}")
        if (self.P < 0).any():
            raise ValueError("P has negative entries")
        if self.failure is not None:
            if self.failure.shape != (S,):
                raise ValueError(f"failure has shape {self.failure.shape}, expected {(S,)}")
            if np.any(self.failure & ~self.terminal):
                raise ValueError("failure states must be terminal")
        term = np.flatnonzero(self.terminal)
        if term.size:
            self_loop = self.P[term][:, :, term].diagonal(axis1=0, axis2=2)
            if not np.allclose(self_loop, 1.0):
                raise ValueError("terminal states must be absorbing under every action")
            if not np.allclose(self.R[term], 0.0):
                raise ValueError("terminal states must have zero reward")

    # -- policies ---------------------------------------------------------

    def as_stochastic(self, policy: IntArray | FloatArray) -> FloatArray:
        """Lift a deterministic ``(S,)`` action map to a ``(S, A)`` distribution."""
        arr = np.asarray(policy)
        if arr.ndim == 1:
            out = np.zeros((self.n_states, self.n_actions))
            out[np.arange(self.n_states), arr.astype(np.int64)] = 1.0
            return out
        if arr.shape != (self.n_states, self.n_actions):
            raise ValueError(
                f"policy has shape {arr.shape}, expected {(self.n_states, self.n_actions)}"
            )
        return arr.astype(np.float64)

    def policy_phase(self, policy: IntArray | FloatArray) -> AffineMap:
        """One step under ``policy``, as a discounted affine value map."""
        pi = self.as_stochastic(policy)
        return AffineMap(
            offset=np.einsum("sa,sa->s", pi, self.R),
            kernel=self.gamma * np.einsum("sa,sat->st", pi, self.P),
            steps=1,
        )

    def action_phase(self, action: int, repeat: int = 1) -> AffineMap:
        """``repeat`` consecutive steps of a single fixed action."""
        one = AffineMap(
            offset=self.R[:, action].copy(),
            kernel=self.gamma * self.P[:, action, :],
            steps=1,
        )
        return one.repeated(repeat)

    def policy_value(self, policy: IntArray | FloatArray) -> FloatArray:
        """Exact ``V^pi`` by linear solve."""
        return self.policy_phase(policy).solve_fixed_point()

    def q_from_v(self, V: FloatArray) -> FloatArray:
        """``Q(s,a) = R(s,a) + gamma * sum_s' P(s,a,s') V(s')``."""
        return self.R + self.gamma * (self.P @ V)

    def value_iteration(
        self, tol: float = 1e-12, max_iters: int = 100_000
    ) -> tuple[FloatArray, FloatArray, IntArray]:
        """Optimal ``(V*, Q*, pi*)`` to numerical tolerance."""
        V = np.zeros(self.n_states)
        for _ in range(max_iters):
            Q = self.q_from_v(V)
            V_next = Q.max(axis=1)
            if np.max(np.abs(V_next - V)) < tol:
                V = V_next
                break
            V = V_next
        Q = self.q_from_v(V)
        return V, Q, Q.argmax(axis=1).astype(np.int64)

    # -- the real-time process --------------------------------------------

    def reflex_phase(self, reflex: IntArray | FloatArray, delay: int) -> AffineMap:
        """``delay`` steps of the reflex, in ``O(log delay)`` matrix products.

        Cached: a budget sweep induces many distinct delays over a handful of
        reflexes, and recomputing the powers for each would dominate the solve.
        """
        pi = self.as_stochastic(reflex)
        key = (hash(pi.tobytes()), delay)
        cached = self._phase_cache.get(key)
        if cached is None:
            cached = self.policy_phase(pi).repeated(delay)
            self._phase_cache[key] = cached
        return cached

    def realtime_value(
        self,
        *,
        reflex: IntArray | FloatArray,
        planner_action: IntArray,
        delay: int,
        commit: int = 1,
        arrival: ArrivalMode = STALE,
    ) -> FloatArray:
        """Exact value of one real-time decision epoch, repeated forever.

        The epoch is: at ``s_0`` launch the planner; for ``delay`` ticks the
        reflex acts while the world moves; the planner's action then lands at
        ``s_delay`` and is committed for ``commit`` ticks; a new epoch begins.

        ``arrival`` selects the counterfactual arm:

        - ``STALE`` — the arriving action is ``planner_action[s_0]``, decided
          from the state the planner *saw*. This is the real process, and the
          reason the chain does not factorize trivially: the action applied at
          ``s_delay`` depends on ``s_0``.
        - ``FRESH`` — the arriving action is ``planner_action[s_delay]``, i.e.
          the same planner quality with the staleness removed. The gap between
          the two arms *is* arrival regret, in return units.

        Returns ``(S,)``: the value of every start state, from one solve.
        """
        phase = self.reflex_phase(reflex, delay)
        actions = np.asarray(planner_action, dtype=np.int64)

        if arrival == FRESH:
            # The arriving action is keyed by the state it is applied in, so the
            # commit window is an ordinary affine phase and composes directly.
            return phase.then(self.held_action_phase(actions, commit)).solve_fixed_point()

        if arrival != STALE:
            raise ValueError(f"unknown arrival mode {arrival!r}")

        # Stale arrival is the case that does not factorize: the committed action
        # is keyed by s_0 while the state it acts on is distributed as row s_0 of
        # the reflex kernel. Materializing that as an (S, S, S) tensor would be
        # cubic in memory, so instead group start states by the action they will
        # commit — one matrix product per distinct action.
        offset = np.zeros(self.n_states)
        kernel = np.zeros((self.n_states, self.n_states))
        for a in np.unique(actions):
            rows = np.flatnonzero(actions == a)
            commit_phase = self.action_phase(int(a), commit)
            offset[rows] = phase.offset[rows] + phase.kernel[rows] @ commit_phase.offset
            kernel[rows] = phase.kernel[rows] @ commit_phase.kernel

        return AffineMap(offset=offset, kernel=kernel, steps=delay + commit).solve_fixed_point()

    def held_action_phase(self, actions: IntArray, commit: int) -> AffineMap:
        """Commit ``actions[s]`` for ``commit`` ticks, holding it fixed throughout.

        The distinction from :meth:`policy_phase` repeated ``commit`` times is
        not cosmetic and is easy to get wrong: a repeated policy phase re-selects
        an action at every tick from whatever state it has reached, whereas a
        committed decision is chosen once and held. Those coincide only at
        ``commit == 1``. Getting this wrong makes the ``fresh`` arm silently
        stronger than it should be and inflates measured arrival regret.
        """
        offset = np.zeros(self.n_states)
        kernel = np.zeros((self.n_states, self.n_states))
        for a in np.unique(np.asarray(actions, dtype=np.int64)):
            rows = np.flatnonzero(actions == a)
            phase = self.action_phase(int(a), commit)
            offset[rows] = phase.offset[rows]
            kernel[rows] = phase.kernel[rows]
        return AffineMap(offset=offset, kernel=kernel, steps=commit)

    def reflex_only_value(self, reflex: IntArray | FloatArray) -> FloatArray:
        """Value of never planning at all — the floor the atlas measures against."""
        return self.policy_value(reflex)


def bellman_trace(mdp: TabularMDP, depths: list[int]) -> dict[int, IntArray]:
    """Greedy action maps for depth-limited lookahead, at every requested depth.

    A depth-``k`` planner is the ``k``-th Bellman iterate started from zero, and
    iterate ``k`` is a prefix of iterate ``k'`` for ``k' > k``. So one run to
    ``max(depths)`` yields the whole budget axis: ``O(k_max)`` rather than
    ``O(sum_k k)``. This mirrors ``planner.recommend_trace`` in the sampled arm,
    and it is exact — these are the same action maps independent searches would
    return, not approximations of them.
    """
    if not depths:
        return {}
    if min(depths) < 0:
        raise ValueError("depths must be non-negative")

    wanted = set(depths)
    out: dict[int, IntArray] = {}
    V = np.zeros(mdp.n_states)
    if 0 in wanted:
        # Depth 0 sees no future: greedy on immediate reward alone.
        out[0] = mdp.R.argmax(axis=1).astype(np.int64)
    for k in range(1, max(depths) + 1):
        Q = mdp.q_from_v(V)
        V = Q.max(axis=1)
        if k in wanted:
            out[k] = Q.argmax(axis=1).astype(np.int64)
    return out


def reflex_phase(mdp: TabularMDP, reflex: IntArray | FloatArray, delay: int) -> AffineMap:
    """Module-level alias for :meth:`TabularMDP.reflex_phase`."""
    return mdp.reflex_phase(reflex, delay)
