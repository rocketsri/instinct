"""Tabular environments: the same dynamics, reachable two ways.

Every environment here is a :class:`~instinct.core.mdp.TabularMDP` *and* a
batched :class:`~instinct.core.env.BatchedEnv` over the same transition tensor.
That duality is deliberate and is what makes the tabular arm the backbone of P1:
the exact solver and the sampled rollout engine can be pointed at literally the
same dynamics, so any disagreement is a bug in one of them rather than a
difference in setup.

:func:`corridor_with_pit` is the one built for a specific measurement. Most of
its states are recoverable, but a few are absorbing failures, so
``L_irreversible`` is genuinely non-zero and separable from ``L_arrival``. In an
environment where every mistake is recoverable, those two terms are confounded
and the decomposition cannot be validated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from instinct.core.env import EnvState, StepResult
from instinct.core.mdp import TabularMDP
from instinct.core.rng import SeedScope

__all__ = ["TabularEnv", "chase_chain", "corridor_with_pit"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class TabularEnv:
    """A batched environment driven by an exact MDP's transition tensor.

    Purity matters here: the successor state is a function of ``(state, action,
    u)`` where ``u`` comes from the counter-based stream keyed by
    ``(episode, tick, lane)``. Two arms that reach the same state at the same
    tick therefore see the same ``u``, which is the pairing the decomposition
    relies on.
    """

    mdp: TabularMDP
    name: str = "tabular"
    start_state: int = 0

    @property
    def n_actions(self) -> int:
        return self.mdp.n_actions

    def _cdf(self) -> FloatArray:
        return np.cumsum(self.mdp.P, axis=2)

    def reset(self, lane_ids: IntArray, *, scope: SeedScope, episode: int = 0) -> EnvState:
        return EnvState(
            lane_ids=np.asarray(lane_ids, dtype=np.int64),
            fields={"s": np.full(len(lane_ids), self.start_state, dtype=np.int64)},
        )

    def step(
        self,
        state: EnvState,
        actions: IntArray,
        *,
        scope: SeedScope,
        episode: int = 0,
        tick: int = 0,
    ) -> StepResult:
        s = state["s"].astype(np.int64)
        a = np.asarray(actions, dtype=np.int64)
        u = scope.stream("transition").uniform(
            state.lane_ids, count=1, episode=episode, tick=tick
        )[:, 0]

        cdf = self._cdf()[s, a]  # (L, S)
        nxt = (u[:, None] >= cdf).sum(axis=1).astype(np.int64)
        nxt = np.minimum(nxt, self.mdp.n_states - 1)

        reward = self.mdp.R[s, a].astype(np.float64)
        # Terminal states are absorbing with zero reward, so a lane that has
        # already failed contributes nothing further.
        reward = np.where(self.mdp.terminal[s], 0.0, reward)
        return StepResult(
            state=state.replace_fields(s=nxt),
            reward=reward,
            done=self.mdp.terminal[nxt],
        )


def _absorb(P: FloatArray, R: FloatArray, terminal: npt.NDArray[np.bool_]) -> None:
    """Make terminal states absorbing and rewardless, in place."""
    for t in np.flatnonzero(terminal):
        P[t] = 0.0
        P[t, :, t] = 1.0
        R[t] = 0.0


def corridor_with_pit(
    length: int = 8,
    gamma: float = 0.95,
    slip: float = 0.1,
    pit_penalty: float = -1.0,
    goal_reward: float = 1.0,
) -> TabularMDP:
    """A corridor with an absorbing pit, so irreversibility is measurable.

    States ``0..length-1`` are corridor positions, ``length`` is the goal and
    ``length + 1`` is the pit. Actions are advance, wait, retreat. With
    probability ``slip`` an advance overshoots into the pit, which absorbs.

    The pit is what makes ``L_irreversible`` identifiable. Waiting is safe but
    earns nothing, advancing is productive but risks an unrecoverable state, and
    a delayed decision can arrive when the safe action is no longer the one that
    was planned. That is the regime the decomposition needs in order to separate
    "the decision arrived stale" from "something unrecoverable happened while no
    slow intervention was available".
    """
    n_states = length + 2
    goal, pit = length, length + 1
    n_actions = 3  # 0 advance, 1 wait, 2 retreat
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))

    for s in range(length):
        forward = min(s + 1, goal)
        P[s, 0, forward] += 1.0 - slip
        P[s, 0, pit] += slip
        R[s, 0] = -0.01  # moving costs a little
        P[s, 1, s] = 1.0
        R[s, 1] = -0.02  # waiting costs slightly more, so freezing is not free
        P[s, 2, max(s - 1, 0)] = 1.0
        R[s, 2] = -0.03
    R[length - 1, 0] = goal_reward
    R[:, :][pit] = pit_penalty

    terminal = np.zeros(n_states, dtype=bool)
    terminal[[goal, pit]] = True
    _absorb(P, R, terminal)

    mdp = TabularMDP(P=P, R=R, gamma=gamma, terminal=terminal, name="corridor_with_pit")
    mdp.validate()
    return mdp


def chase_chain(n_positions: int = 6, gamma: float = 0.95, drift: float = 0.3) -> TabularMDP:
    """A target that moves while you decide, with no irreversible states.

    The deliberate contrast to :func:`corridor_with_pit`: here every state is
    recoverable, so ``L_irreversible`` should measure as zero and any arrival
    cost is pure staleness. Having one environment of each kind is how the
    decomposition's terms get told apart rather than merely computed.

    State encodes ``(agent, target)`` positions on a ring; the target drifts with
    probability ``drift`` each tick, so a decision computed a few ticks ago may
    chase where the target no longer is.
    """
    n_states = n_positions * n_positions
    n_actions = 3  # 0 left, 1 stay, 2 right

    def idx(agent: int, target: int) -> int:
        return agent * n_positions + target

    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))
    for agent in range(n_positions):
        for target in range(n_positions):
            s = idx(agent, target)
            for a, step in enumerate((-1, 0, 1)):
                nxt_agent = (agent + step) % n_positions
                # Target drifts either way with probability drift/2, else holds.
                for t_step, prob in ((-1, drift / 2), (0, 1 - drift), (1, drift / 2)):
                    nxt_target = (target + t_step) % n_positions
                    P[s, a, idx(nxt_agent, nxt_target)] += prob
                R[s, a] = 1.0 if nxt_agent == target else -0.05

    mdp = TabularMDP(
        P=P, R=R, gamma=gamma, terminal=np.zeros(n_states, dtype=bool), name="chase_chain"
    )
    mdp.validate()
    return mdp
