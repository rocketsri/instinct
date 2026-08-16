"""Splitting the value of computation into causes.

The headline quantity of P1 is

    sigma(k) = [J(k) - C_hw(k)] - [J(base) - C_hw(base)]

the net gain from planning with budget ``k`` instead of the cheapest budget,
once the cost of the extra computation is charged. That number alone says
whether more thinking paid off; it says nothing about *why*. The decomposition
attributes it to four separately measured causes:

``G_plan``
    the better decision, with staleness removed. Measured on the ``instant``
    arm, where the planner's answer is applied to the state it was computed
    from.
``R_intermediate``
    reward the reflex actually banked while the planner was thinking. A longer
    wait is not pure loss — the fast policy is still acting.
``L_arrival``
    the decision's loss of relevance by the time it lands. Measured as the gap
    between ``fresh`` and ``actual``, which differ *only* in whether the planner
    saw the state its answer would be applied in. This is arrival regret in
    return units, and it is the quantity the deleted action-gap ratio was a poor
    proxy for.
``L_irreversible``
    the rest of what the wait cost, over and above what the reflex earned back.

The terms form a **telescoping chain** between adjacent counterfactual arms,

    base -> instant(base) -> instant(k) -> fresh(k) -> actual(k)

so they sum exactly rather than approximately, and ``eps_cross`` collapses to a
single identified quantity: the base arm's own delay cost, which is exactly zero
whenever the base budget is cheap enough to land immediately.

A correction worth recording, because the first version of this module got it
wrong. Those five terms were originally measured independently, and nothing then
forced them to add up — the residual came out around 4.9 return units on a scale
where the whole budget effect is a fraction of that. The residual was not noise.
It was absorbing a large real effect with no name: the plain **discounting** cost
of waiting. Everything after a delay of ``d`` ticks is worth ``gamma**d`` times
what it would have been, whether or not anything unrecoverable happened.

So ``L_irreversible`` here means *the whole cost of having waited, net of reflex
earnings*, which mixes discounting with genuinely unrecoverable damage. Those are
different things and P1 needs them apart, so the unrecoverable part is measured
separately as

``L_unrecoverable``
    the gap that survives when both arms are handed an *optimal, instantaneous*
    future from the handoff state onward. Damage that outlives unlimited future
    planning is unrecoverable by definition; the remainder,
    ``L_irreversible - L_unrecoverable``, is the recoverable opportunity cost of
    the delay.

``L_unrecoverable`` sits outside the identity on purpose. It is a measurement of
a sub-component, not a fifth way to make the books balance, and it is the term
that should distinguish an environment with absorbing failures from one without.

Nothing here fits a curve or assumes a functional form. This module measures;
``atlas/fit.py`` and ``atlas/transfer.py`` decide whether the measurements have
any structure worth calling a law.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from instinct.atlas.schema import AtlasRow
from instinct.core.env import Timing
from instinct.core.mdp import FRESH, STALE, TabularMDP
from instinct.core.planner import ExactLookaheadPlanner

__all__ = ["Decomposition", "decompose_exact", "exact_atlas_rows"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Decomposition:
    """The decomposition at one budget, for every start state at once.

    Every field is a ``(n_states,)`` array in return units. Return units are the
    only currency used: ratios near a small denominator are exactly how a
    harmless rank flip gets reported as a catastrophe.
    """

    budget: int
    delay: int
    J_actual: FloatArray
    J_instant: FloatArray
    J_fresh: FloatArray
    J_base: FloatArray
    G_plan: FloatArray
    R_intermediate: FloatArray
    L_arrival: FloatArray
    L_irreversible: FloatArray
    C_hw: FloatArray
    eps_cross: FloatArray
    sigma: FloatArray
    # Outside the identity: the part of L_irreversible that no future planning
    # could have recovered. The remainder is discounting and opportunity cost.
    L_unrecoverable: FloatArray
    # The one quantity eps_cross collapses to: what the *base* arm loses to its
    # own delay. Exposed so the residual is an identified term rather than a
    # bucket, and so a test can assert eps_cross is nothing else.
    L_base_delay: FloatArray

    def max_residual(self) -> float:
        return float(np.max(np.abs(self.eps_cross)))

    def recoverable_share(self) -> FloatArray:
        """How much of the wait's cost later planning could in principle undo."""
        return self.L_irreversible - self.L_unrecoverable


def _reflex_reward(mdp: TabularMDP, reflex: IntArray, delay: int) -> FloatArray:
    """Discounted reward the reflex banks over ``delay`` ticks, per start state."""
    return mdp.reflex_phase(reflex, delay).offset


def _optimal_handoff(
    mdp: TabularMDP, reflex: IntArray, delay: int, V_star: FloatArray
) -> FloatArray:
    """**Undiscounted** expected optimal value at the handoff state.

    The reflex runs for ``delay`` ticks and an oracle then takes over with
    unlimited budget and no latency. What survives that most generous
    continuation is damage planning cannot undo.

    The discount factor is divided back out, and that is the whole point.
    ``reflex_phase(...).kernel`` is ``gamma**delay * P_mu**delay``, so comparing
    two delays with it in place mostly compares ``gamma**d1`` against
    ``gamma**d2`` — pure time preference. An earlier version left it in and duly
    reported ~4.9 units of "unrecoverable damage" in an environment with no
    absorbing states at all. Dividing it out makes this measure *where the reflex
    left you*, independent of *when* it left you there, which is the only version
    of the question that distinguishes a pit from a head start.
    """
    kernel = mdp.reflex_phase(reflex, delay).kernel
    discount = mdp.gamma**delay
    if discount < 1e-300:  # pathological only; delays in the sweep are small
        return np.zeros(mdp.n_states)
    return (kernel / discount) @ V_star


def decompose_exact(
    mdp: TabularMDP,
    *,
    reflex: IntArray,
    budget: int,
    timing: Timing,
    base_budget: int = 1,
    commit: int = 1,
    cost_per_simulation: float = 0.0,
    planner: ExactLookaheadPlanner | None = None,
    V_star: FloatArray | None = None,
) -> Decomposition:
    """Decompose the value of budget ``budget`` against ``base_budget``, exactly.

    No sampling anywhere: the real-time process is a finite Markov chain, so
    every arm is a linear solve and every start state comes back from the same
    solve. That makes the whole state axis free, and because the terms telescope,
    the "decomposition noise swamps the budget effect" kill condition is cleared
    by construction rather than by spending seeds on it.
    """
    planner = planner or ExactLookaheadPlanner(mdp)
    if V_star is None:
        V_star, _, _ = mdp.value_iteration()

    delay = timing.delay_ticks(budget)
    base_delay = timing.delay_ticks(base_budget)

    action_maps = planner.action_maps([budget, base_budget])
    a_k, a_base = action_maps[budget], action_maps[base_budget]

    def value(actions: IntArray, d: int, arrival: str) -> FloatArray:
        return mdp.realtime_value(
            reflex=reflex, planner_action=actions, delay=d, commit=commit, arrival=arrival
        )

    # -- the four arms ----------------------------------------------------
    J_actual = value(a_k, delay, STALE)
    J_base = value(a_base, base_delay, STALE)
    # `instant` removes the delay entirely: the decision is applied where it was
    # computed, isolating planner quality from everything temporal.
    J_instant = value(a_k, 0, STALE)
    J_instant_base = value(a_base, 0, STALE)
    # `fresh` keeps the full delay but lets the planner read the state its answer
    # lands in. It differs from `actual` in staleness alone.
    J_fresh = value(a_k, delay, FRESH)

    # -- the telescoping chain: base -> instant(base) -> instant(k) -> fresh(k) -> actual(k)
    #
    # Each term is the gap between two adjacent arms that differ in exactly one
    # respect, so the chain sums exactly and each link still means something on
    # its own. Measuring the terms independently instead lets them fail to add
    # up, and the leftover then silently absorbs whatever effect has no name.
    G_plan = J_instant - J_instant_base
    R_intermediate = _reflex_reward(mdp, reflex, delay) - _reflex_reward(mdp, reflex, base_delay)
    L_arrival = J_fresh - J_actual
    # The whole cost of having waited, net of what the reflex earned back. This
    # bundles discounting with unrecoverable damage; L_unrecoverable below
    # separates them.
    L_irreversible = R_intermediate - (J_fresh - J_instant)
    C_hw = np.full(
        mdp.n_states, cost_per_simulation * float(budget - base_budget), dtype=np.float64
    )

    # Measured independently, and deliberately not part of the identity: the gap
    # that survives when both arms get an optimal, zero-latency future from the
    # handoff onward. What planning cannot undo is what "irreversible" should
    # mean, and in an environment with no absorbing states this must be ~0 even
    # though L_irreversible is large.
    L_unrecoverable = _optimal_handoff(mdp, reflex, base_delay, V_star) - _optimal_handoff(
        mdp, reflex, delay, V_star
    )

    L_base_delay = J_instant_base - J_base

    sigma = (J_actual - C_hw) - J_base
    explained = G_plan + R_intermediate - L_arrival - L_irreversible - C_hw
    # The one thing the chain leaves over: the base arm's own delay cost. Exactly
    # zero whenever the base budget lands immediately.
    eps_cross = sigma - explained

    return Decomposition(
        budget=budget,
        delay=delay,
        J_actual=J_actual,
        J_instant=J_instant,
        J_fresh=J_fresh,
        J_base=J_base,
        G_plan=G_plan,
        R_intermediate=R_intermediate,
        L_arrival=L_arrival,
        L_irreversible=L_irreversible,
        C_hw=C_hw,
        eps_cross=eps_cross,
        sigma=sigma,
        L_unrecoverable=L_unrecoverable,
        L_base_delay=L_base_delay,
    )


def exact_atlas_rows(
    mdp: TabularMDP,
    *,
    env_name: str,
    reflex: IntArray,
    reflex_name: str,
    budgets: list[int],
    timing: Timing,
    states: list[int] | None = None,
    base_budget: int = 1,
    commit: int = 1,
    cost_per_simulation: float = 0.0,
) -> list[AtlasRow]:
    """Emit schema rows for a whole budget grid at one speed and latency.

    The planner's action maps for every budget come from a single
    value-iteration run, and each budget's arms come from linear solves that
    cover all start states simultaneously. So the cost of this call scales with
    the number of budgets, not with budgets times states times seeds.
    """
    V_star, _, _ = mdp.value_iteration()
    planner = ExactLookaheadPlanner(mdp)
    picked = states if states is not None else list(range(mdp.n_states))

    rows: list[AtlasRow] = []
    for k in budgets:
        d = decompose_exact(
            mdp,
            reflex=reflex,
            budget=k,
            timing=timing,
            base_budget=base_budget,
            commit=commit,
            cost_per_simulation=cost_per_simulation,
            planner=planner,
            V_star=V_star,
        )
        for s in picked:
            rows.append(
                AtlasRow(
                    env=env_name,
                    reflex=reflex_name,
                    nu_e=timing.nu_e,
                    nu_h=timing.nu_h,
                    budget=k,
                    delay=d.delay,
                    staleness=timing.staleness(k),
                    start_state=s,
                    J_actual=float(d.J_actual[s]),
                    J_instant=float(d.J_instant[s]),
                    J_fresh=float(d.J_fresh[s]),
                    J_base=float(d.J_base[s]),
                    G_plan=float(d.G_plan[s]),
                    R_intermediate=float(d.R_intermediate[s]),
                    L_arrival=float(d.L_arrival[s]),
                    L_irreversible=float(d.L_irreversible[s]),
                    C_hw=float(d.C_hw[s]),
                    eps_cross=float(d.eps_cross[s]),
                    sigma=float(d.sigma[s]),
                    # The exact arm is a solve, not a sample: one "seed", and the
                    # interval is the point itself.
                    n_seeds=1,
                    ci_lo=float(d.sigma[s]),
                    ci_hi=float(d.sigma[s]),
                    exact=True,
                    simulations=k,
                )
            )
    return rows
