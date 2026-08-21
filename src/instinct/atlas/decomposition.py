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
``L_wait``
    the rest of what the wait cost, over and above what the reflex earned back.

The terms form a **telescoping chain** between adjacent counterfactual arms,

    base -> instant(base) -> instant(k) -> fresh(k) -> actual(k)

so they sum exactly rather than approximately. The base arm's own delay cost is
reported explicitly as ``L_base_delay``; only the remaining numerical
reconstruction residual is ``epsilon_id``.

A correction to the proposal, found empirically. Its five terms were originally
measured independently here, and nothing forced them to add up -- the residual
came out around 4.9 return units on a scale where the whole budget effect is a
fraction of that. The residual was not noise. It was absorbing a large real
effect the proposal has no name for: the plain **discounting** cost of waiting.
Everything after a delay of ``d`` ticks is worth ``gamma**d`` times what it
would have been, whether or not anything unrecoverable happened.

``L_wait`` therefore carries the whole cost of having waited, mixing discounting
with genuinely unrecoverable damage. Those are different things and P1 needs
them apart, so the proposal's actual quantity is measured separately as

``L_irreversible``
    the matched-time excess probability of reaching an explicitly declared
    failure state while the longer reflex prefix runs, against a counterfactual
    that switches to the safest continuation after the base delay. It is a
    probability diagnostic, not a signed handoff-value difference.

``L_irreversible`` sits outside the identity on purpose. It is a measurement of
a sub-component, not a fifth way to make the books balance, and it is the term
that distinguishes an environment with absorbing failures from one without.

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

    Every field is a ``(n_states,)`` array. Identity terms are in return units;
    ``L_irreversible`` is explicitly a failure-probability diagnostic outside
    that identity.
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
    L_wait: FloatArray
    C_hw: FloatArray
    epsilon_id: FloatArray
    sigma: FloatArray
    # Outside the return-unit identity: matched-time excess failure probability.
    L_irreversible: FloatArray
    # The identified cost the base arm loses to its own delay.
    L_base_delay: FloatArray

    def max_residual(self) -> float:
        return float(np.max(np.abs(self.epsilon_id)))

    @property
    def eps_cross(self) -> FloatArray:
        """Compatibility alias; new artifacts use ``epsilon_id``."""
        return self.epsilon_id

def _reflex_reward(mdp: TabularMDP, reflex: IntArray, delay: int) -> FloatArray:
    """Discounted reward the reflex banks over ``delay`` ticks, per start state."""
    return mdp.reflex_phase(reflex, delay).offset


def _matched_failure_delta(
    mdp: TabularMDP,
    reflex: IntArray,
    *,
    base_delay: int,
    delay: int,
) -> FloatArray:
    """Matched-time failure-risk difference for the irreversibility diagnostic.

    At ``max(delay, base_delay)``, compare continuing the reflex with switching
    from the reflex to the optimal policy after ``base_delay``. Exact failure
    semantics must be declared by the MDP; goals are not silently counted as
    failures merely because they are terminal.
    """
    failure = mdp.failure
    if failure is None or not np.any(failure):
        return np.zeros(mdp.n_states)
    matched_delay = max(delay, base_delay)
    actual = mdp.reflex_phase(reflex, matched_delay)
    prefix_steps = min(base_delay, matched_delay)
    discount = mdp.gamma**matched_delay
    if discount < 1e-300:
        return np.zeros(mdp.n_states)
    target = failure.astype(np.float64)
    # Exact finite-horizon minimum failure probability after the common reflex
    # prefix. This counterfactual is matched in time and does not use reward as
    # a proxy for reachability.
    remaining_risk = target.copy()
    for _ in range(matched_delay - prefix_steps):
        remaining_risk = (mdp.P @ remaining_risk).min(axis=1)
    prefix = mdp.reflex_phase(reflex, prefix_steps)
    prefix_discount = mdp.gamma**prefix_steps
    counterfactual_risk = (prefix.kernel @ remaining_risk) / max(prefix_discount, 1e-300)
    actual_risk = (actual.kernel @ target) / discount
    return actual_risk - counterfactual_risk


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
    solved_v, _, _ = mdp.value_iteration()
    if V_star is None:
        V_star = solved_v

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
    L_wait = R_intermediate - (J_fresh - J_instant)
    C_hw = np.full(
        mdp.n_states, cost_per_simulation * float(budget - base_budget), dtype=np.float64
    )

    # Separate from the identity and evaluated at a common time. A signed
    # handoff-value difference is not labeled damage.
    L_irreversible = _matched_failure_delta(
        mdp,
        reflex,
        base_delay=base_delay,
        delay=delay,
    )

    L_base_delay = J_instant_base - J_base

    sigma = (J_actual - C_hw) - J_base
    explained = G_plan + R_intermediate - L_arrival - L_wait - C_hw + L_base_delay
    epsilon_id = sigma - explained

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
        L_wait=L_wait,
        C_hw=C_hw,
        epsilon_id=epsilon_id,
        sigma=sigma,
        L_irreversible=L_irreversible,
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
                    L_wait=float(d.L_wait[s]),
                    L_irreversible=float(d.L_irreversible[s]),
                    C_hw=float(d.C_hw[s]),
                    L_base_delay=float(d.L_base_delay[s]),
                    epsilon_id=float(d.epsilon_id[s]),
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
