"""The real-time rollout loop, and the counterfactual arms P1 measures.

One decision epoch:

    at ``s_0`` the planner starts thinking; for ``delay`` ticks the reflex acts
    while the world moves; the planner's decision lands at ``s_delay`` and is
    committed for ``commit`` ticks; a new epoch begins.

Four arms differ only in how that decision is produced, which is what lets the
decomposition attribute value to specific causes:

===========  ==========================================================
``actual``   full delay; action computed from ``s_0``. The real process.
``instant``  no delay; action computed from ``s_0``. Isolates planning
             benefit with staleness removed.
``fresh``    full delay, but the action is computed from ``s_delay``.
             Same planner, same waiting, no staleness. The gap to
             ``actual`` *is* arrival regret, in return units.
``base``     the cheapest budget, at its own delay. The floor.
===========  ==========================================================

``fresh`` deserves a word, because it is the one that is easy to get wrong. It is
not "plan instantly at ``s_delay``" — it still pays the full delay, and the
reflex still acts through it. The only thing removed is the mismatch between the
state the planner *saw* and the state its answer *arrives in*. Building it any
other way conflates staleness with the cost of waiting, and those are precisely
the two terms the decomposition exists to separate.

Performance
-----------
:func:`simulate_arm` is the reference: one arm, one budget, a literal loop.
:func:`simulate_lanes` is the fast path, and its saving is **batching, not
skipped work**. Every ``(seed, arm, budget)`` combination becomes a lane, and all
lanes advance in a single vectorized ``env.step`` per tick instead of one Python
loop iteration each.

A note on what is *not* shareable, since an earlier draft of this module claimed
more than it delivered. It is tempting to reuse one reflex rollout across
budgets. That is valid only until the first decision lands: after that, different
budgets have committed different actions at different times and their states have
genuinely diverged, so there is no common prefix left to share. The same applies
to ``actual`` versus ``fresh``. Within a single epoch they do share the reflex
prefix exactly — and running them as sibling lanes captures that — but across
epochs they are different trajectories. Lanes for the same seed carry the *same*
``lane_id``, so they still draw identical environment noise; that shared
randomness is the pairing, and it is what the estimators actually need.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from instinct.core.env import BatchedEnv, EnvState, Timing
from instinct.core.rng import SeedScope

__all__ = [
    "ARMS",
    "ArmTrace",
    "LaneSpec",
    "PlannerFn",
    "RealTimeConfig",
    "ReflexFn",
    "simulate_arm",
    "simulate_lanes",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

ACTUAL, INSTANT, FRESH, BASE = "actual", "instant", "fresh", "base"
ARMS: tuple[str, ...] = (ACTUAL, INSTANT, FRESH, BASE)

# Chooses an action per lane, given the *root* state the planner sees and a budget.
PlannerFn = Callable[[EnvState, int, int], IntArray]
# The fast reflex: state and tick only. It never sees the budget, which is why it
# cannot depend on it.
ReflexFn = Callable[[EnvState, int], IntArray]


@dataclass(frozen=True, slots=True)
class RealTimeConfig:
    budgets: tuple[int, ...]
    timing: Timing = field(default_factory=Timing)
    commit: int = 1
    horizon: int = 64
    gamma: float = 0.95
    base_budget: int = 1

    def delay_for(self, arm: str, budget: int) -> int:
        """Ticks of delay an arm pays at a budget.

        ``instant`` is the only arm that skips the wait. ``fresh`` pays it in
        full: it is a counterfactual about staleness, not about speed.
        """
        if arm == INSTANT:
            return 0
        if arm == BASE:
            return self.timing.delay_ticks(self.base_budget)
        return self.timing.delay_ticks(budget)

    def budget_for(self, arm: str, budget: int) -> int:
        return self.base_budget if arm == BASE else budget


@dataclass(frozen=True, slots=True)
class LaneSpec:
    arm: str
    budget: int


@dataclass(slots=True)
class ArmTrace:
    """Per-lane outcome of one arm at one budget."""

    total_return: FloatArray
    intermediate: FloatArray  # discounted reflex reward banked during delays
    committed: FloatArray  # discounted reward from the planner's committed action
    epochs: int = 0
    planner_calls: int = 0
    simulations: int = 0  # planning simulations consumed: the raw input to C_hw

    @staticmethod
    def empty(n_lanes: int) -> ArmTrace:
        z = np.zeros(n_lanes)
        return ArmTrace(total_return=z.copy(), intermediate=z.copy(), committed=z.copy())


def simulate_arm(
    env: BatchedEnv,
    state0: EnvState,
    *,
    arm: str,
    budget: int,
    cfg: RealTimeConfig,
    reflex: ReflexFn,
    planner: PlannerFn,
    scope: SeedScope,
    episode: int = 0,
) -> ArmTrace:
    """Reference implementation: one arm, one budget, written the obvious way.

    Slow and deliberately unclever. :func:`simulate_lanes` must reproduce it
    exactly; ``tests/test_rollout.py`` asserts that.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

    state = state0
    trace = ArmTrace.empty(state0.n_lanes)
    alive = np.ones(state0.n_lanes, dtype=bool)
    k = cfg.budget_for(arm, budget)
    delay = cfg.delay_for(arm, budget)
    discount = 1.0
    tick = 0

    while tick < cfg.horizon and alive.any():
        epoch_root = state  # what the planner sees when it starts thinking

        for _ in range(delay):  # the reflex fills the wait
            if tick >= cfg.horizon:
                break
            res = env.step(state, reflex(state, tick), scope=scope, episode=episode, tick=tick)
            credited = discount * res.reward * alive
            trace.total_return += credited
            trace.intermediate += credited
            state, alive = res.state, alive & ~res.done
            discount *= cfg.gamma
            tick += 1
        if tick >= cfg.horizon:
            break

        # The decision lands. `fresh` re-reads the world; the others do not.
        landed = planner(state if arm == FRESH else epoch_root, k, tick)
        trace.planner_calls += 1
        trace.simulations += k

        for _ in range(cfg.commit):
            if tick >= cfg.horizon:
                break
            res = env.step(state, landed, scope=scope, episode=episode, tick=tick)
            credited = discount * res.reward * alive
            trace.total_return += credited
            trace.committed += credited
            state, alive = res.state, alive & ~res.done
            discount *= cfg.gamma
            tick += 1
        trace.epochs += 1

    return trace


def _replicate(state: EnvState, times: int) -> EnvState:
    """Stack ``times`` copies of every lane, keeping lane identities intact.

    Repeating the ids rather than renumbering them is the whole point: sibling
    lanes for one seed must draw the *same* environment noise so the arms stay
    paired.
    """
    return EnvState(
        lane_ids=np.tile(state.lane_ids, times),
        fields={k: np.tile(v, (times,) + (1,) * (v.ndim - 1)) for k, v in state.fields.items()},
    )


def simulate_lanes(
    env: BatchedEnv,
    state0: EnvState,
    specs: Sequence[LaneSpec],
    *,
    cfg: RealTimeConfig,
    reflex: ReflexFn,
    planner: PlannerFn,
    scope: SeedScope,
    episode: int = 0,
) -> dict[tuple[str, int], ArmTrace]:
    """Run every ``(arm, budget)`` combination as lanes of one batched rollout.

    Returns ``{(arm, budget): ArmTrace}``. Each combination occupies a contiguous
    block of ``state0.n_lanes`` lanes, and all blocks step together, so the cost
    is one vectorized ``env.step`` per tick rather than one per combination.

    Lanes are in different phases at any given tick — some filling a delay, some
    committing a landed action — so actions are assembled by masking rather than
    branching. That is the price of vectorizing over heterogeneous delays, and it
    is well worth paying.
    """
    n_seeds = state0.n_lanes
    n_combos = len(specs)
    if n_combos == 0:
        return {}

    state = _replicate(state0, n_combos)
    total_lanes = n_seeds * n_combos
    block = np.repeat(np.arange(n_combos), n_seeds)  # which spec each lane follows

    combo_budgets = np.array([cfg.budget_for(s.arm, s.budget) for s in specs], dtype=np.int64)
    delays = np.array([cfg.delay_for(s.arm, s.budget) for s in specs], dtype=np.int64)[block]
    budgets = combo_budgets[block]
    is_fresh = np.array([s.arm == FRESH for s in specs], dtype=bool)[block]

    alive = np.ones(total_lanes, dtype=bool)
    total = np.zeros(total_lanes)
    intermediate = np.zeros(total_lanes)
    committed = np.zeros(total_lanes)
    planner_calls = np.zeros(n_combos, dtype=np.int64)
    simulations = np.zeros(n_combos, dtype=np.int64)
    epochs = np.zeros(n_combos, dtype=np.int64)

    # Phase bookkeeping. `remaining` counts down the current phase; `in_commit`
    # says which phase that is.
    remaining = delays.copy()
    in_commit = delays == 0
    landed = np.zeros(total_lanes, dtype=np.int64)
    root = state  # snapshot of each lane's epoch root, for the stale arms
    discount = 1.0

    # Lanes that start with no delay must plan immediately, before the first step.
    if in_commit.any():
        landed = _plan_into(
            landed, root, state, in_commit, is_fresh, budgets, planner, tick=0
        )
        remaining = np.where(in_commit, cfg.commit, remaining)
        _tally(planner_calls, simulations, block, in_commit, combo_budgets)

    for tick in range(cfg.horizon):
        if not alive.any():
            break

        reflex_actions = reflex(state, tick)
        actions = np.where(in_commit, landed, reflex_actions)

        res = env.step(state, actions, scope=scope, episode=episode, tick=tick)
        credited = discount * res.reward * alive
        total += credited
        committed += credited * in_commit
        intermediate += credited * ~in_commit
        state = res.state
        alive &= ~res.done
        discount *= cfg.gamma

        remaining -= 1
        finished = remaining <= 0
        # Snapshot the phase *before* any transition: whether a lane is due a
        # decision depends on which phase it just completed, not the one it is
        # about to enter. Reading `in_commit` after the epoch rollover below
        # makes a lane that just finished committing look like it finished a
        # delay, and it would plan again immediately instead of waiting.
        was_commit = in_commit.copy()

        # Lanes finishing a commit phase start a new epoch; their root is now.
        new_epoch = finished & was_commit
        if new_epoch.any():
            epochs[np.unique(block[new_epoch])] += 1
            root = EnvState(
                lane_ids=root.lane_ids,
                fields={
                    k: np.where(
                        new_epoch.reshape((-1,) + (1,) * (v.ndim - 1)), state.fields[k], v
                    )
                    for k, v in root.fields.items()
                },
            )
            remaining = np.where(new_epoch, delays, remaining)
            in_commit = in_commit & ~new_epoch

        # A decision lands for lanes that just finished a delay, plus lanes whose
        # new epoch has no delay at all and so lands at once.
        # A decision taken on the final tick could never be executed, so it is
        # not taken and not charged. The reference loop exits before planning in
        # that case; without this guard the batched path books a phantom planner
        # call and overstates C_hw by one epoch's worth of simulations.
        landing = (finished & ~was_commit) | (new_epoch & (delays == 0))
        if tick + 1 >= cfg.horizon:
            landing = np.zeros_like(landing)
        if landing.any():
            landed = _plan_into(
                landed, root, state, landing, is_fresh, budgets, planner, tick=tick + 1
            )
            _tally(planner_calls, simulations, block, landing, combo_budgets)
            in_commit = in_commit | landing
            remaining = np.where(landing, cfg.commit, remaining)

    out: dict[tuple[str, int], ArmTrace] = {}
    for i, spec in enumerate(specs):
        sl = slice(i * n_seeds, (i + 1) * n_seeds)
        out[(spec.arm, spec.budget)] = ArmTrace(
            total_return=total[sl].copy(),
            intermediate=intermediate[sl].copy(),
            committed=committed[sl].copy(),
            epochs=int(epochs[i]),
            planner_calls=int(planner_calls[i]),
            simulations=int(simulations[i]),
        )
    return out


def _plan_into(
    landed: IntArray,
    root: EnvState,
    current: EnvState,
    mask: BoolArray,
    is_fresh: BoolArray,
    budgets: IntArray,
    planner: PlannerFn,
    *,
    tick: int,
) -> IntArray:
    """Fill in decisions for the masked lanes.

    Grouped by budget because the planner takes a scalar budget; ``fresh`` lanes
    read the current state while the rest read their epoch root, which is the
    single line that distinguishes stale from fresh arrival.
    """
    landed = landed.copy()
    for k in np.unique(budgets[mask]):
        for fresh in (False, True):
            sel = mask & (budgets == k) & (is_fresh == fresh)
            if not sel.any():
                continue
            source = current if fresh else root
            idx = np.flatnonzero(sel)
            landed[idx] = planner(source.take(idx), int(k), tick)
    return landed


def _tally(
    planner_calls: IntArray,
    simulations: IntArray,
    block: IntArray,
    mask: BoolArray,
    combo_budgets: IntArray,
) -> None:
    """Charge one planner call, and its simulations, per planning *combination*.

    Counted per combination rather than per lane. Lanes within a block are
    seeds of the same configuration and stay phase-locked, so one decision
    serves all of them — charging each lane separately would inflate the cost by
    the seed count and corrupt ``C_hw``, which is measured, not assumed.
    """
    if not mask.any():
        return
    planning = np.unique(block[mask])
    planner_calls[planning] += 1
    simulations[planning] += combo_budgets[planning]
