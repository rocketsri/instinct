"""Batched environments and the real-time timing model.

Two ideas carry most of the weight here.

**Lanes carry identity, not position.** An :class:`EnvState` holds a leading
lane axis plus the ``lane_ids`` those lanes belong to. Subsetting, forking and
reordering all preserve the ids, so the counter-based streams in
:mod:`instinct.core.rng` keep handing each lane its own noise no matter how the
batch is sliced. Drop the ids and every paired comparison silently loses its
pairing.

**Environment speed and hardware latency are separate axes.** ``nu_e`` is how
fast the world moves; ``nu_h`` is how slow the planner is. Their product times
the budget gives a staleness ``delta = nu_e * nu_h * k``, and it is tempting to
treat ``delta`` as the only thing that matters. Whether the compute-freshness
surface actually collapses onto that single number is one of P1's questions, so
:class:`Timing` deliberately keeps the two knobs independent and never folds
them together behind the caller's back.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from instinct.core.rng import SeedScope

__all__ = ["EnvState", "Timing", "BatchedEnv", "StepResult"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class EnvState:
    """A batch of environment states sharing one leading lane axis.

    ``fields`` maps a name to an array whose first axis is the lane axis. The
    class stays agnostic about what those arrays mean so gridworlds, bitboard
    arcade games and tabular MDPs can all use it.
    """

    lane_ids: IntArray
    fields: dict[str, npt.NDArray[np.generic]]

    def __post_init__(self) -> None:
        n = self.lane_ids.shape[0]
        for name, arr in self.fields.items():
            if arr.shape[0] != n:
                raise ValueError(
                    f"field {name!r} has leading axis {arr.shape[0]}, expected {n} lanes"
                )

    @property
    def n_lanes(self) -> int:
        return int(self.lane_ids.shape[0])

    def __getitem__(self, name: str) -> npt.NDArray[np.generic]:
        return self.fields[name]

    def take(self, idx: IntArray | BoolArray) -> EnvState:
        """Select a subset of lanes, carrying their identities along.

        This is what makes forking safe: the selected lanes keep drawing the
        same noise they would have drawn in the full batch.
        """
        return EnvState(
            lane_ids=self.lane_ids[idx],
            fields={k: v[idx] for k, v in self.fields.items()},
        )

    def copy(self) -> EnvState:
        return EnvState(
            lane_ids=self.lane_ids.copy(),
            fields={k: v.copy() for k, v in self.fields.items()},
        )

    def replace_fields(self, **updates: npt.NDArray[np.generic]) -> EnvState:
        return replace(self, fields={**self.fields, **updates})

    def relabel(self, lane_ids: IntArray) -> EnvState:
        """Reassign lane identities.

        Needed when one logical lane is expanded into several counterfactual
        arms that must *not* share noise — for example the action branches in
        P3's CRN branching, where sharing would defeat the comparison.
        """
        if lane_ids.shape[0] != self.n_lanes:
            raise ValueError("lane_ids length must match the lane axis")
        return replace(self, lane_ids=lane_ids.astype(np.int64))


@dataclass(frozen=True, slots=True)
class StepResult:
    state: EnvState
    reward: FloatArray
    done: BoolArray


@dataclass(frozen=True, slots=True)
class Timing:
    """Maps a planning budget to a delay in environment ticks.

    ``nu_e``: environment ticks per unit wall-clock (how fast the world moves).
    ``nu_h``: wall-clock per planner simulation (how slow the hardware is).

    A caution that matters when reading P1's results: delays are integers, so a
    range of budgets can map to the same delay. That produces genuine plateaus
    and step edges in the budget curve which are artifacts of discretization,
    not of the environment. Since one of P1's outputs is a taxonomy that
    includes a *threshold regime*, this is a live confound — hence
    :meth:`distinct_delays`, so a sweep can report how much of its apparent
    structure is just rounding.
    """

    nu_e: float = 1.0
    nu_h: float = 1.0
    rounding: str = "floor"

    def staleness(self, budget: int) -> float:
        """The continuous ``delta = nu_e * nu_h * k``, before discretization.

        Exposed so the sweep can test whether the surface collapses onto this
        single number — a hypothesis, never an assumption.
        """
        return self.nu_e * self.nu_h * float(budget)

    def delay_ticks(self, budget: int) -> int:
        raw = self.staleness(budget)
        if self.rounding == "floor":
            return int(np.floor(raw))
        if self.rounding == "round":
            return int(np.rint(raw))
        if self.rounding == "ceil":
            return int(np.ceil(raw))
        raise ValueError(f"unknown rounding {self.rounding!r}")

    def distinct_delays(self, budgets: list[int]) -> dict[int, list[int]]:
        """Group budgets by the delay they produce.

        Any group with more than one budget is a set of points that differ only
        in planner quality, with staleness held exactly fixed. Those groups are
        the cleanest available read on ``G_plan``, and equally they are where a
        naive reading would mistake a rounding plateau for a threshold.
        """
        groups: dict[int, list[int]] = {}
        for k in budgets:
            groups.setdefault(self.delay_ticks(k), []).append(k)
        return groups


@runtime_checkable
class BatchedEnv(Protocol):
    """A real-time environment that advances every lane in lockstep.

    Implementations must be pure with respect to ``(state, actions, episode,
    tick)``: all randomness comes from ``scope`` keyed by the lane ids in
    ``state``, never from internal mutable RNG. That purity is what allows a
    rollout to be forked, replayed or resumed and still land on identical
    trajectories.
    """

    n_actions: int
    name: str

    def reset(self, lane_ids: IntArray, *, scope: SeedScope, episode: int = 0) -> EnvState: ...

    def step(
        self,
        state: EnvState,
        actions: IntArray,
        *,
        scope: SeedScope,
        episode: int = 0,
        tick: int = 0,
    ) -> StepResult: ...
