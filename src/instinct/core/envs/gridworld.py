"""Pursuit on a grid: the environment built to decay *smoothly*.

P1's taxonomy needs at least one environment where staleness hurts a little bit
more for every extra tick of delay, and at least one where it does nothing until
it suddenly does everything. This module is the first kind; :mod:`instinct.core.envs.arcade`
is the second. Having both is what turns "the shape of the staleness curve" from
an assumption into a measurement.

The mechanism that produces the smooth decay is deliberate and worth stating,
because it is the thing a reader should be suspicious of. The agent's reward is
dominated by a shaping term proportional to its distance from the target, and
the target performs a bounded random walk. A plan computed ``d`` ticks ago aimed
at where the target *was*; the target has since diffused about ``sqrt(d)`` cells
away, so the mismatch grows continuously in ``d`` with no cliff anywhere. Nothing
in the environment is absorbing, so ``L_irreversible`` is zero **by construction**
here — that is the point. It is the contrast partner for the arcade games, in the
same way :func:`~instinct.core.envs.tabular.chase_chain` is the contrast partner for
:func:`~instinct.core.envs.tabular.corridor_with_pit`.

Two representation choices carry the vectorization:

*Positions are ``(lane, 2)`` integer arrays*, not flat cell indices. Every move
is one array add plus a clip, and the wall test is a single 2-D gather. Flat
indices would need a divmod on both ends of every move for no gain.

*Free cells are precomputed once* as an ``(n_free, 2)`` table. Placing an agent
or respawning a target is then a uniform draw into that table — O(1), exact, and
crucially **rejection-free**. Rejection sampling would make the number of random
words a lane consumes depend on the values it drew, which destroys the O(1)
counter addressability that :mod:`instinct.core.rng` is built on and would
silently break CRN pairing between arms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from instinct.core.env import EnvState, StepResult
from instinct.core.rng import SeedScope

__all__ = ["GridPursuit", "open_field", "pillar_field"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

# Action 0 is an explicit no-op. A pursuit agent that cannot choose to hold still
# has "do nothing" spelled as "walk into a wall", which is not the same decision
# and would contaminate the move cost.
STAY, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4
_MOVES: IntArray = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)
# The target's four drift directions, indexed to match _MOVES[1:].
_DRIFTS: IntArray = _MOVES[1:]
N_DRIFTS = 4


@dataclass(frozen=True, eq=False)
class GridPursuit:
    """A batched gridworld where the target keeps moving while you deliberate.

    ``walls`` is an ``(H, W)`` boolean mask of impassable cells; both the agent
    and the target are blocked by it and by the grid boundary. ``drift`` is the
    per-tick probability that the target moves at all, and ``evasion`` bends its
    direction choice away from the agent (0 gives a pure random walk).

    ``eq=False`` because the dataclass fields are arrays: a generated
    ``__eq__`` would return an array and raise on truth-testing, which is a
    footgun nobody needs from an environment object.
    """

    walls: BoolArray
    drift: float = 0.5
    evasion: float = 0.0
    catch_reward: float = 1.0
    move_cost: float = 0.01
    distance_weight: float = 0.25
    name: str = "grid_pursuit"

    # Derived once at construction; see the module docstring on why the free-cell
    # table exists rather than rejection sampling.
    free_cells: IntArray = field(init=False, repr=False)
    max_distance: int = field(init=False, repr=False)
    bounds: IntArray = field(init=False, repr=False)
    any_walls: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        w = np.asarray(self.walls, dtype=bool)
        if w.ndim != 2:
            raise ValueError(f"walls must be 2-D, got shape {w.shape}")
        free = np.argwhere(~w).astype(np.int64)
        if free.shape[0] < 2:
            # The respawn fix-up below assumes at least one alternative cell.
            raise ValueError("grid needs at least two free cells")
        if not 0.0 <= self.drift <= 1.0:
            raise ValueError(f"drift={self.drift} must lie in [0, 1]")
        if not 0.0 <= self.evasion <= 1.0:
            raise ValueError(f"evasion={self.evasion} must lie in [0, 1]")
        object.__setattr__(self, "walls", w)
        object.__setattr__(self, "free_cells", free)
        object.__setattr__(self, "max_distance", int(w.shape[0] + w.shape[1] - 2))
        object.__setattr__(self, "bounds", np.array(w.shape, dtype=np.int64) - 1)
        object.__setattr__(self, "any_walls", bool(w.any()))

    @property
    def n_actions(self) -> int:
        return int(_MOVES.shape[0])

    @property
    def height(self) -> int:
        return int(self.walls.shape[0])

    @property
    def width(self) -> int:
        return int(self.walls.shape[1])

    @property
    def n_free(self) -> int:
        return int(self.free_cells.shape[0])

    # -- dynamics pieces ---------------------------------------------------

    def _move(self, pos: IntArray, delta: IntArray) -> IntArray:
        """Apply a step, refusing moves into walls or off the grid.

        Clipping first and testing the wall mask second means one gather covers
        both failure modes; a separate bounds test would be a second pass over
        the same data.
        """
        raw = pos + delta
        clipped = np.stack(
            (
                np.clip(raw[:, 0], 0, self.height - 1),
                np.clip(raw[:, 1], 0, self.width - 1),
            ),
            axis=1,
        )
        blocked = np.any(clipped != raw, axis=1) | self.walls[clipped[:, 0], clipped[:, 1]]
        return np.where(blocked[:, None], pos, clipped).astype(np.int64)

    def _drift_probs(self, agent: IntArray, target: IntArray) -> FloatArray:
        """Direction distribution for the target: uniform, tilted away by ``evasion``.

        Written as an explicit ``(lanes, 4)`` probability table rather than a
        branch on "is the agent north or south", because a table plus one
        inverse-CDF draw is the same cost for every lane and stays exact when
        ``evasion`` is fractional.
        """
        n = agent.shape[0]
        uniform = np.full((n, N_DRIFTS), 1.0 / N_DRIFTS)
        if self.evasion == 0.0:
            return uniform
        dr = target[:, 0] - agent[:, 0]
        dc = target[:, 1] - agent[:, 1]
        # Flee along whichever axis it is already ahead on; ties go to the row.
        use_row = np.abs(dr) >= np.abs(dc)
        away = np.where(use_row, np.where(dr >= 0, DOWN, UP), np.where(dc >= 0, RIGHT, LEFT)) - 1
        onehot = np.arange(N_DRIFTS)[None, :] == away[:, None]
        probs = (1.0 - self.evasion) / N_DRIFTS + self.evasion * onehot
        # Sitting on the agent leaves "away" undefined, so fall back to uniform.
        return np.where(((dr == 0) & (dc == 0))[:, None], uniform, probs)

    def _free_index(self, u: FloatArray) -> IntArray:
        """Uniform index into the free-cell table, one per lane."""
        return np.minimum((u * self.n_free).astype(np.int64), self.n_free - 1)

    # -- BatchedEnv --------------------------------------------------------

    def reset(self, lane_ids: IntArray, *, scope: SeedScope, episode: int = 0) -> EnvState:
        lanes = np.asarray(lane_ids, dtype=np.int64)
        u = scope.stream("spawn").uniform(lanes, count=2, episode=episode, tick=0)
        agent_idx = self._free_index(u[:, 0])
        target_idx = self._free_index(u[:, 1])
        # Nudge rather than resample on a collision: a second draw conditioned on
        # the first would make the number of words a lane consumes depend on the
        # values it drew, which is exactly the data-dependence that breaks O(1)
        # counter addressability. The agent occupies one free cell, so a single
        # +1 always resolves the clash and stays exactly reproducible.
        target_idx = np.where(target_idx == agent_idx, (target_idx + 1) % self.n_free, target_idx)
        return EnvState(
            lane_ids=lanes,
            fields={
                "agent": self.free_cells[agent_idx].copy(),
                "target": self.free_cells[target_idx].copy(),
            },
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
        agent = state["agent"].astype(np.int64)
        target = state["target"].astype(np.int64)
        a = np.asarray(actions, dtype=np.int64)
        if np.any((a < 0) | (a >= self.n_actions)):
            raise ValueError("action outside [0, n_actions)")

        # One draw site per tick: three words from a single Threefry call is
        # cheaper than three calls and keeps each lane's noise addressed by
        # (episode, tick, lane) alone.
        u = scope.stream("world").uniform(state.lane_ids, count=3, episode=episode, tick=tick)

        agent_next = self._move(agent, _MOVES[a])

        cdf = np.cumsum(self._drift_probs(agent, target), axis=1)
        # Clamped because the cumulative sum's last entry can land a few ulps
        # below 1.0, which would otherwise index one past the last direction.
        direction = np.minimum((u[:, 1, None] >= cdf).sum(axis=1), N_DRIFTS - 1)
        moving = (u[:, 0] < self.drift)[:, None]
        target_next = self._move(target, np.where(moving, _DRIFTS[direction], 0))

        # A swap counts as a catch. Without this the target could walk straight
        # through the agent whenever they trade cells, which reads as a bug in
        # any trace and would understate the value of a fresh plan.
        landed_on = np.all(agent_next == target_next, axis=1)
        swapped = np.all(agent_next == target, axis=1) & np.all(target_next == agent, axis=1)
        caught = landed_on | swapped

        distance = np.abs(agent_next - target_next).sum(axis=1)
        reward = (
            self.catch_reward * caught
            - self.move_cost * (a != STAY)
            # Normalized by the grid diameter so the shaping term stays in [0, 1]
            # and `distance_weight` means the same thing at every grid size.
            - self.distance_weight * np.where(caught, 0.0, distance / self.max_distance)
        )

        # Respawn only where a catch happened, so an uneventful tick pays nothing
        # for the branch.
        if bool(caught.any()):
            respawn = self.free_cells[self._free_index(u[:, 2])]
            target_next = np.where(caught[:, None], respawn, target_next)

        return StepResult(
            state=state.replace_fields(agent=agent_next, target=target_next),
            reward=reward.astype(np.float64),
            # Nothing here absorbs: L_irreversible is zero by construction, which
            # is exactly what makes this the clean read on pure arrival cost.
            done=np.zeros(state.n_lanes, dtype=bool),
        )


def open_field(
    size: int = 9,
    *,
    drift: float = 0.5,
    evasion: float = 0.0,
    distance_weight: float = 0.25,
) -> GridPursuit:
    """An empty ``size x size`` grid: pursuit with no occlusion at all.

    The baseline shape. A stale plan is wrong only by the distance the target
    drifted, never by being routed around something, so the decay here is as
    close to pure diffusion as the atlas gets.
    """
    return GridPursuit(
        walls=np.zeros((size, size), dtype=bool),
        drift=drift,
        evasion=evasion,
        distance_weight=distance_weight,
        name="grid_pursuit_open",
    )


def pillar_field(
    size: int = 11,
    spacing: int = 3,
    *,
    drift: float = 0.5,
    evasion: float = 0.0,
    distance_weight: float = 0.25,
) -> GridPursuit:
    """A regular lattice of one-cell pillars.

    The contrast to :func:`open_field`: here a stale plan can be wrong in *kind*
    (committed to the wrong side of a pillar) and not merely in degree. If the
    decay curve comes out the same in both, the smoothness is a property of the
    diffusing target rather than of the geometry — a claim worth being able to
    check rather than assert.
    """
    walls = np.zeros((size, size), dtype=bool)
    walls[spacing::spacing, spacing::spacing] = True
    # The boundary ring is kept clear so the grid stays fully connected however
    # the spacing divides the size; a pillar in a corner can seal cells off.
    walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = False
    return GridPursuit(
        walls=walls,
        drift=drift,
        evasion=evasion,
        distance_weight=distance_weight,
        name="grid_pursuit_pillars",
    )
