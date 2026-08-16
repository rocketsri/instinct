"""Two integer-state arcade games: the threshold end of the atlas.

:mod:`instinct.core.envs.gridworld` is the environment where a stale plan degrades a
little more each tick. These two are the opposite shape, and P1's taxonomy is
only worth reporting if both shapes are present and measured rather than
assumed. In :class:`SnakeLite` a plan that is one tick out of date is usually
free and occasionally fatal; in :class:`TetrisLite` a misplaced piece costs
nothing at all until the stack reaches the ceiling. Both absorb on failure, so
``L_irreversible`` is genuinely non-zero and separable from ``L_arrival`` — the
same reason :func:`~instinct.core.envs.tabular.corridor_with_pit` has a pit.

State representations
---------------------
Both games are chosen for how they *vectorize*, because the cost model of this
repo is "one batched NumPy call per tick over every lane", and any design that
needs a Python-side loop over lanes is disqualified no matter how faithful it is.

*Snake* keeps occupancy as **one ``uint64`` bitboard per lane**, so the
self-collision test for the whole batch is a shift, an ``and`` and a compare —
three array ops, no per-lane masking. The board is capped at 64 cells for exactly
this reason. The body order still has to be known (the tail vacates a cell every
tick), so it lives in a fixed-capacity **ring buffer** of cell indices with a
head pointer; a ring buffer never shifts elements, so growing the snake is a
single scatter rather than an O(length) roll.

*Tetris* keeps a **height per column** plus the current piece. A full cell grid
would make "where does this piece land" a per-lane scan; a height vector makes it
a masked max. The consequence is that the game is the *skyline* abstraction of
Tetris rather than Tetris: a piece rests on the tallest column it spans and fills
everything beneath it, so buried holes do not exist and a completed row is
exactly ``min(heights)``. This is stated plainly rather than hidden, because the
threshold behaviour P1 measures comes from the ceiling, not from hole-making, and
an honest simple game beats an approximate faithful one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from instinct.core.env import EnvState, StepResult
from instinct.core.rng import SeedScope

__all__ = ["SnakeLite", "TetrisLite"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
U64Array = npt.NDArray[np.uint64]

_ONE = np.uint64(1)

# Snake headings, in the order (up, down, left, right).
_SNAKE_DR: IntArray = np.array([-1, 1, 0, 0], dtype=np.int64)
_SNAKE_DC: IntArray = np.array([0, 0, -1, 1], dtype=np.int64)
_OPPOSITE: IntArray = np.array([1, 0, 3, 2], dtype=np.int64)

# Snake cells are stored as int16: a 64-cell board indexes fine in 16 bits, and
# the ring buffer is copied once per step, so a 4x smaller copy is 4x less
# memory traffic on the hot path.
_CELL_DTYPE = np.int16


def _free_cell(occ: U64Array, u: FloatArray, n_cells: int) -> IntArray:
    """Pick a uniformly random unoccupied cell from each lane's bitboard.

    Rank-select rather than the obvious "draw a cell, retry if taken": retrying
    makes the number of random words a lane consumes depend on the values it
    drew, which destroys the O(1) counter addressability CRN pairing rests on.
    Expanding the 64-bit board into a ``(lanes, 64)`` prefix sum costs a fixed
    64x on a step that only happens when something was eaten, and is exact.
    """
    positions = np.arange(n_cells, dtype=np.uint64)
    free = (~(occ[:, None] >> positions[None, :]) & _ONE).astype(np.int64)
    n_free = free.sum(axis=1)
    rank = np.minimum((u * n_free).astype(np.int64), np.maximum(n_free - 1, 0))
    # argmax on a boolean row returns the first True, i.e. the rank-th free cell.
    return np.argmax(np.cumsum(free, axis=1) > rank[:, None], axis=1).astype(np.int64)


@dataclass(frozen=True, eq=False)
class SnakeLite:
    """Snake on a board of at most 64 cells, one ``uint64`` bitboard per lane.

    Actions are the four absolute headings. A command that reverses the current
    heading is ignored rather than fatal: reversal death is an artifact of the
    control scheme, not of the board, and letting it kill would credit staleness
    for deaths that have nothing to do with the world having moved.

    Fields: ``body`` (ring buffer of cell indices), ``head_ptr``, ``length``,
    ``heading``, ``food``, ``occ`` (bitboard), ``dead``.
    """

    height: int = 8
    width: int = 8
    food_reward: float = 1.0
    step_cost: float = 0.01
    death_penalty: float = 1.0
    name: str = "snake_lite"

    n_cells: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = self.height * self.width
        if not 4 <= n <= 64:
            raise ValueError(f"board has {n} cells; the uint64 bitboard holds 4..64")
        object.__setattr__(self, "n_cells", n)

    @property
    def n_actions(self) -> int:
        return 4

    # -- helpers -----------------------------------------------------------

    def _bit(self, cell: IntArray) -> U64Array:
        return (_ONE << cell.astype(np.uint64)).astype(np.uint64)

    def _step_cell(self, cell: IntArray, heading: IntArray) -> tuple[IntArray, BoolArray]:
        """Advance a flat cell index, reporting whether it left the board.

        Returns the *unchanged* cell when the move is off-board so the caller can
        keep using it as an index; the accompanying mask says the move was fatal.
        """
        row = cell // self.width
        col = cell % self.width
        nrow = row + _SNAKE_DR[heading]
        ncol = col + _SNAKE_DC[heading]
        off = (nrow < 0) | (nrow >= self.height) | (ncol < 0) | (ncol >= self.width)
        return np.where(off, cell, nrow * self.width + ncol), off

    # -- BatchedEnv --------------------------------------------------------

    def reset(self, lane_ids: IntArray, *, scope: SeedScope, episode: int = 0) -> EnvState:
        lanes = np.asarray(lane_ids, dtype=np.int64)
        n = lanes.shape[0]
        u = scope.stream("spawn").uniform(lanes, count=3, episode=episode, tick=0)

        head = np.minimum((u[:, 0] * self.n_cells).astype(np.int64), self.n_cells - 1)
        heading = np.minimum((u[:, 1] * 4).astype(np.int64), 3)
        occ = self._bit(head)

        body = np.zeros((n, self.n_cells), dtype=_CELL_DTYPE)
        body[:, 0] = head.astype(_CELL_DTYPE)
        return EnvState(
            lane_ids=lanes,
            fields={
                "body": body,
                "head_ptr": np.zeros(n, dtype=np.int64),
                # Length 1 on purpose: a longer seeded snake needs a placement
                # rule for its tail, and every such rule is a hidden bias in the
                # initial state distribution that no measurement here wants.
                "length": np.ones(n, dtype=np.int64),
                "heading": heading,
                "food": _free_cell(occ, u[:, 2], self.n_cells),
                "occ": occ,
                "dead": np.zeros(n, dtype=bool),
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
        body = state["body"]
        head_ptr = state["head_ptr"].astype(np.int64)
        length = state["length"].astype(np.int64)
        heading = state["heading"].astype(np.int64)
        food = state["food"].astype(np.int64)
        occ = state["occ"].astype(np.uint64)
        dead = state["dead"].astype(bool)
        a = np.asarray(actions, dtype=np.int64)
        if np.any((a < 0) | (a >= self.n_actions)):
            raise ValueError("action outside [0, n_actions)")

        n = state.n_lanes
        rows = np.arange(n)
        cap = self.n_cells

        reversing = (a == _OPPOSITE[heading]) & (length > 1)
        new_heading = np.where(reversing, heading, a)

        head = body[rows, head_ptr].astype(np.int64)
        new_head, off_board = self._step_cell(head, new_heading)

        ate = ~off_board & (new_head == food)
        # The tail vacates its cell before the head arrives unless the snake grew,
        # so chasing your own tail is legal. Folding that into the occupancy mask
        # keeps the collision test a single bit lookup.
        tail = body[rows, (head_ptr - length + 1) % cap].astype(np.int64)
        occ_free = occ & ~np.where(ate, np.uint64(0), self._bit(tail))
        self_hit = ~off_board & (((occ_free >> new_head.astype(np.uint64)) & _ONE) != 0)

        collide = off_board | self_hit
        died = ~dead & collide
        advance = ~dead & ~collide

        # Every update below is gated on `advance`, so a dead lane is bit-for-bit
        # frozen: absorbing states have to be absorbing in the state as well as in
        # the reward, or a replay from a terminal state diverges.
        ptr_next = np.where(advance, (head_ptr + 1) % cap, head_ptr)
        body_next = body.copy()
        # A no-op write for frozen lanes (same slot, same value) keeps this a
        # single scatter instead of a mask-and-gather.
        body_next[rows, ptr_next] = np.where(advance, new_head, head).astype(_CELL_DTYPE)

        grew = advance & ate
        occ_next = np.where(advance, occ_free | self._bit(new_head), occ)
        length_next = length + grew
        heading_next = np.where(advance, new_heading, heading)

        food_next = food
        if bool(grew.any()):
            u = scope.stream("food").uniform(state.lane_ids, count=1, episode=episode, tick=tick)
            food_next = np.where(grew, _free_cell(occ_next, u[:, 0], cap), food)

        reward = np.where(
            dead,
            0.0,
            self.food_reward * grew - self.step_cost - self.death_penalty * died,
        )
        dead_next = dead | died
        return StepResult(
            state=state.replace_fields(
                body=body_next,
                head_ptr=ptr_next,
                length=length_next,
                heading=heading_next,
                food=food_next,
                occ=occ_next,
                dead=dead_next,
            ),
            reward=reward.astype(np.float64),
            done=dead_next,
        )


# Pieces are axis-aligned rectangles ``(width, height)``. Only rectangles: any
# other shape would leave the surface underspecified by a height vector, and a
# representation that is exact for the dynamics beats one that is nearly faithful
# to Tetris and approximate for the thing being measured.
_PIECES: IntArray = np.array(
    [[1, 1], [2, 1], [1, 2], [2, 2], [3, 1], [1, 3], [4, 1], [1, 4]], dtype=np.int64
)


@dataclass(frozen=True, eq=False)
class TetrisLite:
    """Skyline Tetris: a height per column, plus the piece waiting to be placed.

    An action names the target column. It is clamped so the piece always fits
    horizontally, which makes **every action legal in every state**. That is a
    deliberate choice: an illegal-action failure mode would be a second way for a
    stale plan to fail, confounded with the one under study.

    Fields: ``heights`` (lanes, width), ``piece``, ``dead``.
    """

    width: int = 6
    ceiling: int = 10
    survive_reward: float = 0.05
    clear_reward: float = 1.0
    death_penalty: float = 1.0
    name: str = "tetris_lite"

    def __post_init__(self) -> None:
        if self.width < int(_PIECES[:, 0].max()):
            raise ValueError(f"width={self.width} is narrower than the widest piece")
        if self.ceiling < int(_PIECES[:, 1].max()):
            raise ValueError(f"ceiling={self.ceiling} is lower than the tallest piece")

    @property
    def n_actions(self) -> int:
        return self.width

    @property
    def n_pieces(self) -> int:
        return int(_PIECES.shape[0])

    def _draw_piece(
        self, lanes: IntArray, *, scope: SeedScope, episode: int, tick: int
    ) -> IntArray:
        return scope.stream("piece").integers(
            lanes, self.n_pieces, count=1, episode=episode, tick=tick
        )[:, 0]

    def reset(self, lane_ids: IntArray, *, scope: SeedScope, episode: int = 0) -> EnvState:
        lanes = np.asarray(lane_ids, dtype=np.int64)
        return EnvState(
            lane_ids=lanes,
            fields={
                "heights": np.zeros((lanes.shape[0], self.width), dtype=np.int64),
                # The piece visible at tick t is drawn by the step at tick t-1, so
                # the spawn draw sits on its own stream rather than colliding with
                # the transition stream at tick 0.
                "piece": self._draw_piece(lanes, scope=scope, episode=episode, tick=0),
                "dead": np.zeros(lanes.shape[0], dtype=bool),
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
        heights = state["heights"].astype(np.int64)
        piece = state["piece"].astype(np.int64)
        dead = state["dead"].astype(bool)
        a = np.asarray(actions, dtype=np.int64)
        if np.any((a < 0) | (a >= self.n_actions)):
            raise ValueError("action outside [0, n_actions)")

        pw = _PIECES[piece, 0]
        ph = _PIECES[piece, 1]
        col = np.minimum(a, self.width - pw)

        # The span mask is what lets a per-lane variable piece width stay
        # vectorized: the max over "the columns this piece covers" becomes one
        # masked reduction over the full width instead of a ragged slice.
        idx = np.arange(self.width)[None, :]
        span = (idx >= col[:, None]) & (idx < (col + pw)[:, None])
        base = np.max(np.where(span, heights, -1), axis=1)
        top = base + ph

        topped_out = top > self.ceiling
        died = ~dead & topped_out
        advance = ~dead & ~topped_out

        placed = np.where(span, top[:, None], heights)
        # With no buried holes by construction, the number of completed rows is
        # exactly the shortest column.
        cleared = np.where(advance, placed.min(axis=1), 0)
        heights_next = np.where(advance[:, None], placed - cleared[:, None], heights)

        drawn = self._draw_piece(state.lane_ids, scope=scope, episode=episode, tick=tick)
        piece_next = np.where(advance, drawn, piece)

        reward = np.where(
            dead,
            0.0,
            self.survive_reward * advance
            + self.clear_reward * cleared
            - self.death_penalty * died,
        )
        dead_next = dead | died
        return StepResult(
            state=state.replace_fields(
                heights=heights_next, piece=piece_next, dead=dead_next
            ),
            reward=reward.astype(np.float64),
            done=dead_next,
        )
