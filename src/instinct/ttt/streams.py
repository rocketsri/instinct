"""Synthetic streams whose harmful writes are known in advance.

An oracle that scores writes is only as trustworthy as the check you can run on
it, and on real data there is no check: nothing tells you which writes were the
destructive ones, so a broken oracle and a true negative result look the same.
These streams exist to close that hole. Harmful chunks are *constructed*, the
mechanism of the harm is stated, and the ground-truth flags come out attached
to the chunks. If :mod:`instinct.ttt.oracle` cannot recover them with near-perfect
recall here, the oracle is wrong — not the hypothesis.

Two mechanisms, because "harmful" is not one thing:

**Drifting regression.** The stream teaches ``y = A_t x`` with ``A_t`` rotating
slowly, so a memory that refuses to write falls behind and genuinely needs
plasticity. A minority of chunks come from a conflicting task ``y = -A_t x``.
Fitting one damages every retained base-task association at once: the harm is
*global and directional*.

**Associative recall.** The stream writes distinct ``(key, value)`` pairs. A
harmful chunk re-uses a key that was written earlier and pairs it with a
different value. Nothing else in the memory is touched: the harm is *local and
targeted*, and the damaged probe is identified by name.

A gate that catches the first and misses the second has learned update
magnitude, not harm. Keeping both is what makes that distinguishable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from instinct.core.rng import SeedScope

__all__ = [
    "Chunk",
    "SyntheticStream",
    "associative_recall_stream",
    "drifting_regression_stream",
]

FloatArray = npt.NDArray[np.float64]

BASE_TASK = 0
CONFLICT_TASK = 1


@dataclass(frozen=True, slots=True)
class Chunk:
    """One large-chunk test-time update's worth of data.

    ``harmful`` is ground truth from the generator, never a label anything
    downstream is allowed to read before it is being scored against.
    ``collides_with`` names the earlier chunk whose association this one
    overwrites, when the mechanism is a collision; it is the finest-grained
    ground truth available and the recall test uses it.
    """

    t: int
    keys: Tensor
    values: Tensor
    task_id: int = BASE_TASK
    harmful: bool = False
    collides_with: int | None = None

    @property
    def n_rows(self) -> int:
        return int(self.keys.shape[0])


@dataclass(frozen=True, slots=True)
class SyntheticStream:
    """A chunked stream plus the ground truth that makes it a known-answer test."""

    name: str
    d_in: int
    d_out: int
    chunks: tuple[Chunk, ...]
    dtype: torch.dtype = torch.float64
    notes: str = ""
    _meta: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def harmful_mask(self) -> npt.NDArray[np.bool_]:
        return np.array([c.harmful for c in self.chunks], dtype=bool)

    @property
    def harmful_indices(self) -> npt.NDArray[np.int64]:
        return np.flatnonzero(self.harmful_mask).astype(np.int64)


def _normal(scope: SeedScope, purpose: str, rows: int, cols: int, *, tick: int = 0) -> FloatArray:
    """``(rows, cols)`` standard normals, one lane per row.

    Row-per-lane rather than one flat draw so that adding rows later leaves the
    earlier rows' values untouched — the same common-random-number discipline
    the rest of the repo runs on.
    """
    st = scope.stream(purpose)
    return np.asarray(st.normal(np.arange(rows, dtype=np.int64), cols, tick=tick), dtype=np.float64)


def _unit_rows(a: FloatArray) -> FloatArray:
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def drifting_regression_stream(
    *,
    d_in: int = 6,
    d_out: int = 4,
    n_chunks: int = 40,
    chunk_size: int = 8,
    drift: float = 0.04,
    conflict_every: int = 7,
    conflict_offset: int = 3,
    noise: float = 0.0,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> SyntheticStream:
    """``y = A_t x`` with slow drift, punctuated by sign-flipped conflict chunks.

    ``drift`` sets how fast ``A_t`` rotates; with ``drift = 0`` there is nothing
    to adapt to and every write is a waste, which makes the frontier degenerate.
    A nonzero drift is what puts a real plasticity term on the other side of the
    trade-off.

    The conflicting task is ``-A_t`` rather than an unrelated random map. An
    unrelated map is mostly *orthogonal* to the base task in high dimension and
    so does surprisingly little damage; the sign flip guarantees that fitting it
    moves the memory directly away from every retained association. That is a
    deliberately easy case: it makes the known-answer test decisive, and it
    means near-perfect recall here is a *necessary*, not sufficient, condition
    on the oracle.

    ``noise`` adds label noise. Zero by default: phase zero is about whether the
    oracle can see a harm that is definitely there, and observation noise is a
    separate question that deserves its own sweep.
    """
    if conflict_every < 2:
        raise ValueError("conflict_every must be >= 2 or every chunk conflicts")
    scope = SeedScope(root_seed=seed).child("stream", "drift-regression")

    A0 = _normal(scope, "teacher-A0", d_out, d_in) / math.sqrt(d_in)
    D = _normal(scope, "teacher-drift", d_out, d_in) / math.sqrt(d_in)
    X = _normal(scope, "inputs", n_chunks * chunk_size, d_in).reshape(n_chunks, chunk_size, d_in)
    E = _normal(scope, "label-noise", n_chunks * chunk_size, d_out).reshape(
        n_chunks, chunk_size, d_out
    )

    chunks: list[Chunk] = []
    for t in range(n_chunks):
        A_t = A0 + drift * t * D
        conflicting = t % conflict_every == conflict_offset
        teacher = -A_t if conflicting else A_t
        y = X[t] @ teacher.T + noise * E[t]
        chunks.append(
            Chunk(
                t=t,
                keys=torch.from_numpy(np.ascontiguousarray(X[t])).to(dtype),
                values=torch.from_numpy(np.ascontiguousarray(y)).to(dtype),
                task_id=CONFLICT_TASK if conflicting else BASE_TASK,
                harmful=conflicting,
            )
        )

    return SyntheticStream(
        name="drifting-regression",
        d_in=d_in,
        d_out=d_out,
        chunks=tuple(chunks),
        dtype=dtype,
        notes=(
            "Harm mechanism: conflicting chunks teach -A_t, so a write moves the memory "
            "directly away from every retained base-task association."
        ),
        _meta={"drift": drift, "conflict_every": conflict_every},
    )


def associative_recall_stream(
    *,
    d_in: int = 8,
    d_out: int = 6,
    n_chunks: int = 40,
    pairs_per_chunk: int = 4,
    collide_every: int = 6,
    collide_offset: int = 4,
    min_collision_age: int = 8,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> SyntheticStream:
    """Distinct ``(key, value)`` writes, punctuated by colliding overwrites.

    A colliding chunk repeats a key first written at least ``min_collision_age``
    chunks ago and pairs it with a fresh, unrelated value. Writing it is not
    "large" and not "surprising" in any global sense — the memory has never seen
    this value before, so surprise is high, but so it is for every genuinely new
    association. What distinguishes the collision is *which* aged probe it
    destroys, which is exactly the thing a cheap pre-write feature cannot see
    and the oracle can.

    The age floor matters. A collision against a key written one chunk ago is
    barely retention at all; the fast weights have hardly consolidated it, so
    the measured damage is small and the case is uninteresting. Requiring real
    age is what makes this a retention test.
    """
    scope = SeedScope(root_seed=seed).child("stream", "assoc-recall")

    n_pairs = n_chunks * pairs_per_chunk
    keys = _unit_rows(_normal(scope, "keys", n_pairs, d_in))
    values = _unit_rows(_normal(scope, "values", n_pairs, d_out))
    alt_values = _unit_rows(_normal(scope, "alt-values", n_chunks, d_out))

    chunks: list[Chunk] = []
    for t in range(n_chunks):
        sl = slice(t * pairs_per_chunk, (t + 1) * pairs_per_chunk)
        k = keys[sl].copy()
        v = values[sl].copy()
        colliding = t % collide_every == collide_offset and t >= min_collision_age
        victim: int | None = None
        if colliding:
            # Overwrite the *first* slot with an aged key, keeping chunk size and
            # therefore the per-chunk compute identical to a benign chunk.
            victim = t - min_collision_age
            k[0] = keys[victim * pairs_per_chunk]
            v[0] = alt_values[t]
        chunks.append(
            Chunk(
                t=t,
                keys=torch.from_numpy(np.ascontiguousarray(k)).to(dtype),
                values=torch.from_numpy(np.ascontiguousarray(v)).to(dtype),
                task_id=BASE_TASK,
                harmful=colliding,
                collides_with=victim,
            )
        )

    return SyntheticStream(
        name="associative-recall",
        d_in=d_in,
        d_out=d_out,
        chunks=tuple(chunks),
        dtype=dtype,
        notes=(
            "Harm mechanism: a colliding chunk rebinds an aged key to a new value, "
            "destroying one named earlier association and nothing else."
        ),
        _meta={"pairs_per_chunk": pairs_per_chunk, "min_collision_age": min_collision_age},
    )
