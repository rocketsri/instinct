"""Counter-based common-random-number (CRN) machinery.

This is the enabler for the whole batched design, so it is worth being explicit
about why it looks the way it does.

Every measurement in this repo is a *paired* comparison between counterfactual
arms that differ only in an intervention: plan with budget ``k`` versus ``k'``,
apply a fast-weight update versus don't, take reflex action ``a`` versus ``a'``.
Pairing is what makes the differences resolvable at feasible sample sizes, and
pairing requires that corresponding lanes in different arms see *identical*
environment noise.

A sequential PRNG cannot deliver that once arms are batched. If arm A steps
lanes ``[0, 1, 2]`` and arm B steps only the survivors ``[0, 2]``, a sequential
stream hands lane 2 different draws in the two arms, and the pairing silently
breaks. Nothing crashes; the variance just quietly inflates and the estimator
becomes wrong in a way that looks like noise.

So randomness here is **counter-based**: a draw is a pure function

    bits = F(key(scope, purpose), counter(episode, tick, lane, index))

with no mutable state anywhere. Lane 2's noise is lane 2's noise regardless of
which arm asks, what order it asks in, or which other lanes are present. That
property is what lets :mod:`instinct.core.rollout` simulate a shared prefix once
and fork it, and what lets the oracle in :mod:`instinct.ttt.oracle` evaluate
many forks in one batched pass.

The generator is Threefry-2x64-20 (Salmon et al., *Parallel Random Numbers: As
Easy as 1, 2, 3*), the same construction JAX uses. It is add/xor/rotate only, so
it vectorizes in plain NumPy over a whole ``(lanes, draws)`` array at once — no
Python loop over lanes, which is the entire point.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt

__all__ = [
    "SeedScope",
    "Stream",
    "threefry2x64",
]

U64 = np.uint64
UInt64Array = npt.NDArray[np.uint64]
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# Threefry-2x64 constants (Random123).
_PARITY = U64(0x1BD11BDAA9FC1A22)
_ROTATIONS = (U64(16), U64(42), U64(12), U64(31), U64(16), U64(32), U64(24), U64(21))
_ROUNDS = 20
_SIXTY_FOUR = U64(64)

# Bit budget for each field packed into the 128-bit counter.
_FIELD_BITS = U64(32)
_FIELD_MAX = 1 << 32


def _rotl(x: UInt64Array, r: U64) -> UInt64Array:
    """Rotate left within 64 bits. NumPy shifts need uint64 operands throughout."""
    return ((x << r) | (x >> (_SIXTY_FOUR - r))).astype(np.uint64)


def threefry2x64(
    key0: U64,
    key1: U64,
    ctr0: UInt64Array | U64,
    ctr1: UInt64Array | U64,
) -> tuple[UInt64Array, UInt64Array]:
    """Vectorized Threefry-2x64-20.

    ``ctr0``/``ctr1`` are arbitrarily shaped (broadcast-compatible) uint64
    arrays; the return is two uint64 arrays of the broadcast shape, i.e. 128
    pseudorandom bits per counter value. Unsigned overflow wraps, which is the
    defined behaviour the cipher relies on.
    """
    ks0, ks1 = U64(key0), U64(key1)
    ks2 = ks0 ^ ks1 ^ _PARITY
    schedule = (ks0, ks1, ks2)

    x0 = (np.asarray(ctr0, dtype=np.uint64) + ks0).astype(np.uint64)
    x1 = (np.asarray(ctr1, dtype=np.uint64) + ks1).astype(np.uint64)
    x0, x1 = np.broadcast_arrays(x0, x1)
    x0, x1 = x0.astype(np.uint64, copy=True), x1.astype(np.uint64, copy=True)

    for rnd in range(_ROUNDS):
        x0 = (x0 + x1).astype(np.uint64)
        x1 = _rotl(x1, _ROTATIONS[rnd % 8])
        x1 = x1 ^ x0
        if (rnd + 1) % 4 == 0:
            j = (rnd + 1) // 4
            x0 = (x0 + schedule[j % 3]).astype(np.uint64)
            x1 = (x1 + schedule[(j + 1) % 3] + U64(j)).astype(np.uint64)

    return x0, x1


@lru_cache(maxsize=4096)
def _derive_key(root_seed: int, path: tuple[str, ...], purpose: str) -> tuple[U64, U64]:
    """Hash a scope path to a 128-bit key.

    BLAKE2b rather than :func:`hash` because the latter is salted per process:
    a run would not reproduce across invocations, which would defeat the point.
    Cached because the key depends only on static scope identity — episode and
    tick live in the counter, so this is not on the per-step hot path.
    """
    canonical = "\x1f".join((str(root_seed), *path, purpose)).encode("utf-8")
    digest = hashlib.blake2b(canonical, digest_size=16).digest()
    return (
        U64(int.from_bytes(digest[:8], "little")),
        U64(int.from_bytes(digest[8:], "little")),
    )


def _check_field(value: int, name: str) -> None:
    if not 0 <= value < _FIELD_MAX:
        raise ValueError(f"{name}={value} outside the packable range [0, 2**32)")


@dataclass(frozen=True, slots=True)
class Stream:
    """A named, stateless source of randomness bound to one scope and purpose.

    All draw methods take ``lanes`` (an integer array of *lane identities*, not
    positions) and return an array whose leading axis matches. Two calls that
    share a lane identity return the same numbers for that lane, whatever else
    is in the batch — that is the CRN guarantee the estimators rely on.
    """

    key0: U64
    key1: U64
    purpose: str

    # -- raw bits ---------------------------------------------------------

    def bits(
        self,
        lanes: IntArray | int,
        count: int = 1,
        *,
        episode: int = 0,
        tick: int = 0,
        offset: int = 0,
    ) -> UInt64Array:
        """Return ``(n_lanes, count)`` uint64 words.

        The counter packs ``(episode, tick)`` into its low word and
        ``(lane, draw_index)`` into its high word, so distinct draw sites never
        collide and every field is addressable in O(1).
        """
        _check_field(episode, "episode")
        _check_field(tick, "tick")
        if count < 0:
            raise ValueError(f"count={count} must be non-negative")

        lane_arr = np.atleast_1d(np.asarray(lanes, dtype=np.int64))
        if lane_arr.ndim != 1:
            raise ValueError(f"lanes must be 1-D, got shape {lane_arr.shape}")
        if lane_arr.size and (lane_arr.min() < 0 or lane_arr.max() >= _FIELD_MAX):
            raise ValueError("lane ids must lie in [0, 2**32)")

        ctr0 = U64((episode << 32) | tick)
        # Threefry emits two words per counter, so we need half as many counters.
        n_words = count + (count & 1)
        idx = np.arange(offset // 2, offset // 2 + n_words // 2, dtype=np.uint64)
        lane_hi = (lane_arr.astype(np.uint64) << _FIELD_BITS)
        ctr1 = lane_hi[:, None] | idx[None, :]

        w0, w1 = threefry2x64(self.key0, self.key1, ctr0, ctr1)
        # Interleave the two output words so `count` and `offset` index a single
        # flat sequence per lane.
        out = np.empty((lane_arr.size, n_words), dtype=np.uint64)
        out[:, 0::2] = w0
        out[:, 1::2] = w1
        return out[:, :count]

    # -- distributions ----------------------------------------------------

    def uniform(
        self,
        lanes: IntArray | int,
        count: int = 1,
        *,
        episode: int = 0,
        tick: int = 0,
        offset: int = 0,
        low: float = 0.0,
        high: float = 1.0,
    ) -> FloatArray:
        """Uniform on ``[low, high)``, shape ``(n_lanes, count)``."""
        raw = self.bits(lanes, count, episode=episode, tick=tick, offset=offset)
        # Top 53 bits give an exactly-representable float64 in [0, 1).
        unit = (raw >> U64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)
        return low + (high - low) * unit

    def normal(
        self,
        lanes: IntArray | int,
        count: int = 1,
        *,
        episode: int = 0,
        tick: int = 0,
        offset: int = 0,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> FloatArray:
        """Standard normal via Box-Muller, shape ``(n_lanes, count)``.

        Box-Muller rather than the faster ziggurat on purpose: ziggurat uses
        rejection, so the number of uniforms a lane consumes depends on the
        values it drew. That would make a lane's noise depend on its own history
        and break the O(1) addressability the whole design rests on. Box-Muller
        consumes exactly two uniforms per pair of normals, always.
        """
        n_pairs = (count + 1) // 2
        u = self.uniform(lanes, 2 * n_pairs, episode=episode, tick=tick, offset=offset)
        u1 = np.maximum(u[:, 0::2], np.finfo(np.float64).tiny)  # log(0) guard
        u2 = u[:, 1::2]
        radius = np.sqrt(-2.0 * np.log(u1))
        angle = 2.0 * np.pi * u2
        pair = np.empty((u.shape[0], 2 * n_pairs), dtype=np.float64)
        pair[:, 0::2] = radius * np.cos(angle)
        pair[:, 1::2] = radius * np.sin(angle)
        return loc + scale * pair[:, :count]

    def integers(
        self,
        lanes: IntArray | int,
        high: int,
        count: int = 1,
        *,
        episode: int = 0,
        tick: int = 0,
        offset: int = 0,
    ) -> IntArray:
        """Uniform integers on ``[0, high)``, shape ``(n_lanes, count)``.

        Lemire multiply-shift on the top 32 bits: branch-free, so it stays
        vectorized and keeps one word per draw. The bias is bounded by
        ``high / 2**32`` (below 1e-6 for any ``high`` we use); rejection
        sampling would remove it but reintroduce data-dependent draw counts,
        which costs more than the bias does.
        """
        if not 0 < high < _FIELD_MAX:
            raise ValueError(f"high={high} must lie in (0, 2**32)")
        raw = self.bits(lanes, count, episode=episode, tick=tick, offset=offset)
        return ((raw >> U64(32)) * U64(high) >> U64(32)).astype(np.int64)

    def categorical(
        self,
        lanes: IntArray | int,
        probs: FloatArray,
        *,
        episode: int = 0,
        tick: int = 0,
        offset: int = 0,
    ) -> IntArray:
        """Sample one category per lane by inverse CDF.

        ``probs`` is ``(n_categories,)`` shared across lanes, or
        ``(n_lanes, n_categories)`` per lane. Returns shape ``(n_lanes,)``.
        """
        p = np.asarray(probs, dtype=np.float64)
        u = self.uniform(lanes, 1, episode=episode, tick=tick, offset=offset)[:, 0]
        if p.ndim == 1:
            cdf = np.cumsum(p)
            return np.searchsorted(cdf / cdf[-1], u, side="right").astype(np.int64)
        cdf = np.cumsum(p, axis=1)
        cdf = cdf / cdf[:, -1:]
        # One searchsorted per row would loop; compare against the whole CDF instead.
        return (u[:, None] >= cdf).sum(axis=1).astype(np.int64)

    def spawn_generator(self, lane: int, *, episode: int = 0, tick: int = 0) -> np.random.Generator:
        """A stock NumPy generator seeded from this stream.

        Escape hatch for scalar reference implementations and for third-party
        code that insists on a ``Generator``. The fast paths do not use it, but
        seeding it from the same counter keeps a naive reference reproducible
        alongside its optimized counterpart.
        """
        seed = self.bits(np.array([lane]), 4, episode=episode, tick=tick)[0]
        return np.random.default_rng([int(w) for w in seed])


@dataclass(frozen=True, slots=True)
class SeedScope:
    """A hierarchical namespace for randomness.

    Scopes are values, not stateful objects: ``child`` returns a new scope and
    mutates nothing, so handing a scope to a subroutine cannot perturb the
    caller's stream. Build one per run and derive everything from it.

    >>> root = SeedScope(root_seed=0)
    >>> env = root.child("gridworld", "arm=actual").stream("transition")
    >>> a = env.normal(lanes=np.array([0, 1, 2]), count=3, tick=5)
    >>> b = env.normal(lanes=np.array([2, 0]), count=3, tick=5)
    >>> bool(np.allclose(a[2], b[0]) and np.allclose(a[0], b[1]))
    True
    """

    root_seed: int
    path: tuple[str, ...] = ()

    def child(self, *names: str) -> SeedScope:
        return SeedScope(self.root_seed, self.path + tuple(names))

    def stream(self, purpose: str) -> Stream:
        """A stream for a named draw site.

        ``purpose`` separates independent sources within a scope — environment
        transitions, planner tie-breaks, disturbance draws — so that changing
        how many samples the planner takes cannot shift the environment's noise.
        Keeping these decoupled is what makes an intervention the *only*
        difference between two arms.
        """
        key0, key1 = _derive_key(self.root_seed, self.path, purpose)
        return Stream(key0=key0, key1=key1, purpose=purpose)

    def label(self) -> str:
        return "/".join(self.path) if self.path else "<root>"
