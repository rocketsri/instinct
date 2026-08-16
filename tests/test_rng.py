"""Counter-based CRN guarantees.

The properties here are the ones the estimators actually depend on. In
particular :func:`test_lane_identity_survives_batch_composition` is the reason
the whole batched design is safe: if it ever fails, every paired comparison in
the repo silently loses its pairing and the variance inflation looks like noise
rather than a bug.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from instinct.core.rng import SeedScope, threefry2x64


@pytest.fixture
def stream():
    return SeedScope(root_seed=20260816).child("env", "arm=actual").stream("transition")


# -- the CRN guarantee ----------------------------------------------------


def test_lane_identity_survives_batch_composition(stream) -> None:
    """A lane's draws depend on its identity, not its position in the batch.

    This is the property a sequential PRNG cannot provide, and the reason for
    the counter-based design. Arms that step different lane subsets, in
    different orders, must still agree lane-for-lane.
    """
    full = stream.normal(np.array([0, 1, 2, 3, 4]), count=6, tick=9)
    shuffled = stream.normal(np.array([4, 2, 0]), count=6, tick=9)
    for batch_pos, lane in enumerate([4, 2, 0]):
        assert np.allclose(shuffled[batch_pos], full[lane]), f"lane {lane} drifted"


def test_singleton_matches_batch(stream) -> None:
    """Asking for one lane must equal asking for many and indexing in."""
    batch = stream.uniform(np.arange(64), count=3, tick=2)
    for lane in (0, 17, 63):
        assert np.allclose(stream.uniform(np.array([lane]), count=3, tick=2)[0], batch[lane])


def test_draws_are_stateless_and_repeatable(stream) -> None:
    a = stream.normal(np.arange(10), count=4, episode=2, tick=3)
    _ = stream.uniform(np.arange(999), count=50)  # interleaved unrelated work
    b = stream.normal(np.arange(10), count=4, episode=2, tick=3)
    assert np.array_equal(a, b)


def test_scope_is_reproducible_across_construction() -> None:
    """Keys come from BLAKE2b, not Python's per-process-salted hash()."""
    mk = lambda: SeedScope(7).child("a", "b").stream("noise")  # noqa: E731
    assert np.array_equal(mk().bits(np.arange(4), 4), mk().bits(np.arange(4), 4))


# -- independence between draw sites --------------------------------------


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (dict(episode=0, tick=0), dict(episode=0, tick=1)),
        (dict(episode=0, tick=0), dict(episode=1, tick=0)),
        (dict(episode=3, tick=5), dict(episode=5, tick=3)),
    ],
)
def test_distinct_counters_decorrelate(stream, left, right) -> None:
    a = stream.normal(np.arange(3000), count=1, **left)[:, 0]
    b = stream.normal(np.arange(3000), count=1, **right)[:, 0]
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.05


def test_purposes_are_independent() -> None:
    """Changing how many samples the planner draws must not move the environment.

    Separate purposes are what keep an intervention the *only* difference
    between two arms; if they shared a stream, altering planner behaviour would
    perturb the environment's noise and destroy the pairing.
    """
    scope = SeedScope(1).child("env")
    a = scope.stream("transition").normal(np.arange(3000), count=1)[:, 0]
    b = scope.stream("planner_tiebreak").normal(np.arange(3000), count=1)[:, 0]
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.05


def test_scope_paths_are_independent() -> None:
    a = SeedScope(1).child("gridworld").stream("t").normal(np.arange(3000))[:, 0]
    b = SeedScope(1).child("arcade").stream("t").normal(np.arange(3000))[:, 0]
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.05


def test_root_seeds_are_independent() -> None:
    a = SeedScope(0).child("e").stream("t").normal(np.arange(3000))[:, 0]
    b = SeedScope(1).child("e").stream("t").normal(np.arange(3000))[:, 0]
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.05


# -- distributional correctness -------------------------------------------


def test_uniform_passes_kolmogorov_smirnov(stream) -> None:
    u = stream.uniform(np.arange(2000), count=20).ravel()
    assert stats.kstest(u, "uniform").pvalue > 0.001
    assert u.min() >= 0.0 and u.max() < 1.0


def test_normal_passes_kolmogorov_smirnov(stream) -> None:
    n = stream.normal(np.arange(2000), count=20, tick=1).ravel()
    assert stats.kstest(n, "norm").pvalue > 0.001


@pytest.mark.parametrize("high", [2, 3, 7, 256])
def test_integers_are_uniform_and_in_range(stream, high) -> None:
    draws = stream.integers(np.arange(4000), high, count=10, tick=2)
    assert draws.min() >= 0 and draws.max() < high
    counts = np.bincount(draws.ravel(), minlength=high)
    assert stats.chisquare(counts).pvalue > 0.001


def test_categorical_respects_its_weights(stream) -> None:
    probs = np.array([0.1, 0.6, 0.3])
    draws = stream.categorical(np.arange(30_000), probs, tick=4)
    freq = np.bincount(draws, minlength=3) / draws.size
    assert np.allclose(freq, probs, atol=0.01)


def test_categorical_accepts_per_lane_weights(stream) -> None:
    """Per-lane weights are needed for batched stochastic policies."""
    n = 20_000
    probs = np.tile(np.array([[0.8, 0.2], [0.2, 0.8]]), (n // 2, 1))
    draws = stream.categorical(np.arange(n), probs, tick=5)
    assert abs(draws[0::2].mean() - 0.2) < 0.02
    assert abs(draws[1::2].mean() - 0.8) < 0.02


# -- the underlying cipher ------------------------------------------------


def test_threefry_avalanche() -> None:
    """One counter bit flipped should change about half the output bits."""
    base = np.zeros(64, dtype=np.uint64)
    flips = (np.uint64(1) << np.arange(64, dtype=np.uint64))
    w0, w1 = threefry2x64(np.uint64(0xDEAD), np.uint64(0xBEEF), base, base)
    f0, f1 = threefry2x64(np.uint64(0xDEAD), np.uint64(0xBEEF), base, flips)
    changed = [
        bin(int(a) ^ int(b)).count("1")
        for a, b in zip(np.concatenate([w0, w1]), np.concatenate([f0, f1]))
    ]
    assert 24 < float(np.mean(changed)) < 40


def test_threefry_is_injective_over_a_counter_range() -> None:
    n = 40_000
    ctr = np.arange(n, dtype=np.uint64)
    w0, _ = threefry2x64(np.uint64(3), np.uint64(4), np.zeros(n, dtype=np.uint64), ctr)
    assert np.unique(w0).size == n


def test_threefry_broadcasts() -> None:
    ctr0 = np.arange(3, dtype=np.uint64)[:, None]
    ctr1 = np.arange(5, dtype=np.uint64)[None, :]
    w0, w1 = threefry2x64(np.uint64(1), np.uint64(1), ctr0, ctr1)
    assert w0.shape == (3, 5) and w1.shape == (3, 5)


# -- addressing -----------------------------------------------------------


def test_offset_indexes_one_contiguous_sequence_per_lane(stream) -> None:
    """`offset` must address a flat per-lane sequence, so callers can resume."""
    whole = stream.bits(np.array([5]), 8, tick=3)[0]
    for start in (0, 2, 4, 6):
        part = stream.bits(np.array([5]), 2, tick=3, offset=start)[0]
        assert np.array_equal(part, whole[start : start + 2])


def test_odd_counts_are_exact(stream) -> None:
    """Threefry emits words in pairs; an odd request must not leak the spare."""
    assert stream.bits(np.array([1]), 3).shape == (1, 3)
    assert np.array_equal(stream.bits(np.array([1]), 3)[0], stream.bits(np.array([1]), 4)[0, :3])


def test_zero_count_is_empty(stream) -> None:
    assert stream.bits(np.array([1]), 0).shape == (1, 0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(lanes=np.array([-1]), count=1), "lane ids"),
        (dict(lanes=np.array([0]), count=1, episode=-1), "episode"),
        (dict(lanes=np.array([0]), count=1, tick=2**32), "tick"),
        (dict(lanes=np.zeros((2, 2), dtype=np.int64), count=1), "1-D"),
    ],
)
def test_out_of_range_addressing_is_rejected(stream, kwargs, match) -> None:
    """Silent counter wraparound would alias two draw sites onto one stream."""
    with pytest.raises(ValueError, match=match):
        stream.bits(**kwargs)


def test_integers_rejects_impossible_bounds(stream) -> None:
    with pytest.raises(ValueError, match="high"):
        stream.integers(np.array([0]), 0)


def test_spawn_generator_is_deterministic(stream) -> None:
    a = stream.spawn_generator(3, tick=1).normal(size=5)
    b = stream.spawn_generator(3, tick=1).normal(size=5)
    c = stream.spawn_generator(4, tick=1).normal(size=5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
