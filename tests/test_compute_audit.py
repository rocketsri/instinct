"""Compute accounting and audit plumbing: known answers and fast-path equality.

These are the last two modules from the parallel-worker batch that had no
known-answer coverage. Neither feeds a headline number directly, but both gate
whether headline numbers are *admissible*: matched-compute is rule 4, and the
blind-spot detector is what stops a learned trigger from being evaluated only on
the decisions it chose to make.

Each module ships a ``_naive`` reference, so rule 7 is checkable here rather than
merely asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from instinct.audit.blindspot import (
    _nearest_audit_distance_naive,
    detect_blindspots,
    nearest_audit_distance,
    quantile_cells,
)
from instinct.audit.counterfactual import (
    CounterfactualError,
    assert_unchanged,
    fork,
    fork_scope,
    states_equal,
)
from instinct.audit.propensity import (
    PropensityError,
    _doubly_robust_naive,
    _ips_naive,
    doubly_robust,
    effective_sample_size,
    importance_weights,
    ips,
    self_normalized_ips,
)
from instinct.compute.budget import (
    ComputeLedger,
    ComputeMismatchError,
    _extremes,
    _extremes_naive,
    assert_matched,
    check_matched,
)

# -- matched compute (rule 4) --------------------------------------------


def test_matched_arms_pass_and_mismatched_arms_raise() -> None:
    """The whole point: a comparison must refuse to render when arms diverge."""
    led = ComputeLedger("t")
    led.record("actual", wall_clock_s=1.00, simulations=512)
    led.record("fresh", wall_clock_s=1.02, simulations=512)
    assert check_matched(led, rtol=0.05).matched
    assert_matched(led, rtol=0.05)

    led.record("fresh", wall_clock_s=5.0)  # now wildly ahead
    assert not check_matched(led, rtol=0.05).matched
    with pytest.raises(ComputeMismatchError):
        assert_matched(led, rtol=0.05)


def test_the_cheap_base_arm_can_be_excluded() -> None:
    """`base` is *supposed* to be cheaper; including it would fail for the wrong reason."""
    led = ComputeLedger("t")
    led.record("actual", wall_clock_s=1.0)
    led.record("fresh", wall_clock_s=1.0)
    led.record("base", wall_clock_s=0.05)
    assert not check_matched(led, rtol=0.05).matched
    assert check_matched(led, rtol=0.05, arms=["actual", "fresh"]).matched


def test_wall_clock_is_the_default_basis() -> None:
    """The T4 profile showed a 64x FLOP swing at constant wall-clock.

    So arms that agree on wall-clock and differ hugely in FLOPs must pass by
    default, and the mismatch must be visible when FLOPs are asked for
    explicitly. Getting this backwards would make the accounting reject correct
    comparisons and accept incorrect ones in exactly the regime we measured.
    """
    led = ComputeLedger("t")
    led.record("a", wall_clock_s=1.0, flops=10**6)
    led.record("b", wall_clock_s=1.0, flops=64 * 10**6)
    assert check_matched(led, rtol=0.05).matched
    assert not check_matched(led, basis="flops", rtol=0.05).matched


def test_a_single_arm_is_trivially_matched() -> None:
    led = ComputeLedger("t")
    led.record("only", wall_clock_s=1.0)
    assert check_matched(led).matched


def test_unknown_arm_is_an_error_not_a_silent_pass() -> None:
    led = ComputeLedger("t")
    led.record("a", wall_clock_s=1.0)
    with pytest.raises(KeyError):
        check_matched(led, arms=["a", "ghost"])


def test_extremes_fast_path_matches_reference() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        vals = list(rng.normal(size=int(rng.integers(2, 12))))
        assert _extremes(vals) == _extremes_naive(vals)


# -- off-policy estimation ------------------------------------------------


def test_ips_is_unbiased_when_the_log_covers_the_target() -> None:
    """Known answer: uniform logging, deterministic target, analytic mean."""
    rng = np.random.default_rng(1)
    n = 20_000
    # Two actions; reward is 1.0 for action 1 and 0.0 for action 0.
    action = rng.integers(0, 2, size=n)
    rewards = action.astype(np.float64)
    behavior = np.full(n, 0.5)
    # Target always picks action 1, so its true value is exactly 1.0.
    target = np.where(action == 1, 1.0, 0.0)
    est = ips(rewards, behavior, target, n_boot=200)
    assert abs(est.value - 1.0) < 0.05, est.value


def test_snips_matches_ips_when_weights_are_balanced() -> None:
    rng = np.random.default_rng(2)
    n = 5_000
    rewards = rng.normal(1.0, 0.2, size=n)
    probs = np.full(n, 0.5)
    a = ips(rewards, probs, probs, n_boot=200).value
    b = self_normalized_ips(rewards, probs, probs, n_boot=200).value
    assert abs(a - b) < 1e-9


def test_doubly_robust_beats_ips_when_the_model_is_good() -> None:
    """DR should inherit the model's accuracy, not the weights' variance."""
    rng = np.random.default_rng(3)
    n = 4_000
    action = rng.integers(0, 2, size=n)
    truth = action.astype(np.float64)
    rewards = truth + rng.normal(0, 0.05, size=n)
    behavior = np.full(n, 0.5)
    target = np.where(action == 1, 1.0, 0.0)
    q_hat = np.stack([np.zeros(n), np.ones(n)], axis=1)  # a near-perfect model
    target_dist = np.stack([np.zeros(n), np.ones(n)], axis=1)  # always action 1

    dr = doubly_robust(rewards, action, behavior, target_dist, q_hat, n_boot=200)
    plain = ips(rewards, behavior, target, n_boot=200)
    assert abs(dr.value - 1.0) <= abs(plain.value - 1.0) + 1e-9


def test_zero_propensity_is_rejected() -> None:
    """No support means no estimate. Silently clipping would fake coverage."""
    with pytest.raises(PropensityError):
        importance_weights(np.array([0.0, 0.5]), np.array([0.5, 0.5]))


def test_effective_sample_size_endpoints() -> None:
    equal = np.ones(100)
    assert effective_sample_size(equal) == pytest.approx(100.0)
    degenerate = np.zeros(100)
    degenerate[0] = 1.0
    assert effective_sample_size(degenerate) == pytest.approx(1.0)


def test_propensity_fast_paths_match_their_references() -> None:
    rng = np.random.default_rng(5)
    n = 300
    rewards = rng.normal(size=n)
    behavior = rng.uniform(0.2, 0.8, size=n)
    target = rng.uniform(0.2, 0.8, size=n)
    assert ips(rewards, behavior, target, n_boot=1).value == pytest.approx(
        _ips_naive(rewards, behavior, target), abs=1e-10
    )

    action = rng.integers(0, 2, size=n)
    q_hat = rng.normal(size=(n, 2))
    target_dist = rng.uniform(0.2, 0.8, size=(n, 2))
    target_dist /= target_dist.sum(axis=1, keepdims=True)
    fast = doubly_robust(rewards, action, behavior, target_dist, q_hat, n_boot=1).value
    assert fast == pytest.approx(
        _doubly_robust_naive(rewards, action, behavior, target_dist, q_hat), abs=1e-10
    )


# -- selective-label blind spots ------------------------------------------


def test_a_decline_region_with_no_audits_is_flagged() -> None:
    """The failure mode the module exists for.

    A trigger that always declines one region never learns what it would have
    got there. If that is not flagged, the trigger looks well-calibrated on
    exactly the data it chose to generate.
    """
    x = np.linspace(0.0, 1.0, 200).reshape(-1, 1)
    declined = x[:, 0] > 0.5
    audited = np.zeros(200, dtype=bool)
    audited[:20] = True  # audits only in the accepted region

    report = detect_blindspots(x, declined, audited, n_bins=4)
    assert report.flagged
    assert report.n_cells_uncovered > 0


def test_audits_spread_across_the_decline_region_clear_the_flag() -> None:
    """The positive control: the detector must be able to *not* fire."""
    x = np.linspace(0.0, 1.0, 200).reshape(-1, 1)
    declined = x[:, 0] > 0.5
    audited = np.zeros(200, dtype=bool)
    audited[declined] = np.arange(int(declined.sum())) % 3 == 0

    report = detect_blindspots(x, declined, audited, n_bins=4)
    assert not report.flagged, report.n_cells_uncovered


def test_accepted_audits_do_not_count_as_decline_coverage() -> None:
    """An audit only informs the decline region if the trigger declined it."""
    x = np.linspace(0.0, 1.0, 120).reshape(-1, 1)
    declined = x[:, 0] > 0.5
    accepted_audits = (~declined).copy()  # every accepted record observed
    report = detect_blindspots(x, declined, accepted_audits, n_bins=4)
    assert report.flagged


def test_nearest_audit_distance_matches_reference() -> None:
    rng = np.random.default_rng(7)
    # Both arguments are coordinate matrices: the declined points and the
    # audited points, not a matrix plus a mask.
    declined_pts = rng.normal(size=(120, 3))
    audited_pts = rng.normal(size=(30, 3))
    fast = nearest_audit_distance(declined_pts, audited_pts)
    slow = _nearest_audit_distance_naive(declined_pts, audited_pts)
    assert np.allclose(fast, slow, atol=1e-10)


def test_quantile_cells_are_balanced() -> None:
    # Returns (n_records, n_features) bin indices, one column per covariate.
    x = np.linspace(0.0, 1.0, 100).reshape(-1, 1)
    cells = quantile_cells(x, 4)
    assert cells.shape == (100, 1)
    counts = np.bincount(cells[:, 0], minlength=4)
    assert counts.min() >= 20 and counts.sum() == 100


# -- exact counterfactual forking -----------------------------------------


def test_fork_is_independent_of_its_source() -> None:
    state = {"w": np.ones(4), "n": 3}
    branch = fork(state)
    branch["w"] += 1.0
    branch["n"] = 99
    assert np.allclose(state["w"], 1.0), "the fork aliased its source"
    assert state["n"] == 3


def test_states_equal_is_exact_not_approximate() -> None:
    a = {"w": np.ones(3)}
    b = {"w": np.ones(3)}
    assert states_equal(a, b)
    b["w"][0] += 1e-12
    assert not states_equal(a, b), "a tiny leak must still count as a change"


def test_fork_scope_verifies_the_original_survived() -> None:
    state = {"w": np.zeros(3)}
    with fork_scope(state) as branch:
        branch["w"] += 5.0
    assert np.allclose(state["w"], 0.0)


def test_assert_unchanged_catches_a_mutation() -> None:
    before = {"w": np.zeros(2)}
    after = {"w": np.array([0.0, 1.0])}
    with pytest.raises(CounterfactualError):
        assert_unchanged(before, after)
