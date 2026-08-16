"""Statistics layer: coverage, pairing discipline, and anytime validity.

Two tests here are about the guarantees themselves rather than the code:
:func:`test_bca_interval_covers_at_the_nominal_rate` checks the bootstrap
actually covers, and :func:`test_confidence_sequence_survives_optional_stopping`
checks that the sequential machinery does not break under the peeking that the
sweep's adaptive allocation performs constantly.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from instinct.core.confseq import (
    ConfidenceSequence,
    empirical_bernstein_radius,
    normal_mixture_radius,
)
from instinct.core.stats import (
    control_variate,
    kendall_tau,
    paired_bootstrap_ci,
    paired_permutation_test,
)

# -- pairing discipline ---------------------------------------------------


def test_mismatched_arms_are_rejected() -> None:
    """Unpaired data must error, not silently produce a wider interval."""
    with pytest.raises(ValueError, match="not paired"):
        paired_bootstrap_ci(np.zeros(10), np.zeros(9))


def test_two_dimensional_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="1-D"):
        paired_bootstrap_ci(np.zeros((4, 2)), np.zeros((4, 2)))


def test_pairing_beats_treating_arms_as_independent() -> None:
    """The whole reason for common random numbers, made quantitative."""
    rng = np.random.default_rng(0)
    shared = rng.normal(0, 5.0, size=400)  # environment noise, common to both arms
    a = shared + rng.normal(0.30, 0.1, size=400)
    b = shared + rng.normal(0.25, 0.1, size=400)

    paired_width = paired_bootstrap_ci(a, b, n_boot=2000, seed=1).width
    # What an unpaired analysis would see: same marginals, pairing destroyed.
    unpaired_width = paired_bootstrap_ci(
        a, rng.permutation(b), n_boot=2000, seed=1
    ).width
    assert paired_width < unpaired_width / 5, (
        f"pairing should shrink the interval sharply: {paired_width:.4f} vs {unpaired_width:.4f}"
    )


# -- bootstrap correctness ------------------------------------------------


def test_bca_interval_covers_at_the_nominal_rate() -> None:
    """Nominal 95% coverage, checked by simulation on a skewed distribution.

    Skewed on purpose: differences of returns are skewed when a few seeds hit an
    absorbing failure, and that is exactly where a symmetric percentile interval
    mis-covers.
    """
    rng = np.random.default_rng(7)
    true_mean = 1.0
    hits = 0
    trials = 200
    for t in range(trials):
        sample = rng.exponential(scale=true_mean, size=60)
        est = paired_bootstrap_ci(sample, np.zeros(60), n_boot=800, seed=t)
        hits += est.lo <= true_mean <= est.hi
    coverage = hits / trials
    assert 0.88 <= coverage <= 1.0, f"coverage {coverage:.3f} far from nominal 0.95"


def test_point_estimate_is_the_paired_mean_difference() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.5, 1.0, 2.5, 2.0])
    # differences [0.5, 1.0, 0.5, 2.0] -> mean 1.0
    assert paired_bootstrap_ci(a, b, n_boot=500).value == pytest.approx(1.0)


def test_identical_arms_give_a_zero_interval() -> None:
    x = np.linspace(0, 1, 32)
    est = paired_bootstrap_ci(x, x, n_boot=500)
    assert est.value == 0.0 and est.width == pytest.approx(0.0, abs=1e-12)
    assert not est.excludes_zero()


def test_percentile_and_bca_agree_on_symmetric_data() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(2.0, 1.0, size=300)
    z = np.zeros(300)
    bca = paired_bootstrap_ci(x, z, n_boot=4000, seed=5, bca=True)
    pct = paired_bootstrap_ci(x, z, n_boot=4000, seed=5, bca=False)
    assert abs(bca.lo - pct.lo) < 0.06 and abs(bca.hi - pct.hi) < 0.06


# -- permutation test -----------------------------------------------------


def test_permutation_detects_a_real_shift() -> None:
    rng = np.random.default_rng(11)
    shared = rng.normal(0, 1, size=200)
    assert paired_permutation_test(shared + 0.5, shared, n_perm=2000) < 0.01


def test_permutation_is_calibrated_under_the_null() -> None:
    rng = np.random.default_rng(13)
    p = [
        paired_permutation_test(
            rng.normal(0, 1, size=50), rng.normal(0, 1, size=50), n_perm=500, seed=i
        )
        for i in range(60)
    ]
    assert 0.02 <= float(np.mean(np.array(p) < 0.05)) <= 0.2


def test_p_value_is_never_exactly_zero() -> None:
    shared = np.arange(40, dtype=float)
    assert paired_permutation_test(shared + 100.0, shared, n_perm=200) > 0.0


# -- control variates -----------------------------------------------------


def test_control_variate_reduces_variance_and_sharpens_the_estimate() -> None:
    """The adjusted mean should move *away* from the raw sample mean.

    Subtracting ``beta * (c - known_mean)`` corrects for the covariate's
    sampling error, so it deliberately shifts the estimate — toward the truth.
    Asserting the mean is unchanged would be asserting the method does nothing.
    """
    rng = np.random.default_rng(17)
    true_mean = 3.0
    covariate = rng.normal(0, 1, size=500)
    target = true_mean + 2.0 * covariate + rng.normal(0, 0.3, size=500)

    adjusted, beta = control_variate(target, covariate, known_mean=0.0)

    assert beta == pytest.approx(2.0, abs=0.15)
    assert np.var(adjusted) < np.var(target) / 5, "variance should collapse"
    assert abs(adjusted.mean() - true_mean) < abs(target.mean() - true_mean), (
        "the adjusted estimate should sit closer to the truth than the raw mean"
    )


def test_control_variate_is_unbiased_across_repetitions() -> None:
    """Unbiasedness holds for any coefficient, which is why fitting beta in-sample is fine."""
    rng = np.random.default_rng(19)
    true_mean = 3.0
    means = []
    for _ in range(300):
        c = rng.normal(0, 1, size=40)
        t = true_mean + 2.0 * c + rng.normal(0, 0.3, size=40)
        means.append(control_variate(t, c, known_mean=0.0)[0].mean())
    assert float(np.mean(means)) == pytest.approx(true_mean, abs=0.02)


def test_control_variate_is_inert_on_a_constant_covariate() -> None:
    target = np.arange(20, dtype=float)
    adjusted, beta = control_variate(target, np.ones(20))
    assert beta == 0.0 and np.array_equal(adjusted, target)


# -- rank agreement -------------------------------------------------------


def test_kendall_tau_endpoints() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert kendall_tau(x, x) == pytest.approx(1.0)
    assert kendall_tau(x, -x) == pytest.approx(-1.0)


def test_kendall_tau_catches_a_curve_that_fits_but_misorders() -> None:
    """A fit can track values closely and still rank budgets wrongly.

    Which is why P1 scores frontiers by rank agreement and budget regret rather
    than by R-squared.
    """
    oracle = np.array([1.0, 1.01, 3.0, 3.01])
    fitted = np.array([1.01, 1.0, 3.01, 3.0])  # tiny errors, two inversions
    assert np.allclose(oracle, fitted, atol=0.02)
    assert kendall_tau(oracle, fitted) < 1.0


# -- anytime validity -----------------------------------------------------


def test_confidence_sequence_survives_optional_stopping() -> None:
    """Peek after every wave and stop when the interval looks good.

    This is precisely the procedure the sweep's adaptive allocator uses, and the
    procedure a fixed-sample interval would be invalidated by. Coverage must
    hold at the *stopped* sample size.
    """
    rng = np.random.default_rng(23)
    true_mean = 0.4
    misses = 0
    trials = 120
    for _ in range(trials):
        cs = ConfidenceSequence(alpha=0.05, value_range=1.0)
        draws = rng.binomial(1, true_mean, size=2000).astype(float)
        for start in range(0, 2000, 50):
            cs.update(draws[start : start + 50])
            if cs.resolved(target_width=0.15):
                break  # a data-dependent stopping time
        lo, hi = cs.interval()
        misses += not (lo <= true_mean <= hi)
    assert misses / trials <= 0.05, f"coverage failed under optional stopping: {misses}/{trials}"


def test_radius_shrinks_with_more_samples() -> None:
    cs = ConfidenceSequence(alpha=0.05, value_range=1.0)
    rng = np.random.default_rng(29)
    widths = []
    for _ in range(6):
        cs.update(rng.normal(0.5, 0.1, size=200))
        widths.append(cs.width())
    assert all(later < earlier for earlier, later in itertools.pairwise(widths))


def test_empirical_bernstein_beats_the_range_bound_on_tight_data() -> None:
    """Variance adaptivity is what makes adaptive allocation pay off.

    Paired differences are far tighter than their nominal range, so a bound that
    only knows the range would keep sampling long after the answer is clear.
    """
    n, rng_width = 500, 2.0
    tight = empirical_bernstein_radius(n, variance=0.001, rng=rng_width)
    worst_case = normal_mixture_radius(n, sigma=rng_width / 2)
    assert tight < worst_case / 3


def test_unresolved_before_any_samples() -> None:
    cs = ConfidenceSequence()
    assert not cs.resolved(target_width=1e9)
    assert cs.radius() == float("inf")


def test_unknown_method_is_rejected() -> None:
    cs = ConfidenceSequence(method="wishful").update(np.zeros(5))
    with pytest.raises(ValueError, match="unknown method"):
        cs.radius()
