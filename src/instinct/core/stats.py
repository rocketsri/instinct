"""Paired statistics for counterfactual comparisons.

Everything measured in this repo is a difference between arms that saw the same
randomness, so the estimators here are paired by default. That is not a stylistic
preference: it is where most of the statistical power comes from. Two arms of a
real-time rollout share their environment noise, so their returns are strongly
correlated, and the variance of the *difference* can be one or two orders of
magnitude below the variance of either arm. Comparing them as independent samples
throws that away and can turn a resolvable effect into noise.

:func:`paired_bootstrap_ci` therefore resamples *pairs*, never the arms
separately, and :func:`assert_paired` exists so that a caller who accidentally
hands over unpaired data gets an error instead of a plausible-looking wrong
answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats as scipy_stats

__all__ = [
    "Estimate",
    "assert_paired",
    "control_variate",
    "kendall_tau",
    "paired_bootstrap_ci",
    "paired_permutation_test",
    "spearman",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Estimate:
    """A point estimate with an interval. Never report one without the other."""

    value: float
    lo: float
    hi: float
    n: int
    method: str = "paired_bca"

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0

    def __str__(self) -> str:
        return f"{self.value:+.5f} [{self.lo:+.5f}, {self.hi:+.5f}] (n={self.n})"


def assert_paired(a: FloatArray, b: FloatArray, *, what: str = "arms") -> None:
    """Fail loudly when two arms are not aligned sample-for-sample.

    An unpaired comparison of paired data does not crash — it just quietly loses
    power and reports a wider interval than it should. This turns that into an
    error, which is the only reason it gets caught.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(
            f"{what} are not paired: shapes {a.shape} and {b.shape} differ. "
            "Paired comparisons require one sample per arm per seed, aligned."
        )
    if a.ndim != 1:
        raise ValueError(f"{what} must be 1-D per-seed arrays, got {a.ndim}-D")


def paired_bootstrap_ci(
    a: FloatArray,
    b: FloatArray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    bca: bool = True,
) -> Estimate:
    """Bootstrap CI for the paired mean difference ``mean(a - b)``.

    Resamples pairs, preserving the correlation that pairing bought. With
    ``bca`` the interval is bias-corrected and accelerated, which matters here
    because differences of returns are frequently skewed — a symmetric percentile
    interval mis-covers when a small fraction of seeds fall into an absorbing
    failure and drag one tail out.
    """
    assert_paired(a, b)
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = d.size
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    theta = float(d.mean())
    if n == 1:
        return Estimate(theta, theta, theta, n, "degenerate")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)

    if not bca:
        lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
        return Estimate(theta, float(lo), float(hi), n, "paired_percentile")

    # Bias correction from the fraction of resamples below the point estimate.
    prop = float(np.mean(boots < theta))
    prop = min(max(prop, 1.0 / n_boot), 1.0 - 1.0 / n_boot)
    z0 = scipy_stats.norm.ppf(prop)

    # Acceleration from the jackknife's third moment.
    total = d.sum()
    jack = (total - d) / (n - 1)
    jack_dev = jack.mean() - jack
    denom = 6.0 * (np.sum(jack_dev**2) ** 1.5)
    acc = float(np.sum(jack_dev**3) / denom) if denom > 1e-300 else 0.0

    z_lo, z_hi = scipy_stats.norm.ppf([alpha / 2, 1 - alpha / 2])
    def adjust(z: float) -> float:
        return float(scipy_stats.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z))))

    q_lo, q_hi = adjust(z_lo), adjust(z_hi)
    lo, hi = np.quantile(boots, [np.clip(q_lo, 0, 1), np.clip(q_hi, 0, 1)])
    return Estimate(theta, float(lo), float(hi), n, "paired_bca")


def paired_permutation_test(
    a: FloatArray, b: FloatArray, *, n_perm: int = 10_000, seed: int = 0
) -> float:
    """Two-sided p-value for a zero paired mean difference.

    Permutes the *sign* of each pair's difference, which is the exchangeability
    that pairing actually provides. Shuffling arm labels across seeds would test
    a different and wrong null.
    """
    assert_paired(a, b)
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    observed = abs(float(d.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, d.size))
    null = np.abs((signs * d).mean(axis=1))
    # +1 in both terms keeps the p-value from ever being exactly zero.
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))


def control_variate(
    target: FloatArray, covariate: FloatArray, *, known_mean: float = 0.0
) -> tuple[FloatArray, float]:
    """Reduce variance using a correlated quantity with a known mean.

    Returns the adjusted samples and the coefficient used. The adjustment is
    unbiased for any coefficient, so the fitted one only affects efficiency —
    which is why fitting it on the same data is acceptable here.

    Used for the tail term of the decomposition, where a value-function estimate
    at the handoff state correlates strongly with the realized return.
    """
    t = np.asarray(target, dtype=np.float64)
    c = np.asarray(covariate, dtype=np.float64)
    assert_paired(t, c, what="target and covariate")
    var_c = float(np.var(c))
    if var_c < 1e-15:
        return t, 0.0
    beta = float(np.cov(t, c, ddof=1)[0, 1] / var_c)
    return t - beta * (c - known_mean), beta


def kendall_tau(a: FloatArray, b: FloatArray) -> float:
    """Rank agreement between two orderings.

    P1 scores budget frontiers by whether a fitted curve *ranks* budgets the way
    the oracle does, which is what a downstream controller would actually use.
    Rank agreement is the honest summary; R-squared on the curve is not, since a
    curve can fit well and still order the budgets wrongly.
    """
    a, b = np.asarray(a), np.asarray(b)
    assert_paired(a, b, what="rankings")
    if a.size < 2:
        return float("nan")
    return float(scipy_stats.kendalltau(a, b).statistic)


def spearman(a: FloatArray, b: FloatArray) -> float:
    """Rank correlation for a paired identifiability check.

    Mirrors :func:`kendall_tau`'s shape exactly: paired-by-default, NaN below
    two points, no NaN-to-zero substitution. A caller gating on
    ``spearman(...) >= threshold`` therefore fails closed on degenerate input
    rather than silently passing or failing it.
    """
    a, b = np.asarray(a), np.asarray(b)
    assert_paired(a, b, what="rankings")
    if a.size < 2:
        return float("nan")
    return float(scipy_stats.spearmanr(a, b).statistic)
