"""Anytime-valid confidence sequences.

Built in Stage 0 rather than with the rest of P6, because P1's sweep needs them
first. The sweep allocates seeds adaptively — sample a cell in waves, stop once
its interval is narrow relative to the effect being measured — and that is
optional stopping. A fixed-sample confidence interval is only valid at the
sample size you committed to in advance; peeking at it and stopping when it
looks good inflates the error rate, sometimes badly.

A confidence sequence is valid *uniformly over time*: the guarantee

    P( mu in CI_n  for all n simultaneously ) >= 1 - alpha

holds no matter what rule you use to decide when to stop, including rules that
depend on the data. That is exactly the licence adaptive allocation needs.

The construction is the sub-Gaussian normal-mixture boundary of Howard, Ramdas,
McAuliffe and Sekhon (*Time-uniform Chernoff bounds via nonnegative
supermartingales*), plus its variance-adaptive empirical-Bernstein sibling for
the common case where the range is known but is a poor description of the
spread.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

__all__ = ["ConfidenceSequence", "empirical_bernstein_radius", "normal_mixture_radius"]

FloatArray = npt.NDArray[np.float64]


def normal_mixture_radius(
    n: int, sigma: float, alpha: float = 0.05, rho: float | None = None
) -> float:
    """Half-width of a time-uniform interval for a sub-Gaussian mean.

    ``sigma`` is the sub-Gaussian variance proxy. For a bounded quantity on a
    range of width ``w``, Hoeffding's lemma gives ``sigma = w / 2``, which is
    always valid but often loose.

    ``rho`` tunes *when* the boundary is tightest. The mixture is optimized for
    a particular sample size; the default targets a moderate one, since most
    sweep cells resolve early. Larger ``rho`` trades early tightness for late.

    The radius shrinks like ``sqrt(log log n / n)`` rather than ``sqrt(1/n)`` —
    the price of validity at every ``n`` at once. It is a small price: a factor
    of roughly 1.5-2 in width, against an error rate that would otherwise creep
    toward 1 under repeated peeking.
    """
    if n <= 0:
        return float("inf")
    if sigma <= 0:
        return 0.0
    if rho is None:
        rho = max(1.0, n / 8.0)
    v = n  # intrinsic time for an i.i.d. mean
    inner = np.sqrt((v + rho) / rho) / alpha
    radius = sigma * np.sqrt(2.0 * (v + rho) * np.log(inner)) / n
    return float(radius)


def empirical_bernstein_radius(
    n: int, variance: float, rng: float, alpha: float = 0.05, rho: float | None = None
) -> float:
    """Variance-adaptive time-uniform half-width for a bounded mean.

    Uses the observed spread instead of the worst case, so a cell whose paired
    differences are tightly clustered resolves far sooner than the range alone
    would allow. This matters a lot here: paired differences of returns are
    usually much tighter than their theoretical range, which is precisely why
    pairing was worth doing.

    A range-based term is retained so the bound stays honest when the variance
    estimate is itself built from few samples.
    """
    if n <= 0:
        return float("inf")
    if rho is None:
        rho = max(1.0, n / 8.0)
    v = max(n * variance, 1e-12)
    inner = np.sqrt((v + rho) / rho) / alpha
    log_term = np.log(inner)
    # Bernstein form: a variance term that dominates once the spread is known,
    # plus a range term that carries the tail.
    return float(np.sqrt(2.0 * (v + rho) * log_term) / n + rng * log_term / (3.0 * n))


@dataclass
class ConfidenceSequence:
    """A running, always-valid interval for the mean of a stream of samples.

    Feed it samples in waves with :meth:`update`; read :meth:`interval` whenever
    you like, including to decide whether to keep sampling. That is the whole
    point — the interval does not become invalid because you looked at it.

    >>> cs = ConfidenceSequence(alpha=0.05, value_range=2.0)
    >>> _ = cs.update(np.random.default_rng(0).normal(0.5, 0.1, size=500))
    >>> lo, hi = cs.interval()
    >>> bool(lo < 0.5 < hi)
    True
    """

    alpha: float = 0.05
    value_range: float = 1.0
    method: str = "empirical_bernstein"
    n: int = 0
    _sum: float = 0.0
    _sum_sq: float = 0.0
    _history: list[int] = field(default_factory=list, repr=False)

    def update(self, samples: FloatArray) -> ConfidenceSequence:
        x = np.asarray(samples, dtype=np.float64).ravel()
        if x.size:
            self.n += int(x.size)
            self._sum += float(x.sum())
            self._sum_sq += float(np.square(x).sum())
            self._history.append(self.n)
        return self

    @property
    def mean(self) -> float:
        return self._sum / self.n if self.n else 0.0

    @property
    def variance(self) -> float:
        if self.n < 2:
            return self.value_range**2 / 4.0
        var = self._sum_sq / self.n - self.mean**2
        return max(var, 0.0)

    def radius(self) -> float:
        if self.method == "empirical_bernstein":
            return empirical_bernstein_radius(
                self.n, self.variance, self.value_range, self.alpha
            )
        if self.method == "normal_mixture":
            return normal_mixture_radius(self.n, self.value_range / 2.0, self.alpha)
        raise ValueError(f"unknown method {self.method!r}")

    def interval(self) -> tuple[float, float]:
        r = self.radius()
        return self.mean - r, self.mean + r

    def width(self) -> float:
        return 2.0 * self.radius()

    def resolved(self, target_width: float) -> bool:
        """Has this cell been measured precisely enough to stop?

        The stopping rule for the sweep's adaptive allocator. Because the
        interval is time-uniform, consulting it here does not invalidate it.
        """
        return self.n > 1 and self.width() <= target_width

    def excludes_zero(self) -> bool:
        lo, hi = self.interval()
        return lo > 0.0 or hi < 0.0
