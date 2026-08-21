"""A theorem-matched finite-family Bernoulli confidence process."""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import beta


def polynomial_spend(total: float, index: int) -> float:
    if not 0 < total < 1:
        raise ValueError("total spend must lie in (0,1)")
    if index < 1:
        raise ValueError("spending index starts at one")
    return total * 6.0 / (3.141592653589793**2 * index**2)


def clopper_pearson_upper(successes: int, n: int, delta: float) -> float:
    """Exact one-sided upper bound for a Bernoulli bad-event probability."""
    if n < 0 or not 0 <= successes <= n:
        raise ValueError("successes must lie between zero and n")
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0,1)")
    if n == 0 or successes == n:
        return 1.0
    return float(beta.ppf(1.0 - delta, successes + 1, n - successes))


@dataclass
class BernoulliUpperProcess:
    """One-sided CP bounds with summable look and finite-family spending."""

    confidence_delta: float
    family_size: int
    successes: int = 0
    n: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.confidence_delta < 1:
            raise ValueError("confidence_delta must lie in (0,1)")
        if self.family_size < 1:
            raise ValueError("family_size must be positive")

    def update(self, bad_event: int | bool) -> None:
        if bad_event not in (0, 1, False, True):
            raise ValueError("Bernoulli observations must be zero or one")
        self.successes += int(bad_event)
        self.n += 1

    def upper(self, look_index: int) -> float:
        if self.n == 0:
            return 1.0
        local_delta = polynomial_spend(self.confidence_delta, look_index) / self.family_size
        return clopper_pearson_upper(self.successes, self.n, local_delta)


def longest_certified(bounds: dict[int, float], risk_budget: float) -> int:
    """Longest safe prefix; zero is deterministic abstention."""
    if not 0 <= risk_budget <= 1:
        raise ValueError("risk_budget must lie in [0,1]")
    return max((p for p, upper in bounds.items() if upper <= risk_budget), default=0)
