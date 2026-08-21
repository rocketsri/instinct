"""Common interfaces for P6 certificate and execution comparators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from instinct.p6_certification.certificates import (
    clopper_pearson_upper,
    longest_certified,
    polynomial_spend,
)


@dataclass(frozen=True)
class ComparatorSnapshot:
    """Bounds and action chosen using only rollout observations."""

    bounds: dict[int, float]
    selected_prefix: int


class PrefixComparator(ABC):
    """Stateless comparator evaluated on cumulative Bernoulli counts."""

    name: str
    validity: str

    def __init__(self, confidence_delta: float, n_prefixes: int, n_looks: int) -> None:
        if not 0 < confidence_delta < 1:
            raise ValueError("confidence_delta must lie in (0,1)")
        if n_prefixes < 1 or n_looks < 1:
            raise ValueError("prefix and look counts must be positive")
        self.confidence_delta = confidence_delta
        self.n_prefixes = n_prefixes
        self.n_looks = n_looks

    @abstractmethod
    def local_delta(self, look_index: int) -> float:
        """Confidence error assigned to one prefix at one look."""

    def bounds(self, bad_counts: Sequence[int], n: int, look_index: int) -> dict[int, float]:
        if len(bad_counts) != self.n_prefixes:
            raise ValueError("bad_counts length must equal declared prefix family")
        if not 1 <= look_index <= self.n_looks:
            raise ValueError("look index outside declared schedule")
        raw = np.array(
            [
                clopper_pearson_upper(int(count), n, self.local_delta(look_index))
                for count in bad_counts
            ],
            dtype=np.float64,
        )
        # Prefix failure events are nested, so their risks are nondecreasing.
        # Raising later bounds to the cumulative maximum preserves coverage.
        nested = np.maximum.accumulate(raw)
        return {prefix: float(nested[prefix - 1]) for prefix in range(1, self.n_prefixes + 1)}

    def select(self, bounds: dict[int, float], risk_budget: float) -> int:
        return longest_certified(bounds, risk_budget)

    def snapshot(
        self,
        bad_counts: Sequence[int],
        n: int,
        look_index: int,
        risk_budget: float,
    ) -> ComparatorSnapshot:
        bounds = self.bounds(bad_counts, n, look_index)
        return ComparatorSnapshot(bounds, self.select(bounds, risk_budget))


class FixedTimePointwiseComparator(PrefixComparator):
    """Valid at one fixed prefix/look; negative control under adaptation."""

    name = "fixed_time_pointwise"
    validity = "fixed prefix and fixed look only"

    def local_delta(self, look_index: int) -> float:
        del look_index
        return self.confidence_delta


class BonferroniFiniteFamilyComparator(PrefixComparator):
    """Selection-valid across prefixes at one preregistered final look."""

    name = "bonferroni_finite_family"
    validity = "simultaneous prefixes at one fixed look"

    def local_delta(self, look_index: int) -> float:
        del look_index
        return self.confidence_delta / self.n_prefixes


class DeclaredExitUnionComparator(PrefixComparator):
    """Finite union bound over every declared (exit, prefix) pair."""

    name = "nested_exit_union"
    validity = "simultaneous declared exits and prefixes"

    def local_delta(self, look_index: int) -> float:
        del look_index
        return self.confidence_delta / (self.n_prefixes * self.n_looks)


class AnytimeMaxCertifiedComparator(PrefixComparator):
    """Per-prefix anytime process followed by longest-prefix selection."""

    name = "anytime_max_certified"
    validity = "anytime over looks and simultaneous over prefixes"

    def local_delta(self, look_index: int) -> float:
        return polynomial_spend(self.confidence_delta, look_index) / self.n_prefixes


class AlwaysAbstainComparator(PrefixComparator):
    name = "always_abstain"
    validity = "deterministic abstention"

    def local_delta(self, look_index: int) -> float:
        del look_index
        return self.confidence_delta / self.n_prefixes

    def select(self, bounds: dict[int, float], risk_budget: float) -> int:
        del bounds, risk_budget
        return 0


class AlwaysExecuteComparator(PrefixComparator):
    name = "always_execute"
    validity = "uncertified execution control"

    def local_delta(self, look_index: int) -> float:
        del look_index
        return self.confidence_delta / self.n_prefixes

    def select(self, bounds: dict[int, float], risk_budget: float) -> int:
        del bounds, risk_budget
        return self.n_prefixes


COMPARATOR_TYPES: tuple[type[PrefixComparator], ...] = (
    FixedTimePointwiseComparator,
    BonferroniFiniteFamilyComparator,
    DeclaredExitUnionComparator,
    AnytimeMaxCertifiedComparator,
    AlwaysAbstainComparator,
    AlwaysExecuteComparator,
)


def make_comparators(
    confidence_delta: float, n_prefixes: int, n_looks: int
) -> list[PrefixComparator]:
    return [kind(confidence_delta, n_prefixes, n_looks) for kind in COMPARATOR_TYPES]
