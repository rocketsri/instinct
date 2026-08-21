"""P6.0/P6.1 simulations with exact truth confined to evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from instinct.p6_certification.certificates import polynomial_spend
from instinct.p6_certification.comparators import COMPARATOR_TYPES, PrefixComparator


@dataclass(frozen=True)
class ExactPrefixMDP:
    """Absorbing hazard chain used as hidden P6.1 evaluation truth."""

    hazards: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.hazards or any(not 0 <= hazard <= 1 for hazard in self.hazards):
            raise ValueError("hazards must be a nonempty sequence in [0,1]")

    def exact_prefix_risks(self) -> np.ndarray:
        """Dynamic-programming truth; callers must not pass this to a comparator."""
        return 1.0 - np.cumprod(1.0 - np.asarray(self.hazards, dtype=np.float64))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if n < 1:
            raise ValueError("sample count must be positive")
        step_failures = rng.random((n, len(self.hazards))) < np.asarray(self.hazards)
        return np.maximum.accumulate(step_failures, axis=1)


@dataclass(frozen=True)
class BenchmarkResult:
    comparator: str
    optional_stopping_coverage: float
    prefix_selection_coverage: float
    prefix_false_certification_rate: float
    repeated_spending_coverage: float
    episode_violation_rate: float
    mean_selected_prefix: float
    abstention_rate: float
    nonvacuity_rate: float


@dataclass
class _Counters:
    optional_covered: int = 0
    prefix_covered: int = 0
    prefix_false: int = 0
    repeated_covered: int = 0
    episode_violations: int = 0
    prefix_total: int = 0
    abstentions: int = 0


def hazards_from_cumulative(risks: Sequence[float]) -> tuple[float, ...]:
    values = np.asarray(risks, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("risks must be a nonempty vector")
    if np.any(values < 0) or np.any(values >= 1) or np.any(np.diff(values) < 0):
        raise ValueError("cumulative risks must be nondecreasing and lie in [0,1)")
    survival = 1.0
    hazards: list[float] = []
    for risk in values:
        next_survival = 1.0 - float(risk)
        hazards.append(1.0 - next_survival / survival)
        survival = next_survival
    return tuple(hazards)


def _comparator(
    kind: type[PrefixComparator], delta: float, n_prefixes: int, n_looks: int
) -> PrefixComparator:
    # The comparator constructor has no truth argument by design. Exact risks
    # remain in this evaluation module and are used only after selection.
    return kind(delta, n_prefixes, n_looks)


def _optional_stopping_experiment(
    rng: np.random.Generator,
    counters: dict[str, _Counters],
    confidence_delta: float,
    repetitions: int,
    exits: tuple[int, ...],
) -> None:
    truth = 0.12
    adversarial_certification_threshold = 0.115
    for _ in range(repetitions):
        draws = rng.random(exits[-1]) < truth
        cumulative = np.cumsum(draws)
        for kind in COMPARATOR_TYPES:
            comparator = _comparator(kind, confidence_delta, 1, len(exits))
            final_upper = 1.0
            for look_index, n in enumerate(exits, start=1):
                snapshot = comparator.snapshot(
                    [int(cumulative[n - 1])], n, look_index, adversarial_certification_threshold
                )
                final_upper = snapshot.bounds[1]
                if snapshot.selected_prefix > 0:
                    break
            counters[comparator.name].optional_covered += int(truth <= final_upper)


def _prefix_selection_experiment(
    rng: np.random.Generator,
    counters: dict[str, _Counters],
    confidence_delta: float,
    risk_budget: float,
    repetitions: int,
    exits: tuple[int, ...],
) -> None:
    mdp = ExactPrefixMDP(hazards_from_cumulative((0.035, 0.08, 0.17)))
    truths = mdp.exact_prefix_risks()
    for _ in range(repetitions):
        outcomes = mdp.sample(rng, exits[-1])
        counts = outcomes.sum(axis=0).astype(int)
        for kind in COMPARATOR_TYPES:
            comparator = _comparator(kind, confidence_delta, len(truths), len(exits))
            snapshot = comparator.snapshot(counts, exits[-1], len(exits), risk_budget)
            selected = snapshot.selected_prefix
            selected_covered = selected == 0 or truths[selected - 1] <= snapshot.bounds[selected]
            false_certification = selected > 0 and truths[selected - 1] > risk_budget
            counter = counters[comparator.name]
            counter.prefix_covered += int(selected_covered)
            counter.prefix_false += int(false_certification)
            counter.prefix_total += selected
            counter.abstentions += int(selected == 0)


def _repeated_spending_experiment(
    rng: np.random.Generator,
    counters: dict[str, _Counters],
    confidence_delta: float,
    episode_risk: float,
    repetitions: int,
    exits: tuple[int, ...],
    decisions: int,
) -> None:
    for _ in range(repetitions):
        execution_uniforms = rng.random(decisions)
        all_covered = {kind.name: True for kind in COMPARATOR_TYPES}
        any_violation = {kind.name: False for kind in COMPARATOR_TYPES}
        for decision in range(1, decisions + 1):
            alpha_t = polynomial_spend(episode_risk, decision)
            delta_t = polynomial_spend(confidence_delta, decision)
            target_risks = (0.25 * alpha_t, 0.60 * alpha_t, min(1.10 * alpha_t, 0.99))
            mdp = ExactPrefixMDP(hazards_from_cumulative(target_risks))
            truths = mdp.exact_prefix_risks()
            outcomes = mdp.sample(rng, exits[-1])
            cumulative = np.cumsum(outcomes, axis=0)
            for kind in COMPARATOR_TYPES:
                comparator = _comparator(kind, delta_t, len(truths), len(exits))
                selected = 0
                selected_upper = 1.0
                for look_index, n in enumerate(exits, start=1):
                    snapshot = comparator.snapshot(
                        cumulative[n - 1].astype(int), n, look_index, alpha_t
                    )
                    selected = snapshot.selected_prefix
                    selected_upper = snapshot.bounds.get(selected, 1.0)
                    if selected > 0:
                        break
                truth = 0.0 if selected == 0 else float(truths[selected - 1])
                all_covered[comparator.name] &= truth <= selected_upper and truth <= alpha_t
                any_violation[comparator.name] |= execution_uniforms[decision - 1] < truth
        for name, counter in counters.items():
            counter.repeated_covered += int(all_covered[name])
            counter.episode_violations += int(any_violation[name])


def run_benchmark(
    seed: int,
    confidence_delta: float,
    episode_risk: float,
    repetitions: int,
    exits: Sequence[int],
    repeated_decisions: int,
) -> list[BenchmarkResult]:
    """Run common-random-number P6.0/P6.1 comparator experiments."""
    exit_tuple = tuple(int(value) for value in exits)
    if (
        not exit_tuple
        or exit_tuple[0] < 1
        or any(right <= left for left, right in pairwise(exit_tuple))
    ):
        raise ValueError("exits must be strictly increasing positive sample counts")
    if repetitions < 1 or repeated_decisions < 1:
        raise ValueError("repetitions and repeated decisions must be positive")
    rng = np.random.default_rng(seed)
    counters = {kind.name: _Counters() for kind in COMPARATOR_TYPES}
    _optional_stopping_experiment(rng, counters, confidence_delta, repetitions, exit_tuple)
    _prefix_selection_experiment(
        rng, counters, confidence_delta, episode_risk, repetitions, exit_tuple
    )
    _repeated_spending_experiment(
        rng,
        counters,
        confidence_delta,
        episode_risk,
        repetitions,
        exit_tuple,
        repeated_decisions,
    )
    max_prefix = 3.0
    return [
        BenchmarkResult(
            comparator=name,
            optional_stopping_coverage=counter.optional_covered / repetitions,
            prefix_selection_coverage=counter.prefix_covered / repetitions,
            prefix_false_certification_rate=counter.prefix_false / repetitions,
            repeated_spending_coverage=counter.repeated_covered / repetitions,
            episode_violation_rate=counter.episode_violations / repetitions,
            mean_selected_prefix=counter.prefix_total / repetitions,
            abstention_rate=counter.abstentions / repetitions,
            nonvacuity_rate=counter.prefix_total / (repetitions * max_prefix),
        )
        for name, counter in counters.items()
    ]
