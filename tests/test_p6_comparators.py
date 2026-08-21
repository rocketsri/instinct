from __future__ import annotations

import numpy as np

from instinct.p6_certification.comparators import (
    AlwaysAbstainComparator,
    AlwaysExecuteComparator,
    AnytimeMaxCertifiedComparator,
    DeclaredExitUnionComparator,
    make_comparators,
)
from instinct.p6_certification.simulation import (
    ExactPrefixMDP,
    hazards_from_cumulative,
    run_benchmark,
)


def test_exact_prefix_truth_is_not_a_comparator_input() -> None:
    risks = np.array([0.03, 0.08, 0.17])
    mdp = ExactPrefixMDP(hazards_from_cumulative(risks))
    assert np.allclose(mdp.exact_prefix_risks(), risks)
    comparator = AnytimeMaxCertifiedComparator(0.05, n_prefixes=3, n_looks=4)
    snapshot = comparator.snapshot([1, 2, 4], n=40, look_index=2, risk_budget=0.2)
    assert set(snapshot.bounds) == {1, 2, 3}


def test_declared_exit_and_anytime_allocations_are_familywise() -> None:
    union = DeclaredExitUnionComparator(0.05, n_prefixes=3, n_looks=5)
    assert union.local_delta(1) * 3 * 5 == 0.05
    anytime = AnytimeMaxCertifiedComparator(0.05, n_prefixes=3, n_looks=5)
    allocated = sum(anytime.local_delta(look) for look in range(1, 50)) * 3
    assert allocated <= 0.05


def test_abstain_and_execute_controls_are_extremes() -> None:
    bounds = {1: 0.1, 2: 0.2, 3: 0.3}
    assert AlwaysAbstainComparator(0.05, 3, 2).select(bounds, 0.2) == 0
    assert AlwaysExecuteComparator(0.05, 3, 2).select(bounds, 0.2) == 3
    assert len(make_comparators(0.05, 3, 2)) == 6


def test_common_random_benchmark_exposes_validity_nonvacuity_tradeoff() -> None:
    results = {
        result.comparator: result
        for result in run_benchmark(
            seed=17,
            confidence_delta=0.05,
            episode_risk=0.2,
            repetitions=80,
            exits=(20, 40, 80, 160),
            repeated_decisions=4,
        )
    }
    assert results["always_abstain"].mean_selected_prefix == 0
    assert results["always_execute"].mean_selected_prefix == 3
    assert results["always_execute"].repeated_spending_coverage == 0
    assert (
        results["anytime_max_certified"].optional_stopping_coverage
        >= results["fixed_time_pointwise"].optional_stopping_coverage
    )
