from __future__ import annotations

import inspect

import numpy as np

from instinct.p6_certification.comparators import AnytimeMaxCertifiedComparator
from instinct.p6_certification.mismatch import (
    CSAStyleEProcessComparator,
    ShiftCondition,
    calibrate_shift_envelope,
    robust_snapshot,
    run_mismatch_benchmark,
)
from instinct.p6_certification.simulation import ExactPrefixMDP, hazards_from_cumulative


def test_shift_certificate_interfaces_have_no_exact_risk_argument() -> None:
    calibration_parameters = set(inspect.signature(calibrate_shift_envelope).parameters)
    snapshot_parameters = set(inspect.signature(robust_snapshot).parameters)
    assert "true_risks" not in calibration_parameters
    assert "model_risks" not in calibration_parameters
    assert "true_risks" not in snapshot_parameters
    assert "model_risks" not in snapshot_parameters


def test_paired_discordance_envelope_is_sampled_and_prefix_simultaneous() -> None:
    model = ExactPrefixMDP(hazards_from_cumulative((0.03, 0.08, 0.15)))
    shifted = ExactPrefixMDP(hazards_from_cumulative((0.07, 0.16, 0.26)))
    uniforms = np.random.default_rng(8).random((2_000, 3))
    model_bad = np.maximum.accumulate(
        uniforms < np.asarray(model.hazards, dtype=np.float64), axis=1
    )
    shifted_bad = np.maximum.accumulate(
        uniforms < np.asarray(shifted.hazards, dtype=np.float64), axis=1
    )
    envelope = calibrate_shift_envelope(
        model_bad, shifted_bad, confidence_delta=0.025
    )
    assert 0 < envelope[0] <= envelope[1] <= envelope[2] < 1
    comparator = AnytimeMaxCertifiedComparator(0.025, 3, 4)
    snapshot = robust_snapshot(comparator, [2, 8, 15], 100, 2, 0.2, envelope)
    assert set(snapshot.bounds) == {1, 2, 3}


def test_csa_style_analogue_is_a_threshold_only_eprocess() -> None:
    comparator = CSAStyleEProcessComparator(0.05, n_prefixes=3, n_looks=5)
    safe = comparator.snapshot([0, 1, 2], n=400, look_index=3, risk_budget=0.2)
    unsafe = comparator.snapshot([100, 120, 140], n=400, look_index=3, risk_budget=0.2)
    assert safe.selected_prefix > 0
    assert unsafe.selected_prefix == 0


def test_model_validity_and_true_validity_separate_under_shift() -> None:
    results, learned = run_mismatch_benchmark(
        pilot_seed=611,
        eval_seed=621,
        confidence_delta=0.05,
        episode_risk=0.2,
        repetitions=80,
        exits=(80, 160, 320, 640),
        nominal_model_risks=(0.03, 0.08, 0.15),
        pilot_model_samples=1_200,
        shift_calibration_samples=320,
        conditions=(ShiftCondition("adverse", (0.07, 0.16, 0.26)),),
    )
    assert len(learned) == 3
    by_method = {result.method: result for result in results}
    unadjusted = by_method["anytime_max_certified"]
    robust = by_method["shift_robust_anytime"]
    assert unadjusted.model_false_certification_rate == 0
    assert unadjusted.true_false_certification_rate > 0
    assert robust.true_false_certification_rate == 0
    assert robust.mean_selected_prefix > 0
    assert by_method["always_abstain"].mean_selected_prefix == 0
    assert by_method["always_execute"].true_false_certification_rate == 1
