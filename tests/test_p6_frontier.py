from __future__ import annotations

import inspect

from instinct.core.configschema import load_proposal_config
from instinct.p6_certification.frontier import (
    avcrc_correction,
    avcrc_shift_snapshot,
    csa_shift_snapshot,
    run_frontier_benchmark,
    validate_p6_3,
)
from instinct.p6_certification.mismatch import ShiftCondition


def test_avcrc_stitched_correction_shrinks_with_trajectory_scale() -> None:
    early = avcrc_correction(1_024, 0.2, 0.025)
    late = avcrc_correction(8_192, 0.2, 0.025)
    assert 0 < late < early < 0.2
    selected = avcrc_shift_snapshot([2, 5, 9], 8_192, 0.2, 0.025, [0.01] * 3)
    assert selected == 3


def test_csa_specialization_uses_no_exact_risk_input() -> None:
    parameters = set(inspect.signature(csa_shift_snapshot).parameters)
    assert "true_risks" not in parameters
    assert "model_risks" not in parameters
    safe = csa_shift_snapshot([2, 4, 8], 1_000, 0.2, 0.025, [0.01] * 3, 0.02)
    unsafe = csa_shift_snapshot([190, 200, 210], 1_000, 0.2, 0.025, [0.01] * 3, 0.02)
    assert safe == 3
    assert unsafe == 0


def test_frontier_reports_risk_nonvacuity_samples_and_latency() -> None:
    results, learned = run_frontier_benchmark(
        pilot_seed=31,
        eval_seed=41,
        confidence_delta=0.05,
        episode_risk=0.2,
        repetitions=16,
        sample_sizes=(128, 256),
        declared_looks=256,
        nominal_model_risks=(0.03, 0.08, 0.14, 0.19),
        pilot_model_samples=800,
        conditions=(ShiftCondition("shift", (0.05, 0.11, 0.18, 0.25)),),
        csa_bet_margin=0.02,
    )
    assert len(learned) == 4
    assert {result.samples for result in results} == {128, 256}
    assert all(result.trajectory_prefix_evaluations == result.samples * 4 for result in results)
    assert all(result.mean_certificate_latency_us >= 0 for result in results)
    at_scale = {result.method: result for result in results if result.samples == 256}
    assert at_scale["always_abstain"].mean_selected_prefix == 0
    assert at_scale["always_execute"].true_false_certification_rate == 1


def test_frozen_p6_3_config_is_valid_and_uses_fresh_units() -> None:
    config = load_proposal_config("configs/p6/p6_3/cpu.yaml")
    assert config.params["stage"] == "p6.3"
    assert config.params["declared_looks"] == 8_192
    assert config.params["gain_actions"] == 0.3
    assert all(check.passed for check in validate_p6_3(config))
