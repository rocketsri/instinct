from __future__ import annotations

import inspect

import numpy as np

from instinct.core.configschema import load_proposal_config
from instinct.p6_certification.learned_dynamics import (
    StructuralShift,
    TransitionRiskModel,
    certify_prefixes,
    run_dynamics_benchmark,
    validate_p6_4,
)


def _model() -> TransitionRiskModel:
    matrices = np.asarray(
        [
            [[0.96, 0.03, 0.01], [0.2, 0.7, 0.1], [0.0, 0.0, 1.0]],
            [[0.94, 0.04, 0.02], [0.1, 0.75, 0.15], [0.0, 0.0, 1.0]],
        ]
    )
    return TransitionRiskModel(matrices, (0, 1, 0, 1))


def test_transition_model_is_actually_fitted_from_pilot_trajectories() -> None:
    model = _model()
    states = model.sample_states_with_uniforms(np.random.default_rng(3).random((20_000, 4)))
    fitted = TransitionRiskModel.fit(states, model.action_schedule, n_actions=2)
    assert fitted.matrices.shape == (2, 3, 3)
    assert np.mean(np.abs(fitted.matrices - model.matrices)) < 0.01
    risks = fitted.exact_prefix_risks()
    assert np.all(np.diff(risks) >= 0)


def test_stable_certificate_api_has_no_exact_truth_input() -> None:
    parameters = set(inspect.signature(certify_prefixes).parameters)
    assert "true_risks" not in parameters
    assert "model_risks" not in parameters
    selected = certify_prefixes(
        "shift_robust_csa_specialization",
        [0, 1, 2, 3],
        n=1_000,
        look_index=1_000,
        declared_looks=1_000,
        episode_risk=0.2,
        confidence_delta=0.05,
        shift_envelope=[0.01] * 4,
        csa_bet_margin=0.02,
    )
    assert selected > 0


def test_learned_dynamics_integration_reports_cost_axes() -> None:
    model = _model()
    shifted = model.matrices.copy()
    shifted[:, 0, 2] += 0.03
    shifted[:, 0, 0] -= 0.03
    results, fitted, profile = run_dynamics_benchmark(
        nominal=model,
        shifts=(StructuralShift("direct_hazard", shifted),),
        pilot_seed=11,
        eval_seed=12,
        pilot_trajectories=1_000,
        repetitions=8,
        sample_sizes=(128, 256),
        declared_looks=256,
        episode_risk=0.2,
        confidence_delta=0.05,
        csa_bet_margin=0.02,
    )
    assert fitted.horizon == 4
    assert profile.pilot_transitions == 4_000
    assert {result.samples for result in results} == {128, 256}
    assert all(result.model_rollout_buffer_kb > 0 for result in results)
    assert all(result.paired_calibration_buffer_kb > result.model_rollout_buffer_kb for result in results)
    assert all(result.mean_model_rollout_latency_us > 0 for result in results)


def test_frozen_p6_4_config_validates() -> None:
    config = load_proposal_config("configs/p6/p6_4/cpu.yaml")
    assert config.params["stage"] == "p6.4"
    assert all(check.passed for check in validate_p6_4(config))
