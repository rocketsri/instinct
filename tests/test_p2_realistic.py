"""P2.2 bounded natural-image pilot invariants and fail-closed gates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from instinct.core.configschema import load_proposal_config
from instinct.p2_harmful_write.realistic import (
    P2PilotConfig,
    _select_top,
    run_natural_image_pilot,
)
from instinct.p2_harmful_write.run import validate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def result():
    return run_natural_image_pilot(221, P2PilotConfig())


def test_config_registers_p2_2_without_requiring_torch() -> None:
    config = load_proposal_config(ROOT / "configs/p2/p2_2/smoke.yaml")
    checks = validate(config)
    assert all(check.passed for check in checks)
    assert config.params["stage"] == "p2_2"


def test_streams_horizons_and_utility_decomposition_are_auditable(result) -> None:
    assert result.streams_disjoint
    assert result.full_horizons_only
    assert result.candidates
    for candidate in result.candidates:
        assert candidate.utility == pytest.approx(
            candidate.benefit - candidate.damage - candidate.cost
        )
        assert len(candidate.oracle_probe_ids) == 4
    for arm in result.arms:
        assert arm.utility == pytest.approx(arm.benefit - arm.damage - arm.cost)


def test_candidate_restriction_precedes_every_top_m_selection(result) -> None:
    eligible = set(result.eligible_timesteps)
    selected_arms = [
        arm
        for arm in result.arms
        if arm.method in {"validation_only", "oracle_ceiling"}
        or arm.method.startswith("random-")
    ]
    assert result.matched_count_passed
    assert all(arm.n_writes == result.m for arm in selected_arms)
    assert all(set(arm.selected_timesteps) <= eligible for arm in selected_arms)

    # An adversarial score cannot smuggle an excluded late timestep into top-m:
    # _select_top accepts only an already-restricted score vector.
    scores = np.arange(len(result.eligible_timesteps), dtype=np.float64)
    chosen = _select_top(scores, result.eligible_timesteps, 2)
    assert set(chosen) <= eligible
    with pytest.raises(ValueError, match="already be restricted"):
        _select_top(np.append(scores, 1e9), result.eligible_timesteps, 2)


def test_required_baselines_are_present_and_deployable_compute_is_matched(result) -> None:
    methods = {arm.method for arm in result.arms}
    assert {"always", "never", "validation_only", "oracle_ceiling"} <= methods
    assert sum(method.startswith("random-") for method in methods) == 8
    deployable = [
        arm.deployable_compute_units
        for arm in result.arms
        if arm.method == "validation_only" or arm.method.startswith("random-")
    ]
    assert result.matched_compute_passed
    assert len(set(deployable)) == 1
    oracle = next(arm for arm in result.arms if arm.method == "oracle_ceiling")
    assert oracle.offline_oracle_units > 0


def test_realism_and_useful_sparsity_fail_closed_so_p7_stays_locked(result) -> None:
    assert result.identifiability_passed
    assert not result.realistic_scale_passed
    assert result.oracle_advantage < 0.0
    assert result.useful_sparsity == 0.0
    assert not result.qualifies_p7


def test_seeded_pilot_is_deterministic(result) -> None:
    repeated = run_natural_image_pilot(221, P2PilotConfig())
    assert repeated.source_sha256 == result.source_sha256
    assert repeated.eligible_timesteps == result.eligible_timesteps
    assert repeated.oracle_advantage == result.oracle_advantage
    assert repeated.arms == result.arms
