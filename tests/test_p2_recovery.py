"""P2.2 recovery-cycle-1 invariants and qualification guard."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from instinct.core.configschema import load_proposal_config
from instinct.p2_harmful_write.recovery import RecoveryConfig, run_recovery
from instinct.p2_harmful_write.run import validate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def recovery():
    return run_recovery(231, RecoveryConfig())


def test_recovery_config_is_preregistered_as_a_fresh_stage() -> None:
    config = load_proposal_config(ROOT / "configs/p2/p2_2/recovery1.yaml")
    assert config.params["stage"] == "p2_2_r1"
    assert all(check.passed for check in validate(config))
    assert config.theory.theory_id == "spec-5.1-5.10"


def test_recovery_freezes_eligibility_before_exact_subset_search(recovery) -> None:
    assert recovery.full_horizons_only
    assert recovery.streams_disjoint
    assert recovery.oracle_subsets_evaluated == math.comb(len(recovery.eligible_timesteps), 3)
    eligible = set(recovery.eligible_timesteps)
    for arm in recovery.arms:
        if arm.method in {"validation_only", "oracle_ceiling"} or arm.method.startswith(
            "random-"
        ):
            assert arm.n_writes == recovery.m
            assert set(arm.selected_timesteps) <= eligible


def test_recovery_preserves_utility_decomposition_and_offline_oracle_cost(recovery) -> None:
    for arm in recovery.arms:
        assert arm.utility == pytest.approx(arm.benefit - arm.damage - arm.cost)
    oracle = next(arm for arm in recovery.arms if arm.method == "oracle_ceiling")
    assert oracle.offline_oracle_units > 0
    assert recovery.matched_count_passed
    assert recovery.matched_compute_passed


def test_recovery_has_all_required_baselines(recovery) -> None:
    methods = {arm.method for arm in recovery.arms}
    assert {"always", "never", "validation_only", "oracle_ceiling"} <= methods
    assert sum(method.startswith("random-") for method in methods) == 8


def test_recovery_finds_provisional_sparsity_but_cannot_unlock_p7(recovery) -> None:
    assert recovery.identifiability_passed
    assert recovery.oracle_advantage > 0.0
    assert recovery.oracle_gain_over_never_mse > 0.0
    assert recovery.useful_sparsity == pytest.approx(25 / 28)
    assert not recovery.realistic_scale_passed
    assert not recovery.qualifies_p7
