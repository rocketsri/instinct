"""CPU-only F2 repair plumbing: fresh-data provenance and the archival gate.

Neither of these touches Torch, CUDA, or an actual CIFAR-10 image -- they are
the pure-Python pieces of the F2 repair (see
``docs/reviews/p2/v6/theory_literature_synthesis.md``'s "One permissible
pilot-only F2 repair" and "Exact archival condition") that can be verified
without spending any Colab budget. The Torch-gated scoring/replay/rescoring
functions this plumbing will eventually feed are a later, separately reviewed
step.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from instinct.core.configschema import load_proposal_config
from instinct.p2_harmful_write.recovery2_benchmark import (
    F2_REPAIR_SOURCE_OFFSET,
    BenchmarkConfig,
    FULL_EVAL_CONFIRMATION,
    build_f2_repair_partition_plan,
    evaluate_f2_repair_archival_condition,
)
from instinct.p2_harmful_write.run import validate

ROOT = Path(__file__).resolve().parents[1]

# -- config shape ------------------------------------------------------------


def test_f2_repair_config_names_its_stage_and_scoring_mode() -> None:
    config = load_proposal_config(ROOT / "configs/p2/p2_2/f2_repair.yaml")
    assert config.params["stage"] == "p2_2_r2"
    assert config.params["scoring_mode"] == "f2_repair"
    assert config.params["eval_partition_seed"] == 241
    assert config.params["f2_repair_source_offset"] == F2_REPAIR_SOURCE_OFFSET
    assert config.params["spearman_gate"] == 0.25
    assert config.params["minimum_accuracy_gain"] == 0.005
    assert config.seed == 251
    assert config.seeds == [251, 252]
    assert all(check.passed for check in validate(config))


def test_f2_repair_config_uses_fresh_seeds_disjoint_from_recovery2() -> None:
    repair = load_proposal_config(ROOT / "configs/p2/p2_2/f2_repair.yaml")
    recovery2 = load_proposal_config(ROOT / "configs/p2/p2_2/recovery2.yaml")
    assert set(repair.seeds).isdisjoint(set(recovery2.seeds))
    # The partition seed is a different, code-locked field and must stay 241.
    assert repair.params["eval_partition_seed"] == recovery2.params["eval_partition_seed"] == 241


# -- fresh-data provenance -------------------------------------------------


def _pilot_config(**overrides: object) -> BenchmarkConfig:
    defaults = dict(mode="pilot", n_steps=12, source_epochs=1)
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)  # type: ignore[arg-type]


def test_source_offset_is_past_every_prior_pilot_artifact() -> None:
    # pilot.py used pilot_source_images=2048; pilot2.py used 4096.
    assert F2_REPAIR_SOURCE_OFFSET == 4096
    assert F2_REPAIR_SOURCE_OFFSET >= 2048


def test_repair_partition_is_reproducible_for_a_fixed_seed() -> None:
    config = _pilot_config()
    first = build_f2_repair_partition_plan(config, audit_pool_size=64, downstream_label_pool_size=32)
    second = build_f2_repair_partition_plan(config, audit_pool_size=64, downstream_label_pool_size=32)
    assert first.audit_pool_indices == second.audit_pool_indices
    assert first.downstream_label_pool_indices == second.downstream_label_pool_indices
    assert first.manifest() == second.manifest()


def test_repair_partition_is_disjoint_from_itself_and_every_fixed_stream() -> None:
    config = _pilot_config(pilot_source_images=4096, pilot_stream_images=256)
    plan = build_f2_repair_partition_plan(config, audit_pool_size=4096, downstream_label_pool_size=1536)
    assert plan.disjoint
    assert set(plan.audit_pool_indices).isdisjoint(set(plan.downstream_label_pool_indices))
    for indices in plan.base.stream_indices().values():
        assert set(plan.audit_pool_indices).isdisjoint(set(indices))
        assert set(plan.downstream_label_pool_indices).isdisjoint(set(indices))
    assert set(plan.audit_pool_indices).isdisjoint(set(plan.base.source_train_indices))
    assert set(plan.downstream_label_pool_indices).isdisjoint(set(plan.base.source_train_indices))


def test_repair_partition_detects_overlap_when_pools_collide() -> None:
    from dataclasses import replace

    config = _pilot_config()
    plan = build_f2_repair_partition_plan(config, audit_pool_size=64, downstream_label_pool_size=32)
    colliding = replace(plan, downstream_label_pool_indices=plan.audit_pool_indices[:5])
    assert not colliding.disjoint


def test_repair_partition_rejects_full_mode() -> None:
    config = BenchmarkConfig(mode="full", full_eval_confirmation=FULL_EVAL_CONFIRMATION)
    with pytest.raises(ValueError, match="pilot-mode"):
        build_f2_repair_partition_plan(config, audit_pool_size=64, downstream_label_pool_size=32)


def test_repair_partition_rejects_pools_exceeding_the_source_band() -> None:
    config = _pilot_config()
    with pytest.raises(ValueError, match="exceed"):
        build_f2_repair_partition_plan(
            config, audit_pool_size=30_000, downstream_label_pool_size=10_000
        )


def test_repair_offset_must_stay_inside_the_source_pool() -> None:
    config = _pilot_config()
    with pytest.raises(ValueError, match="source pool"):
        build_f2_repair_partition_plan(
            config, repair_offset=40_000, audit_pool_size=64, downstream_label_pool_size=32
        )


def test_repair_pool_sizes_must_be_positive() -> None:
    config = _pilot_config()
    with pytest.raises(ValueError, match="positive"):
        build_f2_repair_partition_plan(config, audit_pool_size=0, downstream_label_pool_size=32)


# -- exact archival condition ----------------------------------------------


def test_archival_condition_fails_closed_when_identifiability_misses_any_fraction() -> None:
    result = evaluate_f2_repair_archival_condition(
        identifiability_by_fraction={
            0.1: (0.30, 0.30),
            0.25: (0.10, 0.30),  # damage rho misses the gate at this fraction
            0.5: (0.30, 0.30),
        },
        mechanism_gain_by_fraction={0.1: 0.01, 0.25: 0.01, 0.5: 0.01},
        oracle_beats_every_control_by_fraction={0.1: True, 0.25: True, 0.5: True},
    )
    assert not result["identifiability_passed"]
    assert result["disposition"] == "archive_unidentifiable_under_frozen_benchmark"


def test_archival_condition_archives_as_mechanism_failure_when_identifiable_but_no_gain() -> None:
    result = evaluate_f2_repair_archival_condition(
        identifiability_by_fraction={0.1: (0.30, 0.30), 0.25: (0.30, 0.30)},
        mechanism_gain_by_fraction={0.1: 0.001, 0.25: 0.004},
        oracle_beats_every_control_by_fraction={0.1: True, 0.25: True},
    )
    assert result["identifiability_passed"]
    assert not result["mechanism_ceiling_passed"]
    assert result["disposition"] == "archive_valid_mechanism_failure"


def test_archival_condition_archives_as_mechanism_failure_when_a_control_matches_the_oracle() -> None:
    """Gain alone is not enough -- reset/dense matching the oracle also fails the ceiling."""
    result = evaluate_f2_repair_archival_condition(
        identifiability_by_fraction={0.5: (0.30, 0.30)},
        mechanism_gain_by_fraction={0.5: 0.02},
        oracle_beats_every_control_by_fraction={0.5: False},
    )
    assert result["identifiability_passed"]
    assert not result["mechanism_ceiling_passed"]
    assert result["disposition"] == "archive_valid_mechanism_failure"


def test_archival_condition_proceeds_when_both_gates_clear_at_one_fraction() -> None:
    result = evaluate_f2_repair_archival_condition(
        identifiability_by_fraction={0.1: (0.20, 0.20), 0.5: (0.30, 0.30)},
        mechanism_gain_by_fraction={0.1: 0.001, 0.5: 0.01},
        oracle_beats_every_control_by_fraction={0.1: True, 0.5: True},
    )
    assert not result["identifiability_passed"]  # 0.1 misses the gate
    assert result["disposition"] == "archive_unidentifiable_under_frozen_benchmark"


def test_archival_condition_only_needs_one_fraction_to_clear_the_mechanism_ceiling() -> None:
    result = evaluate_f2_repair_archival_condition(
        identifiability_by_fraction={0.1: (0.25, 0.25), 0.25: (0.26, 0.40), 0.5: (0.40, 0.40)},
        mechanism_gain_by_fraction={0.1: 0.0, 0.25: 0.0, 0.5: 0.005},
        oracle_beats_every_control_by_fraction={0.1: True, 0.25: True, 0.5: True},
    )
    assert result["identifiability_passed"]
    assert result["mechanism_ceiling_passed"]
    assert result["mechanism_ceiling_by_fraction"] == {0.1: False, 0.25: False, 0.5: True}
    assert result["disposition"] == "proceed_reserved_test_partition_may_open"


def test_archival_condition_thresholds_are_inclusive_at_exact_equality() -> None:
    result = evaluate_f2_repair_archival_condition(
        identifiability_by_fraction={0.25: (0.25, 0.25)},
        mechanism_gain_by_fraction={0.25: 0.005},
        oracle_beats_every_control_by_fraction={0.25: True},
    )
    assert result["identifiability_passed"]
    assert result["mechanism_ceiling_passed"]
    assert result["disposition"] == "proceed_reserved_test_partition_may_open"


def test_archival_condition_requires_at_least_one_fraction() -> None:
    with pytest.raises(ValueError, match="at least one write fraction"):
        evaluate_f2_repair_archival_condition(
            identifiability_by_fraction={},
            mechanism_gain_by_fraction={},
            oracle_beats_every_control_by_fraction={},
        )
