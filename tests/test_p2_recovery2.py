"""Final P2 realistic-recovery artifact contract and fail-closed behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from instinct.core.configschema import load_proposal_config
from instinct.p2_harmful_write.recovery2 import (
    CIFAR10_ARCHIVE,
    RESNET18_WEIGHTS,
    inspect_recovery2,
)
from instinct.p2_harmful_write.recovery2_benchmark import (
    ARTIFACT_SCHEMA,
    FULL_EVAL_CONFIRMATION,
    BenchmarkConfig,
    build_partition_plan,
    compute_utility,
    evaluate_p7_qualification,
    restricted_top_m,
    write_artifact,
)
from instinct.p2_harmful_write.run import validate

ROOT = Path(__file__).resolve().parents[1]


def test_recovery2_config_names_the_final_frozen_stage() -> None:
    config = load_proposal_config(ROOT / "configs/p2/p2_2/recovery2.yaml")
    assert config.params["stage"] == "p2_2_r2"
    assert all(check.passed for check in validate(config))
    assert config.params["require_measured_wall_clock_match"] is True


def test_exact_official_artifact_contract_is_frozen() -> None:
    assert CIFAR10_ARCHIVE.name == "cifar-10-python.tar.gz"
    assert CIFAR10_ARCHIVE.expected_bytes == 170_498_071
    assert CIFAR10_ARCHIVE.checksum == "c58f30108f718f92721af3b95e74349a"
    assert RESNET18_WEIGHTS.name == "resnet18-f37072fd.pth"
    assert RESNET18_WEIGHTS.expected_bytes == 46_830_571
    assert RESNET18_WEIGHTS.checksum == (
        "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
    )


def test_missing_artifacts_fail_before_eval_units_are_opened() -> None:
    report = inspect_recovery2({})
    assert not report.realistic_contract_passed
    assert not report.qualifies_p7
    assert all(not artifact.passed for artifact in report.artifacts)
    ranges = list(report.fixed_partitions.values())
    assert ranges == [(0, 2500), (2500, 5000), (5000, 7500), (7500, 10000)]


def test_wrong_size_and_checksum_cannot_masquerade_as_artifacts(tmp_path: Path) -> None:
    dataset = tmp_path / CIFAR10_ARCHIVE.name
    weights = tmp_path / RESNET18_WEIGHTS.name
    dataset.write_bytes(b"not cifar")
    weights.write_bytes(b"not weights")
    report = inspect_recovery2({"cifar10_archive": str(dataset), "resnet18_weights": str(weights)})
    assert all(artifact.exists for artifact in report.artifacts)
    assert all(not artifact.size_matches for artifact in report.artifacts)
    assert all(not artifact.checksum_matches for artifact in report.artifacts)
    assert not report.realistic_contract_passed
    assert not report.qualifies_p7


def test_pilot_partition_never_opens_or_overlaps_reserved_test_units() -> None:
    config = BenchmarkConfig(
        mode="pilot",
        n_steps=12,
        source_epochs=1,
        pilot_source_images=128,
        pilot_stream_images=48,
    )
    plan = build_partition_plan(config)
    assert plan.dataset_split == "train"
    assert not plan.opens_reserved_test_partition
    assert plan.disjoint
    assert len(plan.source_train_indices) == 128
    assert all(len(indices) == 48 for indices in plan.stream_indices().values())


def test_full_partition_is_exact_seeded_test_quarters_and_requires_confirmation() -> None:
    with pytest.raises(ValueError, match="confirmation"):
        BenchmarkConfig(mode="full")
    config = BenchmarkConfig(
        mode="full",
        full_eval_confirmation=FULL_EVAL_CONFIRMATION,
    )
    plan = build_partition_plan(config)
    assert plan.dataset_split == "test"
    assert plan.opens_reserved_test_partition
    assert plan.disjoint
    assert len(plan.source_train_indices) == 50_000
    assert [len(indices) for indices in plan.stream_indices().values()] == [2500] * 4
    assert set().union(*(set(values) for values in plan.stream_indices().values())) == set(
        range(10_000)
    )


def test_candidate_restriction_is_required_before_top_m() -> None:
    eligible = (4, 5, 6, 7)
    assert restricted_top_m(eligible, [0.2, 0.9, 0.1, 0.8], 2) == (5, 7)
    with pytest.raises(ValueError, match="already be restricted"):
        restricted_top_m(eligible, [0.2, 0.9, 0.1, 0.8, 100.0], 2)
    with pytest.raises(ValueError, match="unique"):
        restricted_top_m((4, 4), [0.2, 0.9], 1)


def test_multi_horizon_terms_keep_frozen_signs_and_separate_outputs() -> None:
    terms = compute_utility(0.4, 0.1, 0.02, retention_lambda=1.5)
    assert terms.benefit == 0.4
    assert terms.damage == 0.1
    assert terms.cost == 0.02
    assert terms.utility == pytest.approx(0.4 - 1.5 * 0.1 - 0.02)


def test_p7_qualification_fails_closed_for_smoke_even_if_other_gates_pass() -> None:
    common = dict(
        artifacts_verified=True,
        t4_verified=True,
        partitions_disjoint=True,
        downstream_opened_after_lock=True,
        full_horizons=True,
        candidate_restriction_passed=True,
        exact_counts=True,
        compute_matched=True,
        baselines_complete=True,
        source_accuracy=0.8,
        minimum_source_accuracy=0.45,
        positive_utility_fraction=0.25,
        minimum_positive_fraction=0.05,
        maximum_positive_fraction=0.9,
        oracle_frontier_gain=0.02,
        minimum_accuracy_gain=0.005,
    )
    pilot = evaluate_p7_qualification(mode="pilot", **common)
    assert not pilot["checks"]["full_realistic_mode"]
    assert not pilot["qualifies_p7"]
    full = evaluate_p7_qualification(mode="full", **common)
    assert all(full["checks"].values())
    assert full["qualifies_p7"]


def test_structured_artifact_writer_is_strict_and_atomic(tmp_path: Path) -> None:
    destination = tmp_path / "p2-r2.json"
    write_artifact(
        {
            "schema": ARTIFACT_SCHEMA,
            "theory": {"theory_id": "spec-5.1-5.10"},
            "qualifies_p7": False,
        },
        destination,
    )
    assert json.loads(destination.read_text())["qualifies_p7"] is False
    assert not destination.with_suffix(".json.partial").exists()
