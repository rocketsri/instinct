"""P1.1 failure-rich recovery, lookup labels, and external preflights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from instinct.core.configschema import load_proposal_config
from instinct.p1_atlas.ftt_adapter import FTTAdapter, inspect_external_ftt, load_t4_cells
from instinct.p1_atlas.pilot import PilotArtifacts, generate_pilot
from instinct.p1_atlas.recovery import validate_recovery

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def recovery() -> PilotArtifacts:
    config = load_proposal_config(ROOT / "configs/p1/p1_1/recovery1.yaml")
    return generate_pilot(config)


def test_recovery_config_uses_fresh_longer_units() -> None:
    config = load_proposal_config(ROOT / "configs/p1/p1_1/recovery1.yaml")
    assert all(check.passed for check in validate_recovery(config))
    assert set(config.pilot_seeds).isdisjoint(config.eval_seeds)
    assert min(config.pilot_seeds) >= 301
    assert config.params["horizon"] == 24


def test_failure_rich_distribution_produces_margins_failures_and_survivals(
    recovery: PilotArtifacts,
) -> None:
    recurring = recovery.metrics.loc[
        (recovery.metrics["record_type"] == "recurring_episode")
        & (recovery.metrics["family"] == "control")
    ]
    starts = recurring.drop_duplicates(["context_id", "unit_role", "unit_id"])
    margins = starts["initial_recovery_margin"].to_numpy(dtype=np.float64)
    assert (margins < 0).any() and (margins >= 0).any()
    assert recurring["failed"].astype(bool).any()
    assert (~recurring["failed"].astype(bool)).any()


def test_recovery_preserves_episodewise_sigma_identity(recovery: PilotArtifacts) -> None:
    decomposition = recovery.metrics.loc[
        recovery.metrics["record_type"] == "handoff_decomposition"
    ]
    assert decomposition["epsilon_id"].abs().max() < 1e-10


def test_ftt_adapter_matches_finite_budget_interface_without_claiming_equivalence() -> None:
    adapter = FTTAdapter((1, 4, 8), lambda observation: (observation[:, 0] > 0).astype(np.int64))
    selected = adapter.choose(
        np.array([[-1.0], [1.0]]),
        np.zeros((2, 3)),
        np.zeros((2, 1)),
    )
    assert selected.tolist() == [1, 4]
    preflight = inspect_external_ftt({})
    assert not preflight.official_equivalence_available


def test_t4_preflight_fails_closed_without_or_with_incomplete_cells(tmp_path: Path) -> None:
    passed, detail = load_t4_cells(None)
    assert not passed and "no T4" in detail
    path = tmp_path / "timing.json"
    path.write_text('{"cells": [{"device": "T4"}]}')
    passed, detail = load_t4_cells(path)
    assert not passed and "cells=1" in detail
