from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from instinct.core.configschema import load_proposal_config
from instinct.p1_atlas.pilot import PilotArtifacts, generate_pilot

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def pilot() -> PilotArtifacts:
    config = load_proposal_config(ROOT / "configs/p1/p1_1/smoke.yaml")
    return generate_pilot(config)


def test_p11_uses_whole_context_splits_and_all_heldout_crosses(
    pilot: PilotArtifacts,
) -> None:
    contexts = pilot.contexts
    assert set(contexts["split"]) == {"train", "validation", "test"}
    test = contexts.loc[contexts["split"] == "test"]
    assert set(zip(test["speed_index"], test["runtime_index"], strict=True)) == {
        (1, 2),
        (2, 1),
    }
    assert test.groupby(["family", "reflex"]).size().eq(2).all()

    recurring = pilot.metrics.loc[pilot.metrics["record_type"] == "recurring_episode"]
    budget_counts = recurring.groupby(["context_id", "unit_role", "unit_id"])[
        "budget"
    ].nunique()
    assert budget_counts.eq(8).all(), "budget points may not cross context/unit splits"


def test_p11_oracle_and_regret_units_are_disjoint_and_arms_fail_closed(
    pilot: PilotArtifacts,
) -> None:
    recurring = pilot.metrics.loc[pilot.metrics["record_type"] == "recurring_episode"]
    oracle = set(recurring.loc[recurring["unit_role"] == "oracle", "unit_id"])
    regret = set(recurring.loc[recurring["unit_role"] == "regret", "unit_id"])
    assert oracle.isdisjoint(regret)

    handoff = pilot.metrics.loc[pilot.metrics["record_type"] == "handoff_decomposition"]
    required = ["J_actual", "J_fresh", "J_instant", "J_base", "J_instant_base"]
    assert handoff[required].notna().all(axis=None)
    assert handoff["epsilon_id"].abs().max() < 1e-10


def test_p11_runner_has_exact_frozen_family_coverage(pilot: PilotArtifacts) -> None:
    failed = {check.name for check in pilot.controls if not check.passed}
    assert not failed
    assert pilot.full_family_coverage
    assert set(pilot.contexts["family"]) == {"pursuit", "tetris", "control"}
    summaries = pilot.metrics.loc[pilot.metrics["record_type"] == "model_summary"]
    assert {"M0", "M2", "M4", "M5", "FTT_LITE"} <= set(summaries["model_id"])
    assert pd.to_numeric(summaries["context_regret_ucb95"]).notna().all()
