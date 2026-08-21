"""P1 final recovery: shrinkage leakage guards and fresh-unit integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from instinct.core.configschema import load_proposal_config
from instinct.p1_atlas.controls import conditional_surface
from instinct.p1_atlas.models import uncertainty_shrunk_atlas
from instinct.p1_atlas.pilot import PilotArtifacts, generate_pilot
from instinct.p1_atlas.recovery2 import validate_recovery2

ROOT = Path(__file__).resolve().parents[1]


def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        frame.loc[frame["split"] == "train"].copy(),
        frame.loc[frame["split"] == "test"].copy(),
    )


def test_shrinkage_predictions_do_not_read_evaluation_sigma() -> None:
    train, test = _split(conditional_surface())
    expected = uncertainty_shrunk_atlas(train, test)
    perturbed = test.copy()
    perturbed["sigma"] = np.linspace(-1e6, 1e6, len(perturbed))
    actual = uncertainty_shrunk_atlas(train, perturbed)
    assert actual.choices == expected.choices
    assert actual.predictions["prediction"].to_numpy() == pytest.approx(
        expected.predictions["prediction"].to_numpy(), abs=1e-12
    )
    assert actual.predictions["m4_weight"].between(0.0, 1.0).all()


def test_recovery2_config_preregisters_fresh_disjoint_units() -> None:
    config = load_proposal_config(ROOT / "configs/p1/p1_1/recovery2.yaml")
    assert all(check.passed for check in validate_recovery2(config))
    assert set(config.pilot_seeds).isdisjoint(config.eval_seeds)
    assert min(config.pilot_seeds) >= 501
    assert config.params["shrinkage_confidence"] == 0.95


@pytest.fixture(scope="module")
def recovery2() -> PilotArtifacts:
    config = load_proposal_config(ROOT / "configs/p1/p1_1/recovery2.yaml")
    return generate_pilot(config)


def test_recovery2_reports_original_registry_shrinkage_and_r1_champion(
    recovery2: PilotArtifacts,
) -> None:
    models = set(recovery2.model_summary["model_id"])
    assert {f"M{index}" for index in range(6)} <= models
    assert {"M4S", "SPEED", "R1_SPEED_CHAMPION"} <= models
    speed = recovery2.model_summary.set_index("model_id")
    assert speed.loc["SPEED", "mean_context_regret"] == pytest.approx(
        speed.loc["R1_SPEED_CHAMPION", "mean_context_regret"], abs=1e-12
    )
    shrinkage = recovery2.metrics.loc[
        recovery2.metrics["record_type"] == "recovery2_shrinkage"
    ]
    assert len(shrinkage) == 1
    assert not bool(shrinkage.iloc[0]["fit_uses_evaluation_sigma"])


def test_recovery2_keeps_whole_contexts_and_sigma_identity_disjoint(
    recovery2: PilotArtifacts,
) -> None:
    contexts = recovery2.contexts
    train = set(contexts.loc[contexts["split"] == "train", "context_id"])
    test = set(contexts.loc[contexts["split"] == "test", "context_id"])
    assert train.isdisjoint(test)
    decomposition = recovery2.metrics.loc[
        recovery2.metrics["record_type"] == "handoff_decomposition"
    ]
    assert decomposition["epsilon_id"].abs().max() < 1e-10
