from __future__ import annotations

import pandas as pd
import pytest

from instinct.p1_atlas.controls import conditional_surface
from instinct.p1_atlas.models import CONTEXT_COLUMNS, MODEL_SPECS, evaluate_registry


def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        frame.loc[frame.split == "train"].copy(),
        frame.loc[frame.split == "test"].copy(),
    )


def test_registry_freezes_m0_m5_information_restrictions() -> None:
    assert [spec.model_id for spec in MODEL_SPECS] == [f"M{i}" for i in range(6)]
    information = {spec.model_id: set(spec.information) for spec in MODEL_SPECS}
    assert information["M0"] == {"budget"}
    assert information["M1"] == {"budget"}
    assert information["M2"] == {"budget", "nu_e*nu_h*budget"}
    assert information["M3"] == {"budget", "nu_e", "nu_h"}
    assert set(CONTEXT_COLUMNS) <= information["M4"]
    assert set(CONTEXT_COLUMNS) <= information["M5"]


def test_registry_rejects_context_leakage() -> None:
    train, test = _split(conditional_surface())
    leaked = test.copy()
    leaked.loc[leaked.context_id == leaked.context_id.iloc[0], "context_id"] = (
        train.context_id.iloc[0]
    )
    with pytest.raises(ValueError, match="context overlap"):
        evaluate_registry(train, leaked)


def test_conditional_model_wins_on_separate_hardware_surface() -> None:
    result = evaluate_registry(*_split(conditional_surface()))
    m4 = result.score("M4")
    assert m4.identifiable
    assert m4.mean_budget_regret < 0.2 * m4.target_range
    assert all(
        m4.mean_budget_regret < result.score(model).mean_budget_regret
        for model in ("M0", "M2", "M5")
    )


def test_product_model_recovers_imposed_collapse() -> None:
    result = evaluate_registry(*_split(conditional_surface(collapse=True)))
    assert result.score("M2").mean_budget_regret < 1e-9


def test_flat_frontiers_are_not_identifiable() -> None:
    result = evaluate_registry(*_split(conditional_surface(flat=True)))
    assert not any(score.identifiable for score in result.scores)


def test_registry_scores_are_invariant_to_row_order() -> None:
    train, test = _split(conditional_surface())
    expected = evaluate_registry(train, test)
    shuffled = evaluate_registry(
        train.sample(frac=1.0, random_state=7),
        test.sample(frac=1.0, random_state=11),
    )
    for spec in MODEL_SPECS:
        left = expected.score(spec.model_id)
        right = shuffled.score(spec.model_id)
        assert right.mean_budget_regret == pytest.approx(left.mean_budget_regret, abs=1e-12)
        assert right.exact_argmax_rate == pytest.approx(left.exact_argmax_rate, abs=1e-12)
