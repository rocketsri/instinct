from __future__ import annotations

import numpy as np

from instinct.p4_slack.inertial_surrogate import (
    fit_inertial_surrogate,
    inertial_recovery_dataset,
    inertial_surrogate_metrics,
)


def test_inertial_surrogate_respects_structural_terminals_and_freezes() -> None:
    train = inertial_recovery_dataset(
        [("a", ((1, 1), (2, 1)), 0.1, 8)], v_fail=-1.0
    )
    calibration = inertial_recovery_dataset(
        [("b", ((1, 2), (2, 2)), 0.2, 10)], v_fail=-1.0
    )
    surrogate = fit_inertial_surrogate(
        train,
        calibration,
        v_fail=-1.0,
        alpha=0.2,
        state_coverage=0.8,
    )
    frozen = surrogate.coefficients.copy()
    prediction = surrogate.predict(calibration)
    assert np.all(prediction[calibration["failure"].astype(bool)] == -1.0)
    assert np.all(prediction[calibration["goal"].astype(bool)] == 0.0)
    metrics = inertial_surrogate_metrics(surrogate, calibration)
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert np.array_equal(surrogate.coefficients, frozen)


def test_inertial_dataset_keeps_complete_layout_condition_ids() -> None:
    conditions = [
        ("layout", ((1, 1), (2, 1)), 0.05, 8),
        ("layout", ((1, 1), (2, 1)), 0.15, 12),
    ]
    frame = inertial_recovery_dataset(conditions, v_fail=-1.0)
    assert frame["condition_id"].nunique() == 2
    assert set(frame["layout"]) == {"layout"}
    assert frame.groupby("condition_id").size().nunique() == 1
