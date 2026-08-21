from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from instinct.p4_slack.surrogate import (
    condition_coverage_metrics,
    condition_split_registry,
    exact_recovery_dataset,
    fit_calibrated_surrogate,
    fit_condition_calibrated_surrogate,
    surrogate_metrics,
)


def test_calibration_is_frozen_and_covers_heldout_conditions() -> None:
    train = exact_recovery_dataset(
        [(5, 0.05, 4), (5, 0.3, 12), (7, 0.05, 12), (7, 0.3, 4)],
        v_fail=-1.0,
    )
    calibration = exact_recovery_dataset([(6, 0.15, 6), (6, 0.15, 10)], v_fail=-1.0)
    test = exact_recovery_dataset([(8, 0.2, 8)], v_fail=-1.0)
    surrogate = fit_calibrated_surrogate(train, calibration, alpha=0.1)
    frozen = surrogate.coefficients.copy()
    metrics = surrogate_metrics(surrogate, test)
    # This frozen split is a deliberate extrapolation failure: the instrument
    # must expose undercoverage rather than tune its radius on the test set.
    assert metrics["coverage"] == pytest.approx(0.7)
    assert metrics["mae"] < 0.5
    assert np.array_equal(surrogate.coefficients, frozen)


def test_invalid_calibration_parameters_are_rejected() -> None:
    frame = exact_recovery_dataset([(5, 0.1, 4)], v_fail=-1.0)
    for alpha in (0.0, 1.0):
        try:
            fit_calibrated_surrogate(frame, frame, alpha=alpha)
        except ValueError:
            pass
        else:  # pragma: no cover - explicit failure branch
            raise AssertionError("invalid alpha accepted")


def test_v4_condition_registry_is_label_blind_disjoint_and_reproducible() -> None:
    first = condition_split_registry(
        [4, 5, 6, 7, 8, 9, 10],
        [0.08, 0.14, 0.22, 0.28],
        [5, 7, 9, 11],
        split_seed=4417,
        excluded=[(8, 0.2, 8)],
    )
    second = condition_split_registry(
        [4, 5, 6, 7, 8, 9, 10],
        [0.08, 0.14, 0.22, 0.28],
        [5, 7, 9, 11],
        split_seed=4417,
        excluded=[(8, 0.2, 8)],
    )
    assert first == second
    assert (len(first.train), len(first.calibration), len(first.test)) == (56, 28, 28)
    split_sets = [set(first.train), set(first.calibration), set(first.test)]
    assert split_sets[0].isdisjoint(split_sets[1])
    assert split_sets[0].isdisjoint(split_sets[2])
    assert split_sets[1].isdisjoint(split_sets[2])
    assert all((8, 0.2, 8) not in split for split in split_sets)


def test_v4_structural_terminals_and_condition_balanced_calibration() -> None:
    train = exact_recovery_dataset(
        [(4, 0.08, 5), (6, 0.14, 9), (8, 0.22, 7), (10, 0.28, 11)],
        v_fail=-1.0,
    )
    calibration = exact_recovery_dataset(
        [(5, 0.08, 7), (7, 0.14, 11), (9, 0.22, 5), (10, 0.28, 9)],
        v_fail=-1.0,
    )
    surrogate = fit_condition_calibrated_surrogate(
        train,
        calibration,
        v_fail=-1.0,
        alpha=0.25,
        state_coverage=0.8,
    )
    prediction = surrogate.predict(calibration)
    np.testing.assert_array_equal(
        prediction[calibration["failure"].astype(bool).to_numpy()], np.array([-1.0] * 4)
    )
    np.testing.assert_array_equal(
        prediction[calibration["goal"].astype(bool).to_numpy()], np.array([0.0] * 4)
    )
    first_condition = calibration.iloc[0]["condition_id"]
    duplicated_condition = pd.concat(
        [calibration, calibration.loc[calibration["condition_id"] == first_condition]],
        ignore_index=True,
    )
    duplicated_fit = fit_condition_calibrated_surrogate(
        train,
        duplicated_condition,
        v_fail=-1.0,
        alpha=0.25,
        state_coverage=0.8,
    )
    assert duplicated_fit.residual_radius == pytest.approx(surrogate.residual_radius)
    diagnostics = condition_coverage_metrics(surrogate, calibration)
    assert diagnostics["n_conditions"] == 4
