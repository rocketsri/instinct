"""Small exact-label recovery surrogate for the bounded P4.1 pilot."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np
import numpy.typing as npt
import pandas as pd

from instinct.core.envs.tabular import corridor_with_pit
from instinct.core.rng import SeedScope
from instinct.p4_slack.slack import finite_horizon_replan_value

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RidgeRecoverySurrogate:
    coefficients: FloatArray
    residual_radius: float

    def predict(self, frame: pd.DataFrame) -> FloatArray:
        return _features(frame) @ self.coefficients

    def interval(self, frame: pd.DataFrame) -> tuple[FloatArray, FloatArray]:
        prediction = self.predict(frame)
        return prediction - self.residual_radius, prediction + self.residual_radius


@dataclass(frozen=True, slots=True)
class ConditionSplit:
    """Condition-level registry fixed before any exact labels are generated."""

    train: tuple[tuple[int, float, int], ...]
    calibration: tuple[tuple[int, float, int], ...]
    test: tuple[tuple[int, float, int], ...]


@dataclass(frozen=True, slots=True)
class StructuredRecoverySurrogate:
    """Polynomial recovery model with exact terminal boundary conditions."""

    coefficients: FloatArray
    residual_radius: float
    v_fail: float
    state_coverage: float

    def predict(self, frame: pd.DataFrame) -> FloatArray:
        prediction = _structured_features(frame) @ self.coefficients
        failure = frame["failure"].to_numpy(dtype=bool)
        goal = frame["goal"].to_numpy(dtype=bool)
        return np.where(failure, self.v_fail, np.where(goal, 0.0, prediction))

    def interval(self, frame: pd.DataFrame) -> tuple[FloatArray, FloatArray]:
        prediction = self.predict(frame)
        terminal = frame["failure"].to_numpy(dtype=bool) | frame["goal"].to_numpy(
            dtype=bool
        )
        radius = np.where(terminal, 0.0, self.residual_radius)
        return prediction - radius, prediction + radius


def _features(frame: pd.DataFrame) -> FloatArray:
    position = frame["position"].to_numpy(dtype=np.float64)
    length = frame["length"].to_numpy(dtype=np.float64)
    slip = frame["slip"].to_numpy(dtype=np.float64)
    horizon = frame["horizon"].to_numpy(dtype=np.float64) / 12.0
    progress = position / np.maximum(length, 1.0)
    failure = frame["failure"].to_numpy(dtype=np.float64)
    goal = frame["goal"].to_numpy(dtype=np.float64)
    return np.column_stack(
        (
            np.ones(len(frame)),
            progress,
            progress**2,
            slip,
            slip**2,
            horizon,
            horizon**2,
            progress * slip,
            progress * horizon,
            slip * horizon,
            failure,
            goal,
        )
    )


def _structured_features(frame: pd.DataFrame) -> FloatArray:
    """Cubic condition/state basis fixed before the v4 evaluation split.

    Inputs are scaled by the preregistered condition-grid center and span. The
    basis includes every monomial through degree three, avoiding a hand-picked
    interaction chosen after evaluation.
    """
    position = frame["position"].to_numpy(dtype=np.float64)
    length = frame["length"].to_numpy(dtype=np.float64)
    base = np.column_stack(
        (
            position / np.maximum(length, 1.0),
            (length - 7.0) / 3.0,
            (frame["slip"].to_numpy(dtype=np.float64) - 0.18) / 0.12,
            (frame["horizon"].to_numpy(dtype=np.float64) - 8.0) / 4.0,
        )
    )
    columns: list[FloatArray] = [np.ones(len(frame), dtype=np.float64)]
    for degree in range(1, 4):
        for indices in combinations_with_replacement(range(base.shape[1]), degree):
            columns.append(np.prod(base[:, indices], axis=1))
    return np.column_stack(columns)


def condition_split_registry(
    lengths: Sequence[int],
    slips: Sequence[float],
    horizons: Sequence[int],
    *,
    split_seed: int,
    train_fraction: float = 0.5,
    calibration_fraction: float = 0.25,
    excluded: Sequence[tuple[int, float, int]] = (),
) -> ConditionSplit:
    """Create a reproducible label-blind split over complete conditions."""
    if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1:
        raise ValueError("split fractions must lie in (0,1)")
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("train and calibration fractions must leave test conditions")
    excluded_set = {(int(length), float(slip), int(horizon)) for length, slip, horizon in excluded}
    conditions = tuple(
        (int(length), float(slip), int(horizon))
        for length in lengths
        for slip in slips
        for horizon in horizons
        if (int(length), float(slip), int(horizon)) not in excluded_set
    )
    if len(set(conditions)) != len(conditions) or len(conditions) < 4:
        raise ValueError("condition grid must contain at least four unique conditions")
    condition_ids = np.arange(len(conditions), dtype=np.int64)
    keys = SeedScope(split_seed).stream("p4.1-v4-condition-split").uniform(
        condition_ids, count=1
    )[:, 0]
    order = np.argsort(keys, kind="stable")
    n_train = max(1, int(np.floor(train_fraction * len(conditions))))
    n_calibration = max(1, int(np.floor(calibration_fraction * len(conditions))))
    if n_train + n_calibration >= len(conditions):
        raise ValueError("split fractions leave no test conditions")

    def selected(indices: npt.NDArray[np.int64]) -> tuple[tuple[int, float, int], ...]:
        return tuple(conditions[int(index)] for index in indices)

    return ConditionSplit(
        train=selected(order[:n_train]),
        calibration=selected(order[n_train : n_train + n_calibration]),
        test=selected(order[n_train + n_calibration :]),
    )


def exact_recovery_dataset(
    conditions: list[tuple[int, float, int]],
    *,
    v_fail: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for length, slip, horizon in conditions:
        mdp = corridor_with_pit(length=length, gamma=0.95, slip=slip)
        values = finite_horizon_replan_value(mdp, horizon=horizon, v_fail=v_fail)
        failure_mask = mdp.failure
        if failure_mask is None:  # pragma: no cover - corridor contract
            raise ValueError("recovery dataset requires explicit failure states")
        for state, value in enumerate(values):
            rows.append(
                {
                    "condition_id": f"L{length}-S{slip:.6f}-H{horizon}",
                    "length": length,
                    "slip": slip,
                    "horizon": horizon,
                    "position": min(state, length),
                    "failure": int(bool(failure_mask[state])),
                    "goal": int(state == length),
                    "exact_value": float(value),
                }
            )
    return pd.DataFrame(rows)


def fit_calibrated_surrogate(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    alpha: float = 0.1,
    ridge: float = 1e-6,
) -> RidgeRecoverySurrogate:
    if not 0 < alpha < 1 or ridge < 0:
        raise ValueError("alpha must lie in (0,1) and ridge must be nonnegative")
    if train.empty or calibration.empty:
        raise ValueError("train and calibration sets must be nonempty")
    x = _features(train)
    y = train["exact_value"].to_numpy(dtype=np.float64)
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    calibration_error = np.abs(
        calibration["exact_value"].to_numpy(dtype=np.float64)
        - _features(calibration) @ coefficients
    )
    level = min(1.0, np.ceil((len(calibration_error) + 1) * (1 - alpha)) / len(calibration_error))
    radius = float(np.quantile(calibration_error, level, method="higher"))
    return RidgeRecoverySurrogate(coefficients, radius)


def fit_condition_calibrated_surrogate(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    v_fail: float,
    alpha: float = 0.1,
    state_coverage: float = 0.9,
    ridge: float = 1e-5,
) -> StructuredRecoverySurrogate:
    """Fit and condition-calibrate without treating correlated states as iid.

    Each calibration condition first contributes its empirical
    ``state_coverage`` residual quantile. A finite-sample split-conformal upper
    quantile is then taken across conditions. Thus long corridors cannot
    silently receive more calibration weight merely because they contain more
    state labels.
    """
    if not 0 < alpha < 1 or not 0 < state_coverage <= 1 or ridge < 0:
        raise ValueError("invalid alpha, state_coverage, or ridge")
    if train.empty or calibration.empty:
        raise ValueError("train and calibration sets must be nonempty")
    if "condition_id" not in calibration:
        raise ValueError("calibration rows require condition_id")
    train_nonterminal = train.loc[~(train["failure"].astype(bool) | train["goal"].astype(bool))]
    if train_nonterminal.empty:
        raise ValueError("training requires nonterminal exact labels")
    x = _structured_features(train_nonterminal)
    y = train_nonterminal["exact_value"].to_numpy(dtype=np.float64)
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    provisional = StructuredRecoverySurrogate(coefficients, 0.0, v_fail, state_coverage)
    calibration = calibration.copy()
    calibration["absolute_error"] = np.abs(
        calibration["exact_value"].to_numpy(dtype=np.float64)
        - provisional.predict(calibration)
    )
    condition_scores = calibration.groupby("condition_id")["absolute_error"].agg(
        lambda values: float(
            np.quantile(values.to_numpy(dtype=np.float64), state_coverage, method="higher")
        )
    )
    n_conditions = len(condition_scores)
    level = min(
        1.0,
        np.ceil((n_conditions + 1) * (1 - alpha)) / n_conditions,
    )
    radius = float(np.quantile(condition_scores, level, method="higher"))
    return StructuredRecoverySurrogate(coefficients, radius, v_fail, state_coverage)


def surrogate_metrics(
    surrogate: RidgeRecoverySurrogate | StructuredRecoverySurrogate,
    frame: pd.DataFrame,
) -> dict[str, float]:
    truth = frame["exact_value"].to_numpy(dtype=np.float64)
    prediction = surrogate.predict(frame)
    lower, upper = surrogate.interval(frame)
    return {
        "mae": float(np.mean(np.abs(prediction - truth))),
        "coverage": float(np.mean((truth >= lower) & (truth <= upper))),
        "worst_overestimate": float(np.max(prediction - truth)),
        "radius": surrogate.residual_radius,
        "n": float(len(frame)),
    }


def condition_coverage_metrics(
    surrogate: StructuredRecoverySurrogate,
    frame: pd.DataFrame,
) -> dict[str, float]:
    """Condition-balanced interval diagnostics for the repaired estimator."""
    lower, upper = surrogate.interval(frame)
    evaluated = frame.copy()
    truth = evaluated["exact_value"].to_numpy(dtype=np.float64)
    evaluated["covered"] = (truth >= lower) & (truth <= upper)
    by_condition = evaluated.groupby("condition_id")["covered"].mean()
    return {
        "mean_condition_coverage": float(by_condition.mean()),
        "minimum_condition_coverage": float(by_condition.min()),
        "condition_success_fraction": float(
            np.mean(by_condition.to_numpy(dtype=np.float64) >= surrogate.state_coverage)
        ),
        "n_conditions": float(len(by_condition)),
    }
