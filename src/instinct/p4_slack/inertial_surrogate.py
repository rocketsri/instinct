"""Condition-calibrated recovery surrogate for the exact inertial grid family."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from instinct.p4_slack.inertial import inertial_grid, one_step_failure_floor
from instinct.p4_slack.slack import finite_horizon_replan_value

FloatArray = npt.NDArray[np.float64]
Condition = tuple[str, tuple[tuple[int, int], ...], float, int]


@dataclass(frozen=True, slots=True)
class InertialRecoverySurrogate:
    coefficients: FloatArray
    residual_radius: float
    v_fail: float
    state_coverage: float
    feature_version: str = "v1-dynamics"

    def predict(self, frame: pd.DataFrame) -> FloatArray:
        prediction = _features(frame, self.feature_version) @ self.coefficients
        failure = frame["failure"].to_numpy(dtype=bool)
        goal = frame["goal"].to_numpy(dtype=bool)
        return np.where(failure, self.v_fail, np.where(goal, 0.0, prediction))

    def interval(self, frame: pd.DataFrame) -> tuple[FloatArray, FloatArray]:
        prediction = self.predict(frame)
        terminal = (
            frame["failure"].to_numpy(dtype=bool)
            | frame["goal"].to_numpy(dtype=bool)
        )
        radius = np.where(terminal, 0.0, self.residual_radius)
        return prediction - radius, prediction + radius


def inertial_recovery_dataset(
    conditions: Sequence[Condition], *, v_fail: float
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for layout, obstacles, slip, horizon in conditions:
        frame = inertial_state_frame(
            layout=layout,
            obstacles=obstacles,
            slip=slip,
            horizon=horizon,
        )
        grid = inertial_grid(obstacles=obstacles, actuator_slip=slip)
        frame["exact_value"] = finite_horizon_replan_value(
            grid.mdp, horizon=horizon, v_fail=v_fail
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def inertial_state_frame(
    *,
    layout: str,
    obstacles: tuple[tuple[int, int], ...],
    slip: float,
    horizon: int,
) -> pd.DataFrame:
    """Create model inputs without computing or exposing exact recovery truth."""

    if len(obstacles) != 2:
        raise ValueError("inertial surrogate conditions require exactly two obstacles")
    rows: list[dict[str, float | int | str]] = []
    grid = inertial_grid(obstacles=obstacles, actuator_slip=slip)
    failure_floor = one_step_failure_floor(grid)
    action_failure = grid.mdp.P[:, :, grid.failure_state]
    for state, encoding in enumerate(grid.states):
        terminal_goal = state == grid.goal_state
        terminal_failure = state == grid.failure_state
        if encoding is None:
            x, y, vx, vy = grid.goal[0], grid.goal[1], 0, 0
        else:
            x, y, vx, vy = encoding
        row: dict[str, float | int | str] = {
            "condition_id": f"{layout}-S{slip:.3f}-H{horizon}",
            "layout": layout,
            "slip": slip,
            "horizon": horizon,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "goal": int(terminal_goal),
            "failure": int(terminal_failure),
            "failure_floor": float(failure_floor[state]),
            "progress": float(grid.progress()[state]),
        }
        sorted_obstacles = sorted(obstacles)
        for obstacle_index, (obstacle_x, obstacle_y) in enumerate(sorted_obstacles):
            row[f"obstacle_{obstacle_index}_x"] = obstacle_x
            row[f"obstacle_{obstacle_index}_y"] = obstacle_y
        for action in range(grid.mdp.n_actions):
            row[f"action_failure_{action}"] = float(action_failure[state, action])
        rows.append(row)
    return pd.DataFrame(rows)


def _features(frame: pd.DataFrame, version: str) -> FloatArray:
    common = [
        frame["x"].to_numpy(dtype=np.float64) / 3.0,
        frame["y"].to_numpy(dtype=np.float64) / 3.0,
        frame["vx"].to_numpy(dtype=np.float64) / 2.0,
        frame["vy"].to_numpy(dtype=np.float64) / 2.0,
        frame["slip"].to_numpy(dtype=np.float64) / 0.3,
        frame["horizon"].to_numpy(dtype=np.float64) / 16.0,
        frame["progress"].to_numpy(dtype=np.float64),
        frame["failure_floor"].to_numpy(dtype=np.float64),
        *(
            frame[f"action_failure_{action}"].to_numpy(dtype=np.float64)
            for action in range(5)
        ),
    ]
    if version == "v2-layout-geometry":
        x = frame["x"].to_numpy(dtype=np.float64)
        y = frame["y"].to_numpy(dtype=np.float64)
        for obstacle_index in range(2):
            obstacle_x = frame[f"obstacle_{obstacle_index}_x"].to_numpy(
                dtype=np.float64
            )
            obstacle_y = frame[f"obstacle_{obstacle_index}_y"].to_numpy(
                dtype=np.float64
            )
            common.extend(
                (
                    obstacle_x / 3.0,
                    obstacle_y / 3.0,
                    (x - obstacle_x) / 3.0,
                    (y - obstacle_y) / 3.0,
                    (np.abs(x - obstacle_x) + np.abs(y - obstacle_y)) / 6.0,
                )
            )
    elif version != "v1-dynamics":
        raise ValueError(f"unknown inertial feature version {version!r}")
    base = np.column_stack(common)
    columns: list[FloatArray] = [np.ones(len(frame), dtype=np.float64)]
    columns.extend(base[:, index] for index in range(base.shape[1]))
    columns.extend(base[:, index] ** 2 for index in range(base.shape[1]))
    for left in range(base.shape[1]):
        for right in range(left + 1, base.shape[1]):
            columns.append(base[:, left] * base[:, right])
    return np.column_stack(columns)


def fit_inertial_surrogate(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    v_fail: float,
    alpha: float = 0.1,
    state_coverage: float = 0.9,
    ridge: float = 1e-4,
    feature_version: str = "v1-dynamics",
) -> InertialRecoverySurrogate:
    if train.empty or calibration.empty:
        raise ValueError("train and calibration data must be nonempty")
    if not 0 < alpha < 1 or not 0 < state_coverage <= 1 or ridge < 0:
        raise ValueError("invalid calibration or ridge parameter")
    nonterminal = train.loc[
        ~(train["failure"].astype(bool) | train["goal"].astype(bool))
    ]
    x = _features(nonterminal, feature_version)
    y = nonterminal["exact_value"].to_numpy(dtype=np.float64)
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    provisional = InertialRecoverySurrogate(
        coefficients, 0.0, v_fail, state_coverage, feature_version
    )
    scored = calibration.copy()
    scored["absolute_error"] = np.abs(
        scored["exact_value"].to_numpy(dtype=np.float64)
        - provisional.predict(scored)
    )
    condition_scores = scored.groupby("condition_id")["absolute_error"].agg(
        lambda values: float(
            np.quantile(
                values.to_numpy(dtype=np.float64),
                state_coverage,
                method="higher",
            )
        )
    )
    n_conditions = len(condition_scores)
    level = min(1.0, np.ceil((n_conditions + 1) * (1 - alpha)) / n_conditions)
    radius = float(np.quantile(condition_scores, level, method="higher"))
    return InertialRecoverySurrogate(
        coefficients, radius, v_fail, state_coverage, feature_version
    )


def inertial_surrogate_metrics(
    surrogate: InertialRecoverySurrogate, frame: pd.DataFrame
) -> dict[str, float]:
    truth = frame["exact_value"].to_numpy(dtype=np.float64)
    prediction = surrogate.predict(frame)
    lower, upper = surrogate.interval(frame)
    evaluated = frame.copy()
    evaluated["covered"] = (truth >= lower) & (truth <= upper)
    condition_coverage = evaluated.groupby("condition_id")["covered"].mean()
    return {
        "mae": float(np.mean(np.abs(prediction - truth))),
        "coverage": float(np.mean(evaluated["covered"])),
        "condition_success_fraction": float(
            np.mean(
                condition_coverage.to_numpy(dtype=np.float64)
                >= surrogate.state_coverage
            )
        ),
        "minimum_condition_coverage": float(condition_coverage.min()),
        "radius": surrogate.residual_radius,
        "value_range": float(truth.max() - truth.min()),
        "n_conditions": float(len(condition_coverage)),
        "n": float(len(frame)),
    }
