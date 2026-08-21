"""P1's preregistered M0--M5 comparison on whole held-out contexts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats as scipy_stats

FloatArray = npt.NDArray[np.float64]

CONTEXT_COLUMNS = (
    "nu_e",
    "nu_h",
    "state_descriptor",
    "reflex_descriptor",
    "environment_descriptor",
    "hardware_quality",
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    label: str
    information: tuple[str, ...]


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("M0", "global fixed budget", ("budget",)),
    ModelSpec("M1", "budget-only curve", ("budget",)),
    ModelSpec("M2", "product-collapse curve", ("budget", "nu_e*nu_h*budget")),
    ModelSpec("M3", "separate speed/hardware axes", ("budget", "nu_e", "nu_h")),
    ModelSpec(
        "M4",
        "conditional descriptor model",
        ("budget", *CONTEXT_COLUMNS),
    ),
    ModelSpec("M5", "nearest-context lookup", ("budget", *CONTEXT_COLUMNS)),
)


@dataclass(frozen=True, slots=True)
class ModelScore:
    model_id: str
    mean_budget_regret: float
    max_budget_regret: float
    exact_argmax_rate: float
    n_contexts: int
    target_range: float
    identifiable: bool


@dataclass(frozen=True, slots=True)
class RegistryResult:
    scores: tuple[ModelScore, ...]
    predictions: pd.DataFrame

    def score(self, model_id: str) -> ModelScore:
        return next(score for score in self.scores if score.model_id == model_id)


@dataclass(frozen=True, slots=True)
class ShrunkAtlasResult:
    """Training-only uncertainty gate and predictions for recovery2's M4S."""

    choices: dict[str, int]
    predictions: pd.DataFrame
    raw_alpha: float
    active_alpha: float
    improvement_lcb95: float
    distance_scale: float


def _require_frame(frame: pd.DataFrame) -> None:
    required = {"context_id", "budget", "sigma", *CONTEXT_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"conditional atlas frame missing {sorted(missing)}")
    counts = frame.groupby("context_id")["budget"].nunique()
    if counts.nunique() != 1:
        raise ValueError("every context must expose the same candidate budget grid")


def _features(frame: pd.DataFrame, model_id: str) -> FloatArray:
    k = frame["budget"].to_numpy(dtype=np.float64)
    columns: list[FloatArray] = [np.ones(len(frame)), k, k**2]
    if model_id == "M2":
        delta = frame["nu_e"].to_numpy() * frame["nu_h"].to_numpy() * k
        columns.extend((delta, delta * k))
    elif model_id == "M3":
        for name in ("nu_e", "nu_h"):
            z = frame[name].to_numpy(dtype=np.float64)
            columns.extend((z, z * k, z * k**2))
        product = frame["nu_e"].to_numpy() * frame["nu_h"].to_numpy()
        columns.extend((product, product * k, product * k**2))
    elif model_id == "M4":
        for name in CONTEXT_COLUMNS:
            z = frame[name].to_numpy(dtype=np.float64)
            columns.extend((z, z * k, z * k**2))
        product = frame["nu_e"].to_numpy() * frame["nu_h"].to_numpy()
        columns.extend((product, product * k, product * k**2))
    return np.column_stack(columns)


def _ridge_predict(train: pd.DataFrame, test: pd.DataFrame, model_id: str) -> FloatArray:
    x_train = _features(train, model_id)
    x_test = _features(test, model_id)
    y = train["sigma"].to_numpy(dtype=np.float64)
    penalty = np.eye(x_train.shape[1]) * 1e-8
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y)
    return np.asarray(x_test @ beta, dtype=np.float64)


def _m0_predict(train: pd.DataFrame, test: pd.DataFrame) -> FloatArray:
    means = train.groupby("budget")["sigma"].mean()
    chosen = int(means.idxmax())
    return (test["budget"].to_numpy(dtype=np.int64) == chosen).astype(np.float64)


def _m5_predict(train: pd.DataFrame, test: pd.DataFrame) -> FloatArray:
    train_context = train.drop_duplicates("context_id").set_index("context_id")
    test_context = test.drop_duplicates("context_id").set_index("context_id")
    train_x = train_context[list(CONTEXT_COLUMNS)].to_numpy(dtype=np.float64)
    test_x = test_context[list(CONTEXT_COLUMNS)].to_numpy(dtype=np.float64)
    scale = np.std(train_x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    distance = np.sum(((test_x[:, None, :] - train_x[None, :, :]) / scale) ** 2, axis=2)
    nearest = np.argmin(distance, axis=1)
    nearest_id = {
        test_id: train_context.index[index]
        for test_id, index in zip(test_context.index, nearest, strict=True)
    }
    lookup = train.set_index(["context_id", "budget"])["sigma"]
    return np.array(
        [lookup.loc[(nearest_id[row.context_id], row.budget)] for row in test.itertuples()],
        dtype=np.float64,
    )


def _one_sided_lower(values: FloatArray, confidence: float) -> float:
    if values.size < 2 or not np.isfinite(values).all():
        return float("-inf")
    sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
    critical = float(scipy_stats.t.ppf(confidence, values.size - 1))
    return float(np.mean(values)) - critical * sem


def _standardized_nearest_distances(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[dict[str, float], float]:
    train_context = train.drop_duplicates("context_id").sort_values("context_id")
    test_context = test.drop_duplicates("context_id").sort_values("context_id")
    train_x = train_context[list(CONTEXT_COLUMNS)].to_numpy(dtype=np.float64)
    test_x = test_context[list(CONTEXT_COLUMNS)].to_numpy(dtype=np.float64)
    scale = np.std(train_x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    train_scaled = train_x / scale
    test_scaled = test_x / scale

    train_pairwise = np.sqrt(
        np.sum((train_scaled[:, None, :] - train_scaled[None, :, :]) ** 2, axis=2)
    )
    np.fill_diagonal(train_pairwise, np.inf)
    loo_distances = np.min(train_pairwise, axis=1)
    distance_scale = max(float(np.quantile(loo_distances, 0.9)), 1e-12)
    test_distances = np.sqrt(
        np.sum((test_scaled[:, None, :] - train_scaled[None, :, :]) ** 2, axis=2)
    ).min(axis=1)
    return (
        {
            str(context_id): float(distance)
            for context_id, distance in zip(
                test_context["context_id"], test_distances, strict=True
            )
        },
        distance_scale,
    )


def uncertainty_shrunk_atlas(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    confidence: float = 0.95,
) -> ShrunkAtlasResult:
    """Fit recovery2's preregistered M4/M5 stack without test-surface leakage.

    The convex coefficient and its uncertainty gate use leave-one-whole-context-
    out predictions on ``train``.  Test descriptors affect only the prespecified
    support-distance attenuation; test ``sigma`` values are never read.
    """

    _require_frame(train)
    _require_frame(test)
    overlap = set(train["context_id"]) & set(test["context_id"])
    if overlap:
        raise ValueError(f"train/test context overlap: {sorted(overlap)[:3]}")
    context_ids = sorted(str(value) for value in train["context_id"].unique())
    if len(context_ids) < 4:
        raise ValueError("M4S requires at least four whole training contexts")
    if not 0.5 < confidence < 1.0:
        raise ValueError("M4S confidence must lie strictly between 0.5 and 1")

    oof_rows: list[pd.DataFrame] = []
    for context_id in context_ids:
        heldout = train.loc[train["context_id"].astype(str) == context_id].copy()
        fitted = train.loc[train["context_id"].astype(str) != context_id].copy()
        fold = heldout[["context_id", "budget", "sigma"]].copy()
        fold["m4_prediction"] = _ridge_predict(fitted, heldout, "M4")
        fold["m5_prediction"] = _m5_predict(fitted, heldout)
        oof_rows.append(fold)
    oof = pd.concat(oof_rows, ignore_index=True)
    delta = (
        oof["m4_prediction"].to_numpy(dtype=np.float64)
        - oof["m5_prediction"].to_numpy(dtype=np.float64)
    )
    target = (
        oof["sigma"].to_numpy(dtype=np.float64)
        - oof["m5_prediction"].to_numpy(dtype=np.float64)
    )
    denominator = float(delta @ delta)
    raw_alpha = (
        float(np.clip(float(delta @ target) / denominator, 0.0, 1.0))
        if denominator > 1e-12
        else 0.0
    )
    oof["stack_prediction"] = oof["m5_prediction"] + raw_alpha * delta
    context_improvements = np.asarray(
        [
            float(
                np.mean(
                    np.square(group["m5_prediction"] - group["sigma"])
                    - np.square(group["stack_prediction"] - group["sigma"])
                )
            )
            for _, group in oof.groupby("context_id", sort=True)
        ],
        dtype=np.float64,
    )
    improvement_lcb = _one_sided_lower(context_improvements, confidence)
    active_alpha = raw_alpha if improvement_lcb > 0.0 else 0.0

    m4_prediction = _ridge_predict(train, test, "M4")
    m5_prediction = _m5_predict(train, test)
    distances, distance_scale = _standardized_nearest_distances(train, test)
    context_distance = test["context_id"].astype(str).map(distances).to_numpy(dtype=np.float64)
    excess = np.maximum(context_distance / distance_scale - 1.0, 0.0)
    weights = active_alpha * np.exp(-np.square(excess))
    stacked = weights * m4_prediction + (1.0 - weights) * m5_prediction
    predictions = test[["context_id", "budget"]].copy()
    predictions["record_type"] = "recovery2_prediction"
    predictions["model_id"] = "M4S"
    predictions["m4_prediction"] = m4_prediction
    predictions["m5_prediction"] = m5_prediction
    predictions["prediction"] = stacked
    predictions["m4_weight"] = weights
    predictions["nearest_context_distance"] = context_distance
    choices = {
        str(context_id): int(cast(Any, group.loc[group["prediction"].idxmax(), "budget"]))
        for context_id, group in predictions.groupby("context_id", sort=True)
    }
    return ShrunkAtlasResult(
        choices=choices,
        predictions=predictions,
        raw_alpha=raw_alpha,
        active_alpha=active_alpha,
        improvement_lcb95=improvement_lcb,
        distance_scale=distance_scale,
    )


def evaluate_registry(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    flat_tolerance: float = 1e-9,
) -> RegistryResult:
    """Fit every frozen model on train contexts and score untouched contexts."""
    _require_frame(train)
    _require_frame(test)
    overlap = set(train["context_id"]) & set(test["context_id"])
    if overlap:
        raise ValueError(f"train/test context overlap: {sorted(overlap)[:3]}")

    predicted: dict[str, FloatArray] = {
        "M0": _m0_predict(train, test),
        "M1": _ridge_predict(train, test, "M1"),
        "M2": _ridge_predict(train, test, "M2"),
        "M3": _ridge_predict(train, test, "M3"),
        "M4": _ridge_predict(train, test, "M4"),
        "M5": _m5_predict(train, test),
    }
    rows: list[dict[str, object]] = []
    by_context = list(test.groupby("context_id", sort=True))
    context_ranges = [float(group["sigma"].max() - group["sigma"].min()) for _, group in by_context]
    target_range = float(np.median(context_ranges))
    identifiable = target_range > flat_tolerance
    scores: list[ModelScore] = []
    for model_id, prediction in predicted.items():
        regrets: list[float] = []
        hits: list[bool] = []
        for context_id, group in by_context:
            indices = group.index.to_numpy()
            positions = test.index.get_indexer(pd.Index(indices))
            pred = prediction[positions]
            truth = group["sigma"].to_numpy(dtype=np.float64)
            budgets = group["budget"].to_numpy(dtype=np.int64)
            chosen = int(np.argmax(pred))
            optimum = float(np.max(truth))
            regret = optimum - float(truth[chosen])
            optimal_mask = np.isclose(truth, optimum, atol=1e-12, rtol=0.0)
            hit = bool(optimal_mask[chosen])
            regrets.append(regret)
            hits.append(hit)
            rows.append(
                {
                    "context_id": context_id,
                    "model_id": model_id,
                    "chosen_budget": int(budgets[chosen]),
                    "optimal_budget": int(budgets[int(np.argmax(truth))]),
                    "budget_regret": regret,
                    "argmax_hit": hit,
                }
            )
        scores.append(
            ModelScore(
                model_id=model_id,
                mean_budget_regret=float(np.mean(regrets)),
                max_budget_regret=float(np.max(regrets)),
                exact_argmax_rate=float(np.mean(hits)),
                n_contexts=len(regrets),
                target_range=target_range,
                identifiable=identifiable,
            )
        )
    return RegistryResult(tuple(scores), pd.DataFrame(rows))


def model_registry_rows() -> list[dict[str, object]]:
    return [
        {"model_id": spec.model_id, "label": spec.label, "information": list(spec.information)}
        for spec in MODEL_SPECS
    ]
