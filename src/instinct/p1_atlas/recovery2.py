"""Final P1.1 recovery: uncertainty-gated shrinkage toward deployable M5."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from instinct.core.configschema import ProposalRunConfig
from instinct.core.results import ResultsWriter
from instinct.core.verdict import FailureCode, PrerequisiteCheck, Status, VerdictReport
from instinct.p1_atlas.ftt_adapter import FTTAdapter, inspect_external_ftt, load_t4_cells
from instinct.p1_atlas.pilot import generate_pilot, validate_sampled

__all__ = ["resume_recovery2", "run_recovery2", "validate_recovery2"]


def validate_recovery2(config: ProposalRunConfig) -> list[PrerequisiteCheck]:
    checks = validate_sampled(config)
    prior_units = set(range(101, 109)) | set(range(201, 213))
    prior_units |= set(range(301, 307)) | set(range(401, 409))
    declared = set(config.pilot_seeds) | set(config.eval_seeds)
    checks.extend(
        (
            PrerequisiteCheck(
                "final uncertainty-shrinkage recovery selected",
                config.params.get("stage") == "p1.1-r2"
                and config.params.get("failure_rich_control") is True,
            ),
            PrerequisiteCheck(
                "fresh recovery2 seed registry",
                set(config.pilot_seeds).isdisjoint(config.eval_seeds)
                and declared.isdisjoint(prior_units),
            ),
            PrerequisiteCheck(
                "decisive bounded CPU unit counts",
                int(config.params.get("horizon", 0)) >= 24
                and int(config.params.get("n_oracle", 0)) >= 8
                and int(config.params.get("n_regret", 0)) >= 12,
            ),
            PrerequisiteCheck(
                "frozen 95% shrinkage gate",
                float(config.params.get("shrinkage_confidence", 0.0)) == 0.95,
            ),
        )
    )
    return checks


def _one_sided_lower(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    if values.size < 2 or not np.isfinite(values).all():
        return float("-inf")
    sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
    return float(np.mean(values)) - float(scipy_stats.t.ppf(0.95, values.size - 1)) * sem


def _adapter_check(budgets: tuple[int, ...]) -> bool:
    adapter = FTTAdapter(budgets, lambda observation: np.zeros(len(observation), dtype=np.int64))
    selected = adapter.choose(
        np.zeros((2, 3), dtype=np.float64),
        np.zeros((2, 4), dtype=np.float64),
        np.zeros((2, 1), dtype=np.float64),
    )
    return bool(np.array_equal(selected, np.full(2, budgets[0])))


def run_recovery2(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    initial = validate_recovery2(config)
    if any(not check.passed for check in initial):
        failed = [check.name for check in initial if not check.passed]
        raise ValueError(f"P1.1 recovery2 validation failed closed: {failed}")

    artifacts = generate_pilot(config)
    recurring = artifacts.metrics.loc[
        artifacts.metrics["record_type"] == "recurring_episode"
    ].copy()
    control = recurring.loc[recurring["family"] == "control"]
    starts = control.drop_duplicates(["context_id", "unit_role", "unit_id"])
    margins = pd.to_numeric(starts["initial_recovery_margin"]).to_numpy(np.float64)
    failed_count = int(control["failed"].sum())
    survived_count = int((~control["failed"].astype(bool)).sum())
    failure_rich = bool(
        failed_count > 0
        and survived_count > 0
        and (margins < 0).any()
        and (margins >= 0).any()
    )

    contexts = artifacts.contexts
    train_contexts = set(contexts.loc[contexts["split"] == "train", "context_id"])
    test_contexts = set(contexts.loc[contexts["split"] == "test", "context_id"])
    whole_context_disjoint = not (train_contexts & test_contexts)

    summary = artifacts.model_summary.set_index("model_id")
    required = {f"M{index}" for index in range(6)} | {
        "M4S",
        "SPEED",
        "R1_SPEED_CHAMPION",
    }
    registry_complete = required <= set(summary.index)
    m4s = summary.loc["M4S"]
    m5 = summary.loc["M5"]
    speed = summary.loc["R1_SPEED_CHAMPION"]
    m4s_ucb = float(cast(Any, m4s["context_regret_ucb95"]))
    m5_ucb = float(cast(Any, m5["context_regret_ucb95"]))
    speed_ucb = float(cast(Any, speed["context_regret_ucb95"]))
    target_range = float(cast(Any, m4s["target_range"]))
    n_test_contexts = int(cast(Any, m4s["n_test_contexts"]))

    regrets = artifacts.metrics.loc[
        artifacts.metrics["record_type"] == "context_regret",
        ["context_id", "model_id", "paired_regret"],
    ]
    pivot = regrets.pivot(index="context_id", columns="model_id", values="paired_regret")
    improvement = (
        pivot["M5"].to_numpy(dtype=np.float64)
        - pivot["M4S"].to_numpy(dtype=np.float64)
    )
    improvement_lcb = _one_sided_lower(improvement)
    identifiable = target_range > float(config.params.get("flat_tolerance", 0.01))
    absolute_pass = m4s_ucb < 0.2 * target_range
    beats_m5 = improvement_lcb > 0.0

    shrinkage_row = artifacts.metrics.loc[
        artifacts.metrics["record_type"] == "recovery2_shrinkage"
    ].iloc[0]
    no_eval_sigma = not bool(cast(Any, shrinkage_row["fit_uses_evaluation_sigma"]))
    active_alpha = float(cast(Any, shrinkage_row["active_alpha"]))
    raw_alpha = float(cast(Any, shrinkage_row["raw_alpha"]))
    training_lcb = float(cast(Any, shrinkage_row["training_improvement_lcb95"]))

    ftt = inspect_external_ftt(config.params)
    t4_passed, t4_detail = load_t4_cells(config.params.get("t4_timing_artifact"))
    budgets = tuple(sorted(int(value) for value in config.params["budgets"]))
    adapter_ok = _adapter_check(budgets)
    lookup_oracle = pd.DataFrame(
        [
            {
                "record_type": "model_summary",
                "model_id": "LOOKUP_ORACLE_EVAL_ONLY",
                "mean_context_regret": 0.0,
                "context_regret_ucb95": 0.0,
                "n_test_contexts": n_test_contexts,
                "target_range": target_range,
                "deployable": False,
                "description": "held-out oracle surface argmax; evaluation-only ceiling",
            }
        ]
    )
    final_gate = pd.DataFrame(
        [
            {
                "record_type": "recovery2_gate",
                "model_id": "M4S",
                "m4s_context_regret_ucb95": m4s_ucb,
                "m5_context_regret_ucb95": m5_ucb,
                "r1_speed_champion_ucb95": speed_ucb,
                "m5_minus_m4s_improvement_lcb95": improvement_lcb,
                "absolute_threshold": 0.2 * target_range,
                "raw_alpha": raw_alpha,
                "active_alpha": active_alpha,
                "training_improvement_lcb95": training_lcb,
                "conditional_atlas_archived": not (identifiable and absolute_pass and beats_m5),
                "official_ftt_equivalence": ftt.official_equivalence_available,
                "ftt_detail": ftt.detail,
                "t4_timing_available": t4_passed,
                "t4_detail": t4_detail,
            }
        ]
    )
    writer.write_metrics(
        pd.concat((artifacts.metrics, lookup_oracle, final_gate), ignore_index=True, sort=False)
    )

    controls = [
        *artifacts.controls,
        PrerequisiteCheck(
            "whole training/evaluation contexts disjoint", whole_context_disjoint
        ),
        PrerequisiteCheck("M4S fit excludes evaluation Sigma", no_eval_sigma),
        PrerequisiteCheck(
            "M0-M5 and recovery1 champion reported",
            registry_complete,
            f"models={sorted(set(summary.index))}",
        ),
        PrerequisiteCheck(
            "natural continuous-control failures and survivals",
            failure_rich,
            f"failed_rows={failed_count}, survived_rows={survived_count}, "
            f"margin=[{np.min(margins):.3f},{np.max(margins):.3f}]",
        ),
        PrerequisiteCheck("FTT-compatible adapter boundary", adapter_ok),
        PrerequisiteCheck(
            "official FTT source/checkpoint equivalence",
            ftt.official_equivalence_available,
            ftt.detail,
        ),
        PrerequisiteCheck("measured T4 held-out timing cells", t4_passed, t4_detail),
        PrerequisiteCheck(
            "identifiable recovery2 target", identifiable, f"target_range={target_range:.6f}"
        ),
        PrerequisiteCheck(
            "M4S below frozen practical tolerance",
            absolute_pass,
            f"UCB={m4s_ucb:.6f}, threshold={0.2 * target_range:.6f}",
        ),
        PrerequisiteCheck(
            "M4S beats deployable M5",
            beats_m5,
            f"paired M5-minus-M4S LCB95={improvement_lcb:.6f}",
        ),
        PrerequisiteCheck(
            "lookup oracle labeled evaluation-only",
            True,
            "LOOKUP_ORACLE_EVAL_ONLY is not a deployable transfer method",
        ),
    ]
    external_names = {
        "official FTT source/checkpoint equivalence",
        "measured T4 held-out timing cells",
    }
    success_names = {
        "identifiable recovery2 target",
        "M4S below frozen practical tolerance",
        "M4S beats deployable M5",
    }
    instrumentation = all(
        check.passed
        for check in controls
        if check.name not in external_names | success_names
    )
    mechanism = all(check.passed for check in controls if check.name in success_names)
    external = ftt.official_equivalence_available and t4_passed
    status = (
        Status.GO
        if instrumentation and mechanism and external
        else Status.INCONCLUSIVE
        if not instrumentation or mechanism
        else Status.STOP
    )
    failure_code = (
        None
        if status in {Status.GO, Status.INCONCLUSIVE} and instrumentation
        else FailureCode.F0_CODE_INVARIANT
        if not instrumentation
        else FailureCode.F4_BASELINE_DOMINANCE
        if not beats_m5
        else FailureCode.F3_MECHANISM_ABSENT
    )
    archive = instrumentation and not mechanism
    writer.event(
        stage="p1.1-r2",
        status=status.value,
        m4s_context_regret_ucb95=m4s_ucb,
        m5_context_regret_ucb95=m5_ucb,
        m5_minus_m4s_improvement_lcb95=improvement_lcb,
        active_alpha=active_alpha,
        conditional_atlas_archived=archive,
    )
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=status,
        evidence_level="E2",
        primary_result={
            "metric": "M5_minus_M4S_paired_improvement_lcb95",
            "value": improvement_lcb,
            "unit": "return",
        },
        baselines_run=[
            "M0-M5 (unchanged)",
            "M4S uncertainty-gated convex shrinkage",
            "recovery1 SPEED champion",
            "smallest/largest budget",
            "FTT-lite (not official-equivalent)",
            "evaluation-only held-out lookup oracle",
        ],
        baselines_skipped={
            "official Finding-the-Time-to-Think PPO gate": (
                "released JAX source/checkpoint stack not configured"
            ),
            "measured T4 latency transfer": "no verified T4 timing-cell artifact configured",
        },
        controls=controls,
        failure_code=failure_code,
        interpretation=(
            "The final fresh CPU recovery preserves Sigma and tests a training-only, "
            "uncertainty-gated shrinkage of M4 toward deployable M5. Original M4 and all "
            "M0-M5 baselines remain unchanged. No universal collapse is asserted."
        ),
        non_claim=config.preregistration.non_claim,
        decision=(
            "Archive the conditional-atlas mechanism after two valid recoveries; retain "
            "the exact Sigma decomposition and empirical M5/SPEED evidence."
            if archive
            else "Supply official FTT and verified T4 artifacts before any GO claim."
            if mechanism
            else "Repair instrumentation before interpreting recovery2."
        ),
    )


def resume_recovery2(
    config: ProposalRunConfig,
    writer: ResultsWriter,
    checkpoint_state: Mapping[str, Any],
) -> VerdictReport:
    del checkpoint_state
    return run_recovery2(config, writer)
