"""Fresh P1.1 failure-rich CPU recovery with external-equivalence preflights."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd

from instinct.core.configschema import ProposalRunConfig
from instinct.core.results import ResultsWriter
from instinct.core.verdict import FailureCode, PrerequisiteCheck, Status, VerdictReport
from instinct.p1_atlas.ftt_adapter import FTTAdapter, inspect_external_ftt, load_t4_cells
from instinct.p1_atlas.pilot import generate_pilot

__all__ = ["run_recovery", "validate_recovery"]


def validate_recovery(config: ProposalRunConfig) -> list[PrerequisiteCheck]:
    from instinct.p1_atlas.pilot import validate_sampled

    checks = validate_sampled(config)
    checks.extend(
        (
            PrerequisiteCheck(
                "fresh failure-rich recovery selected",
                config.params.get("stage") == "p1.1-r1"
                and config.params.get("failure_rich_control") is True,
            ),
            PrerequisiteCheck(
                "longer than preserved smoke",
                int(config.params.get("horizon", 0)) >= 24
                and int(config.params.get("n_regret", 0)) >= 8,
            ),
        )
    )
    return checks


def _adapter_check(budgets: tuple[int, ...]) -> bool:
    adapter = FTTAdapter(budgets, lambda observation: np.zeros(len(observation), dtype=np.int64))
    selected = adapter.choose(
        np.zeros((2, 3), dtype=np.float64),
        np.zeros((2, 4), dtype=np.float64),
        np.zeros((2, 1), dtype=np.float64),
    )
    return bool(np.array_equal(selected, np.full(2, budgets[0])))


def run_recovery(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    initial = validate_recovery(config)
    if any(not check.passed for check in initial):
        failed = [check.name for check in initial if not check.passed]
        raise ValueError(f"P1.1 recovery validation failed closed: {failed}")

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

    ftt = inspect_external_ftt(config.params)
    t4_passed, t4_detail = load_t4_cells(config.params.get("t4_timing_artifact"))
    budgets = tuple(sorted(int(value) for value in config.params["budgets"]))
    adapter_ok = _adapter_check(budgets)
    summary = artifacts.model_summary.set_index("model_id")
    m4 = summary.loc["M4"]
    m5 = summary.loc["M5"]
    m4_ucb = float(cast(Any, m4["context_regret_ucb95"]))
    m5_ucb = float(cast(Any, m5["context_regret_ucb95"]))
    target_range = float(cast(Any, m4["target_range"]))
    n_test_contexts = int(cast(Any, m4["n_test_contexts"]))
    m4_beats_lookup = float(cast(Any, m5["improvement_over_m4_lcb95"])) > 0
    lookup_oracle = pd.DataFrame(
        [
            {
                "record_type": "model_summary",
                "model_id": "LOOKUP_ORACLE_EVAL_ONLY",
                "mean_context_regret": 0.0,
                "context_regret_ucb95": 0.0,
                "improvement_over_m4_lcb95": float("nan"),
                "n_test_contexts": n_test_contexts,
                "target_range": target_range,
                "deployable": False,
                "description": "held-out oracle surface argmax; evaluation-only ceiling",
            }
        ]
    )
    preflight = pd.DataFrame(
        [
            {
                "record_type": "recovery_preflight",
                "natural_control_failures": failed_count,
                "natural_control_survivals": survived_count,
                "initial_margin_min": float(np.min(margins)),
                "initial_margin_max": float(np.max(margins)),
                "ftt_adapter_api_passed": adapter_ok,
                "official_ftt_equivalence": ftt.official_equivalence_available,
                "ftt_detail": ftt.detail,
                "t4_timing_available": t4_passed,
                "t4_detail": t4_detail,
                "m4_beats_m5_lcb95": m4_beats_lookup,
            }
        ]
    )
    writer.write_metrics(
        pd.concat((artifacts.metrics, lookup_oracle, preflight), ignore_index=True, sort=False)
    )

    controls = [
        *artifacts.controls,
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
            "M4 beats deployable M5 lookup",
            m4_beats_lookup,
            f"M4 UCB={m4_ucb:.6f}, M5 UCB={m5_ucb:.6f}",
        ),
        PrerequisiteCheck(
            "lookup oracle labeled evaluation-only",
            True,
            "LOOKUP_ORACLE_EVAL_ONLY is not a deployable transfer method",
        ),
    ]
    hard_cpu = all(
        check.passed
        for check in controls
        if check.name
        not in {"official FTT source/checkpoint equivalence", "measured T4 held-out timing cells"}
    )
    if not hard_cpu:
        failure_code = (
            FailureCode.F4_BASELINE_DOMINANCE
            if not m4_beats_lookup
            else FailureCode.F1_UNIDENTIFIABLE
        )
    else:
        failure_code = None
    status = (
        Status.GO
        if hard_cpu and ftt.official_equivalence_available and t4_passed
        else Status.INCONCLUSIVE
    )
    writer.event(
        stage="p1.1-r1",
        status=status.value,
        natural_control_failures=failed_count,
        natural_control_survivals=survived_count,
        m4_context_regret_ucb95=m4_ucb,
        m5_context_regret_ucb95=m5_ucb,
        official_ftt_equivalence=ftt.official_equivalence_available,
        t4_timing_available=t4_passed,
    )
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=status,
        evidence_level="E2",
        primary_result={
            "metric": "M4_context_regret_ucb95",
            "value": m4_ucb,
            "unit": "return",
        },
        baselines_run=[
            "M0-M5",
            "smallest/largest budget",
            "speed schedule",
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
            "A fresh longer CPU recovery with sampled states on both sides of the inertial "
            "recovery frontier. The Sigma identity is unchanged. M5 is a deployable nearest-"
            "training-context lookup; the zero-regret held-out lookup is separately labeled "
            "evaluation-only. No universal collapse is asserted."
        ),
        non_claim=config.preregistration.non_claim,
        decision=(
            "Preserve this CPU recovery. Supply the official realtime-rl-code checkout plus "
            "matching gating checkpoint and a verified T4 timing-cell artifact before any "
            "FTT-equivalence, hardware-transfer, or GO claim."
        ),
    )


def resume_recovery(
    config: ProposalRunConfig,
    writer: ResultsWriter,
    checkpoint_state: Mapping[str, Any],
) -> VerdictReport:
    del checkpoint_state
    return run_recovery(config, writer)
