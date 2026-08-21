from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from instinct.atlas.decomposition import exact_atlas_rows
from instinct.atlas.schema import AtlasRow, to_frame, validate_frame
from instinct.atlas.transfer import collapse_test
from instinct.core.configschema import ProposalRunConfig
from instinct.core.env import Timing
from instinct.core.envs.tabular import chase_chain, corridor_with_pit
from instinct.core.proposal import common_checks, resume_by_rerun
from instinct.core.results import ResultsWriter
from instinct.core.verdict import PrerequisiteCheck, Status, VerdictReport
from instinct.p1_atlas.controls import conditional_surface
from instinct.p1_atlas.models import MODEL_SPECS, evaluate_registry
from instinct.p1_atlas.pilot import resume_sampled, run_sampled, validate_sampled
from instinct.p1_atlas.recovery import resume_recovery, run_recovery, validate_recovery
from instinct.p1_atlas.recovery2 import (
    resume_recovery2,
    run_recovery2,
    validate_recovery2,
)


def validate(config: ProposalRunConfig) -> list[PrerequisiteCheck]:
    if config.params.get("stage") == "p1.1-r2":
        return validate_recovery2(config)
    if config.params.get("stage") == "p1.1-r1":
        return validate_recovery(config)
    if config.params.get("stage", "p1.0") == "p1.1":
        return validate_sampled(config)
    checks = common_checks(config, "p1_atlas")
    budgets = config.params.get("budgets", [1, 2, 4])
    checks.append(PrerequisiteCheck("budget grid", isinstance(budgets, list) and len(budgets) >= 3))
    return checks


def _planted_surface(*, hardware_quality: bool) -> Any:
    rows = []
    for speed, latency in ((1.0, 2.0), (2.0, 1.0)):
        for budget in (1, 2, 4, 8, 16):
            sigma = 1.0 - np.exp(-budget / 5.0) - 0.025 * speed * latency * budget
            if hardware_quality:
                sigma += 0.08 * budget if latency > 1 else -0.004 * budget**1.5
            rows.append(
                AtlasRow(
                    env="planted",
                    reflex="fixed",
                    nu_e=speed,
                    nu_h=latency,
                    budget=budget,
                    delay=int(speed * latency * budget),
                    staleness=speed * latency * budget,
                    start_state=0,
                    J_actual=sigma,
                    J_instant=sigma,
                    J_fresh=sigma,
                    J_base=0.0,
                    G_plan=sigma,
                    R_intermediate=0.0,
                    L_arrival=0.0,
                    L_wait=0.0,
                    L_irreversible=0.0,
                    C_hw=0.0,
                    L_base_delay=0.0,
                    epsilon_id=0.0,
                    sigma=sigma,
                    n_seeds=1,
                    ci_lo=sigma,
                    ci_hi=sigma,
                    exact=True,
                    simulations=budget,
                )
            )
    return to_frame(rows)


def _run_p1_0(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    checks = validate(config)
    budgets = [int(k) for k in config.params.get("budgets", [1, 2, 4])]
    rows = []
    for mdp, name, reflex in (
        (chase_chain(n_positions=4, gamma=0.9, drift=0.3), "chase", None),
        (corridor_with_pit(length=5, gamma=0.9, slip=0.2), "pit", None),
    ):
        mu = (
            np.ones(mdp.n_states, dtype=np.int64)
            if reflex is None and name == "chase"
            else np.zeros(mdp.n_states, dtype=np.int64)
        )
        for speed, latency in ((0.5, 0.5), (1.0, 0.25), (1.0, 0.5)):
            rows.extend(
                exact_atlas_rows(
                    mdp,
                    env_name=name,
                    reflex=mu,
                    reflex_name="hold" if name == "chase" else "advance",
                    budgets=budgets,
                    timing=Timing(nu_e=speed, nu_h=latency),
                    states=list(range(min(4, mdp.n_states))),
                )
            )
    frame = to_frame(rows)
    validate_frame(frame)
    residual = float(frame["epsilon_id"].abs().max())
    matched_time = bool(
        frame.loc[frame.env == "chase", "L_irreversible"].eq(0).all()
        and frame.loc[frame.env == "pit", "L_irreversible"].max() > 0
    )
    collapsed = collapse_test(_planted_surface(hardware_quality=False), family="pwlinear3")
    noncollapsed = collapse_test(_planted_surface(hardware_quality=True), family="pwlinear3")
    collapse_ok = collapsed is not None and collapsed.mean_budget_regret < 1e-9
    noncollapse_ok = noncollapsed is not None and noncollapsed.mean_budget_regret > 0.01

    conditional = conditional_surface()
    conditional_result = evaluate_registry(
        conditional.loc[conditional.split == "train"].copy(),
        conditional.loc[conditional.split == "test"].copy(),
    )
    collapsed_conditional = conditional_surface(collapse=True)
    collapsed_result = evaluate_registry(
        collapsed_conditional.loc[collapsed_conditional.split == "train"].copy(),
        collapsed_conditional.loc[collapsed_conditional.split == "test"].copy(),
    )
    flat = conditional_surface(flat=True)
    flat_result = evaluate_registry(
        flat.loc[flat.split == "train"].copy(),
        flat.loc[flat.split == "test"].copy(),
    )
    m4 = conditional_result.score("M4")
    comparison_ids = ("M0", "M2", "M5")
    m4_dominates = all(
        m4.mean_budget_regret < conditional_result.score(model).mean_budget_regret
        for model in comparison_ids
    )
    m2_collapse = collapsed_result.score("M2").mean_budget_regret < 1e-9
    flat_inconclusive = not any(score.identifiable for score in flat_result.scores)
    controls = [
        *checks,
        PrerequisiteCheck("frozen identity", residual < 1e-9, f"max={residual:.3e}"),
        PrerequisiteCheck("matched-time failure control", matched_time),
        PrerequisiteCheck("planted product collapse", collapse_ok),
        PrerequisiteCheck("planted hardware-quality noncollapse", noncollapse_ok),
        PrerequisiteCheck(
            "M0-M5 registry complete",
            {spec.model_id for spec in MODEL_SPECS} == {f"M{i}" for i in range(6)},
        ),
        PrerequisiteCheck(
            "M4 conditional transfer dominates M0/M2/M5",
            m4_dominates,
            f"M4 regret={m4.mean_budget_regret:.3e}",
        ),
        PrerequisiteCheck(
            "M4 below practical transfer tolerance",
            m4.mean_budget_regret < 0.2 * m4.target_range,
            f"regret={m4.mean_budget_regret:.3e}, range={m4.target_range:.3e}",
        ),
        PrerequisiteCheck("M2 recovers imposed product collapse", m2_collapse),
        PrerequisiteCheck("flat frontiers are inconclusive", flat_inconclusive),
    ]
    decomposition_metrics = frame.copy()
    decomposition_metrics["record_type"] = "decomposition"
    summary_metrics = pd.DataFrame(
        [
            {
                "record_type": "model_summary",
                "model_id": score.model_id,
                "mean_budget_regret": score.mean_budget_regret,
                "max_budget_regret": score.max_budget_regret,
                "exact_argmax_rate": score.exact_argmax_rate,
                "n_contexts": score.n_contexts,
                "target_range": score.target_range,
                "identifiable": score.identifiable,
            }
            for score in conditional_result.scores
        ]
    )
    prediction_metrics = conditional_result.predictions.copy()
    prediction_metrics["record_type"] = "heldout_prediction"
    registry_metrics = pd.DataFrame(
        [
            {
                "record_type": "model_registry",
                "model_id": spec.model_id,
                "model_label": spec.label,
                "information": ",".join(spec.information),
            }
            for spec in MODEL_SPECS
        ]
    )
    writer.write_metrics(
        pd.concat(
            [decomposition_metrics, summary_metrics, prediction_metrics, registry_metrics],
            ignore_index=True,
            sort=False,
        )
    )
    writer.event(
        stage="p1.0",
        rows=len(frame),
        heldout_contexts=m4.n_contexts,
        max_identity_residual=residual,
        m4_budget_regret=m4.mean_budget_regret,
    )
    passed = all(c.passed for c in controls)
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=Status.GO if passed else Status.INCONCLUSIVE,
        evidence_level="E0",
        primary_result={
            "metric": "M4_heldout_budget_regret",
            "value": m4.mean_budget_regret,
            "unit": "return",
        },
        baselines_run=[
            "M0 global fixed budget",
            "M1 budget-only curve",
            "M2 product collapse",
            "M3 separate axes",
            "M4 conditional descriptors",
            "M5 nearest-context lookup",
            "two exact MDPs",
            "matched-time failure counterfactual",
        ],
        controls=controls,
        interpretation=(
            "P1.0 known-answer validation: the exact identity, collapse identifiability, "
            "flat-frontier handling, and M0-M5 conditional transfer behave as planted."
        ),
        non_claim=config.preregistration.non_claim,
        decision="Proceed to P1.1 sampled real-time latency measurement."
        if passed
        else "Repair failed E0 control.",
    )


def run(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    """Dispatch without changing the frozen P1.0 implementation."""
    stage = config.params.get("stage", "p1.0")
    if stage == "p1.1-r2":
        return run_recovery2(config, writer)
    if stage == "p1.1-r1":
        return run_recovery(config, writer)
    if stage == "p1.1":
        return run_sampled(config, writer)
    if stage != "p1.0":
        raise ValueError(
            f"unknown P1 stage {stage!r}; expected 'p1.0', 'p1.1', "
            "'p1.1-r1', or 'p1.1-r2'"
        )
    return _run_p1_0(config, writer)


def resume(
    config: ProposalRunConfig, writer: ResultsWriter, checkpoint_state: Mapping[str, Any]
) -> VerdictReport:
    if config.params.get("stage") == "p1.1-r2":
        return resume_recovery2(config, writer, checkpoint_state)
    if config.params.get("stage") == "p1.1-r1":
        return resume_recovery(config, writer, checkpoint_state)
    if config.params.get("stage", "p1.0") == "p1.1":
        return resume_sampled(config, writer, checkpoint_state)
    return resume_by_rerun(config, writer, checkpoint_state, run)
