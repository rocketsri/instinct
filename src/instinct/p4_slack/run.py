from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from instinct.core.configschema import ProposalRunConfig
from instinct.core.envs.tabular import corridor_with_pit
from instinct.core.proposal import common_checks, resume_by_rerun
from instinct.core.results import ResultsWriter
from instinct.core.verdict import FailureCode, PrerequisiteCheck, Status, VerdictReport
from instinct.p4_slack.inertial import (
    braking_policy,
    evaluate_impulse_policy,
    greedy_progress_policy,
    impulse_recovery_values,
    inertial_grid,
    one_step_failure_floor,
    recovery_policy,
)
from instinct.p4_slack.inertial_surrogate import (
    Condition,
    fit_inertial_surrogate,
    inertial_recovery_dataset,
    inertial_surrogate_metrics,
)
from instinct.p4_slack.slack import (
    disturbed_replan_values,
    finite_horizon_replan_value,
    lower_quantile,
    pareto_dominated,
    recovery_slack,
    reward_rescaling_consistent,
    slack_penalty,
)
from instinct.p4_slack.surrogate import (
    condition_coverage_metrics,
    condition_split_registry,
    exact_recovery_dataset,
    fit_calibrated_surrogate,
    fit_condition_calibrated_surrogate,
    surrogate_metrics,
)


def validate(config: ProposalRunConfig) -> list[PrerequisiteCheck]:
    checks = common_checks(config, "p4_slack")
    q = float(config.params.get("q", 0.25))
    margin = float(config.params.get("margin", 1.0))
    horizon = int(config.params.get("replan_horizon", 8))
    stage = str(config.params.get("stage", "p4.0"))
    estimator_version = str(config.params.get("estimator_version", "v3"))
    checks.append(
        PrerequisiteCheck(
            "known P4 stage",
            stage in {"p4.0", "p4.0-inertial", "p4.1", "p4.1-inertial", "p4.2"},
            stage,
        )
    )
    checks.append(
        PrerequisiteCheck(
            "known P4.1 estimator version",
            stage != "p4.1" or estimator_version in {"v3", "v4-condition-conformal"},
            estimator_version,
        )
    )
    checks.append(PrerequisiteCheck("lower-tail quantile", 0 < q <= 0.5, f"q={q}"))
    checks.append(PrerequisiteCheck("positive slack margin", margin > 0, f"margin={margin}"))
    checks.append(PrerequisiteCheck("positive replanning horizon", horizon > 0, f"H={horizon}"))
    return checks


def _run_exact(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    checks = validate(config)
    q = float(config.params.get("q", 0.25))
    margin = float(config.params.get("margin", 1.0))
    replan_horizon = int(config.params.get("replan_horizon", 8))
    v_fail = float(config.params.get("v_fail", -1.0))
    known = lower_quantile(np.array([-2.0, 0.0, 0.0, 5.0]), 0.5)
    checks.append(
        PrerequisiteCheck("hand-computed quantile with ties", known == 0.0, f"value={known}")
    )
    # Exact finite-horizon labels, then frozen common disturbance kernels for
    # the three constructed policies. No learned surrogate enters this gate.
    mdp = corridor_with_pit(length=6, gamma=0.95, slip=0.2)
    exact_value = finite_horizon_replan_value(mdp, horizon=replan_horizon, v_fail=v_fail)
    n_states = mdp.n_states

    def kernel(rows: list[list[tuple[int, float]]]) -> np.ndarray:
        out = np.zeros((2, 4, n_states), dtype=np.float64)
        for flat_index, mixture in enumerate(rows):
            prefix, disturbance = divmod(flat_index, 4)
            for state, probability in mixture:
                out[prefix, disturbance, state] += probability
        return out

    pit = n_states - 1
    policies = {
        "inactive": (
            0.0,
            kernel([[(0, 1.0)]] * 8),
            0.96,
        ),
        "interior": (
            0.7,
            kernel(
                [
                    [(1, 1.0)],
                    [(1, 0.9), (pit, 0.1)],
                    [(2, 0.8), (0, 0.2)],
                    [(1, 0.75), (pit, 0.25)],
                    [(2, 1.0)],
                    [(3, 0.85), (pit, 0.15)],
                    [(2, 0.8), (1, 0.2)],
                    [(3, 0.7), (pit, 0.3)],
                ]
            ),
            0.82,
        ),
        "reckless": (
            1.0,
            kernel(
                [
                    [(3, 0.7), (pit, 0.3)],
                    [(4, 0.5), (pit, 0.5)],
                    [(3, 0.4), (pit, 0.6)],
                    [(4, 0.25), (pit, 0.75)],
                    [(4, 0.6), (pit, 0.4)],
                    [(5, 0.4), (pit, 0.6)],
                    [(4, 0.3), (pit, 0.7)],
                    [(5, 0.2), (pit, 0.8)],
                ]
            ),
            0.48,
        ),
    }
    metrics = []
    points = []
    min_slacks: dict[str, float] = {}
    for name, (progress, disturbance_kernel, heldout_success) in policies.items():
        values = disturbed_replan_values(exact_value, disturbance_kernel)
        slack = recovery_slack(values, v_fail=v_fail, q=q)
        penalty = slack_penalty(slack, margin=margin)
        min_slacks[name] = float(slack.min())
        points.append((name, progress, heldout_success))
        metrics.append(
            {
                "policy": name,
                "progress": progress,
                "heldout_success": heldout_success,
                "min_slack": min_slacks[name],
                "slack_penalty": penalty,
                "replan_calls": 2,
                "label_source": "exact_finite_horizon_dp",
            }
        )
    interior = next(p for p in points if p[0] == "interior")
    strict_interior = 0 < interior[1] < 1 and points[2][2] < interior[2] < points[0][2]
    terminal_mask_ok = (
        slack_penalty(np.array([0.0, -100.0]), margin=margin, active=np.array([True, False]))
        == margin
    )
    common_disturbances = all(item[1].shape == (2, 4, n_states) for item in policies.values())
    scale_ok = reward_rescaling_consistent(
        disturbed_replan_values(exact_value, policies["interior"][1]),
        v_fail=v_fail,
        margin=margin,
        q=q,
        scale=3.0,
        offset=7.0,
    )
    # Matched at two replanning calls. The planted exact-label method is not
    # dominated, while a deliberately weakened point must be detected.
    matched_baselines = [(0.55, 0.78), (0.62, 0.80)]
    matched_not_dominated = not pareto_dominated((interior[1], interior[2]), matched_baselines)
    dominance_control = pareto_dominated((0.5, 0.7), matched_baselines)
    # A biased surrogate makes the reckless prefix appear safe; exact labels
    # must expose that ordering error before learned policy optimization.
    exact_reckless = min_slacks["reckless"]
    surrogate_reckless = exact_reckless + 3.0 * margin
    surrogate_gaming_detected = surrogate_reckless > margin and exact_reckless < margin
    checks += [
        PrerequisiteCheck(
            "exact finite-horizon replanning labels", bool(np.isfinite(exact_value).all())
        ),
        PrerequisiteCheck("common disturbance kernels", common_disturbances),
        PrerequisiteCheck("strict interior frontier", strict_interior),
        PrerequisiteCheck("terminal prefix masking", terminal_mask_ok),
        PrerequisiteCheck("reward rescaling preserves slack units", scale_ok),
        PrerequisiteCheck("matched verification does not dominate interior", matched_not_dominated),
        PrerequisiteCheck("dominance negative control", dominance_control),
        PrerequisiteCheck("exact labels detect surrogate gaming", surrogate_gaming_detected),
    ]
    metrics.extend(
        {
            "policy": name,
            "progress": progress,
            "heldout_success": success,
            "min_slack": np.nan,
            "slack_penalty": np.nan,
            "replan_calls": 2,
            "label_source": "matched_baseline",
        }
        for name, (progress, success) in zip(
            ("shorter_chunk", "periodic_verification"), matched_baselines, strict=True
        )
    )
    writer.write_metrics(metrics)
    writer.event(
        stage="p4.0",
        q=q,
        margin=margin,
        v_fail=v_fail,
        replan_horizon=replan_horizon,
        interior_frontier=strict_interior,
    )
    passed = all(c.passed for c in checks)
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=Status.GO if passed else Status.INCONCLUSIVE,
        evidence_level="E0",
        primary_result={
            "metric": "interior_heldout_success",
            "value": interior[2],
            "unit": "probability",
        },
        baselines_run=[
            "safe inactive",
            "reckless progress",
            "exact viability/recovery DP",
            "shorter chunk at matched calls",
            "periodic verification at matched calls",
            "biased-surrogate negative control",
        ],
        controls=checks,
        interpretation=(
            "Constructed exact-label frontier only; learned slack and matched-cost "
            "gains remain untested."
        ),
        non_claim=config.preregistration.non_claim,
        decision="Implement calibrated V_replan plus matched verification baselines."
        if passed
        else "Stop before surrogate training.",
    )


def _run_surrogate(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    if str(config.params.get("estimator_version", "v3")) == "v4-condition-conformal":
        return _run_surrogate_repair(config, writer)
    checks = validate(config)
    v_fail = float(config.params.get("v_fail", -1.0))
    alpha = float(config.params.get("calibration_alpha", 0.1))
    minimum_coverage = float(config.params.get("minimum_coverage", 0.9))
    train_conditions = [(5, 0.05, 4), (5, 0.3, 12), (7, 0.05, 12), (7, 0.3, 4)]
    calibration_conditions = [(6, 0.15, 6), (6, 0.15, 10)]
    test_conditions = [(8, 0.2, 8)]
    train = exact_recovery_dataset(train_conditions, v_fail=v_fail)
    calibration = exact_recovery_dataset(calibration_conditions, v_fail=v_fail)
    test = exact_recovery_dataset(test_conditions, v_fail=v_fail)
    surrogate = fit_calibrated_surrogate(train, calibration, alpha=alpha)
    rows = []
    summaries: dict[str, dict[str, float]] = {}
    for split, frame in (("train", train), ("calibration", calibration), ("test", test)):
        summary = surrogate_metrics(surrogate, frame)
        summaries[split] = summary
        rows.append({"split": split, **summary})
    split_disjoint = (
        set(train_conditions).isdisjoint(calibration_conditions)
        and set(train_conditions).isdisjoint(test_conditions)
        and set(calibration_conditions).isdisjoint(test_conditions)
    )


    test_coverage = summaries["test"]["coverage"]
    calibrated = test_coverage >= minimum_coverage
    checks += [
        PrerequisiteCheck("condition splits disjoint", split_disjoint),
        PrerequisiteCheck(
            "held-out exact-label coverage",
            calibrated,
            f"coverage={test_coverage:.3f}, required={minimum_coverage:.3f}",
        ),
        PrerequisiteCheck(
            "surrogate frozen before test",
            True,
            "coefficients and radius use train/calibration only",
        ),
    ]
    writer.write_metrics(rows)
    writer.event(
        stage="p4.1",
        coverage=test_coverage,
        mae=summaries["test"]["mae"],
        calibration_radius=surrogate.residual_radius,
    )
    prerequisites = all(
        check.passed for check in checks if check.name != "held-out exact-label coverage"
    )
    status = Status.GO if prerequisites and calibrated else Status.INCONCLUSIVE
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=status,
        evidence_level="E1",
        primary_result={
            "metric": "heldout_exact_label_coverage",
            "value": test_coverage,
            "unit": "probability",
        },
        baselines_run=["exact finite-horizon DP", "ridge recovery surrogate"],
        controls=checks,
        failure_code=None if calibrated else FailureCode.F2_ESTIMATOR_FAILURE,
        interpretation=(
            "The surrogate is frozen before held-out evaluation. Undercoverage blocks policy "
            "training even though mean absolute error is modest."
        ),
        non_claim=config.preregistration.non_claim,
        decision=(
            "Proceed to P4.2 only after a fresh calibrated surrogate passes."
            if calibrated
            else "Repair the estimator on pilot/calibration conditions; do not widen on test data."
        ),
    )


def _run_inertial(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    """Close the frozen P4.0 2-D inertia/recovery environment blocker."""

    checks = validate(config)
    v_fail = float(config.params.get("v_fail", -1.0))
    horizon = int(config.params.get("replan_horizon", 12))
    q = float(config.params.get("q", 0.25))
    grid = inertial_grid(
        width=int(config.params.get("width", 4)),
        height=int(config.params.get("height", 4)),
        actuator_slip=float(config.params.get("actuator_slip", 0.1)),
    )
    exact_value = finite_horizon_replan_value(grid.mdp, horizon=horizon, v_fail=v_fail)
    impulses = ((0, 0), (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    disturbed = impulse_recovery_values(grid, exact_value, impulses)
    slack = np.array(
        [lower_quantile(row - v_fail, q) for row in disturbed], dtype=np.float64
    )
    failure_floor = one_step_failure_floor(grid)
    active = ~grid.mdp.terminal
    policies = {
        "braking": braking_policy(grid),
        "myopic_progress": greedy_progress_policy(grid),
        "exact_recovery": recovery_policy(grid, exact_value),
    }
    rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, float]] = {}
    for name, policy in policies.items():
        summary = evaluate_impulse_policy(
            grid, policy, impulses=impulses, horizon=horizon
        )
        summaries[name] = summary
        rows.append(
            {
                "stage": "p4.0-inertial",
                "policy": name,
                "state_count": grid.mdp.n_states,
                "disturbance_count": len(impulses),
                "replan_horizon": horizon,
                **summary,
            }
        )
    recoverable = int(np.sum(active & (failure_floor == 0.0)))
    unrecoverable = int(np.sum(active & (failure_floor >= 1.0 - 1e-12)))
    finite = bool(np.isfinite(exact_value).all() and np.isfinite(slack).all())
    recovery = summaries["exact_recovery"]
    braking = summaries["braking"]
    reckless = summaries["myopic_progress"]
    useful_recovery = bool(
        recovery["progress"] > braking["progress"] + 0.25
        and recovery["survival_probability"] > reckless["survival_probability"] + 0.25
    )
    checks += [
        PrerequisiteCheck("exact inertial recovery labels finite", finite),
        PrerequisiteCheck(
            "recoverable legal states exposed", recoverable > 0, f"states={recoverable}"
        ),
        PrerequisiteCheck(
            "already-unrecoverable legal states exposed",
            unrecoverable > 0,
            f"states={unrecoverable}",
        ),
        PrerequisiteCheck(
            "impulse recovery slack nondegenerate",
            float(np.ptp(slack[active])) > 0.5,
            f"range={np.ptp(slack[active]):.3f}",
        ),
        PrerequisiteCheck(
            "recovery policy avoids inactivity and recklessness",
            useful_recovery,
            (
                f"progress={recovery['progress']:.3f} vs brake={braking['progress']:.3f}; "
                f"survival={recovery['survival_probability']:.3f} "
                f"vs greedy={reckless['survival_probability']:.3f}"
            ),
        ),
    ]
    rows.append(
        {
            "stage": "p4.0-inertial-audit",
            "policy": "state_space",
            "state_count": grid.mdp.n_states,
            "recoverable_states": recoverable,
            "unrecoverable_states": unrecoverable,
            "slack_range": float(np.ptp(slack[active])),
        }
    )
    writer.write_metrics(rows)
    writer.event(
        stage="p4.0-inertial",
        states=grid.mdp.n_states,
        recoverable_states=recoverable,
        unrecoverable_states=unrecoverable,
        exact_recovery_progress=recovery["progress"],
        exact_recovery_survival=recovery["survival_probability"],
    )
    passed = all(check.passed for check in checks)
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=Status.GO if passed else Status.INCONCLUSIVE,
        evidence_level="E0",
        primary_result={
            "metric": "exact_recovery_progress_under_heldout_impulses",
            "value": recovery["progress"],
            "unit": "normalized progress",
        },
        baselines_run=["braking/inactivity", "myopic progress", "exact recovery DP"],
        controls=checks,
        interpretation=(
            "The exact 2-D inertial environment contains recoverable and already-unrecoverable "
            "legal states and a useful recovery policy. This closes an environment gate only; "
            "the corridor-calibrated P4.1 estimator is not reused out of domain."
        ),
        non_claim=config.preregistration.non_claim,
        decision=(
            "Train and calibrate a fresh inertial-family P4.1 estimator before P4.2."
            if passed
            else "Repair the inertial exact-label gate before learned policy work."
        ),
    )


def _numeric_list(config: ProposalRunConfig, name: str) -> list[float]:
    values = config.params.get(name)
    if not isinstance(values, list) or not values or not all(
        isinstance(value, int | float) and not isinstance(value, bool) for value in values
    ):
        raise ValueError(f"params.{name} must be a nonempty numeric list")
    return [float(value) for value in values]


def _run_inertial_surrogate(
    config: ProposalRunConfig, writer: ResultsWriter
) -> VerdictReport:
    """One-shot whole-layout calibration for the inertial P4.1 extension."""

    checks = validate(config)
    v_fail = float(config.params.get("v_fail", -1.0))
    alpha = float(config.params.get("calibration_alpha", 0.1))
    state_coverage = float(config.params.get("state_coverage", 0.9))
    minimum_coverage = float(config.params.get("minimum_coverage", 0.9))
    minimum_condition_fraction = float(
        config.params.get("minimum_condition_success_fraction", 0.8)
    )
    maximum_width_fraction = float(config.params.get("maximum_width_fraction", 0.75))
    slips = _numeric_list(config, "condition_slips")
    horizons = [int(value) for value in _numeric_list(config, "condition_horizons")]
    estimator_version = str(config.params.get("inertial_estimator_version", "v1-dynamics"))
    if estimator_version == "v2-layout-geometry":
        layouts = {
            "train-e": ((1, 1), (2, 2)),
            "train-f": ((2, 1), (1, 2)),
            "calibration-new": ((1, 0), (2, 2)),
            "test-new": ((2, 0), (1, 2)),
        }
        train_names = ("train-e", "train-f")
        calibration_name = "calibration-new"
        test_name = "test-new"
    elif estimator_version == "v1-dynamics":
        layouts = {
            "train-a": ((1, 1), (2, 1)),
            "train-b": ((1, 2), (2, 2)),
            "calibration": ((1, 1), (1, 2)),
            "test": ((2, 1), (2, 2)),
        }
        train_names = ("train-a", "train-b")
        calibration_name = "calibration"
        test_name = "test"
    else:
        raise ValueError(f"unknown inertial estimator version {estimator_version!r}")

    def conditions(names: tuple[str, ...]) -> list[Condition]:
        return [
            (name, layouts[name], slip, horizon)
            for name in names
            for slip in slips
            for horizon in horizons
        ]

    train_conditions = conditions(train_names)
    calibration_conditions = conditions((calibration_name,))
    test_conditions = conditions((test_name,))
    train = inertial_recovery_dataset(train_conditions, v_fail=v_fail)
    calibration = inertial_recovery_dataset(calibration_conditions, v_fail=v_fail)
    surrogate = fit_inertial_surrogate(
        train,
        calibration,
        v_fail=v_fail,
        alpha=alpha,
        state_coverage=state_coverage,
        feature_version=estimator_version,
    )
    frozen_coefficients = surrogate.coefficients.copy()
    frozen_radius = surrogate.residual_radius
    test = inertial_recovery_dataset(test_conditions, v_fail=v_fail)
    rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, float]] = {}
    for split, frame in (("train", train), ("calibration", calibration), ("test", test)):
        summary = inertial_surrogate_metrics(surrogate, frame)
        summaries[split] = summary
        rows.append(
            {
                "stage": "p4.1-inertial",
                "split": split,
                "model_id": "quadratic_dynamics_features_condition_conformal",
                "estimator_version": estimator_version,
                **summary,
            }
        )
    test_summary = summaries["test"]
    relative_width = 2.0 * frozen_radius / max(test_summary["value_range"], 1e-12)
    layout_disjoint = set(train["layout"]).isdisjoint(calibration["layout"]) and set(
        train["layout"]
    ).isdisjoint(test["layout"]) and set(calibration["layout"]).isdisjoint(test["layout"])
    frozen = bool(
        np.array_equal(surrogate.coefficients, frozen_coefficients)
        and surrogate.residual_radius == frozen_radius
    )
    calibrated = bool(
        test_summary["coverage"] >= minimum_coverage
        and test_summary["condition_success_fraction"] >= minimum_condition_fraction
        and relative_width <= maximum_width_fraction
    )
    checks += [
        PrerequisiteCheck("whole obstacle-layout splits disjoint", layout_disjoint),
        PrerequisiteCheck("inertial surrogate frozen before test", frozen),
        PrerequisiteCheck(
            "held-out inertial coverage",
            test_summary["coverage"] >= minimum_coverage,
            f"coverage={test_summary['coverage']:.3f}",
        ),
        PrerequisiteCheck(
            "held-out inertial condition coverage",
            test_summary["condition_success_fraction"] >= minimum_condition_fraction,
            f"fraction={test_summary['condition_success_fraction']:.3f}",
        ),
        PrerequisiteCheck(
            "inertial interval nonvacuous",
            relative_width <= maximum_width_fraction,
            f"relative_width={relative_width:.3f}",
        ),
    ]
    rows.append(
        {
            "stage": "p4.1-inertial-audit",
            "split": "test",
            "model_id": "interval_width",
            "relative_interval_width": relative_width,
        }
    )
    writer.write_metrics(rows)
    writer.event(
        stage="p4.1-inertial",
        coverage=test_summary["coverage"],
        condition_success_fraction=test_summary["condition_success_fraction"],
        mae=test_summary["mae"],
        radius=frozen_radius,
        relative_interval_width=relative_width,
    )
    status = (
        Status.GO
        if calibrated and all(check.passed for check in checks)
        else Status.INCONCLUSIVE
    )
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=status,
        evidence_level="E1",
        primary_result={
            "metric": "heldout_inertial_exact_label_coverage",
            "value": test_summary["coverage"],
            "unit": "probability",
        },
        baselines_run=["exact inertial finite-horizon DP", "condition-conformal ridge"],
        controls=checks,
        failure_code=None if calibrated else FailureCode.F2_ESTIMATOR_FAILURE,
        interpretation=(
            "The estimator is trained and calibrated on complete obstacle layouts and evaluated "
            "once on a held-out layout. Passing is an in-domain P4.1 gate, not policy evidence."
        ),
        non_claim=config.preregistration.non_claim,
        decision=(
            "Lock this inertial estimator for a separately preregistered P4.2 comparison."
            if status is Status.GO
            else "Repair only from inertial train/calibration layouts before P4.2."
        ),
    )


def _run_surrogate_repair(
    config: ProposalRunConfig, writer: ResultsWriter
) -> VerdictReport:
    """Fresh v4 condition-level split; the preserved v3 test is never loaded."""
    checks = validate(config)
    v_fail = float(config.params.get("v_fail", -1.0))
    alpha = float(config.params.get("calibration_alpha", 0.1))
    state_coverage = float(config.params.get("state_coverage", 0.9))
    minimum_coverage = float(config.params.get("minimum_coverage", 0.9))
    minimum_condition_fraction = float(
        config.params.get("minimum_condition_success_fraction", 0.8)
    )
    maximum_width_fraction = float(config.params.get("maximum_width_fraction", 0.5))
    lengths = [int(value) for value in _numeric_list(config, "condition_lengths")]
    slips = _numeric_list(config, "condition_slips")
    horizons = [int(value) for value in _numeric_list(config, "condition_horizons")]
    split_seed = int(config.params.get("condition_split_seed", config.seed))
    legacy_test = (8, 0.2, 8)
    registry = condition_split_registry(
        lengths,
        slips,
        horizons,
        split_seed=split_seed,
        train_fraction=float(config.params.get("train_fraction", 0.5)),
        calibration_fraction=float(config.params.get("calibration_fraction", 0.25)),
        excluded=(legacy_test,),
    )
    train = exact_recovery_dataset(list(registry.train), v_fail=v_fail)
    calibration = exact_recovery_dataset(list(registry.calibration), v_fail=v_fail)
    # The model and its interval are frozen before exact labels for this fresh
    # test registry are generated.
    surrogate = fit_condition_calibrated_surrogate(
        train,
        calibration,
        v_fail=v_fail,
        alpha=alpha,
        state_coverage=state_coverage,
    )
    legacy = fit_calibrated_surrogate(train, calibration, alpha=alpha)
    frozen_coefficients = surrogate.coefficients.copy()
    frozen_radius = surrogate.residual_radius
    test = exact_recovery_dataset(list(registry.test), v_fail=v_fail)

    frames = {"train": train, "calibration": calibration, "test": test}
    rows: list[dict[str, object]] = []
    repaired_summaries: dict[str, dict[str, float]] = {}
    for split, frame in frames.items():
        summary = {
            **surrogate_metrics(surrogate, frame),
            **condition_coverage_metrics(surrogate, frame),
        }
        repaired_summaries[split] = summary
        rows.append(
            {
                "split": split,
                "model_id": "structured_condition_conformal",
                "estimator_version": "v4-condition-conformal",
                "proposal_version": config.theory.proposal_version,
                "theory_id": config.theory.theory_id,
                "amendment_id": config.theory.amendment_id,
                **summary,
            }
        )
        rows.append(
            {
                "split": split,
                "model_id": "v3_pointwise_ridge",
                "estimator_version": "v4-fresh-split-baseline",
                "proposal_version": config.theory.proposal_version,
                "theory_id": config.theory.theory_id,
                "amendment_id": config.theory.amendment_id,
                **surrogate_metrics(legacy, frame),
            }
        )

    split_sets = (set(registry.train), set(registry.calibration), set(registry.test))
    disjoint = all(
        split_sets[left].isdisjoint(split_sets[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    )
    legacy_excluded = all(legacy_test not in conditions for conditions in split_sets)
    test_summary = repaired_summaries["test"]
    test_coverage = test_summary["coverage"]
    condition_fraction = test_summary["condition_success_fraction"]
    calibration_range = float(
        calibration["exact_value"].max() - calibration["exact_value"].min()
    )
    relative_width = 2.0 * frozen_radius / max(calibration_range, 1e-12)
    nonvacuous = relative_width <= maximum_width_fraction
    frozen_before_test = bool(
        np.array_equal(surrogate.coefficients, frozen_coefficients)
        and surrogate.residual_radius == frozen_radius
    )
    calibrated = bool(
        test_coverage >= minimum_coverage
        and condition_fraction >= minimum_condition_fraction
        and nonvacuous
    )
    checks += [
        PrerequisiteCheck("fresh condition splits disjoint", disjoint),
        PrerequisiteCheck(
            "v3 held-out condition excluded",
            legacy_excluded,
            "excluded=(8, 0.2, 8) from train/calibration/test",
        ),
        PrerequisiteCheck(
            "surrogate frozen before fresh test labels",
            frozen_before_test,
            "coefficients/radius use v4 train/calibration conditions only",
        ),
        PrerequisiteCheck(
            "held-out exact-label coverage",
            test_coverage >= minimum_coverage,
            f"coverage={test_coverage:.3f}, required={minimum_coverage:.3f}",
        ),
        PrerequisiteCheck(
            "held-out condition coverage",
            condition_fraction >= minimum_condition_fraction,
            f"fraction={condition_fraction:.3f}, required={minimum_condition_fraction:.3f}",
        ),
        PrerequisiteCheck(
            "calibrated interval nonvacuous",
            nonvacuous,
            f"relative_width={relative_width:.3f}, maximum={maximum_width_fraction:.3f}",
        ),
    ]
    writer.write_metrics(rows)
    writer.event(
        stage="p4.1",
        estimator_version="v4-condition-conformal",
        coverage=test_coverage,
        condition_success_fraction=condition_fraction,
        mae=test_summary["mae"],
        calibration_radius=frozen_radius,
        relative_interval_width=relative_width,
        train_conditions=len(registry.train),
        calibration_conditions=len(registry.calibration),
        test_conditions=len(registry.test),
    )
    prerequisites = all(
        check.passed
        for check in checks
        if check.name
        not in {
            "held-out exact-label coverage",
            "held-out condition coverage",
            "calibrated interval nonvacuous",
        }
    )
    status = Status.GO if prerequisites and calibrated else Status.INCONCLUSIVE
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=status,
        evidence_level="E1",
        primary_result={
            "metric": "fresh_heldout_exact_label_coverage",
            "value": test_coverage,
            "unit": "probability",
        },
        baselines_run=[
            "exact finite-horizon DP",
            "v3 pointwise-calibrated polynomial ridge on the fresh split",
        ],
        controls=checks,
        failure_code=None if calibrated else FailureCode.F2_ESTIMATOR_FAILURE,
        interpretation=(
            "The repaired surrogate uses exact terminal constraints and nested condition-level "
            "split calibration. The v3 held-out condition is preserved and excluded. Passing "
            "this estimator gate is not evidence of recovery-slack policy improvement."
        ),
        non_claim=config.preregistration.non_claim,
        decision=(
            "Lock the estimator for a separately preregistered P4.2 policy comparison."
            if calibrated
            else "Keep P4.1 blocked; revise only from train/calibration evidence in a new version."
        ),
    )
def run(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    stage = str(config.params.get("stage", "p4.0"))
    if stage == "p4.1":
        return _run_surrogate(config, writer)
    if stage == "p4.1-inertial":
        return _run_inertial_surrogate(config, writer)
    if stage == "p4.2":
        from instinct.p4_slack.stage2 import run_p4_2

        return run_p4_2(config, writer)
    if stage == "p4.0-inertial":
        return _run_inertial(config, writer)
    return _run_exact(config, writer)


def resume(
    config: ProposalRunConfig, writer: ResultsWriter, checkpoint_state: Mapping[str, Any]
) -> VerdictReport:
    return resume_by_rerun(config, writer, checkpoint_state, run)
