from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

from scipy.stats import beta

from instinct.core.configschema import ProposalRunConfig
from instinct.core.proposal import common_checks, resume_by_rerun
from instinct.core.results import ResultsWriter
from instinct.core.verdict import PrerequisiteCheck, Status, VerdictReport
from instinct.p6_certification.certificates import polynomial_spend
from instinct.p6_certification.comparators import COMPARATOR_TYPES
from instinct.p6_certification.simulation import BenchmarkResult, run_benchmark


def _exits(config: ProposalRunConfig) -> tuple[int, ...]:
    raw = config.params.get("exits", [20, 40, 80, 160, 320])
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return ()
    try:
        return tuple(int(value) for value in raw)
    except (TypeError, ValueError):
        return ()


def validate(config: ProposalRunConfig) -> list[PrerequisiteCheck]:
    stage = str(config.params.get("stage", "p6.0-p6.1"))
    if stage == "p6.2":
        from instinct.p6_certification.mismatch import validate_p6_2

        return validate_p6_2(config)
    if stage == "p6.3":
        from instinct.p6_certification.frontier import validate_p6_3

        return validate_p6_3(config)
    if stage == "p6.4":
        from instinct.p6_certification.learned_dynamics import validate_p6_4

        return validate_p6_4(config)
    if stage == "p6.5":
        from instinct.p6_certification.realistic import validate_p6_5

        return validate_p6_5(config)
    if stage != "p6.0-p6.1":
        return [PrerequisiteCheck("known P6 stage", False, stage)]
    checks = common_checks(config, "p6_certification")
    delta = float(config.params.get("confidence_delta", 0.05))
    risk = float(config.params.get("episode_risk", 0.2))
    repetitions = int(config.params.get("repetitions", 200))
    decisions = int(config.params.get("repeated_decisions", 4))
    exits = _exits(config)
    checks += [
        PrerequisiteCheck("confidence delta valid", 0 < delta < 1),
        PrerequisiteCheck("risk budget valid", 0 < risk < 1),
        PrerequisiteCheck("repetitions positive", repetitions > 0),
        PrerequisiteCheck("repeated decisions positive", decisions > 0),
        PrerequisiteCheck(
            "declared exits strictly increasing",
            bool(exits) and exits[0] > 0 and all(right > left for left, right in pairwise(exits)),
        ),
    ]
    return checks


def _interval(successes: int, n: int, delta: float = 0.05) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(delta / 2, successes, n - successes + 1))
    upper = 1.0 if successes == n else float(beta.ppf(1 - delta / 2, successes + 1, n - successes))
    return lower, upper


def _metric_rows(results: list[BenchmarkResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = (
        ("optional_stopping_coverage", "probability"),
        ("prefix_selection_coverage", "probability"),
        ("prefix_false_certification_rate", "probability"),
        ("repeated_spending_coverage", "probability"),
        ("episode_violation_rate", "probability"),
        ("mean_selected_prefix", "actions"),
        ("abstention_rate", "probability"),
        ("nonvacuity_rate", "fraction_of_max_prefix"),
    )
    validity = {kind.name: kind.validity for kind in COMPARATOR_TYPES}
    for result in results:
        for metric_name, unit in metrics:
            rows.append(
                {
                    "stage": "p6.0-p6.1",
                    "comparator": result.comparator,
                    "validity_scope": validity[result.comparator],
                    "metric_name": metric_name,
                    "value": getattr(result, metric_name),
                    "unit": unit,
                }
            )
    return rows


def run(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    stage = str(config.params.get("stage", "p6.0-p6.1"))
    if stage == "p6.2":
        from instinct.p6_certification.mismatch import run_p6_2

        return run_p6_2(config, writer)
    if stage == "p6.3":
        from instinct.p6_certification.frontier import run_p6_3

        return run_p6_3(config, writer)
    if stage == "p6.4":
        from instinct.p6_certification.learned_dynamics import run_p6_4

        return run_p6_4(config, writer)
    if stage == "p6.5":
        from instinct.p6_certification.realistic import run_p6_5

        return run_p6_5(config, writer)
    if stage != "p6.0-p6.1":
        raise ValueError(
            f"unknown P6 stage {stage!r}; expected p6.0-p6.1 through p6.5"
        )
    checks = validate(config)
    if not all(check.passed for check in checks):
        return VerdictReport(
            question=config.preregistration.primary_hypothesis,
            status=Status.INCONCLUSIVE,
            evidence_level="E0",
            primary_result={"metric": "configuration_valid", "value": 0, "unit": "boolean"},
            controls=checks,
            interpretation="P6 comparator benchmark did not run because its config was invalid.",
            non_claim=config.preregistration.non_claim,
            decision="Repair the declared P6 simulation schedule and rerun.",
        )

    delta = float(config.params.get("confidence_delta", 0.05))
    risk = float(config.params.get("episode_risk", 0.2))
    repetitions = int(config.params.get("repetitions", 200))
    decisions = int(config.params.get("repeated_decisions", 4))
    tolerance = float(config.params.get("coverage_tolerance", 0.04))
    results = run_benchmark(
        seed=config.seed,
        confidence_delta=delta,
        episode_risk=risk,
        repetitions=repetitions,
        exits=_exits(config),
        repeated_decisions=decisions,
    )
    by_name = {result.comparator: result for result in results}
    reference = by_name["anytime_max_certified"]
    union = by_name["nested_exit_union"]
    fixed = by_name["fixed_time_pointwise"]
    safety_spend = sum(polynomial_spend(risk, index) for index in range(1, decisions + 1))
    confidence_spend = sum(polynomial_spend(delta, index) for index in range(1, decisions + 1))
    target_coverage = 1.0 - delta - tolerance
    checks += [
        PrerequisiteCheck(
            "optional-stopping coverage reported separately",
            reference.optional_stopping_coverage >= target_coverage,
            f"coverage={reference.optional_stopping_coverage:.3f}",
        ),
        PrerequisiteCheck(
            "prefix-selection coverage reported separately",
            reference.prefix_selection_coverage >= target_coverage,
            f"coverage={reference.prefix_selection_coverage:.3f}",
        ),
        PrerequisiteCheck(
            "repeated-spending coverage reported separately",
            reference.repeated_spending_coverage >= target_coverage,
            f"coverage={reference.repeated_spending_coverage:.3f}",
        ),
        PrerequisiteCheck(
            "fixed-time adaptive negative control",
            fixed.optional_stopping_coverage < reference.optional_stopping_coverage,
            (
                f"fixed={fixed.optional_stopping_coverage:.3f}, "
                f"anytime={reference.optional_stopping_coverage:.3f}"
            ),
        ),
        PrerequisiteCheck("episode safety spending bounded", safety_spend <= risk),
        PrerequisiteCheck("episode confidence spending bounded", confidence_spend <= delta),
        PrerequisiteCheck(
            "anytime certificate nonvacuous",
            reference.nonvacuity_rate > 0,
            f"fraction={reference.nonvacuity_rate:.3f}",
        ),
        PrerequisiteCheck(
            "exact truth hidden from certifier",
            True,
            "comparators receive Bernoulli counts, n, look index, and risk budget only",
        ),
    ]
    writer.write_metrics(_metric_rows(results))
    for result in results:
        writer.event(stage="p6.0-p6.1", **result.__dict__)

    coverage_successes = round(reference.repeated_spending_coverage * repetitions)
    coverage_lo, coverage_hi = _interval(coverage_successes, repetitions)
    validity_passed = all(check.passed for check in checks)
    material_gain = reference.mean_selected_prefix >= union.mean_selected_prefix + 0.25
    if not validity_passed:
        status = Status.INCONCLUSIVE
        interpretation = (
            "At least one validity or negative-control prerequisite failed; no certificate verdict "
            "is interpretable until that instrument failure is repaired."
        )
        decision = (
            "Repair the failed P6 prerequisite and rerun the same frozen comparator protocol."
        )
    elif material_gain:
        status = Status.GO
        interpretation = (
            "The anytime max-certified process retained separate validity across stopping, prefix "
            "selection, and episode spending and executed materially longer than the declared-exit "
            "union comparator on the exact-MDP benchmark."
        )
        decision = "Freeze P6.1 and preregister P6.2 model-mismatch analysis."
    else:
        status = Status.NARROW
        interpretation = (
            "Validity and nonvacuity are measurable, but the anytime max-certified "
            "construction did not execute materially longer than the simpler declared-exit "
            "union comparator. This is "
            "a bounded statistical result, not a deployment or novelty claim."
        )
        decision = (
            "Remain NARROW; characterize the finite-look efficiency gap before any P6.2 deployment "
            "claim."
        )
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=status,
        evidence_level="E1",
        primary_result={
            "metric": "repeated_spending_coverage",
            "value": reference.repeated_spending_coverage,
            "ci_lo": coverage_lo,
            "ci_hi": coverage_hi,
            "unit": "probability",
        },
        baselines_run=[kind.name for kind in COMPARATOR_TYPES],
        controls=checks,
        interpretation=interpretation,
        non_claim=config.preregistration.non_claim,
        decision=decision,
    )


def resume(
    config: ProposalRunConfig, writer: ResultsWriter, checkpoint_state: Mapping[str, Any]
) -> VerdictReport:
    return resume_by_rerun(config, writer, checkpoint_state, run)
