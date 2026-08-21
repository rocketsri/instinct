"""P6.3 CPU scale/frontier gate for model-mismatched prefix certificates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import log, log2, pi, sqrt
from time import perf_counter_ns
from typing import Any

import numpy as np
from scipy.stats import beta

from instinct.core.configschema import ProposalRunConfig
from instinct.core.proposal import common_checks
from instinct.core.results import ResultsWriter
from instinct.core.verdict import PrerequisiteCheck, Status, VerdictReport
from instinct.p6_certification.comparators import (
    AnytimeMaxCertifiedComparator,
    DeclaredExitUnionComparator,
)
from instinct.p6_certification.mismatch import (
    ShiftCondition,
    _learn_model,
    _sample_with_uniforms,
    calibrate_shift_envelope,
    robust_snapshot,
)
from instinct.p6_certification.simulation import ExactPrefixMDP, hazards_from_cumulative


@dataclass(frozen=True)
class FrontierResult:
    condition: str
    samples: int
    method: str
    validity_scope: str
    model_false_certification_rate: float
    true_false_certification_rate: float
    mean_selected_model_risk: float
    mean_selected_true_risk: float
    mean_selected_prefix: float
    abstention_rate: float
    nonvacuity_rate: float
    mean_certificate_latency_us: float
    mean_shift_calibration_latency_us: float
    trajectory_prefix_evaluations: int


@dataclass
class _ScaleCounts:
    model_false: int = 0
    true_false: int = 0
    model_risk_total: float = 0.0
    true_risk_total: float = 0.0
    selected_total: int = 0
    abstentions: int = 0
    certificate_ns: int = 0
    calibration_ns: int = 0


METHOD_VALIDITY = {
    "model_only_anytime_cp": "model-relative anytime process; no shift guarantee",
    "shift_robust_union": "declared-exit/prefix union plus paired shift envelope",
    "shift_robust_anytime_cp": "summable CP process plus paired shift envelope",
    "shift_robust_avcrc_specialization": (
        "AVCRC Theorem 4.1 Bernoulli specialization plus paired shift envelope"
    ),
    "shift_robust_csa_specialization": (
        "CSA threshold e-process specialization plus paired shift envelope"
    ),
    "always_abstain": "deterministic abstention",
    "always_execute": "uncertified execution control",
}


def avcrc_correction(
    n: int,
    risk_budget: float,
    confidence_delta: float,
    *,
    bound: float = 1.0,
    intrinsic_time_floor: float = 1.0,
) -> float:
    """Theorem 4.1 stitched correction for bounded monotone loss.

    This is the paper's Bernoulli/miscoverage specialization: ``B=1`` and
    intrinsic time ``alpha * (B-alpha) * n``.  Returning a correction larger
    than ``alpha`` correctly makes the selected family member vacuous.
    """

    if n < 1 or not 0 < risk_budget < bound or not 0 < confidence_delta < 1:
        raise ValueError("AVCRC inputs outside the theorem's bounded-loss domain")
    variance_time = risk_budget * (bound - risk_budget) * n
    scaled = max(variance_time, intrinsic_time_floor) / intrinsic_time_floor
    h = 2.0 * log(log2(scaled) + 1.0) + log(pi**2 / (6.0 * confidence_delta))
    boundary = 1.44 * sqrt(max(variance_time, intrinsic_time_floor) * h) + 2.42 * bound * h
    return boundary / n


def avcrc_shift_snapshot(
    bad_counts: Sequence[int],
    n: int,
    risk_budget: float,
    confidence_delta: float,
    shift_envelope: Sequence[float],
) -> int:
    """Compose faithful model-risk AVCRC with a simultaneous shift envelope.

    A common worst-prefix shift adjustment preserves the monotone loss family
    required by AVCRC.  It is conservative but avoids claiming that the
    paper's importance-weighted shift theorem applies without density ratios.
    """

    if len(bad_counts) != len(shift_envelope) or len(bad_counts) == 0:
        raise ValueError("counts and shift envelope must share a nonempty prefix family")
    adjusted_budget = risk_budget - max(float(value) for value in shift_envelope)
    if adjusted_budget <= 0:
        return 0
    correction = avcrc_correction(n, adjusted_budget, confidence_delta)
    empirical = np.asarray(bad_counts, dtype=np.float64) / n
    certified = empirical <= adjusted_budget - correction
    selected = 0
    for prefix, passed in enumerate(certified, start=1):
        if not passed:
            break
        selected = prefix
    return selected


def csa_shift_snapshot(
    bad_counts: Sequence[int],
    n: int,
    risk_budget: float,
    confidence_delta: float,
    shift_envelope: Sequence[float],
    bet_margin: float,
) -> int:
    """Faithful finite-grid CSA specialization under Bernoulli IID losses."""

    if len(bad_counts) != len(shift_envelope) or len(bad_counts) == 0:
        raise ValueError("counts and shift envelope must share a nonempty prefix family")
    local_delta = confidence_delta / len(bad_counts)
    log_threshold = -log(local_delta)
    selected = 0
    for prefix, (bad, radius) in enumerate(zip(bad_counts, shift_envelope), start=1):
        model_budget = risk_budget - float(radius)
        if model_budget <= 0:
            break
        alternative = max(1e-9, model_budget - bet_margin)
        if alternative >= model_budget:
            raise ValueError("CSA bet margin must be positive")
        good = n - int(bad)
        # A fixed alternative is predictable, and its Bernoulli likelihood
        # ratio is a nonnegative supermartingale for the composite null
        # risk >= model_budget. This is a faithful CSA e-process cell.
        log_e = int(bad) * log(alternative / model_budget) + good * log(
            (1.0 - alternative) / (1.0 - model_budget)
        )
        if log_e < log_threshold:
            break
        selected = prefix
    return selected


def _conditions(config: ProposalRunConfig) -> tuple[ShiftCondition, ...]:
    raw = config.params.get("shift_families", [])
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return ()
    parsed: list[ShiftCondition] = []
    for item in raw:
        if not isinstance(item, Mapping):
            return ()
        risks = item.get("true_risks", [])
        if not isinstance(risks, Sequence) or isinstance(risks, str | bytes):
            return ()
        try:
            parsed.append(
                ShiftCondition(str(item["name"]), tuple(float(value) for value in risks))
            )
        except (KeyError, TypeError, ValueError):
            return ()
    return tuple(parsed)


def validate_p6_3(config: ProposalRunConfig) -> list[PrerequisiteCheck]:
    checks = common_checks(config, "p6_certification")
    sample_sizes_raw = config.params.get("sample_sizes", [])
    nominal_raw = config.params.get("nominal_model_risks", [])
    try:
        sample_sizes = tuple(int(value) for value in sample_sizes_raw)
        nominal = tuple(float(value) for value in nominal_raw)
    except (TypeError, ValueError):
        sample_sizes, nominal = (), ()
    conditions = _conditions(config)
    checks += [
        PrerequisiteCheck("P6.3 stage declared", config.params.get("stage") == "p6.3"),
        PrerequisiteCheck(
            "larger trajectory frontier declared",
            bool(sample_sizes)
            and sample_sizes[0] > 0
            and all(b > a for a, b in pairwise(sample_sizes))
            and sample_sizes[-1] > 640,
        ),
        PrerequisiteCheck(
            "dense computation-exit schedule declared",
            int(config.params.get("declared_looks", 0)) >= len(sample_sizes),
        ),
        PrerequisiteCheck(
            "nested nominal prefix risks",
            bool(nominal)
            and all(0 <= value < 1 for value in nominal)
            and all(b >= a for a, b in pairwise(nominal)),
        ),
        PrerequisiteCheck(
            "fresh shift families match prefix shape",
            len(conditions) >= 3
            and all(
                len(condition.true_risks) == len(nominal)
                and all(0 <= value < 1 for value in condition.true_risks)
                and all(b >= a for a, b in pairwise(condition.true_risks))
                for condition in conditions
            ),
        ),
        PrerequisiteCheck(
            "fresh pilot/evaluation units",
            bool(config.pilot_seeds and config.eval_seeds)
            and not set(config.pilot_seeds).intersection(config.eval_seeds),
        ),
        PrerequisiteCheck(
            "positive scale budgets",
            int(config.params.get("repetitions", 0)) > 0
            and int(config.params.get("pilot_model_samples", 0)) > 0,
        ),
        PrerequisiteCheck(
            "substantive gain locked",
            float(config.params.get("gain_actions", 0.0)) > 0,
        ),
        PrerequisiteCheck(
            "CSA predictable bet margin locked",
            0
            < float(config.params.get("csa_bet_margin", 0.0))
            < float(config.params.get("episode_risk", 0.0)),
        ),
        PrerequisiteCheck(
            "certificate latency gain locked",
            float(config.params.get("latency_speedup", 0.0)) > 1,
        ),
        PrerequisiteCheck(
            "exact truth remains evaluation-only",
            True,
            "decision functions accept counts, risk/confidence budgets, "
            "and sampled shift envelopes",
        ),
    ]
    return checks


def run_frontier_benchmark(
    *,
    pilot_seed: int,
    eval_seed: int,
    confidence_delta: float,
    episode_risk: float,
    repetitions: int,
    sample_sizes: Sequence[int],
    declared_looks: int,
    nominal_model_risks: Sequence[float],
    pilot_model_samples: int,
    conditions: Sequence[ShiftCondition],
    csa_bet_margin: float,
) -> tuple[list[FrontierResult], tuple[float, ...]]:
    sizes = tuple(int(value) for value in sample_sizes)
    if not sizes or sizes[0] < 1 or any(b <= a for a, b in pairwise(sizes)):
        raise ValueError("sample sizes must be positive and strictly increasing")
    if declared_looks < len(sizes):
        raise ValueError("declared looks must cover every reported scale checkpoint")
    nominal = ExactPrefixMDP(hazards_from_cumulative(nominal_model_risks))
    learned = _learn_model(
        nominal, rng=np.random.default_rng(pilot_seed), samples=pilot_model_samples
    )
    model_truth = learned.exact_prefix_risks()
    n_prefixes = len(model_truth)
    if any(len(condition.true_risks) != n_prefixes for condition in conditions):
        raise ValueError("shift families must match the model prefix family")

    rng = np.random.default_rng(eval_seed)
    results: list[FrontierResult] = []
    methods = tuple(METHOD_VALIDITY)
    for condition in conditions:
        environment = ExactPrefixMDP(hazards_from_cumulative(condition.true_risks))
        true_truth = environment.exact_prefix_risks()  # evaluation-only oracle
        counters = {
            (samples, method): _ScaleCounts() for samples in sizes for method in methods
        }
        for _ in range(repetitions):
            calibration_uniforms = rng.random((sizes[-1], n_prefixes))
            model_calibration = _sample_with_uniforms(learned, calibration_uniforms)
            environment_calibration = _sample_with_uniforms(
                environment, calibration_uniforms
            )
            model_outcomes = learned.sample(rng, sizes[-1])
            cumulative = np.cumsum(model_outcomes, axis=0)
            for samples in sizes:
                look_index = max(1, round(declared_looks * samples / sizes[-1]))
                calibration_start = perf_counter_ns()
                envelope = calibrate_shift_envelope(
                    model_calibration[:samples],
                    environment_calibration[:samples],
                    confidence_delta=confidence_delta / 2.0,
                )
                calibration_ns = perf_counter_ns() - calibration_start
                bad_counts = cumulative[samples - 1].astype(int)

                selections: dict[str, tuple[int, int]] = {}
                start = perf_counter_ns()
                model_only = AnytimeMaxCertifiedComparator(
                    confidence_delta, n_prefixes, declared_looks
                ).snapshot(bad_counts, samples, look_index, episode_risk)
                selections["model_only_anytime_cp"] = (
                    model_only.selected_prefix,
                    perf_counter_ns() - start,
                )

                start = perf_counter_ns()
                union = robust_snapshot(
                    DeclaredExitUnionComparator(
                        confidence_delta / 2.0, n_prefixes, declared_looks
                    ),
                    bad_counts,
                    samples,
                    look_index,
                    episode_risk,
                    envelope,
                )
                selections["shift_robust_union"] = (
                    union.selected_prefix,
                    perf_counter_ns() - start,
                )

                start = perf_counter_ns()
                anytime = robust_snapshot(
                    AnytimeMaxCertifiedComparator(
                        confidence_delta / 2.0, n_prefixes, declared_looks
                    ),
                    bad_counts,
                    samples,
                    look_index,
                    episode_risk,
                    envelope,
                )
                selections["shift_robust_anytime_cp"] = (
                    anytime.selected_prefix,
                    perf_counter_ns() - start,
                )

                start = perf_counter_ns()
                avcrc = avcrc_shift_snapshot(
                    bad_counts,
                    samples,
                    episode_risk,
                    confidence_delta / 2.0,
                    envelope,
                )
                selections["shift_robust_avcrc_specialization"] = (
                    avcrc,
                    perf_counter_ns() - start,
                )

                start = perf_counter_ns()
                csa = csa_shift_snapshot(
                    bad_counts,
                    samples,
                    episode_risk,
                    confidence_delta / 2.0,
                    envelope,
                    csa_bet_margin,
                )
                selections["shift_robust_csa_specialization"] = (
                    csa,
                    perf_counter_ns() - start,
                )
                selections["always_abstain"] = (0, 0)
                selections["always_execute"] = (n_prefixes, 0)

                for method, (selected, certificate_ns) in selections.items():
                    model_risk = 0.0 if selected == 0 else float(model_truth[selected - 1])
                    true_risk = 0.0 if selected == 0 else float(true_truth[selected - 1])
                    count = counters[(samples, method)]
                    count.model_false += int(selected > 0 and model_risk > episode_risk)
                    count.true_false += int(selected > 0 and true_risk > episode_risk)
                    count.model_risk_total += model_risk
                    count.true_risk_total += true_risk
                    count.selected_total += selected
                    count.abstentions += int(selected == 0)
                    count.certificate_ns += certificate_ns
                    count.calibration_ns += calibration_ns if method.startswith("shift_") else 0

        for samples in sizes:
            for method in methods:
                count = counters[(samples, method)]
                results.append(
                    FrontierResult(
                        condition=condition.name,
                        samples=samples,
                        method=method,
                        validity_scope=METHOD_VALIDITY[method],
                        model_false_certification_rate=count.model_false / repetitions,
                        true_false_certification_rate=count.true_false / repetitions,
                        mean_selected_model_risk=count.model_risk_total / repetitions,
                        mean_selected_true_risk=count.true_risk_total / repetitions,
                        mean_selected_prefix=count.selected_total / repetitions,
                        abstention_rate=count.abstentions / repetitions,
                        nonvacuity_rate=count.selected_total / (repetitions * n_prefixes),
                        mean_certificate_latency_us=count.certificate_ns
                        / (repetitions * 1_000.0),
                        mean_shift_calibration_latency_us=count.calibration_ns
                        / (repetitions * 1_000.0),
                        trajectory_prefix_evaluations=samples * n_prefixes,
                    )
                )
    return results, tuple(float(value) for value in model_truth)


def _upper(rate: float, repetitions: int, delta: float = 0.05) -> float:
    failures = round(rate * repetitions)
    if failures >= repetitions:
        return 1.0
    return float(beta.ppf(1.0 - delta, failures + 1, repetitions - failures))


def _metric_rows(results: Sequence[FrontierResult]) -> list[dict[str, Any]]:
    units = {
        "model_false_certification_rate": "probability",
        "true_false_certification_rate": "probability",
        "mean_selected_model_risk": "probability",
        "mean_selected_true_risk": "probability",
        "mean_selected_prefix": "actions",
        "abstention_rate": "probability",
        "nonvacuity_rate": "fraction_of_max_prefix",
        "mean_certificate_latency_us": "microseconds",
        "mean_shift_calibration_latency_us": "microseconds",
        "trajectory_prefix_evaluations": "prefix_evaluations",
    }
    rows: list[dict[str, Any]] = []
    for result in results:
        for metric, unit in units.items():
            rows.append(
                {
                    "stage": "p6.3",
                    "condition": result.condition,
                    "samples": result.samples,
                    "method": result.method,
                    "validity_scope": result.validity_scope,
                    "metric_name": metric,
                    "value": getattr(result, metric),
                    "unit": unit,
                }
            )
    return rows


def run_p6_3(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    checks = validate_p6_3(config)
    if not all(check.passed for check in checks):
        return VerdictReport(
            question=config.preregistration.primary_hypothesis,
            status=Status.INCONCLUSIVE,
            evidence_level="E0",
            primary_result={"metric": "configuration_valid", "value": 0, "unit": "boolean"},
            controls=checks,
            interpretation="P6.3 did not run because its frozen frontier protocol is invalid.",
            non_claim=config.preregistration.non_claim,
            decision="Repair instrumentation without inspecting held-out outcomes and rerun.",
        )

    results, learned_risks = run_frontier_benchmark(
        pilot_seed=config.pilot_seeds[0],
        eval_seed=config.eval_seeds[0],
        confidence_delta=float(config.params["confidence_delta"]),
        episode_risk=float(config.params["episode_risk"]),
        repetitions=int(config.params["repetitions"]),
        sample_sizes=tuple(int(value) for value in config.params["sample_sizes"]),
        declared_looks=int(config.params["declared_looks"]),
        nominal_model_risks=tuple(float(value) for value in config.params["nominal_model_risks"]),
        pilot_model_samples=int(config.params["pilot_model_samples"]),
        conditions=_conditions(config),
        csa_bet_margin=float(config.params["csa_bet_margin"]),
    )
    writer.write_metrics(_metric_rows(results))
    for result in results:
        writer.event(stage="p6.3", **result.__dict__)
    writer.event(stage="p6.3-model-fit", learned_model_risks=learned_risks)

    repetitions = int(config.params["repetitions"])
    tolerance = float(config.params["false_certification_tolerance"])
    gain_target = float(config.params["gain_actions"])
    latency_target = float(config.params["latency_speedup"])
    largest = max(result.samples for result in results)
    primary = [
        result
        for result in results
        if result.samples == largest and result.method == "shift_robust_csa_specialization"
    ]
    union = [
        result
        for result in results
        if result.samples == largest and result.method == "shift_robust_union"
    ]
    uppers = [_upper(result.true_false_certification_rate, repetitions) for result in primary]
    valid = all(upper <= tolerance for upper in uppers)
    nonvacuous = all(result.nonvacuity_rate > 0 for result in primary)
    gain = float(np.mean([result.mean_selected_prefix for result in primary])) - float(
        np.mean([result.mean_selected_prefix for result in union])
    )
    latency_speedup = float(
        np.mean([result.mean_certificate_latency_us for result in union])
    ) / float(np.mean([result.mean_certificate_latency_us for result in primary]))
    model_true_gap = max(
        result.true_false_certification_rate - result.model_false_certification_rate
        for result in results
        if result.method == "model_only_anytime_cp"
    )
    checks += [
        PrerequisiteCheck(
            "CSA specialization true validity",
            valid,
            f"worst one-sided 95% upper={max(uppers):.3f}",
        ),
        PrerequisiteCheck(
            "CSA specialization nonvacuous in every largest-scale family",
            nonvacuous,
            f"minimum nonvacuity={min(r.nonvacuity_rate for r in primary):.3f}",
        ),
        PrerequisiteCheck(
            "substantive gain over robust union",
            gain >= gain_target,
            f"gain={gain:.3f} actions; target={gain_target:.3f}",
        ),
        PrerequisiteCheck(
            "certificate-only latency gain over robust union",
            latency_speedup >= latency_target,
            f"speedup={latency_speedup:.2f}x; target={latency_target:.2f}x; "
            "shared calibration separate",
        ),
        PrerequisiteCheck(
            "model/true risk separation retained",
            model_true_gap > 0,
            f"maximum false-certification gap={model_true_gap:.3f}",
        ),
        PrerequisiteCheck(
            "sample and latency scaling reported",
            all(result.mean_certificate_latency_us >= 0 for result in results)
            and len({result.samples for result in results}) >= 4,
        ),
    ]

    if not valid:
        status = Status.STOP
        interpretation = (
            "The strongest frontier method failed true-environment validity under a fresh shift "
            "family, so no scale or deployment claim survives."
        )
        decision = "Archive this frontier version and diagnose the failed shift assumption."
    elif not nonvacuous:
        status = Status.NARROW
        interpretation = "The frontier remained valid but became vacuous in a largest-scale cell."
        decision = "Remain NARROW; do not trade away the locked validity criterion."
    elif gain >= gain_target and latency_speedup >= latency_target:
        status = Status.GO
        interpretation = (
            "The faithful finite-grid CSA specialization remained true-environment valid and "
            "nonvacuous and exceeded the robust union by the preregistered action-prefix and "
            "certificate-latency margins. Shared shift calibration is reported separately. This "
            "is a bounded CPU frontier result, not a general CSA or deployment claim."
        )
        decision = "Proceed to a realistic learned-dynamics integration with fresh units."
    else:
        status = Status.NARROW
        interpretation = (
            "The frontier was valid and nonvacuous but did not achieve both preregistered gain "
            "conditions over the strongest robust union baseline."
        )
        decision = "Preserve the valid frontier and remain NARROW on utility."
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=status,
        evidence_level="E2",
        primary_result={
            "metric": "largest_scale_mean_prefix_gain_over_robust_union",
            "value": gain,
            "unit": "actions",
        },
        baselines_run=list(METHOD_VALIDITY),
        controls=checks,
        interpretation=interpretation,
        non_claim=config.preregistration.non_claim,
        decision=decision,
    )
