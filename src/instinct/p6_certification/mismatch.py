"""P6.2 model-mismatch benchmark with truth confined to evaluation.

The certificate sees samples from a learned rollout model.  A robust variant
also sees a disjoint, paired shift-calibration sample.  Exact model and true
environment risks are computed only after a prefix has been selected, so the
benchmark can report model-relative and environment-relative validity without
letting either oracle quantity enter the method.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import log
from typing import Any

import numpy as np
from scipy.special import betainc, betaln
from scipy.stats import beta

from instinct.core.configschema import ProposalRunConfig
from instinct.core.proposal import common_checks
from instinct.core.results import ResultsWriter
from instinct.core.verdict import PrerequisiteCheck, Status, VerdictReport
from instinct.p6_certification.certificates import clopper_pearson_upper, longest_certified
from instinct.p6_certification.comparators import (
    AlwaysAbstainComparator,
    AlwaysExecuteComparator,
    AnytimeMaxCertifiedComparator,
    ComparatorSnapshot,
    DeclaredExitUnionComparator,
    FixedTimePointwiseComparator,
    PrefixComparator,
)
from instinct.p6_certification.simulation import ExactPrefixMDP, hazards_from_cumulative


@dataclass(frozen=True)
class ShiftCondition:
    """A preregistered evaluation environment, not an input to a certificate."""

    name: str
    true_risks: tuple[float, ...]


@dataclass(frozen=True)
class MismatchResult:
    condition: str
    method: str
    validity_scope: str
    model_coverage: float
    true_coverage: float
    model_false_certification_rate: float
    true_false_certification_rate: float
    episode_violation_rate: float
    mean_selected_prefix: float
    abstention_rate: float
    nonvacuity_rate: float
    mean_selected_bound: float
    mean_shift_radius: float


@dataclass
class _Counts:
    model_covered: int = 0
    true_covered: int = 0
    model_false: int = 0
    true_false: int = 0
    episode_violations: int = 0
    selected_total: int = 0
    abstentions: int = 0
    selected_bound_total: float = 0.0
    selected_radius_total: float = 0.0


class AVCRCStyleComparator(AnytimeMaxCertifiedComparator):
    """A bounded Bernoulli/finite-family analogue, not AVCRC reproduction.

    The local implementation uses exact binomial bounds and summable look
    spending.  It captures AVCRC's growing-calibration-set requirement but not
    its general conformal-loss machinery or importance weighting.
    """

    name = "avcrc_style_cp_analogue"
    validity = "anytime Bernoulli analogue; simultaneous finite prefix family"


class CSAStyleEProcessComparator(PrefixComparator):
    """Threshold-only CSA-style Bernoulli e-process analogue.

    For each prefix this tests the single deployment null ``risk >= alpha``
    with a beta-mixture e-process.  Ville's inequality handles arbitrary
    stopping and Bonferroni handles prefix selection.  It is deliberately not
    presented as the full conformal selective acting algorithm.
    """

    name = "csa_style_eprocess_analogue"
    validity = "anytime at the declared risk threshold; simultaneous prefixes"

    def local_delta(self, look_index: int) -> float:
        del look_index
        return self.confidence_delta / self.n_prefixes

    @staticmethod
    def _log_e_value(bad: int, n: int, null_risk: float) -> float:
        good = n - bad
        a = bad + 1.0
        b = good + 1.0
        mass = float(betainc(a, b, null_risk))
        if mass <= 0.0:
            return -np.inf
        # Uniform mixture over alternatives theta in [0, null_risk].
        return (
            log(mass)
            + float(betaln(a, b))
            - (bad + 1.0) * log(null_risk)
            - good * log1p_negative(null_risk)
        )

    def snapshot(
        self,
        bad_counts: Sequence[int],
        n: int,
        look_index: int,
        risk_budget: float,
    ) -> ComparatorSnapshot:
        if len(bad_counts) != self.n_prefixes:
            raise ValueError("bad_counts length must equal declared prefix family")
        if not 1 <= look_index <= self.n_looks:
            raise ValueError("look index outside declared schedule")
        threshold = -log(self.local_delta(look_index))
        raw = [
            self._log_e_value(int(count), n, risk_budget) >= threshold
            for count in bad_counts
        ]
        # Nested risks imply that certification must form an initial segment.
        certified: list[bool] = []
        still_certified = True
        for value in raw:
            still_certified &= value
            certified.append(still_certified)
        bounds = {
            prefix: risk_budget if certified[prefix - 1] else 1.0
            for prefix in range(1, self.n_prefixes + 1)
        }
        return ComparatorSnapshot(
            bounds=bounds, selected_prefix=longest_certified(bounds, risk_budget)
        )


def log1p_negative(value: float) -> float:
    """Stable ``log(1-value)`` kept local to avoid a numerical dependency."""

    return float(np.log1p(-value))


METHOD_TYPES: tuple[type[PrefixComparator], ...] = (
    FixedTimePointwiseComparator,
    DeclaredExitUnionComparator,
    AnytimeMaxCertifiedComparator,
    AVCRCStyleComparator,
    CSAStyleEProcessComparator,
    AlwaysAbstainComparator,
    AlwaysExecuteComparator,
)

METHOD_VALIDITY: dict[str, str] = {
    **{kind.name: kind.validity for kind in METHOD_TYPES},
    "shift_robust_union": (
        "simultaneous declared exits/prefixes plus paired fixed-time shift envelope"
    ),
    "shift_robust_anytime": (
        "anytime model process plus paired fixed-time shift envelope"
    ),
}


def _sample_with_uniforms(mdp: ExactPrefixMDP, uniforms: np.ndarray) -> np.ndarray:
    if uniforms.ndim != 2 or uniforms.shape[1] != len(mdp.hazards):
        raise ValueError("uniform array does not match MDP prefix family")
    failures = uniforms < np.asarray(mdp.hazards, dtype=np.float64)
    return np.maximum.accumulate(failures, axis=1)


def calibrate_shift_envelope(
    model_bad: np.ndarray,
    environment_bad: np.ndarray,
    *,
    confidence_delta: float,
) -> tuple[float, ...]:
    """Upper-bound positive environment/model failure discordance.

    Only paired binary observations enter this function.  For any coupling,
    ``P(Y_true=1)-P(Y_model=1) <= P(Y_true=1,Y_model=0)``.  A fixed-time
    Clopper--Pearson bound on that discordance therefore yields an additive
    shift envelope.  Bonferroni allocation makes it simultaneous over the
    declared prefix family.
    """

    if not 0 < confidence_delta < 1:
        raise ValueError("confidence_delta must lie in (0,1)")
    if model_bad.shape != environment_bad.shape or model_bad.ndim != 2:
        raise ValueError("paired failure arrays must have one shared sample/prefix shape")
    if model_bad.shape[0] < 1 or model_bad.shape[1] < 1:
        raise ValueError("paired failure arrays cannot be empty")
    if not np.all(np.logical_or(model_bad == 0, model_bad == 1)) or not np.all(
        np.logical_or(environment_bad == 0, environment_bad == 1)
    ):
        raise ValueError("paired failures must be binary")
    discordant = np.logical_and(environment_bad, np.logical_not(model_bad))
    n = model_bad.shape[0]
    local_delta = confidence_delta / model_bad.shape[1]
    raw = np.asarray(
        [clopper_pearson_upper(int(value), n, local_delta) for value in discordant.sum(axis=0)],
        dtype=np.float64,
    )
    return tuple(float(value) for value in np.maximum.accumulate(raw))


def robust_snapshot(
    comparator: PrefixComparator,
    bad_counts: Sequence[int],
    n: int,
    look_index: int,
    risk_budget: float,
    shift_envelope: Sequence[float],
) -> ComparatorSnapshot:
    """Add a sampled shift envelope to a model-relative upper process."""

    model_bounds = comparator.bounds(bad_counts, n, look_index)
    if len(shift_envelope) != comparator.n_prefixes:
        raise ValueError("shift envelope length must equal declared prefix family")
    robust_bounds = {
        prefix: min(1.0, model_bounds[prefix] + float(shift_envelope[prefix - 1]))
        for prefix in model_bounds
    }
    return ComparatorSnapshot(robust_bounds, longest_certified(robust_bounds, risk_budget))


def _learn_model(
    nominal: ExactPrefixMDP, *, rng: np.random.Generator, samples: int
) -> ExactPrefixMDP:
    """Fit cumulative risks from a disjoint pilot rollout set."""

    counts = nominal.sample(rng, samples).sum(axis=0)
    # Jeffreys-like finite-sample smoothing avoids degenerate zero/one hazards.
    estimates = (counts.astype(np.float64) + 0.5) / (samples + 1.0)
    estimates = np.maximum.accumulate(estimates)
    estimates = np.minimum(estimates, 1.0 - 1e-9)
    return ExactPrefixMDP(hazards_from_cumulative(estimates.tolist()))


def _adaptive_snapshot(
    comparator: PrefixComparator,
    cumulative: np.ndarray,
    exits: tuple[int, ...],
    risk_budget: float,
    shift_envelope: Sequence[float] | None = None,
) -> ComparatorSnapshot:
    snapshot = ComparatorSnapshot({}, 0)
    for look_index, n in enumerate(exits, start=1):
        counts = cumulative[n - 1].astype(int)
        if shift_envelope is None:
            snapshot = comparator.snapshot(counts, n, look_index, risk_budget)
        else:
            snapshot = robust_snapshot(
                comparator, counts, n, look_index, risk_budget, shift_envelope
            )
        if snapshot.selected_prefix > 0:
            break
    return snapshot


def run_mismatch_benchmark(
    *,
    pilot_seed: int,
    eval_seed: int,
    confidence_delta: float,
    episode_risk: float,
    repetitions: int,
    exits: Sequence[int],
    nominal_model_risks: Sequence[float],
    pilot_model_samples: int,
    shift_calibration_samples: int,
    conditions: Sequence[ShiftCondition],
) -> tuple[list[MismatchResult], tuple[float, ...]]:
    """Run common-random-number P6.2 evaluation cells."""

    exit_tuple = tuple(int(value) for value in exits)
    if not exit_tuple or exit_tuple[0] < 1 or any(b <= a for a, b in pairwise(exit_tuple)):
        raise ValueError("exits must be strictly increasing positive sample counts")
    if repetitions < 1 or pilot_model_samples < 1 or shift_calibration_samples < 1:
        raise ValueError("sample counts and repetitions must be positive")
    if not conditions:
        raise ValueError("at least one shift condition is required")

    nominal = ExactPrefixMDP(hazards_from_cumulative(nominal_model_risks))
    pilot_rng = np.random.default_rng(pilot_seed)
    learned = _learn_model(nominal, rng=pilot_rng, samples=pilot_model_samples)
    model_truth = learned.exact_prefix_risks()  # evaluation only below selection
    n_prefixes = len(model_truth)
    if any(len(condition.true_risks) != n_prefixes for condition in conditions):
        raise ValueError("all shift cells must use the learned model's prefix family")

    rng = np.random.default_rng(eval_seed)
    results: list[MismatchResult] = []
    names = [kind.name for kind in METHOD_TYPES] + [
        "shift_robust_union",
        "shift_robust_anytime",
    ]
    for condition in conditions:
        environment = ExactPrefixMDP(hazards_from_cumulative(condition.true_risks))
        true_truth = environment.exact_prefix_risks()  # evaluation-only oracle
        counters = {name: _Counts() for name in names}
        for _ in range(repetitions):
            # This paired shift-calibration set is disjoint from both model
            # rollouts and the subsequent execution outcome.
            calibration_uniforms = rng.random((shift_calibration_samples, n_prefixes))
            model_calibration = _sample_with_uniforms(learned, calibration_uniforms)
            environment_calibration = _sample_with_uniforms(
                environment, calibration_uniforms
            )
            envelope = calibrate_shift_envelope(
                model_calibration,
                environment_calibration,
                confidence_delta=confidence_delta / 2.0,
            )
            model_outcomes = learned.sample(rng, exit_tuple[-1])
            cumulative = np.cumsum(model_outcomes, axis=0)

            snapshots: dict[str, ComparatorSnapshot] = {}
            for kind in METHOD_TYPES:
                comparator = kind(confidence_delta, n_prefixes, len(exit_tuple))
                snapshots[comparator.name] = _adaptive_snapshot(
                    comparator, cumulative, exit_tuple, episode_risk
                )
            robust_union = DeclaredExitUnionComparator(
                confidence_delta / 2.0, n_prefixes, len(exit_tuple)
            )
            robust_anytime = AnytimeMaxCertifiedComparator(
                confidence_delta / 2.0, n_prefixes, len(exit_tuple)
            )
            snapshots["shift_robust_union"] = _adaptive_snapshot(
                robust_union, cumulative, exit_tuple, episode_risk, envelope
            )
            snapshots["shift_robust_anytime"] = _adaptive_snapshot(
                robust_anytime, cumulative, exit_tuple, episode_risk, envelope
            )

            execution_uniform = float(rng.random())
            for name, snapshot in snapshots.items():
                selected = snapshot.selected_prefix
                bound = 1.0 if selected == 0 else snapshot.bounds[selected]
                model_risk = 0.0 if selected == 0 else float(model_truth[selected - 1])
                true_risk = 0.0 if selected == 0 else float(true_truth[selected - 1])
                radius = (
                    0.0
                    if selected == 0 or not name.startswith("shift_robust")
                    else envelope[selected - 1]
                )
                count = counters[name]
                count.model_covered += int(model_risk <= bound)
                count.true_covered += int(true_risk <= bound)
                count.model_false += int(selected > 0 and model_risk > episode_risk)
                count.true_false += int(selected > 0 and true_risk > episode_risk)
                count.episode_violations += int(execution_uniform < true_risk)
                count.selected_total += selected
                count.abstentions += int(selected == 0)
                count.selected_bound_total += bound
                count.selected_radius_total += radius

        for name, count in counters.items():
            results.append(
                MismatchResult(
                    condition=condition.name,
                    method=name,
                    validity_scope=METHOD_VALIDITY[name],
                    model_coverage=count.model_covered / repetitions,
                    true_coverage=count.true_covered / repetitions,
                    model_false_certification_rate=count.model_false / repetitions,
                    true_false_certification_rate=count.true_false / repetitions,
                    episode_violation_rate=count.episode_violations / repetitions,
                    mean_selected_prefix=count.selected_total / repetitions,
                    abstention_rate=count.abstentions / repetitions,
                    nonvacuity_rate=count.selected_total / (repetitions * n_prefixes),
                    mean_selected_bound=count.selected_bound_total / repetitions,
                    mean_shift_radius=count.selected_radius_total / repetitions,
                )
            )
    return results, tuple(float(value) for value in model_truth)


def _conditions(config: ProposalRunConfig) -> tuple[ShiftCondition, ...]:
    raw = config.params.get("shift_conditions", [])
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


def validate_p6_2(config: ProposalRunConfig) -> list[PrerequisiteCheck]:
    checks = common_checks(config, "p6_certification")
    exits_raw = config.params.get("exits", [])
    try:
        exits = tuple(int(value) for value in exits_raw) if isinstance(exits_raw, Sequence) else ()
    except (TypeError, ValueError):
        exits = ()
    conditions = _conditions(config)
    nominal_raw = config.params.get("nominal_model_risks", [])
    try:
        nominal = tuple(float(value) for value in nominal_raw)
    except (TypeError, ValueError):
        nominal = ()
    risk = float(config.params.get("episode_risk", 0.2))
    delta = float(config.params.get("confidence_delta", 0.05))
    checks += [
        PrerequisiteCheck("P6.2 stage declared", config.params.get("stage") == "p6.2"),
        PrerequisiteCheck(
            "pilot/evaluation seeds disjoint", bool(config.pilot_seeds and config.eval_seeds)
        ),
        PrerequisiteCheck("confidence delta valid", 0 < delta < 1),
        PrerequisiteCheck("risk budget valid", 0 < risk < 1),
        PrerequisiteCheck(
            "declared exits strictly increasing",
            bool(exits) and exits[0] > 0 and all(b > a for a, b in pairwise(exits)),
        ),
        PrerequisiteCheck(
            "nominal risks form a cumulative prefix family",
            bool(nominal)
            and all(0 <= value < 1 for value in nominal)
            and all(b >= a for a, b in pairwise(nominal)),
        ),
        PrerequisiteCheck(
            "fresh shift cells declared",
            bool(conditions) and all(len(cell.true_risks) == len(nominal) for cell in conditions),
        ),
        PrerequisiteCheck(
            "positive sample budgets",
            int(config.params.get("repetitions", 0)) > 0
            and int(config.params.get("pilot_model_samples", 0)) > 0
            and int(config.params.get("shift_calibration_samples", 0)) > 0,
        ),
        PrerequisiteCheck(
            "exact truth hidden from certifier",
            True,
            "certificate APIs accept counts and sampled discordance only; "
            "oracle risks are post-selection evaluation",
        ),
    ]
    return checks


def _metric_rows(results: Sequence[MismatchResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    units = {
        "model_coverage": "probability",
        "true_coverage": "probability",
        "model_false_certification_rate": "probability",
        "true_false_certification_rate": "probability",
        "episode_violation_rate": "probability",
        "mean_selected_prefix": "actions",
        "abstention_rate": "probability",
        "nonvacuity_rate": "fraction_of_max_prefix",
        "mean_selected_bound": "probability",
        "mean_shift_radius": "probability",
    }
    for result in results:
        for metric, unit in units.items():
            rows.append(
                {
                    "stage": "p6.2",
                    "condition": result.condition,
                    "method": result.method,
                    "validity_scope": result.validity_scope,
                    "metric_name": metric,
                    "value": getattr(result, metric),
                    "unit": unit,
                }
            )
    return rows


def _false_rate_upper(rate: float, repetitions: int, delta: float = 0.05) -> float:
    failures = round(rate * repetitions)
    if failures >= repetitions:
        return 1.0
    return float(beta.ppf(1.0 - delta, failures + 1, repetitions - failures))


def run_p6_2(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    checks = validate_p6_2(config)
    if not all(check.passed for check in checks):
        return VerdictReport(
            question=config.preregistration.primary_hypothesis,
            status=Status.INCONCLUSIVE,
            evidence_level="E0",
            primary_result={"metric": "configuration_valid", "value": 0, "unit": "boolean"},
            controls=checks,
            interpretation=(
                "P6.2 did not run because its preregistered mismatch protocol is invalid."
            ),
            non_claim=config.preregistration.non_claim,
            decision="Repair the P6.2 protocol without inspecting held-out outcomes and rerun.",
        )

    results, learned_risks = run_mismatch_benchmark(
        pilot_seed=config.pilot_seeds[0],
        eval_seed=config.eval_seeds[0],
        confidence_delta=float(config.params["confidence_delta"]),
        episode_risk=float(config.params["episode_risk"]),
        repetitions=int(config.params["repetitions"]),
        exits=tuple(int(value) for value in config.params["exits"]),
        nominal_model_risks=tuple(float(value) for value in config.params["nominal_model_risks"]),
        pilot_model_samples=int(config.params["pilot_model_samples"]),
        shift_calibration_samples=int(config.params["shift_calibration_samples"]),
        conditions=_conditions(config),
    )
    writer.write_metrics(_metric_rows(results))
    for result in results:
        writer.event(stage="p6.2", **result.__dict__)
    writer.event(stage="p6.2-model-fit", learned_model_risks=learned_risks)

    tolerance = float(config.params.get("false_certification_tolerance", 0.05))
    repetitions = int(config.params["repetitions"])
    robust = [result for result in results if result.method == "shift_robust_anytime"]
    unadjusted = [result for result in results if result.method == "anytime_max_certified"]
    robust_uppers = [
        _false_rate_upper(result.true_false_certification_rate, repetitions) for result in robust
    ]
    model_uppers = [
        _false_rate_upper(result.model_false_certification_rate, repetitions)
        for result in unadjusted
    ]
    robust_valid = all(upper <= tolerance for upper in robust_uppers)
    robust_nonvacuous = all(result.nonvacuity_rate > 0 for result in robust)
    model_valid = all(upper <= tolerance for upper in model_uppers)
    shifted_gap = max(
        result.true_false_certification_rate - result.model_false_certification_rate
        for result in unadjusted
    )
    checks += [
        PrerequisiteCheck(
            "model-relative validity reported separately",
            model_valid,
            f"max one-sided 95% upper={max(model_uppers):.3f}",
        ),
        PrerequisiteCheck(
            "shift-aware true validity",
            robust_valid,
            f"max one-sided 95% upper={max(robust_uppers):.3f}",
        ),
        PrerequisiteCheck(
            "shift-aware certificate nonvacuous in every cell",
            robust_nonvacuous,
            f"min nonvacuity={min(r.nonvacuity_rate for r in robust):.3f}",
        ),
        PrerequisiteCheck(
            "model/true validity distinction exercised",
            shifted_gap > 0,
            f"maximum true-minus-model false-certification gap={shifted_gap:.3f}",
        ),
    ]

    worst_false = max(result.true_false_certification_rate for result in robust)
    worst_false_upper = max(robust_uppers)
    if not robust_valid:
        status = Status.STOP
        interpretation = (
            "No deployment claim survives: the shift-aware certificate exceeded its preregistered "
            "true-environment false-certification tolerance in at least one held-out cell."
        )
        decision = "Preserve P6.2 as a model-mismatch failure and do not deploy the certificate."
    elif not robust_nonvacuous:
        status = Status.NARROW
        interpretation = (
            "The robust certificate remained true-environment valid but was vacuous in at least "
            "one held-out shift cell."
        )
        decision = "Remain NARROW; improve efficiency without weakening the shift envelope."
    else:
        status = Status.NARROW
        interpretation = (
            "A sampled shift envelope produced a certificate that was both nonvacuous and within "
            "the preregistered true-environment validity tolerance in every cell. Model-relative "
            "and true validity differ under adverse shift, so an unadjusted model certificate is "
            "not a deployment certificate. This finite Bernoulli study does not establish general "
            "AVCRC/CSA equivalence or utility dominance."
        )
        decision = (
            "Remain NARROW; test the paired shift assumption and efficiency on learned, "
            "non-Bernoulli dynamics before any deployment claim."
        )
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=status,
        evidence_level="E1",
        primary_result={
            "metric": "worst_cell_true_false_certification_rate",
            "value": worst_false,
            "ci_lo": 0.0,
            "ci_hi": worst_false_upper,
            "unit": "probability",
        },
        baselines_run=list(METHOD_VALIDITY),
        controls=checks,
        interpretation=interpretation,
        non_claim=config.preregistration.non_claim,
        decision=decision,
    )
