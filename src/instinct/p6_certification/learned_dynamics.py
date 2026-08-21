"""P6.4 learned-transition integration and bounded CPU scale-readiness gate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from time import perf_counter_ns
from typing import Any, Literal

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
from instinct.p6_certification.frontier import avcrc_shift_snapshot, csa_shift_snapshot
from instinct.p6_certification.mismatch import calibrate_shift_envelope, robust_snapshot

CertificateMethod = Literal[
    "model_only_anytime_cp",
    "shift_robust_union",
    "shift_robust_avcrc_specialization",
    "shift_robust_csa_specialization",
    "always_abstain",
    "always_execute",
]

METHOD_VALIDITY: dict[str, str] = {
    "model_only_anytime_cp": "model-relative anytime process; no structural-shift guarantee",
    "shift_robust_union": "finite exit/prefix union plus paired structural-shift envelope",
    "shift_robust_avcrc_specialization": (
        "AVCRC bounded-loss specialization plus common paired shift envelope"
    ),
    "shift_robust_csa_specialization": (
        "CSA finite-grid e-process specialization plus per-prefix paired shift envelope"
    ),
    "always_abstain": "deterministic abstention",
    "always_execute": "uncertified execution control",
}


@dataclass(frozen=True)
class StructuralShift:
    name: str
    transition_matrices: np.ndarray


@dataclass(frozen=True)
class FitProfile:
    fit_latency_ms: float
    pilot_transitions: int
    transition_l1_error: float


@dataclass(frozen=True)
class DynamicsResult:
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
    mean_model_rollout_latency_us: float
    certificate_input_memory_kb: float
    model_rollout_buffer_kb: float
    paired_calibration_buffer_kb: float
    transitions_evaluated: int


@dataclass
class _Counts:
    model_false: int = 0
    true_false: int = 0
    model_risk: float = 0.0
    true_risk: float = 0.0
    selected: int = 0
    abstained: int = 0
    certificate_ns: int = 0
    calibration_ns: int = 0
    rollout_ns: int = 0


@dataclass(frozen=True)
class TransitionRiskModel:
    """Action-conditioned categorical dynamics with absorbing failure state."""

    matrices: np.ndarray
    action_schedule: tuple[int, ...]

    def __post_init__(self) -> None:
        matrices = np.asarray(self.matrices, dtype=np.float64)
        if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
            raise ValueError("transition matrices must have shape [action,state,state]")
        if not self.action_schedule or max(self.action_schedule) >= matrices.shape[0]:
            raise ValueError("action schedule is empty or references an absent matrix")
        if np.any(matrices < 0) or not np.allclose(matrices.sum(axis=2), 1.0):
            raise ValueError("transition rows must be probability vectors")
        if not np.allclose(matrices[:, -1, -1], 1.0):
            raise ValueError("last state must be absorbing failure")
        object.__setattr__(self, "matrices", matrices)

    @property
    def horizon(self) -> int:
        return len(self.action_schedule)

    @classmethod
    def fit(
        cls,
        states: np.ndarray,
        action_schedule: Sequence[int],
        *,
        n_actions: int,
        smoothing: float = 0.5,
    ) -> TransitionRiskModel:
        """Fit transition probabilities from complete pilot trajectories."""

        schedule = tuple(int(value) for value in action_schedule)
        if states.ndim != 2 or states.shape[1] != len(schedule) + 1:
            raise ValueError("states must contain one more column than the action schedule")
        if smoothing <= 0 or n_actions < 1:
            raise ValueError("smoothing and action count must be positive")
        n_states = int(states.max()) + 1
        counts = np.full((n_actions, n_states, n_states), smoothing, dtype=np.float64)
        for step, action in enumerate(schedule):
            np.add.at(counts[action], (states[:, step], states[:, step + 1]), 1.0)
        counts[:, -1, :] = 0.0
        counts[:, -1, -1] = 1.0
        matrices = counts / counts.sum(axis=2, keepdims=True)
        return cls(matrices, schedule)

    def sample_states_with_uniforms(self, uniforms: np.ndarray) -> np.ndarray:
        if uniforms.ndim != 2 or uniforms.shape[1] != self.horizon:
            raise ValueError("uniforms must have shape [trajectory,horizon]")
        n = uniforms.shape[0]
        states = np.zeros((n, self.horizon + 1), dtype=np.int16)
        for step, action in enumerate(self.action_schedule):
            probabilities = self.matrices[action, states[:, step]]
            cumulative = np.cumsum(probabilities, axis=1)
            states[:, step + 1] = np.sum(
                uniforms[:, step, None] > cumulative, axis=1
            ).astype(np.int16)
        return states

    def sample_failures(self, rng: np.random.Generator, n: int) -> np.ndarray:
        states = self.sample_states_with_uniforms(rng.random((n, self.horizon)))
        return states[:, 1:] == self.matrices.shape[1] - 1

    def exact_prefix_risks(self) -> np.ndarray:
        """Evaluation-only dynamic-programming oracle."""

        distribution = np.zeros(self.matrices.shape[1], dtype=np.float64)
        distribution[0] = 1.0
        risks: list[float] = []
        for action in self.action_schedule:
            distribution = distribution @ self.matrices[action]
            risks.append(float(distribution[-1]))
        return np.asarray(risks, dtype=np.float64)


def certify_prefixes(
    method: CertificateMethod,
    bad_counts: Sequence[int],
    *,
    n: int,
    look_index: int,
    declared_looks: int,
    episode_risk: float,
    confidence_delta: float,
    shift_envelope: Sequence[float],
    csa_bet_margin: float,
) -> int:
    """Stable sampled-data certificate API; no exact risk argument exists."""

    n_prefixes = len(bad_counts)
    if method == "model_only_anytime_cp":
        return AnytimeMaxCertifiedComparator(
            confidence_delta, n_prefixes, declared_looks
        ).snapshot(bad_counts, n, look_index, episode_risk).selected_prefix
    if method == "shift_robust_union":
        return robust_snapshot(
            DeclaredExitUnionComparator(
                confidence_delta / 2.0, n_prefixes, declared_looks
            ),
            bad_counts,
            n,
            look_index,
            episode_risk,
            shift_envelope,
        ).selected_prefix
    if method == "shift_robust_avcrc_specialization":
        return avcrc_shift_snapshot(
            bad_counts, n, episode_risk, confidence_delta / 2.0, shift_envelope
        )
    if method == "shift_robust_csa_specialization":
        return csa_shift_snapshot(
            bad_counts,
            n,
            episode_risk,
            confidence_delta / 2.0,
            shift_envelope,
            csa_bet_margin,
        )
    if method == "always_abstain":
        return 0
    if method == "always_execute":
        return n_prefixes
    raise ValueError(f"unknown certificate method {method!r}")


def _array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _nominal_model(config: ProposalRunConfig) -> TransitionRiskModel:
    matrices = _array(config.params.get("nominal_transition_matrices", []))
    schedule = tuple(int(value) for value in config.params.get("action_schedule", []))
    return TransitionRiskModel(matrices, schedule)


def _shifts(config: ProposalRunConfig) -> tuple[StructuralShift, ...]:
    raw = config.params.get("structural_shifts", [])
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return ()
    parsed: list[StructuralShift] = []
    for item in raw:
        if not isinstance(item, Mapping):
            return ()
        try:
            parsed.append(StructuralShift(str(item["name"]), _array(item["matrices"])))
        except (KeyError, TypeError, ValueError):
            return ()
    return tuple(parsed)


def validate_p6_4(config: ProposalRunConfig) -> list[PrerequisiteCheck]:
    checks = common_checks(config, "p6_certification")
    try:
        nominal = _nominal_model(config)
        shifts = _shifts(config)
        shift_valid = len(shifts) >= 3 and all(
            TransitionRiskModel(shift.transition_matrices, nominal.action_schedule).horizon
            == nominal.horizon
            for shift in shifts
        )
    except (TypeError, ValueError):
        nominal = None
        shift_valid = False
    sizes_raw = config.params.get("sample_sizes", [])
    try:
        sizes = tuple(int(value) for value in sizes_raw)
    except (TypeError, ValueError):
        sizes = ()
    checks += [
        PrerequisiteCheck("P6.4 stage declared", config.params.get("stage") == "p6.4"),
        PrerequisiteCheck("fittable nominal transition model", nominal is not None),
        PrerequisiteCheck("held-out structural shifts valid", shift_valid),
        PrerequisiteCheck(
            "scale frontier exceeds P6.2",
            bool(sizes)
            and sizes[0] > 0
            and all(b > a for a, b in pairwise(sizes))
            and sizes[-1] > 640,
        ),
        PrerequisiteCheck(
            "fresh pilot/evaluation seeds",
            bool(config.pilot_seeds and config.eval_seeds)
            and not set(config.pilot_seeds).intersection(config.eval_seeds),
        ),
        PrerequisiteCheck(
            "positive fit/evaluation budgets",
            int(config.params.get("pilot_trajectories", 0)) > 0
            and int(config.params.get("repetitions", 0)) > 0,
        ),
        PrerequisiteCheck(
            "exact truth remains evaluation-only",
            True,
            "stable certificate API accepts sampled counts and shift envelopes only",
        ),
    ]
    return checks


def run_dynamics_benchmark(
    *,
    nominal: TransitionRiskModel,
    shifts: Sequence[StructuralShift],
    pilot_seed: int,
    eval_seed: int,
    pilot_trajectories: int,
    repetitions: int,
    sample_sizes: Sequence[int],
    declared_looks: int,
    episode_risk: float,
    confidence_delta: float,
    csa_bet_margin: float,
) -> tuple[list[DynamicsResult], TransitionRiskModel, FitProfile]:
    sizes = tuple(int(value) for value in sample_sizes)
    pilot_rng = np.random.default_rng(pilot_seed)
    pilot_states = nominal.sample_states_with_uniforms(
        pilot_rng.random((pilot_trajectories, nominal.horizon))
    )
    fit_start = perf_counter_ns()
    learned = TransitionRiskModel.fit(
        pilot_states,
        nominal.action_schedule,
        n_actions=nominal.matrices.shape[0],
    )
    fit_ms = (perf_counter_ns() - fit_start) / 1_000_000.0
    profile = FitProfile(
        fit_latency_ms=fit_ms,
        pilot_transitions=pilot_trajectories * nominal.horizon,
        transition_l1_error=float(np.mean(np.abs(learned.matrices - nominal.matrices))),
    )
    model_truth = learned.exact_prefix_risks()  # evaluation only
    methods = tuple(METHOD_VALIDITY)
    rng = np.random.default_rng(eval_seed)
    results: list[DynamicsResult] = []

    for shift in shifts:
        environment = TransitionRiskModel(shift.transition_matrices, nominal.action_schedule)
        true_truth = environment.exact_prefix_risks()  # evaluation only
        counters = {(n, method): _Counts() for n in sizes for method in methods}
        for _ in range(repetitions):
            paired_uniforms = rng.random((sizes[-1], nominal.horizon))
            model_calibration = (
                learned.sample_states_with_uniforms(paired_uniforms)[:, 1:]
                == learned.matrices.shape[1] - 1
            )
            environment_calibration = (
                environment.sample_states_with_uniforms(paired_uniforms)[:, 1:]
                == environment.matrices.shape[1] - 1
            )
            rollout_start = perf_counter_ns()
            model_outcomes = learned.sample_failures(rng, sizes[-1])
            rollout_ns = perf_counter_ns() - rollout_start
            cumulative = np.cumsum(model_outcomes, axis=0)
            for n in sizes:
                look_index = max(1, round(declared_looks * n / sizes[-1]))
                calibration_start = perf_counter_ns()
                envelope = calibrate_shift_envelope(
                    model_calibration[:n],
                    environment_calibration[:n],
                    confidence_delta=confidence_delta / 2.0,
                )
                calibration_ns = perf_counter_ns() - calibration_start
                bad_counts = cumulative[n - 1].astype(int)
                for method in methods:
                    start = perf_counter_ns()
                    selected = certify_prefixes(
                        method,  # type: ignore[arg-type]
                        bad_counts,
                        n=n,
                        look_index=look_index,
                        declared_looks=declared_looks,
                        episode_risk=episode_risk,
                        confidence_delta=confidence_delta,
                        shift_envelope=envelope,
                        csa_bet_margin=csa_bet_margin,
                    )
                    certificate_ns = perf_counter_ns() - start
                    model_risk = 0.0 if selected == 0 else float(model_truth[selected - 1])
                    true_risk = 0.0 if selected == 0 else float(true_truth[selected - 1])
                    count = counters[(n, method)]
                    count.model_false += int(selected > 0 and model_risk > episode_risk)
                    count.true_false += int(selected > 0 and true_risk > episode_risk)
                    count.model_risk += model_risk
                    count.true_risk += true_risk
                    count.selected += selected
                    count.abstained += int(selected == 0)
                    count.certificate_ns += certificate_ns
                    count.calibration_ns += calibration_ns if method.startswith("shift_") else 0
                    count.rollout_ns += rollout_ns

        horizon = nominal.horizon
        for n in sizes:
            input_kb = (horizon * 8 * 2) / 1024.0
            rollout_kb = (n * horizon * np.dtype(np.bool_).itemsize) / 1024.0
            paired_kb = 2.0 * rollout_kb
            for method in methods:
                count = counters[(n, method)]
                results.append(
                    DynamicsResult(
                        condition=shift.name,
                        samples=n,
                        method=method,
                        validity_scope=METHOD_VALIDITY[method],
                        model_false_certification_rate=count.model_false / repetitions,
                        true_false_certification_rate=count.true_false / repetitions,
                        mean_selected_model_risk=count.model_risk / repetitions,
                        mean_selected_true_risk=count.true_risk / repetitions,
                        mean_selected_prefix=count.selected / repetitions,
                        abstention_rate=count.abstained / repetitions,
                        nonvacuity_rate=count.selected / (repetitions * horizon),
                        mean_certificate_latency_us=count.certificate_ns
                        / (repetitions * 1_000.0),
                        mean_shift_calibration_latency_us=count.calibration_ns
                        / (repetitions * 1_000.0),
                        mean_model_rollout_latency_us=count.rollout_ns
                        / (repetitions * 1_000.0),
                        certificate_input_memory_kb=input_kb,
                        model_rollout_buffer_kb=rollout_kb,
                        paired_calibration_buffer_kb=paired_kb,
                        transitions_evaluated=n * horizon,
                    )
                )
    return results, learned, profile


def _upper(rate: float, repetitions: int) -> float:
    failures = round(rate * repetitions)
    if failures >= repetitions:
        return 1.0
    return float(beta.ppf(0.95, failures + 1, repetitions - failures))


def _metric_rows(results: Sequence[DynamicsResult]) -> list[dict[str, Any]]:
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
        "mean_model_rollout_latency_us": "microseconds",
        "certificate_input_memory_kb": "kilobytes",
        "model_rollout_buffer_kb": "kilobytes",
        "paired_calibration_buffer_kb": "kilobytes",
        "transitions_evaluated": "transitions",
    }
    rows: list[dict[str, Any]] = []
    for result in results:
        for metric, unit in units.items():
            rows.append(
                {
                    "stage": "p6.4",
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


def run_p6_4(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    checks = validate_p6_4(config)
    if not all(check.passed for check in checks):
        return VerdictReport(
            question=config.preregistration.primary_hypothesis,
            status=Status.INCONCLUSIVE,
            evidence_level="E0",
            primary_result={"metric": "configuration_valid", "value": 0, "unit": "boolean"},
            controls=checks,
            interpretation="P6.4 did not run because its learned-dynamics protocol is invalid.",
            non_claim=config.preregistration.non_claim,
            decision="Repair instrumentation without examining fresh evaluation units.",
        )

    results, learned, profile = run_dynamics_benchmark(
        nominal=_nominal_model(config),
        shifts=_shifts(config),
        pilot_seed=config.pilot_seeds[0],
        eval_seed=config.eval_seeds[0],
        pilot_trajectories=int(config.params["pilot_trajectories"]),
        repetitions=int(config.params["repetitions"]),
        sample_sizes=tuple(int(value) for value in config.params["sample_sizes"]),
        declared_looks=int(config.params["declared_looks"]),
        episode_risk=float(config.params["episode_risk"]),
        confidence_delta=float(config.params["confidence_delta"]),
        csa_bet_margin=float(config.params["csa_bet_margin"]),
    )
    writer.write_metrics(_metric_rows(results))
    for result in results:
        writer.event(stage="p6.4", **result.__dict__)
    writer.event(
        stage="p6.4-fit",
        learned_transition_matrices=learned.matrices.tolist(),
        **profile.__dict__,
    )

    largest = max(result.samples for result in results)
    repetitions = int(config.params["repetitions"])
    tolerance = float(config.params["false_certification_tolerance"])
    gain_target = float(config.params["gain_actions"])
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
    model_true_gap = max(
        result.true_false_certification_rate - result.model_false_certification_rate
        for result in results
        if result.samples == largest and result.method == "model_only_anytime_cp"
    )
    checks += [
        PrerequisiteCheck(
            "learned-dynamics true validity",
            valid,
            f"worst one-sided 95% upper={max(uppers):.3f}",
        ),
        PrerequisiteCheck(
            "learned-dynamics nonvacuity",
            nonvacuous,
            f"minimum={min(result.nonvacuity_rate for result in primary):.3f}",
        ),
        PrerequisiteCheck(
            "gain over robust union",
            gain >= gain_target,
            f"gain={gain:.3f}; target={gain_target:.3f}",
        ),
        PrerequisiteCheck(
            "structural shift separates model and true validity",
            model_true_gap > 0,
            f"maximum gap={model_true_gap:.3f}",
        ),
        PrerequisiteCheck(
            "latency/memory/calibration curves complete",
            len({result.samples for result in results}) >= 4
            and all(result.paired_calibration_buffer_kb > 0 for result in results),
        ),
    ]

    if not valid:
        status = Status.STOP
        interpretation = "The learned-dynamics robust certificate failed true validity."
        decision = "Do not advance this integration; diagnose structural-shift calibration."
    elif not nonvacuous:
        status = Status.NARROW
        interpretation = "The learned-dynamics certificate was valid but vacuous."
        decision = "Remain NARROW and preserve the validity threshold."
    elif gain >= gain_target:
        status = Status.GO
        interpretation = (
            "The bounded learned-transition integration retained true validity and nonvacuity and "
            "met the locked prefix gain over robust union. This is CPU scale-readiness evidence, "
            "not deployment evidence."
        )
        decision = "Proceed only to a fresh realistic simulator or logged-system integration."
    else:
        status = Status.NARROW
        interpretation = (
            "The learned-transition integration was valid and nonvacuous but did not beat the "
            "robust union by the locked amount."
        )
        decision = "Preserve the integration and remain NARROW on utility."
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
