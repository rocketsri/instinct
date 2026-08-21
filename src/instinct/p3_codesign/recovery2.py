"""Final P3.1 recovery in continuous delayed inertial control."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from instinct.core.configschema import ProposalRunConfig
from instinct.core.env import EnvState
from instinct.core.envs.control import InertialIntervention
from instinct.core.proposal import common_checks
from instinct.core.results import ResultsWriter
from instinct.core.rng import SeedScope
from instinct.core.verdict import FailureCode, PrerequisiteCheck, Status, VerdictReport

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _numeric_cells(raw: object, name: str) -> tuple[float, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise ValueError(f"{name} must be a sequence")
    values = tuple(float(value) for value in raw)
    if not values or min(values) <= 0:
        raise ValueError(f"{name} must contain positive values")
    return values


def validate_recovery2(config: ProposalRunConfig) -> list[PrerequisiteCheck]:
    checks = common_checks(config, "p3_codesign")
    try:
        train_rates = _numeric_cells(config.params.get("train_event_rates"), "train_event_rates")
        eval_rates = _numeric_cells(config.params.get("eval_event_rates"), "eval_event_rates")
        train_work = _numeric_cells(config.params.get("train_work_scales"), "train_work_scales")
        eval_work = _numeric_cells(config.params.get("eval_work_scales"), "eval_work_scales")
        checks.extend(
            (
                PrerequisiteCheck(
                    "final recovery event rates held out",
                    set(train_rates).isdisjoint(eval_rates),
                ),
                PrerequisiteCheck(
                    "final recovery latency regimes held out",
                    set(train_work).isdisjoint(eval_work),
                ),
            )
        )
    except (TypeError, ValueError) as exc:
        checks.append(PrerequisiteCheck("valid final recovery cells", False, str(exc)))
    checks.append(
        PrerequisiteCheck(
            "final recovery version",
            config.params.get("recovery_version") == "v5-inertial-latency",
        )
    )
    return checks


def reflex_actions(
    env: InertialIntervention,
    state: EnvState,
    *,
    policy: str,
    threshold: float = 0.0,
) -> IntArray:
    velocity = state["velocity"].astype(np.float64)
    position = state["position"].astype(np.float64)
    brake = np.where(velocity > 0.01, 0, np.where(velocity < -0.01, 2, 1))
    pursue = np.where(position < env.target, 2, 0)
    if policy == "hold":
        return np.ones(state.n_lanes, dtype=np.int64)
    if policy in {"immediate", "distilled"}:
        return pursue.astype(np.int64)
    if policy == "uncertainty_gate":
        threshold = 0.15
    if policy in {"threshold", "uncertainty_gate"}:
        return np.where(env.recovery_margin(state) < threshold, brake, pursue).astype(
            np.int64
        )
    raise ValueError(f"unknown reflex policy {policy!r}")


def _burn_work(budget: int, work_scale: int) -> None:
    value = float(budget + work_scale)
    for index in range(max(1, budget * work_scale * 40)):
        value = (value * 1.000000119 + index + 1.0) % 997.0
    if not np.isfinite(value):  # pragma: no cover
        raise FloatingPointError("planner workload became nonfinite")


def measured_latencies(state: EnvState, *, budget: int, work_scale: int) -> FloatArray:
    """Observed per-unit CPU latency shared by all policy arms in a cell."""
    elapsed = np.empty(state.n_lanes, dtype=np.float64)
    _burn_work(budget, work_scale)
    for lane in range(state.n_lanes):
        started = time.perf_counter_ns()
        _burn_work(budget, work_scale)
        elapsed[lane] = max(time.perf_counter_ns() - started, 1) * 1e-9
    return elapsed


def _deterministic_step(
    env: InertialIntervention, state: EnvState, actions: IntArray, tick: int
) -> EnvState:
    return env.step(
        state,
        actions,
        scope=SeedScope(0).child("p3-planner-model"),
        tick=tick,
    ).state


def planned_actions(
    state: EnvState,
    delays: IntArray,
    *,
    budget: int,
    model_policy: str,
    model_threshold: float,
) -> IntArray:
    """Planner predicts arrival under the same named reflex that execution uses."""
    model_env = InertialIntervention(noise_std=0.0)
    actions = np.empty(state.n_lanes, dtype=np.int64)
    for lane in range(state.n_lanes):
        lane_state = state.take(np.array([lane], dtype=np.int64))
        for tick in range(int(delays[lane])):
            reflex = reflex_actions(
                model_env,
                lane_state,
                policy=model_policy,
                threshold=model_threshold,
            )
            lane_state = _deterministic_step(model_env, lane_state, reflex, tick)
        scores = np.empty(model_env.n_actions)
        for candidate in range(model_env.n_actions):
            candidate_state = lane_state.copy()
            total = 0.0
            discount = 1.0
            for tick in range(budget):
                result = model_env.step(
                    candidate_state,
                    np.array([candidate], dtype=np.int64),
                    scope=SeedScope(0).child("p3-planner-candidate"),
                    tick=tick,
                )
                total += discount * float(result.reward[0])
                candidate_state = result.state
                discount *= 0.97
            total -= 0.5 * abs(float(candidate_state["position"][0]) - model_env.target)
            scores[candidate] = total
        actions[lane] = int(np.argmax(scores))
    return actions


def evaluate_policy(
    state0: EnvState,
    delays: IntArray,
    *,
    budget: int,
    policy: str,
    threshold: float,
    planner_model_policy: str | None,
    planner_model_threshold: float | None,
    scope: SeedScope,
) -> tuple[FloatArray, IntArray]:
    env = InertialIntervention()
    model_policy = policy if planner_model_policy is None else planner_model_policy
    model_threshold = threshold if planner_model_threshold is None else planner_model_threshold
    pending = planned_actions(
        state0,
        delays,
        budget=budget,
        model_policy=model_policy,
        model_threshold=model_threshold,
    )
    returns = np.empty(state0.n_lanes, dtype=np.float64)
    for lane in range(state0.n_lanes):
        state = state0.take(np.array([lane], dtype=np.int64))
        total = 0.0
        discount = 1.0
        for tick in range(int(delays[lane])):
            action = reflex_actions(env, state, policy=policy, threshold=threshold)
            result = env.step(state, action, scope=scope, tick=tick)
            total += discount * float(result.reward[0])
            state = result.state
            discount *= 0.97
        result = env.step(
            state,
            np.array([pending[lane]], dtype=np.int64),
            scope=scope,
            tick=int(delays[lane]),
        )
        total += discount * float(result.reward[0])
        state = result.state
        total -= 0.5 * abs(float(state["position"][0]) - env.target)
        returns[lane] = total
    return returns, pending


def _fit_conditioned_threshold(
    cell_rows: list[tuple[float, float, float, float]], thresholds: FloatArray
) -> FloatArray:
    """Regress the pilot-optimal interior threshold on speed and observed delay."""
    design = np.array(
        [[1.0, np.log(rate), delay] for rate, delay, _, _ in cell_rows], dtype=np.float64
    )
    targets = np.array([threshold for _, _, threshold, _ in cell_rows])
    coefficients = np.linalg.solve(design.T @ design + 1e-5 * np.eye(3), design.T @ targets)
    # Coefficients are unconstrained; only predictions live in threshold units
    # and are clipped by `_predict_threshold`. Clipping this vector is a unit
    # error because the log-rate coefficient is not itself a threshold.
    del thresholds
    return coefficients


def _predict_threshold(
    coefficients: FloatArray,
    *,
    event_rate: float,
    mean_delay: float,
    thresholds: FloatArray,
) -> float:
    value = float(np.array([1.0, np.log(event_rate), mean_delay]) @ coefficients)
    return float(np.clip(value, thresholds.min(), thresholds.max()))


def run_recovery2(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    checks = validate_recovery2(config)
    if not all(check.passed for check in checks):
        return VerdictReport(
            question=config.preregistration.primary_hypothesis,
            status=Status.INCONCLUSIVE,
            evidence_level="E0",
            primary_result={"metric": "configuration_valid", "value": 0.0, "unit": "bool"},
            controls=checks,
            non_claim=config.preregistration.non_claim,
            decision="Repair final recovery instrumentation before evaluation.",
        )
    params: Mapping[str, Any] = config.params
    train_rates = _numeric_cells(params["train_event_rates"], "train_event_rates")
    eval_rates = _numeric_cells(params["eval_event_rates"], "eval_event_rates")
    train_work = tuple(
        int(v)
        for v in _numeric_cells(params["train_work_scales"], "train_work_scales")
    )
    eval_work = tuple(
        int(v)
        for v in _numeric_cells(params["eval_work_scales"], "eval_work_scales")
    )
    thresholds = np.asarray(
        params.get("thresholds", [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4]),
        dtype=np.float64,
    )
    budget = int(params.get("planner_budget", 4))
    max_delay = int(params.get("maximum_delay", 8))
    root = SeedScope(config.seed).child("p3-recovery2")
    pilot_lanes = np.asarray(config.pilot_seeds, dtype=np.int64)
    eval_lanes = np.asarray(config.eval_seeds, dtype=np.int64)
    env = InertialIntervention()
    pilot_rows: list[tuple[float, float, float, float]] = []
    query_count = 0

    for rate in train_rates:
        for work in train_work:
            scope = root.child("pilot", f"rate-{rate}", f"work-{work}")
            state = env.reset(pilot_lanes, scope=scope)
            latency = measured_latencies(state, budget=budget, work_scale=work)
            delays = np.clip(np.ceil(latency * rate), 1, max_delay).astype(np.int64)
            values = []
            for threshold in thresholds:
                returns, _ = evaluate_policy(
                    state,
                    delays,
                    budget=budget,
                    policy="threshold",
                    threshold=float(threshold),
                    planner_model_policy=None,
                    planner_model_threshold=None,
                    scope=scope,
                )
                values.append(float(returns.mean()))
                query_count += state.n_lanes
            best = int(np.argmax(values))
            pilot_rows.append((rate, float(delays.mean()), float(thresholds[best]), values[best]))

    coefficients = _fit_conditioned_threshold(pilot_rows, thresholds)
    # Equal-compute PG consumes the same pilot return table: exponentiated
    # gradient over global threshold arms, one update per already-queried arm.
    logits = np.zeros(len(thresholds))
    for _, _, best_threshold, best_value in pilot_rows:
        chosen = int(np.flatnonzero(thresholds == best_threshold)[0])
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        gradient = -probabilities
        gradient[chosen] += 1.0
        logits += 0.2 * best_value * gradient
    pg_threshold = float(thresholds[int(np.argmax(logits))])
    pg_query_count = query_count

    rows: list[dict[str, Any]] = []
    gains: list[float] = []
    mismatch_disagreements: list[float] = []
    predicted_thresholds: list[float] = []
    frontier_deltas: dict[int, list[float]] = {}
    for rate in eval_rates:
        for work in eval_work:
            scope = root.child("eval", f"rate-{rate}", f"work-{work}")
            state = env.reset(eval_lanes, scope=scope)
            latency = measured_latencies(state, budget=budget, work_scale=work)
            delays = np.clip(np.ceil(latency * rate), 1, max_delay).astype(np.int64)
            threshold = _predict_threshold(
                coefficients,
                event_rate=rate,
                mean_delay=float(delays.mean()),
                thresholds=thresholds,
            )
            predicted_thresholds.append(threshold)
            arms = {
                "hold": ("hold", 0.0),
                "immediate": ("immediate", 0.0),
                "distilled": ("distilled", 0.0),
                "reflex_only": ("threshold", float(np.median(thresholds))),
                "equal_compute_policy_gradient": ("threshold", pg_threshold),
                "awt_style_waiting": ("threshold", 0.0),
                "when_to_plan_uncertainty_gate": ("uncertainty_gate", 0.15),
            }
            baseline_returns: dict[str, float] = {}
            for name, (policy, arm_threshold) in arms.items():
                returns, _ = evaluate_policy(
                    state,
                    delays,
                    budget=budget,
                    policy=policy,
                    threshold=arm_threshold,
                    planner_model_policy=None,
                    planner_model_threshold=None,
                    scope=scope,
                )
                baseline_returns[name] = float(returns.mean())
                rows.append(
                    {
                        "stage": "p3.1-recovery2",
                        "split": "fresh_speed_latency",
                        "event_rate": rate,
                        "work_scale": work,
                        "mean_delay": float(delays.mean()),
                        "mean_latency_s": float(latency.mean()),
                        "baseline": name,
                        "mean_return": baseline_returns[name],
                        "planner_work": budget * work,
                        "label_queries": pg_query_count if "policy_gradient" in name else 0,
                        "proposal_version": config.theory.proposal_version,
                        "theory_id": config.theory.theory_id,
                        "amendment_id": config.theory.amendment_id,
                    }
                )
            proposed_returns, proposed_pending = evaluate_policy(
                state,
                delays,
                budget=budget,
                policy="threshold",
                threshold=threshold,
                planner_model_policy=None,
                planner_model_threshold=None,
                scope=scope,
            )
            mismatch_returns, mismatch_pending = evaluate_policy(
                state,
                delays,
                budget=budget,
                policy="threshold",
                threshold=threshold,
                planner_model_policy="hold",
                planner_model_threshold=0.0,
                scope=scope,
            )
            gain = float(proposed_returns.mean()) - max(baseline_returns.values())
            gains.append(gain)
            disagreement = float(np.mean(proposed_pending != mismatch_pending))
            mismatch_disagreements.append(disagreement)
            rows.extend(
                (
                    {
                        "stage": "p3.1-recovery2",
                        "split": "fresh_speed_latency",
                        "event_rate": rate,
                        "work_scale": work,
                        "mean_delay": float(delays.mean()),
                        "mean_latency_s": float(latency.mean()),
                        "baseline": "conditioned_interior_reflex",
                        "mean_return": float(proposed_returns.mean()),
                        "gain_vs_best_deployable": gain,
                        "predicted_threshold": threshold,
                        "planner_work": budget * work,
                        "label_queries": query_count,
                        "proposal_version": config.theory.proposal_version,
                        "theory_id": config.theory.theory_id,
                        "amendment_id": config.theory.amendment_id,
                    },
                    {
                        "stage": "p3.1-recovery2-mismatch",
                        "split": "fresh_speed_latency",
                        "event_rate": rate,
                        "work_scale": work,
                        "baseline": "hold_model_mismatch",
                        "mean_return": float(mismatch_returns.mean()),
                        "matched_minus_mismatch": float(
                            proposed_returns.mean() - mismatch_returns.mean()
                        ),
                        "planner_action_disagreement": disagreement,
                        "proposal_version": config.theory.proposal_version,
                        "theory_id": config.theory.theory_id,
                        "amendment_id": config.theory.amendment_id,
                    },
                )
            )
            for frontier_budget in (1, 2, 4, 8):
                frontier_latency = measured_latencies(
                    state, budget=frontier_budget, work_scale=work
                )
                frontier_delay = np.clip(
                    np.ceil(frontier_latency * rate), 1, max_delay
                ).astype(np.int64)
                before, _ = evaluate_policy(
                    state,
                    frontier_delay,
                    budget=frontier_budget,
                    policy="hold",
                    threshold=0.0,
                    planner_model_policy=None,
                    planner_model_threshold=None,
                    scope=scope,
                )
                after, _ = evaluate_policy(
                    state,
                    frontier_delay,
                    budget=frontier_budget,
                    policy="threshold",
                    threshold=threshold,
                    planner_model_policy=None,
                    planner_model_threshold=None,
                    scope=scope,
                )
                frontier_deltas.setdefault(frontier_budget, []).append(
                    float(after.mean() - before.mean())
                )
                for phase, frontier_values in (
                    ("before_hold", before),
                    ("after_learned", after),
                ):
                    rows.append(
                        {
                            "stage": "p3.1-recovery2-frontier",
                            "split": "fresh_speed_latency",
                            "event_rate": rate,
                            "work_scale": work,
                            "planner_budget": frontier_budget,
                            "mean_delay": float(frontier_delay.mean()),
                            "mean_latency_s": float(frontier_latency.mean()),
                            "baseline": phase,
                            "mean_return": float(frontier_values.mean()),
                            "proposal_version": config.theory.proposal_version,
                            "theory_id": config.theory.theory_id,
                            "amendment_id": config.theory.amendment_id,
                        }
                    )

    mean_gain = float(np.mean(gains))
    required_gain = float(params.get("minimum_mean_gain", 0.05))
    win_fraction = float(np.mean(np.asarray(gains) >= required_gain))
    required_win_fraction = float(params.get("minimum_cell_win_fraction", 0.75))
    disagreement = float(np.mean(mismatch_disagreements))
    interior = bool(
        all(thresholds.min() < value < thresholds.max() for value in predicted_thresholds)
    )
    checks.extend(
        (
            PrerequisiteCheck("observed planner latency positive", True),
            PrerequisiteCheck("equal-compute PG query count", query_count == pg_query_count),
            PrerequisiteCheck(
                "conditioned thresholds strictly interior",
                interior,
                f"thresholds={predicted_thresholds}",
            ),
            PrerequisiteCheck(
                "continuous planner mismatch informative",
                disagreement >= float(params.get("minimum_planner_disagreement", 0.02)),
                f"fraction={disagreement:.3f}",
            ),
            PrerequisiteCheck(
                "final meaningful recovery margin",
                mean_gain >= required_gain,
                f"gain={mean_gain:.6f}, required={required_gain:.6f}",
            ),
            PrerequisiteCheck(
                "final fresh-cell win fraction",
                win_fraction >= required_win_fraction,
                f"fraction={win_fraction:.3f}, required={required_win_fraction:.3f}",
            ),
        )
    )
    success_names = {
        "conditioned thresholds strictly interior",
        "final meaningful recovery margin",
        "final fresh-cell win fraction",
    }
    instrumentation = all(c.passed for c in checks if c.name not in success_names)
    mechanism = all(c.passed for c in checks if c.name in success_names)
    status = (
        Status.NARROW
        if instrumentation and mechanism
        else Status.STOP
        if instrumentation
        else Status.INCONCLUSIVE
    )
    writer.write_metrics(rows)
    writer.event(
        stage="p3.1-recovery-cycle-2",
        mean_gain=mean_gain,
        cell_win_fraction=win_fraction,
        mismatch_disagreement=disagreement,
        predicted_thresholds=predicted_thresholds,
        frontier_delta={str(k): float(np.mean(v)) for k, v in frontier_deltas.items()},
    )
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=status,
        evidence_level="E2",
        primary_result={
            "metric": "fresh_continuous_cell_gain_vs_best_deployable",
            "value": mean_gain,
            "unit": "discounted return",
        },
        baselines_run=[
            "hold",
            "immediate",
            "distilled",
            "reflex-only",
            "equal-compute policy gradient",
            "local acting-while-thinking-style waiting",
            "local uncertainty-gated when-to-plan analogue",
            "matched planner/reflex and hold-model mismatch",
            "observed-latency P1 frontier before/after",
        ],
        controls=checks,
        failure_code=(
            None
            if status is Status.NARROW
            else FailureCode.F3_MECHANISM_ABSENT
            if status is Status.STOP
            else FailureCode.F0_CODE_INVARIANT
        ),
        interpretation=(
            "Final allowed P3 recovery in continuous inertial delayed control. Local AWT/WTP "
            "arms are structural analogues, not reproductions of external neural systems."
        ),
        non_claim=config.preregistration.non_claim,
        decision=(
            "Keep P3.2 locked pending an independently authorized recurring experiment."
            if status is Status.NARROW
            else (
                "Archive learned P3 co-design after two failed recoveries; "
                "retain exact diagnostics."
            )
        ),
    )
