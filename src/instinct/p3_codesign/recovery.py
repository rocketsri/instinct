"""Fresh multi-cell P3.1 recovery-cycle-1 experiment.

The completed v3 cell is never loaded.  This version conditions action
advantages on both environment speed and remaining planner delay, crosses
three environment/planner cells, and evaluates only speed and delay values
absent from fitting.  The equal-compute policy-gradient baseline consumes the
same exact counterfactual-label examples as the regression optimizer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from instinct.core.configschema import ProposalRunConfig
from instinct.core.envs.tabular import chase_chain, corridor_with_pit
from instinct.core.mdp import TabularMDP
from instinct.core.proposal import common_checks
from instinct.core.results import ResultsWriter
from instinct.core.rng import SeedScope
from instinct.core.verdict import FailureCode, PrerequisiteCheck, Status, VerdictReport
from instinct.p3_codesign.counterfactual import (
    exact_first_action_labels,
    option_values,
    planner_actions_for_reflex,
)
from instinct.p3_codesign.learner import (
    ReflexSnapshot,
    fixed_snapshot,
    snapshot_from_logits,
    uniform_snapshot,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class CellSpec:
    cell_id: str
    family: str
    size: int
    planner_horizon: int


@dataclass(frozen=True, slots=True)
class DelayAdvantageModel:
    name: str
    coefficients: FloatArray
    speed_center: float
    speed_scale: float
    max_delay: int

    def snapshot(self, speed: float) -> ReflexSnapshot:
        _, n_actions, _ = self.coefficients.shape
        logits = np.zeros((self.max_delay + 1, self.coefficients.shape[0], n_actions))
        for remaining in range(1, self.max_delay + 1):
            features = _features(
                speed,
                remaining,
                speed_center=self.speed_center,
                speed_scale=self.speed_scale,
                max_delay=self.max_delay,
            )
            logits[remaining] = np.einsum("saf,f->sa", self.coefficients, features)
        return snapshot_from_logits(self.name, logits)


CELLS = (
    CellSpec("chase3-h2", "chase", 3, 2),
    CellSpec("chase4-h4", "chase", 4, 4),
    CellSpec("corridor5-h3", "corridor", 5, 3),
)


def _float_cells(raw: object, name: str) -> tuple[float, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise ValueError(f"{name} must be a sequence")
    values = tuple(float(value) for value in raw)
    if not values:
        raise ValueError(f"{name} must be nonempty")
    return values


def _int_cells(raw: object, name: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in _float_cells(raw, name))
    if min(values) < 1:
        raise ValueError(f"{name} must contain positive delays")
    return values


def validate_recovery(config: ProposalRunConfig) -> list[PrerequisiteCheck]:
    checks = common_checks(config, "p3_codesign")
    try:
        train_speeds = _float_cells(config.params.get("train_speeds"), "train_speeds")
        eval_speeds = _float_cells(config.params.get("eval_speeds"), "eval_speeds")
        train_delays = _int_cells(config.params.get("train_delays"), "train_delays")
        eval_delays = _int_cells(config.params.get("eval_delays"), "eval_delays")
        checks.extend(
            (
                PrerequisiteCheck(
                    "fresh speeds held out",
                    set(train_speeds).isdisjoint(eval_speeds),
                    f"train={train_speeds}, eval={eval_speeds}",
                ),
                PrerequisiteCheck(
                    "fresh delays held out",
                    set(train_delays).isdisjoint(eval_delays),
                    f"train={train_delays}, eval={eval_delays}",
                ),
            )
        )
    except (TypeError, ValueError) as exc:
        checks.append(PrerequisiteCheck("valid recovery cells", False, str(exc)))
    checks.append(
        PrerequisiteCheck(
            "fresh recovery version",
            config.params.get("recovery_version") == "v4-advantage-regression",
        )
    )
    return checks


def _environment(cell: CellSpec, speed: float, gamma: float) -> TabularMDP:
    if cell.family == "chase":
        return chase_chain(n_positions=cell.size, gamma=gamma, drift=speed)
    if cell.family == "corridor":
        return corridor_with_pit(length=cell.size, gamma=gamma, slip=speed)
    raise ValueError(f"unknown cell family {cell.family!r}")


def finite_planner_q(mdp: TabularMDP, horizon: int) -> FloatArray:
    """Finite-compute planner Q table used as the cell's planner budget."""
    if horizon < 1:
        raise ValueError("planner horizon must be positive")
    value = np.zeros(mdp.n_states, dtype=np.float64)
    q = np.zeros((mdp.n_states, mdp.n_actions), dtype=np.float64)
    for _ in range(horizon):
        q = mdp.R + mdp.gamma * np.einsum("sat,t->sa", mdp.P, value)
        value = np.where(mdp.terminal, value, q.max(axis=1))
    return q


def _features(
    speed: float,
    delay: int,
    *,
    speed_center: float,
    speed_scale: float,
    max_delay: int,
) -> FloatArray:
    speed_value = (speed - speed_center) / speed_scale
    delay_value = delay / max_delay
    return np.array(
        [
            1.0,
            speed_value,
            delay_value,
            speed_value * delay_value,
            speed_value**2,
            delay_value**2,
        ],
        dtype=np.float64,
    )


def optionality_actions(mdp: TabularMDP, planner_q: FloatArray) -> IntArray:
    """Maximize worst reachable planner value, distinct from inactivity."""
    state_value = planner_q.max(axis=1)
    scores = np.full((mdp.n_states, mdp.n_actions), -np.inf)
    for state in range(mdp.n_states):
        for action in range(mdp.n_actions):
            support = mdp.P[state, action] > 0
            scores[state, action] = float(state_value[support].min())
    return scores.argmax(axis=1).astype(np.int64)


def _label_examples(
    cell: CellSpec,
    speeds: tuple[float, ...],
    delays: tuple[int, ...],
    *,
    gamma: float,
    suffix_by_speed: Mapping[float, ReflexSnapshot],
) -> tuple[list[tuple[float, int, int, FloatArray]], int]:
    examples: list[tuple[float, int, int, FloatArray]] = []
    for speed in speeds:
        mdp = _environment(cell, speed, gamma)
        planner_q = finite_planner_q(mdp, cell.planner_horizon)
        suffix = suffix_by_speed[speed]
        for delay in delays:
            pending = planner_actions_for_reflex(
                mdp,
                planner_q=planner_q,
                simulated_reflex=suffix,
                delay=delay,
            )
            for state in np.flatnonzero(~mdp.terminal):
                labels = exact_first_action_labels(
                    mdp,
                    start_state=int(state),
                    pending_action=int(pending[state]),
                    planner_q=planner_q,
                    suffix_reflex=suffix,
                    remaining=delay,
                )
                examples.append((speed, delay, int(state), labels))
    return examples, len(examples)


def fit_advantage_regression(
    cell: CellSpec,
    speeds: tuple[float, ...],
    delays: tuple[int, ...],
    *,
    all_speeds: tuple[float, ...],
    max_delay: int,
    gamma: float,
    blocks: int,
    ridge: float,
) -> tuple[DelayAdvantageModel, int]:
    """Iterated fitted advantages; no evaluation speed or label enters fitting."""
    reference = _environment(cell, speeds[0], gamma)
    speed_center = float(np.mean(all_speeds))
    speed_scale = max(float(np.ptp(np.asarray(all_speeds))), 1e-6)
    hold = fixed_snapshot(
        "initial_hold",
        np.ones(reference.n_states, dtype=np.int64),
        max_delay,
        reference.n_actions,
    )
    suffixes = {speed: hold for speed in speeds}
    query_count = 0
    model: DelayAdvantageModel | None = None
    for _ in range(blocks):
        examples, count = _label_examples(
            cell, speeds, delays, gamma=gamma, suffix_by_speed=suffixes
        )
        query_count += count
        coefficients = np.zeros((reference.n_states, reference.n_actions, 6))
        for state in range(reference.n_states):
            state_examples = [example for example in examples if example[2] == state]
            if not state_examples:
                continue
            design = np.vstack(
                [
                    _features(
                        speed,
                        delay,
                        speed_center=speed_center,
                        speed_scale=speed_scale,
                        max_delay=max_delay,
                    )
                    for speed, delay, _, _ in state_examples
                ]
            )
            targets = np.vstack([labels - labels.mean() for *_, labels in state_examples])
            penalty = ridge * np.eye(design.shape[1])
            coefficients[state] = np.linalg.solve(
                design.T @ design + penalty, design.T @ targets
            ).T
        model = DelayAdvantageModel(
            "delay_advantage_regression",
            coefficients,
            speed_center,
            speed_scale,
            max_delay,
        )
        suffixes = {speed: model.snapshot(speed) for speed in speeds}
    assert model is not None
    return model, query_count


def fit_equal_compute_policy_gradient(
    cell: CellSpec,
    speeds: tuple[float, ...],
    delays: tuple[int, ...],
    *,
    all_speeds: tuple[float, ...],
    max_delay: int,
    gamma: float,
    blocks: int,
    learning_rate: float,
) -> tuple[DelayAdvantageModel, int]:
    """Exact expected policy gradient with one update per queried label vector."""
    reference = _environment(cell, speeds[0], gamma)
    speed_center = float(np.mean(all_speeds))
    speed_scale = max(float(np.ptp(np.asarray(all_speeds))), 1e-6)
    coefficients = np.zeros((reference.n_states, reference.n_actions, 6))
    model = DelayAdvantageModel(
        "equal_compute_policy_gradient",
        coefficients,
        speed_center,
        speed_scale,
        max_delay,
    )
    query_count = 0
    for block in range(blocks):
        suffixes = {speed: model.snapshot(speed) for speed in speeds}
        examples, count = _label_examples(
            cell, speeds, delays, gamma=gamma, suffix_by_speed=suffixes
        )
        query_count += count
        order_keys = SeedScope(3300 + block).stream("p3-pg-order").uniform(
            np.arange(len(examples), dtype=np.int64), count=1
        )[:, 0]
        for index in np.argsort(order_keys):
            speed, delay, state, labels = examples[int(index)]
            features = _features(
                speed,
                delay,
                speed_center=speed_center,
                speed_scale=speed_scale,
                max_delay=max_delay,
            )
            logits = coefficients[state] @ features
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            advantage = labels - float(probabilities @ labels)
            coefficients[state] += learning_rate * np.outer(
                probabilities * advantage, features
            )
        model = DelayAdvantageModel(
            "equal_compute_policy_gradient",
            coefficients.copy(),
            speed_center,
            speed_scale,
            max_delay,
        )
    return model, query_count


def _summary(
    mdp: TabularMDP,
    execution: ReflexSnapshot,
    planner_model: ReflexSnapshot,
    planner_q: FloatArray,
    delay: int,
) -> tuple[float, IntArray]:
    value, actions, _ = option_values(
        mdp,
        execution_reflex=execution,
        planner_model_reflex=planner_model,
        planner_q=planner_q,
        delay=delay,
    )
    return float(value[~mdp.terminal].mean()), actions


def run_recovery(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    checks = validate_recovery(config)
    if not all(check.passed for check in checks):
        return VerdictReport(
            question=config.preregistration.primary_hypothesis,
            status=Status.INCONCLUSIVE,
            evidence_level="E0",
            primary_result={"metric": "configuration_valid", "value": 0.0, "unit": "bool"},
            controls=checks,
            interpretation="The fresh P3 recovery registry was invalid and no labels were used.",
            non_claim=config.preregistration.non_claim,
            decision="Repair configuration before recovery evaluation.",
        )
    params: Mapping[str, Any] = config.params
    train_speeds = _float_cells(params["train_speeds"], "train_speeds")
    eval_speeds = _float_cells(params["eval_speeds"], "eval_speeds")
    train_delays = _int_cells(params["train_delays"], "train_delays")
    eval_delays = _int_cells(params["eval_delays"], "eval_delays")
    all_speeds = (*train_speeds, *eval_speeds)
    max_delay = max(*train_delays, *eval_delays)
    gamma = float(params.get("gamma", 0.9))
    blocks = int(params.get("blocks", 2))
    rows: list[dict[str, Any]] = []
    cell_gains: dict[str, list[float]] = {cell.cell_id: [] for cell in CELLS}
    disagreements: list[float] = []
    optionality_differences: list[float] = []
    immediate_distilled_agreements: list[float] = []
    proposed_immediate_agreements: list[float] = []
    query_matches: list[bool] = []
    frontier_budgets = _int_cells(params.get("frontier_budgets", [1, 2, 4, 8]), "frontier_budgets")

    for cell in CELLS:
        proposed, proposed_queries = fit_advantage_regression(
            cell,
            train_speeds,
            train_delays,
            all_speeds=all_speeds,
            max_delay=max_delay,
            gamma=gamma,
            blocks=blocks,
            ridge=float(params.get("ridge", 1e-4)),
        )
        policy_gradient, pg_queries = fit_equal_compute_policy_gradient(
            cell,
            train_speeds,
            train_delays,
            all_speeds=all_speeds,
            max_delay=max_delay,
            gamma=gamma,
            blocks=blocks,
            learning_rate=float(params.get("policy_gradient_learning_rate", 0.12)),
        )
        query_matches.append(proposed_queries == pg_queries)
        for speed in eval_speeds:
            mdp = _environment(cell, speed, gamma)
            planner_q = finite_planner_q(mdp, cell.planner_horizon)
            proposed_snapshot = proposed.snapshot(speed)
            pg_snapshot = policy_gradient.snapshot(speed)
            hold_actions = np.ones(mdp.n_states, dtype=np.int64)
            hold = fixed_snapshot("hold", hold_actions, max_delay, mdp.n_actions)
            immediate = fixed_snapshot(
                "immediate", mdp.R.argmax(axis=1).astype(np.int64), max_delay, mdp.n_actions
            )
            distilled = fixed_snapshot(
                "planner_distilled",
                planner_q.argmax(axis=1).astype(np.int64),
                max_delay,
                mdp.n_actions,
            )
            optionality_actions_cell = optionality_actions(mdp, planner_q)
            optionality = fixed_snapshot(
                "optionality_only", optionality_actions_cell, max_delay, mdp.n_actions
            )
            random = uniform_snapshot(mdp.n_states, mdp.n_actions, max_delay)
            optionality_differences.append(
                float(np.mean(optionality_actions_cell[~mdp.terminal] != 1))
            )
            immediate_actions = immediate.actions(1)
            distilled_actions = distilled.actions(1)
            immediate_distilled_agreements.append(
                float(
                    np.mean(
                        immediate_actions[~mdp.terminal]
                        == distilled_actions[~mdp.terminal]
                    )
                )
            )
            deployable = {
                "hold": hold,
                "immediate": immediate,
                "planner_distilled": distilled,
                "optionality_only": optionality,
                "random": random,
                "equal_compute_policy_gradient": pg_snapshot,
            }
            for delay in eval_delays:
                proposed_return, proposed_actions = _summary(
                    mdp, proposed_snapshot, proposed_snapshot, planner_q, delay
                )
                proposed_immediate_agreements.append(
                    float(
                        np.mean(
                            proposed_snapshot.actions(delay)[~mdp.terminal]
                            == immediate_actions[~mdp.terminal]
                        )
                    )
                )
                baseline_returns: dict[str, float] = {}
                for name, snapshot in deployable.items():
                    baseline_return, _ = _summary(mdp, snapshot, snapshot, planner_q, delay)
                    baseline_returns[name] = baseline_return
                    rows.append(
                        {
                            "stage": "p3.1-recovery",
                            "split": "fresh_speed_delay",
                            "cell_id": cell.cell_id,
                            "family": cell.family,
                            "speed": speed,
                            "delay": delay,
                            "baseline": name,
                            "mean_return": baseline_return,
                            "planner_work": mdp.n_states * mdp.n_actions * cell.planner_horizon,
                            "label_queries": pg_queries if "policy_gradient" in name else 0,
                            "proposal_version": config.theory.proposal_version,
                            "theory_id": config.theory.theory_id,
                            "amendment_id": config.theory.amendment_id,
                        }
                    )
                best_baseline = max(baseline_returns.values())
                gain = proposed_return - best_baseline
                cell_gains[cell.cell_id].append(gain)
                rows.append(
                    {
                        "stage": "p3.1-recovery",
                        "split": "fresh_speed_delay",
                        "cell_id": cell.cell_id,
                        "family": cell.family,
                        "speed": speed,
                        "delay": delay,
                        "baseline": "delay_advantage_regression",
                        "mean_return": proposed_return,
                        "gain_vs_best_deployable": gain,
                        "action_agreement_with_immediate": proposed_immediate_agreements[-1],
                        "immediate_distilled_action_agreement": immediate_distilled_agreements[-1],
                        "planner_work": mdp.n_states * mdp.n_actions * cell.planner_horizon,
                        "label_queries": proposed_queries,
                        "proposal_version": config.theory.proposal_version,
                        "theory_id": config.theory.theory_id,
                        "amendment_id": config.theory.amendment_id,
                    }
                )
                # Diagnostic-only F0 repair after v4's first artifact: a
                # training-cell audit selected hold because immediate induced
                # zero planner disagreement. This does not alter any deployed
                # arm, fit, label, threshold, or primary return.
                mismatch_return, mismatch_actions = _summary(
                    mdp, proposed_snapshot, hold, planner_q, delay
                )
                disagreement = float(
                    np.mean(proposed_actions[~mdp.terminal] != mismatch_actions[~mdp.terminal])
                )
                disagreements.append(disagreement)
                rows.append(
                    {
                        "stage": "p3.1-recovery-crossplay",
                        "split": "fresh_speed_delay",
                        "cell_id": cell.cell_id,
                        "family": cell.family,
                        "speed": speed,
                        "delay": delay,
                        "baseline": "planner_reflex_mismatch_hold_model",
                        "mean_return": mismatch_return,
                        "matched_minus_mismatch": proposed_return - mismatch_return,
                        "planner_action_disagreement": disagreement,
                        "planner_work": mdp.n_states * mdp.n_actions * cell.planner_horizon,
                        "proposal_version": config.theory.proposal_version,
                        "theory_id": config.theory.theory_id,
                        "amendment_id": config.theory.amendment_id,
                    }
                )
                for budget in frontier_budgets:
                    frontier_q = finite_planner_q(mdp, budget)
                    frontier_phases = (
                        ("before_hold", hold),
                        ("after_learned", proposed_snapshot),
                    )
                    for phase_name, snapshot in frontier_phases:
                        frontier_return, _ = _summary(
                            mdp, snapshot, snapshot, frontier_q, delay
                        )
                        rows.append(
                            {
                                "stage": "p3.1-recovery-frontier",
                                "split": "fresh_speed_delay",
                                "cell_id": cell.cell_id,
                                "family": cell.family,
                                "speed": speed,
                                "delay": delay,
                                "baseline": phase_name,
                                "planner_budget": budget,
                                "mean_return": frontier_return,
                                "planner_work": mdp.n_states * mdp.n_actions * budget,
                                "proposal_version": config.theory.proposal_version,
                                "theory_id": config.theory.theory_id,
                                "amendment_id": config.theory.amendment_id,
                            }
                        )

    mean_cell_gains = {cell: float(np.mean(gains)) for cell, gains in cell_gains.items()}
    mean_gain = float(np.mean(list(mean_cell_gains.values())))
    minimum_margin = float(params.get("minimum_mean_gain", 0.05))
    cell_win_fraction = float(
        np.mean(np.asarray(list(mean_cell_gains.values())) >= minimum_margin)
    )
    required_cell_fraction = float(params.get("minimum_cell_win_fraction", 2 / 3))
    disagreement_fraction = float(np.mean(disagreements))
    minimum_disagreement = float(params.get("minimum_planner_disagreement", 0.02))
    optionality_difference = float(np.mean(optionality_differences))
    checks.extend(
        (
            PrerequisiteCheck("multiple environment/planner cells", len(CELLS) >= 3),
            PrerequisiteCheck(
                "equal-compute policy-gradient labels", all(query_matches)
            ),
            PrerequisiteCheck(
                "planner mismatch changes pending actions",
                disagreement_fraction >= minimum_disagreement,
                f"fraction={disagreement_fraction:.3f}",
            ),
            PrerequisiteCheck(
                "optionality distinct from hold",
                optionality_difference > 0,
                f"different_fraction={optionality_difference:.3f}",
            ),
            PrerequisiteCheck(
                "meaningful mean recovery margin",
                mean_gain >= minimum_margin,
                f"gain={mean_gain:.6f}, required={minimum_margin:.6f}",
            ),
            PrerequisiteCheck(
                "fresh cell win fraction",
                cell_win_fraction >= required_cell_fraction,
                f"fraction={cell_win_fraction:.3f}, required={required_cell_fraction:.3f}",
            ),
        )
    )
    writer.write_metrics(rows)
    writer.event(
        stage="p3.1-recovery-cycle-1",
        mean_gain=mean_gain,
        cell_gains=mean_cell_gains,
        cell_win_fraction=cell_win_fraction,
        planner_disagreement=disagreement_fraction,
        optionality_difference=optionality_difference,
        immediate_distilled_action_agreement=float(
            np.mean(immediate_distilled_agreements)
        ),
        proposed_immediate_action_agreement=float(
            np.mean(proposed_immediate_agreements)
        ),
    )
    success_names = {"meaningful mean recovery margin", "fresh cell win fraction"}
    instrumentation_passed = all(
        check.passed for check in checks if check.name not in success_names
    )
    mechanism_passed = all(
        check.passed for check in checks if check.name in success_names
    )
    status = (
        Status.NARROW
        if instrumentation_passed and mechanism_passed
        else Status.STOP
        if instrumentation_passed
        else Status.INCONCLUSIVE
    )
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=status,
        evidence_level="E1",
        primary_result={
            "metric": "fresh_cell_mean_gain_vs_best_deployable",
            "value": mean_gain,
            "unit": "discounted return",
        },
        baselines_run=[
            "hold",
            "immediate reward",
            "planner distilled",
            "optionality-only maximin recovery",
            "random",
            "equal-compute policy gradient",
            "planner/reflex mismatch cross-play",
            "finite-compute frontier before/after",
        ],
        controls=checks,
        failure_code=(
            None
            if status is Status.NARROW
            else FailureCode.F4_BASELINE_DOMINANCE
            if status is Status.STOP
            else FailureCode.F0_CODE_INVARIANT
        ),
        interpretation=(
            "Fresh multi-cell recovery evidence only. P3.2 remains locked regardless of this "
            "bounded tabular verdict until a separately preregistered recurring co-design run."
        ),
        non_claim=config.preregistration.non_claim,
        decision=(
            "Retain as useful NARROW evidence; keep P3.2 locked pending an independent run."
            if status is Status.NARROW
            else "Preserve recovery-cycle-1 failure and keep P3.2 locked."
        ),
    )
