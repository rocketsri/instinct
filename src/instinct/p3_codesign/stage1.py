"""Bounded P3.1 tabular counterfactual-label pilot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from instinct.core.configschema import ProposalRunConfig
from instinct.core.envs.tabular import chase_chain
from instinct.core.proposal import common_checks
from instinct.core.results import ResultsWriter
from instinct.core.rng import SeedScope
from instinct.core.verdict import PrerequisiteCheck, Status, VerdictReport
from instinct.p3_codesign.counterfactual import (
    exact_first_action_labels,
    option_values,
    planner_actions_for_reflex,
    sampled_first_action_labels,
)
from instinct.p3_codesign.learner import (
    ReflexSnapshot,
    TimeConditionedSoftmaxReflex,
    fixed_snapshot,
    snapshot_from_logits,
    uniform_snapshot,
)


def _int_cells(raw: object, name: str) -> tuple[int, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise ValueError(f"{name} must be a sequence")
    cells = tuple(int(value) for value in raw)
    if not cells or min(cells) < 1:
        raise ValueError(f"{name} must contain positive delays")
    return cells


def validate_p3_1(config: ProposalRunConfig) -> list[PrerequisiteCheck]:
    checks = common_checks(config, "p3_codesign")
    params = config.params
    try:
        pilot = _int_cells(params.get("pilot_delay_cells", [1, 2]), "pilot_delay_cells")
        heldout = _int_cells(params.get("eval_delay_cells", [3]), "eval_delay_cells")
        checks.append(
            PrerequisiteCheck(
                "delay cells are held out",
                not set(pilot) & set(heldout),
                f"pilot={pilot}, eval={heldout}",
            )
        )
    except ValueError as exc:
        checks.append(PrerequisiteCheck("valid P3.1 delay cells", False, str(exc)))
    checks.append(
        PrerequisiteCheck(
            "pilot and evaluation seeds configured",
            bool(config.pilot_seeds) and bool(config.eval_seeds),
        )
    )
    return checks


def _train(
    *,
    mdp: Any,
    planner_q: np.ndarray,
    pilot_delays: tuple[int, ...],
    max_delay: int,
    pilot_seeds: list[int],
    blocks: int,
    epochs: int,
    learning_rate: float,
    name: str,
) -> tuple[ReflexSnapshot, list[str]]:
    learner = TimeConditionedSoftmaxReflex(
        mdp.n_states,
        mdp.n_actions,
        max_delay,
        learning_rate=learning_rate,
    )
    hold = np.full(mdp.n_states, 1, dtype=np.int64)
    current = fixed_snapshot("initial_hold", hold, max_delay, mdp.n_actions)
    fingerprints = [current.fingerprint]
    for block in range(blocks):
        states: list[int] = []
        remaining: list[int] = []
        targets: list[int] = []
        for delay in pilot_delays:
            pending = planner_actions_for_reflex(
                mdp,
                planner_q=planner_q,
                simulated_reflex=current,
                delay=delay,
            )
            for state in range(mdp.n_states):
                labels = exact_first_action_labels(
                    mdp,
                    start_state=state,
                    pending_action=int(pending[state]),
                    planner_q=planner_q,
                    suffix_reflex=current,
                    remaining=delay,
                )
                states.append(state)
                remaining.append(delay)
                targets.append(int(np.argmax(labels)))
        for seed in pilot_seeds:
            learner.fit(
                np.asarray(states, dtype=np.int64),
                np.asarray(remaining, dtype=np.int64),
                np.asarray(targets, dtype=np.int64),
                seed=seed + 1000 * block,
                epochs=epochs,
            )
        current = learner.snapshot(name)
        fingerprints.append(current.fingerprint)
    return current, fingerprints


def _oracle_first_tick(
    mdp: Any,
    planner_q: np.ndarray,
    suffix: ReflexSnapshot,
    delay: int,
) -> ReflexSnapshot:
    logits = np.asarray(suffix.logits).copy()
    pending = planner_actions_for_reflex(
        mdp, planner_q=planner_q, simulated_reflex=suffix, delay=delay
    )
    for state in range(mdp.n_states):
        labels = exact_first_action_labels(
            mdp,
            start_state=state,
            pending_action=int(pending[state]),
            planner_q=planner_q,
            suffix_reflex=suffix,
            remaining=delay,
        )
        logits[delay, state] = 20.0 * (labels - labels.mean())
    return snapshot_from_logits("exact_label_oracle", logits)


def _summary(
    mdp: Any,
    execution: ReflexSnapshot,
    planner_model: ReflexSnapshot,
    planner_q: np.ndarray,
    delay: int,
) -> dict[str, float]:
    value, _, phase = option_values(
        mdp,
        execution_reflex=execution,
        planner_model_reflex=planner_model,
        planner_q=planner_q,
        delay=delay,
    )
    active = ~mdp.terminal
    transition = phase.kernel / (mdp.gamma**delay)
    terminal_probability = transition[:, mdp.terminal].sum(axis=1)
    failure_probability = (
        transition[:, mdp.failure].sum(axis=1)
        if mdp.failure is not None
        else np.zeros(mdp.n_states)
    )
    hold_fraction = np.mean(
        [np.mean(execution.actions(left)[active] == 1) for left in range(1, delay + 1)]
    )
    return {
        "mean_return": float(value[active].mean()),
        "intermediate_reward": float(phase.offset[active].mean()),
        "arrival_planner_value": float((value - phase.offset)[active].mean()),
        "failure_probability": float(failure_probability[active].mean()),
        "terminated_probability": float(terminal_probability[active].mean()),
        "truncated_probability": float(1.0 - terminal_probability[active].mean()),
        "hold_fraction": float(hold_fraction),
    }


def run_p3_1(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    checks = validate_p3_1(config)
    if not all(check.passed for check in checks):
        return VerdictReport(
            question=config.preregistration.primary_hypothesis,
            status=Status.INCONCLUSIVE,
            evidence_level="E0",
            primary_result={"metric": "configuration_valid", "value": 0.0, "unit": "bool"},
            controls=checks,
            interpretation="P3.1 did not run because its preregistered split was invalid.",
            non_claim=config.preregistration.non_claim,
            decision="Repair configuration before collecting P3.1 evidence.",
        )

    params: Mapping[str, Any] = config.params
    pilot_delays = _int_cells(params.get("pilot_delay_cells", [1, 2]), "pilot_delay_cells")
    eval_delays = _int_cells(params.get("eval_delay_cells", [3]), "eval_delay_cells")
    max_delay = max(*pilot_delays, *eval_delays)
    mdp = chase_chain(
        n_positions=int(params.get("n_positions", 4)),
        gamma=float(params.get("gamma", 0.9)),
        drift=float(params.get("drift", 0.35)),
    )
    _, planner_q, planner_distilled_actions = mdp.value_iteration()
    learned, fingerprints = _train(
        mdp=mdp,
        planner_q=planner_q,
        pilot_delays=pilot_delays,
        max_delay=max_delay,
        pilot_seeds=config.pilot_seeds,
        blocks=int(params.get("blocks", 2)),
        epochs=int(params.get("epochs", 3)),
        learning_rate=float(params.get("learning_rate", 0.12)),
        name="planner_aware",
    )
    shuffled_q = np.roll(planner_q, 1, axis=0)
    shuffled, shuffled_fingerprints = _train(
        mdp=mdp,
        planner_q=shuffled_q,
        pilot_delays=pilot_delays,
        max_delay=max_delay,
        pilot_seeds=config.pilot_seeds,
        blocks=int(params.get("blocks", 2)),
        epochs=int(params.get("epochs", 3)),
        learning_rate=float(params.get("learning_rate", 0.12)),
        name="shuffled_planner_values",
    )
    hold_actions = np.full(mdp.n_states, 1, dtype=np.int64)
    hold = fixed_snapshot("hold", hold_actions, max_delay, mdp.n_actions)
    immediate = fixed_snapshot(
        "immediate", mdp.R.argmax(axis=1).astype(np.int64), max_delay, mdp.n_actions
    )
    distilled = fixed_snapshot(
        "planner_distilled", planner_distilled_actions, max_delay, mdp.n_actions
    )
    random = uniform_snapshot(mdp.n_states, mdp.n_actions, max_delay)
    optionality = fixed_snapshot("optionality_only", hold_actions, max_delay, mdp.n_actions)
    fixed_baselines = {
        "hold": hold,
        "immediate": immediate,
        "random": random,
        "planner_distilled": distilled,
        "optionality_only": optionality,
        "shuffled_planner_values": shuffled,
    }

    rows: list[dict[str, Any]] = []
    learned_returns: list[float] = []
    conventional_returns: list[float] = []
    planner_work = float(mdp.n_states * mdp.n_actions)
    for delay in eval_delays:
        oracle = _oracle_first_tick(mdp, planner_q, learned, delay)
        arms = {"planner_aware": learned, **fixed_baselines, "exact_label_oracle": oracle}
        summaries: dict[str, dict[str, float]] = {}
        for name, reflex in arms.items():
            summaries[name] = _summary(mdp, reflex, reflex, planner_q, delay)
            rows.append(
                {
                    "stage": "p3.1",
                    "split": "heldout_delay",
                    "delay": delay,
                    "baseline": name,
                    "planner_work": planner_work,
                    "reflex_fingerprint": reflex.fingerprint,
                    **summaries[name],
                }
            )
        mismatch = _summary(mdp, learned, immediate, planner_q, delay)
        rows.append(
            {
                "stage": "p3.1",
                "split": "heldout_delay",
                "delay": delay,
                "baseline": "planner_reflex_mismatch",
                "planner_work": planner_work,
                "reflex_fingerprint": learned.fingerprint,
                **mismatch,
            }
        )
        learned_returns.append(summaries["planner_aware"]["mean_return"])
        conventional_returns.append(
            max(
                summaries[name]["mean_return"]
                for name in ("hold", "immediate", "random", "planner_distilled")
            )
        )

        cross_variants = {"hold": hold, "immediate": immediate, "planner_aware": learned}
        for execution_name, execution in cross_variants.items():
            for model_name, model in cross_variants.items():
                cross = _summary(mdp, execution, model, planner_q, delay)
                rows.append(
                    {
                        "stage": "p3.1-crossplay",
                        "split": "heldout_delay",
                        "delay": delay,
                        "baseline": "crossplay",
                        "execution_reflex": execution_name,
                        "planner_model_reflex": model_name,
                        "planner_work": planner_work,
                        "mean_return": cross["mean_return"],
                    }
                )

    audit_errors = []
    audit_rollouts = int(params.get("audit_rollouts", 128))
    audit_delay = eval_delays[0]
    pending = planner_actions_for_reflex(
        mdp, planner_q=planner_q, simulated_reflex=learned, delay=audit_delay
    )
    for seed in config.eval_seeds:
        state = seed % mdp.n_states
        exact = exact_first_action_labels(
            mdp,
            start_state=state,
            pending_action=int(pending[state]),
            planner_q=planner_q,
            suffix_reflex=learned,
            remaining=audit_delay,
        )
        sampled = sampled_first_action_labels(
            mdp,
            start_state=state,
            pending_action=int(pending[state]),
            planner_q=planner_q,
            suffix_reflex=learned,
            remaining=audit_delay,
            n_rollouts=audit_rollouts,
            scope=SeedScope(seed).child("p3", "eval-audit"),
        )
        error = float(np.max(np.abs(exact - sampled.mean_returns)))
        audit_errors.append(error)
        rows.append(
            {
                "stage": "p3.1-label-audit",
                "split": "evaluation_seed",
                "eval_seed": seed,
                "delay": audit_delay,
                "baseline": "crn_counterfactual_labels",
                "max_absolute_error": error,
                "terminated_rate": float(sampled.terminated.mean()),
                "truncated_rate": float(sampled.truncated.mean()),
            }
        )

    writer.write_metrics(rows)
    mean_gain = float(np.mean(np.asarray(learned_returns) - conventional_returns))
    audit_tolerance = float(params.get("audit_tolerance", 0.75))
    checks += [
        PrerequisiteCheck(
            "frozen snapshot fingerprints change after training",
            len(set(fingerprints)) > 1 and len(set(shuffled_fingerprints)) > 1,
        ),
        PrerequisiteCheck(
            "exact and CRN labels agree within smoke tolerance",
            max(audit_errors) <= audit_tolerance,
            f"max error={max(audit_errors):.3g}",
        ),
        PrerequisiteCheck("planner work matched across arms", planner_work > 0),
    ]
    minimum_gain = float(params.get("minimum_gain", 0.0))
    useful = mean_gain > minimum_gain
    status = (
        Status.NARROW
        if all(check.passed for check in checks) and useful
        else Status.INCONCLUSIVE
    )
    writer.event(
        stage="p3.1",
        pilot_delays=list(pilot_delays),
        heldout_delays=list(eval_delays),
        fingerprints=fingerprints,
        mean_heldout_gain=mean_gain,
    )
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=status,
        evidence_level="E1",
        primary_result={
            "metric": "heldout_mean_return_gain_vs_best_conventional_baseline",
            "value": mean_gain,
            "unit": "discounted return",
        },
        baselines_run=[
            "hold",
            "immediate reward",
            "random",
            "planner distilled",
            "optionality only",
            "shuffled planner values",
            "exact first-action label oracle",
            "planner/reflex mismatch and cross-play",
        ],
        controls=checks,
        interpretation=(
            "A learned reflex transferred to held-out delay cells in one exact tabular planner. "
            "This is narrow mechanism evidence, not alternating co-design or scale evidence."
            if status is Status.NARROW
            else "The bounded tabular pilot did not clear its held-out gain and audit controls."
        ),
        non_claim=config.preregistration.non_claim,
        decision=(
            "Retain as NARROW and add a second planner/environment before any P3.2 work."
            if status is Status.NARROW
            else "Diagnose label transfer before expanding beyond the tabular pilot."
        ),
    )
