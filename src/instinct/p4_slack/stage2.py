"""Bounded P4.2 open-loop chunk-policy mechanism study."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import numpy.typing as npt

from instinct.core.configschema import ProposalRunConfig
from instinct.core.proposal import common_checks
from instinct.core.results import ResultsWriter
from instinct.core.verdict import PrerequisiteCheck, Status, VerdictReport
from instinct.p4_slack.inertial import (
    InertialGrid,
    braking_policy,
    greedy_progress_policy,
    inertial_grid,
    one_step_failure_floor,
)
from instinct.p4_slack.inertial_surrogate import (
    InertialRecoverySurrogate,
    fit_inertial_surrogate,
    inertial_recovery_dataset,
    inertial_state_frame,
)

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ChunkPolicy:
    name: str
    actions: IntArray
    chunk_length: int
    beta: float


def locked_inertial_surrogate(*, v_fail: float) -> InertialRecoverySurrogate:
    """Rebuild the frozen v2 estimator without loading either held-out layout."""

    slips = (0.05, 0.15, 0.25)
    horizons = (8, 12, 16)
    train_layouts = {
        "train-e": ((1, 1), (2, 2)),
        "train-f": ((2, 1), (1, 2)),
    }
    train = inertial_recovery_dataset(
        [
            (name, obstacles, slip, horizon)
            for name, obstacles in train_layouts.items()
            for slip in slips
            for horizon in horizons
        ],
        v_fail=v_fail,
    )
    calibration = inertial_recovery_dataset(
        [
            ("calibration-new", ((1, 0), (2, 2)), slip, horizon)
            for slip in slips
            for horizon in horizons
        ],
        v_fail=v_fail,
    )
    return fit_inertial_surrogate(
        train,
        calibration,
        v_fail=v_fail,
        alpha=0.1,
        state_coverage=0.9,
        feature_version="v2-layout-geometry",
    )


def _transition_by_action(
    grid: InertialGrid, impulses: tuple[tuple[int, int], ...]
) -> FloatArray:
    disturbance = grid.impulse_kernel(impulses).mean(axis=0)
    return np.stack(
        [grid.mdp.P[:, action, :] @ disturbance for action in range(grid.mdp.n_actions)]
    )


def train_chunk_policy(
    grid: InertialGrid,
    surrogate: InertialRecoverySurrogate,
    *,
    layout: str,
    obstacles: tuple[tuple[int, int], ...],
    slip: float,
    value_horizon: int,
    impulses: tuple[tuple[int, int], ...],
    chunk_length: int,
    beta: float,
    margin: float,
    v_fail: float,
    progress_weight: float,
    name: str,
    use_lower_confidence: bool = True,
) -> ChunkPolicy:
    """Enumerate open-loop chunks using task value plus predicted prefix slack."""

    if chunk_length < 1 or beta < 0 or margin <= 0 or progress_weight <= 0:
        raise ValueError("invalid chunk-training parameter")
    frame = inertial_state_frame(
        layout=layout,
        obstacles=obstacles,
        slip=slip,
        horizon=value_horizon,
    )
    prediction = surrogate.predict(frame)
    lower = (
        prediction - surrogate.residual_radius
        if use_lower_confidence
        else prediction
    )
    terminal = frame["goal"].to_numpy(dtype=bool) | frame["failure"].to_numpy(
        dtype=bool
    )
    lower = np.where(terminal, prediction, lower)
    penalty = np.maximum(margin - (lower - v_fail), 0.0)
    transition = _transition_by_action(grid, impulses)
    progress = grid.progress()
    rewards = grid.mdp.R.T
    sequences = np.asarray(
        list(product(range(grid.mdp.n_actions), repeat=chunk_length)),
        dtype=np.int64,
    )
    best_score = np.full(grid.mdp.n_states, -np.inf)
    best_sequence = np.zeros((grid.mdp.n_states, chunk_length), dtype=np.int64)
    for sequence in sequences:
        task = np.zeros(grid.mdp.n_states, dtype=np.float64)
        slack_cost = np.zeros(grid.mdp.n_states, dtype=np.float64)
        state_penalty = penalty.copy()
        task += progress_weight * progress
        for prefix in range(chunk_length - 1, -1, -1):
            action = int(sequence[prefix])
            task = rewards[action] + grid.mdp.gamma * (transition[action] @ task)
            slack_cost = transition[action] @ (state_penalty + slack_cost)
        score = task - beta * slack_cost
        improved = score > best_score
        best_score = np.where(improved, score, best_score)
        best_sequence[improved] = sequence
    return ChunkPolicy(name, best_sequence, chunk_length, beta)


def evaluate_chunk_policy(
    grid: InertialGrid,
    policy: ChunkPolicy,
    *,
    impulses: tuple[tuple[int, int], ...],
    total_steps: int,
) -> dict[str, float]:
    """Exact rollout while each open-loop chunk remains keyed to its launch state."""

    if total_steps < 1 or total_steps % policy.chunk_length:
        raise ValueError("total_steps must be a positive multiple of chunk length")
    transition = _transition_by_action(grid, impulses)
    current = np.zeros(grid.mdp.n_states, dtype=np.float64)
    current[grid.state_index[(0, 0, 0, 0)]] = 1.0
    total_return = 0.0
    discount = 1.0
    calls = total_steps // policy.chunk_length
    for _ in range(calls):
        joint = np.diag(current)
        for prefix in range(policy.chunk_length):
            next_joint = np.zeros_like(joint)
            for action in range(grid.mdp.n_actions):
                rows = np.flatnonzero(policy.actions[:, prefix] == action)
                if rows.size:
                    total_return += discount * float(
                        np.sum(joint[rows] @ grid.mdp.R[:, action])
                    )
                    next_joint[rows] = joint[rows] @ transition[action]
            joint = next_joint
            discount *= grid.mdp.gamma
        current = joint.sum(axis=0)
    return {
        "return": total_return,
        "goal_probability": float(current[grid.goal_state]),
        "failure_probability": float(current[grid.failure_state]),
        "survival_probability": float(1.0 - current[grid.failure_state]),
        "progress": float(current @ grid.progress()),
        "replan_calls": float(calls),
        "mean_chunk_length": float(policy.chunk_length),
        "hold_fraction": float(np.mean(policy.actions == 0)),
    }


def evaluate_adaptive_verifier(
    grid: InertialGrid,
    base_policy: ChunkPolicy,
    verifier_policy: ChunkPolicy,
    *,
    unsafe: npt.NDArray[np.bool_],
    impulses: tuple[tuple[int, int], ...],
    total_steps: int,
) -> dict[str, float]:
    """Execute a long base chunk with state-triggered one-step recovery checks."""

    if verifier_policy.chunk_length != 1 or total_steps % base_policy.chunk_length:
        raise ValueError("adaptive verifier requires a one-step verifier and whole base chunks")
    gate = np.asarray(unsafe, dtype=bool)
    if gate.shape != (grid.mdp.n_states,):
        raise ValueError("unsafe gate must provide one decision per state")
    transition = _transition_by_action(grid, impulses)
    current = np.zeros(grid.mdp.n_states, dtype=np.float64)
    current[grid.state_index[(0, 0, 0, 0)]] = 1.0
    total_return = 0.0
    verifier_calls = 0.0
    discount = 1.0
    base_calls = total_steps // base_policy.chunk_length
    verifier_actions = verifier_policy.actions[:, 0]
    for _ in range(base_calls):
        joint = np.diag(current)
        for prefix in range(base_policy.chunk_length):
            verifier_calls += float(joint[:, gate].sum())
            next_joint = np.zeros_like(joint)
            base_actions = base_policy.actions[:, prefix]
            for action in range(grid.mdp.n_actions):
                selected = np.where(
                    gate[None, :],
                    verifier_actions[None, :] == action,
                    base_actions[:, None] == action,
                )
                selected_joint = joint * selected
                state_mass = selected_joint.sum(axis=0)
                total_return += discount * float(state_mass @ grid.mdp.R[:, action])
                next_joint += selected_joint @ transition[action]
            joint = next_joint
            discount *= grid.mdp.gamma
        current = joint.sum(axis=0)
    return {
        "return": total_return,
        "goal_probability": float(current[grid.goal_state]),
        "failure_probability": float(current[grid.failure_state]),
        "survival_probability": float(1.0 - current[grid.failure_state]),
        "progress": float(current @ grid.progress()),
        "replan_calls": float(base_calls + verifier_calls),
        "base_replan_calls": float(base_calls),
        "verifier_calls": verifier_calls,
        "mean_chunk_length": float(base_policy.chunk_length),
        "hold_fraction": float("nan"),
    }


def evaluate_budgeted_verifier(
    grid: InertialGrid,
    base_policy: ChunkPolicy,
    verifier_policy: ChunkPolicy,
    *,
    unsafe: npt.NDArray[np.bool_],
    impulses: tuple[tuple[int, int], ...],
    total_steps: int,
    verifier_budget: int,
) -> dict[str, float]:
    """Use exactly ``verifier_budget`` state-dependent checks per episode.

    The controller spends a check when the frozen recovery gate fires. A
    deadline rule spends remaining checks in the final available slots, so
    compute is matched pathwise rather than estimated from a pilot layout.
    """

    if verifier_policy.chunk_length != 1 or total_steps % base_policy.chunk_length:
        raise ValueError("budgeted verifier requires a one-step verifier and whole base chunks")
    if not 0 <= verifier_budget <= total_steps:
        raise ValueError("verifier budget must lie between zero and total steps")
    gate = np.asarray(unsafe, dtype=bool)
    if gate.shape != (grid.mdp.n_states,):
        raise ValueError("unsafe gate must provide one decision per state")
    transition = _transition_by_action(grid, impulses)
    # Axes: checks used, base-plan launch state, current state.
    current = np.zeros(
        (verifier_budget + 1, grid.mdp.n_states, grid.mdp.n_states),
        dtype=np.float64,
    )
    start = grid.state_index[(0, 0, 0, 0)]
    current[0, start, start] = 1.0
    total_return = 0.0
    discount = 1.0
    step = 0
    base_calls = total_steps // base_policy.chunk_length
    verifier_actions = verifier_policy.actions[:, 0]
    for _ in range(base_calls):
        # Replanning resets the row coordinate to the current launch state.
        launch = np.zeros_like(current)
        for used in range(verifier_budget + 1):
            distribution = current[used].sum(axis=0)
            launch[used] = np.diag(distribution)
        current = launch
        for prefix in range(base_policy.chunk_length):
            next_joint = np.zeros_like(current)
            remaining_slots = total_steps - step
            for used in range(verifier_budget + 1):
                remaining_budget = verifier_budget - used
                force = remaining_slots <= remaining_budget
                call = (gate | force) if remaining_budget else np.zeros_like(gate)
                base_actions = base_policy.actions[:, prefix]
                for called in (False, True):
                    destination_used = used + int(called)
                    if destination_used > verifier_budget:
                        continue
                    call_mask = call if called else ~call
                    for action in range(grid.mdp.n_actions):
                        action_mask = (
                            verifier_actions[None, :] == action
                            if called
                            else base_actions[:, None] == action
                        )
                        selected = call_mask[None, :] & action_mask
                        selected_joint = current[used] * selected
                        state_mass = selected_joint.sum(axis=0)
                        total_return += discount * float(state_mass @ grid.mdp.R[:, action])
                        next_joint[destination_used] += selected_joint @ transition[action]
            current = next_joint
            discount *= grid.mdp.gamma
            step += 1
    distribution = current.sum(axis=(0, 1))
    used_probability = current.sum(axis=(1, 2))
    expected_checks = float(used_probability @ np.arange(verifier_budget + 1))
    return {
        "return": total_return,
        "goal_probability": float(distribution[grid.goal_state]),
        "failure_probability": float(distribution[grid.failure_state]),
        "survival_probability": float(1.0 - distribution[grid.failure_state]),
        "progress": float(distribution @ grid.progress()),
        "replan_calls": float(base_calls + expected_checks),
        "base_replan_calls": float(base_calls),
        "verifier_calls": expected_checks,
        "mean_chunk_length": float(base_policy.chunk_length),
        "hold_fraction": float("nan"),
    }


def surrogate_slack(
    grid: InertialGrid,
    surrogate: InertialRecoverySurrogate,
    *,
    layout: str,
    obstacles: tuple[tuple[int, int], ...],
    slip: float,
    horizon: int,
    v_fail: float,
) -> FloatArray:
    frame = inertial_state_frame(
        layout=layout,
        obstacles=obstacles,
        slip=slip,
        horizon=horizon,
    )
    prediction = surrogate.predict(frame) - v_fail
    prediction[grid.mdp.terminal] = np.inf
    return prediction


def _pace_policy(grid: InertialGrid) -> ChunkPolicy:
    braking = braking_policy(grid)
    greedy = greedy_progress_policy(grid)
    boundary = one_step_failure_floor(grid) > 0.0
    actions = np.where(boundary, braking, greedy)[:, None]
    return ChunkPolicy("pace_boundary", actions, 1, 0.0)


def _run_p4_2_recovery2(
    config: ProposalRunConfig, writer: ResultsWriter
) -> VerdictReport:
    """Final recovery: test whether matched-call adaptive verification is the mechanism."""

    checks = common_checks(config, "p4_slack")
    params = config.params
    v_fail = float(params.get("v_fail", -1.0))
    margin = float(params.get("margin", 1.0))
    beta = float(params.get("beta", 0.1))
    total_steps = int(params.get("total_steps", 12))
    target_calls = float(params.get("target_calls", 4.0))
    call_tolerance = float(params.get("call_tolerance", 0.25))
    minimum_gain = float(params.get("minimum_success_gain", 0.01))
    policy_version = str(params.get("policy_version", "r2-adaptive-verifier"))
    budgeted_repair = policy_version == "r2-f2-budgeted-verifier"
    surrogate = locked_inertial_surrogate(v_fail=v_fail)
    training_impulses = ((0, 0),) * 8 + ((-1, 0), (1, 0), (0, -1), (0, 1))
    pilot_impulses = ((0, 0),) * 8 + ((-1, -1), (1, -1), (-1, 1), (1, 1))
    heldout_impulses = ((0, 0),) * 12 + ((-2, 0), (2, 0), (0, -2), (0, 2))

    def build(
        grid: InertialGrid,
        obstacles: tuple[tuple[int, int], ...],
        layout: str,
        name: str,
        length: int,
        method_beta: float,
    ) -> ChunkPolicy:
        return train_chunk_policy(
            grid,
            surrogate,
            layout=layout,
            obstacles=obstacles,
            slip=0.1,
            value_horizon=12,
            impulses=training_impulses,
            chunk_length=length,
            beta=method_beta,
            margin=margin,
            v_fail=v_fail,
            progress_weight=2.0,
            name=name,
            use_lower_confidence=False,
        )

    pilot_obstacles = ((0, 2), (2, 2))
    pilot_grid = inertial_grid(obstacles=pilot_obstacles, actuator_slip=0.1)
    pilot_base = build(pilot_grid, pilot_obstacles, "pilot", "long_task", 6, 0.0)
    pilot_verifier = build(pilot_grid, pilot_obstacles, "pilot", "verifier", 1, beta)
    pilot_slack = surrogate_slack(
        pilot_grid,
        surrogate,
        layout="pilot",
        obstacles=pilot_obstacles,
        slip=0.1,
        horizon=12,
        v_fail=v_fail,
    )
    finite_pilot = pilot_slack[np.isfinite(pilot_slack)]
    candidates = np.quantile(finite_pilot, np.linspace(0.05, 0.95, 19))
    threshold_results: list[tuple[float, dict[str, float]]] = []
    for threshold in candidates:
        result = evaluate_adaptive_verifier(
            pilot_grid,
            pilot_base,
            pilot_verifier,
            unsafe=pilot_slack < threshold,
            impulses=pilot_impulses,
            total_steps=total_steps,
        )
        threshold_results.append((float(threshold), result))
    threshold, pilot_choice = min(
        threshold_results,
        key=lambda item: (abs(item[1]["replan_calls"] - target_calls), item[1]["replan_calls"]),
    )
    if budgeted_repair:
        threshold = float(params["gate_threshold"])

    obstacles = ((0, 1), (2, 1)) if budgeted_repair else ((1, 2), (2, 0))
    grid = inertial_grid(obstacles=obstacles, actuator_slip=0.1)
    layout = "fresh-r2-f2" if budgeted_repair else "fresh-r2"
    task = build(grid, obstacles, layout, "task_only", 3, 0.0)
    slack_policy = build(grid, obstacles, layout, "slack_regularized", 3, beta)
    short = build(grid, obstacles, layout, "shorter_chunk", 2, 0.0)
    base = build(grid, obstacles, layout, "long_task", 6, 0.0)
    verifier = build(grid, obstacles, layout, "verifier", 1, beta)
    fresh_slack = surrogate_slack(
        grid,
        surrogate,
        layout=layout,
        obstacles=obstacles,
        slip=0.1,
        horizon=12,
        v_fail=v_fail,
    )
    adaptive = (
        evaluate_budgeted_verifier(
            grid,
            base,
            verifier,
            unsafe=fresh_slack < threshold,
            impulses=heldout_impulses,
            total_steps=total_steps,
            verifier_budget=round(target_calls - total_steps / base.chunk_length),
        )
        if budgeted_repair
        else evaluate_adaptive_verifier(
            grid,
            base,
            verifier,
            unsafe=fresh_slack < threshold,
            impulses=heldout_impulses,
            total_steps=total_steps,
        )
    )
    outcomes = {
        "task_only": evaluate_chunk_policy(
            grid, task, impulses=heldout_impulses, total_steps=total_steps
        ),
        "slack_regularized": evaluate_chunk_policy(
            grid, slack_policy, impulses=heldout_impulses, total_steps=total_steps
        ),
        "shorter_chunk": evaluate_chunk_policy(
            grid, short, impulses=heldout_impulses, total_steps=total_steps
        ),
        "adaptive_verifier": adaptive,
    }
    rows = [
        {
            "stage": "p4.2-recovery2-f2-repair" if budgeted_repair else "p4.2-recovery2",
            "split": "heldout_magnitude2_impulses",
            "policy": name,
            "gate_threshold": threshold if name == "adaptive_verifier" else np.nan,
            **metrics,
        }
        for name, metrics in outcomes.items()
    ]
    adaptive = outcomes["adaptive_verifier"]
    task_result = outcomes["task_only"]
    verifier_gain = adaptive["goal_probability"] - task_result["goal_probability"]
    slack_gain = (
        outcomes["slack_regularized"]["goal_probability"]
        - task_result["goal_probability"]
    )
    calls_matched = abs(adaptive["replan_calls"] - target_calls) <= call_tolerance
    checks += [
        PrerequisiteCheck(
            "frozen pilot-only gate calibration",
            True,
            f"threshold={threshold:.6f}, pilot calls={pilot_choice['replan_calls']:.3f}",
        ),
        PrerequisiteCheck(
            "adaptive verifier calls matched",
            calls_matched,
            f"calls={adaptive['replan_calls']:.3f}, target={target_calls:.3f}",
        ),
        PrerequisiteCheck(
            "adaptive verifier meaningful heldout gain",
            verifier_gain >= minimum_gain,
            f"gain={verifier_gain:.6f}, required={minimum_gain:.6f}",
        ),
        PrerequisiteCheck(
            "slack-policy result reported separately",
            True,
            f"same-call slack gain={slack_gain:.6f}",
        ),
    ]
    writer.write_metrics(rows)
    writer.event(
        stage="p4.2-recovery2-f2-repair" if budgeted_repair else "p4.2-recovery2",
        threshold=threshold,
        pilot_calls=pilot_choice["replan_calls"],
        heldout_calls=adaptive["replan_calls"],
        verifier_gain=verifier_gain,
        slack_gain=slack_gain,
    )
    useful_verifier = verifier_gain >= minimum_gain and calls_matched
    status = (
        Status.NARROW
        if useful_verifier and all(check.passed for check in checks)
        else Status.STOP
    )
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=status,
        evidence_level="E1",
        primary_result={
            "metric": "matched_call_adaptive_verifier_goal_gain",
            "value": verifier_gain,
            "unit": "probability",
        },
        baselines_run=[
            "same-call task-only chunk",
            "same-call slack-regularized chunk",
            "higher-call shorter chunk",
            "pilot-calibrated adaptive verifier",
        ],
        controls=checks,
        interpretation=(
            "The final recovery tests whether recovery value is useful as an adaptive verifier "
            "even when it does not improve the trained chunk policy. A verifier win narrows the "
            "claim to inference-time intervention; it does not rescue slack-training novelty."
        ),
        non_claim=config.preregistration.non_claim,
        decision=(
            "Retain the exactly budgeted adaptive verifier direction; "
            "archive slack-policy training."
            if status is Status.NARROW and budgeted_repair
            else "Retain the matched-call adaptive verifier direction; "
            "archive slack-policy training."
            if status is Status.NARROW
            else "Archive bounded P4.2 after the preserved F2 repair also failed."
            if budgeted_repair
            else "Preserve this cost-mismatch result and run the preregistered F2 repair."
        ),
    )


def run_p4_2(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    """Fresh held-out-impulse comparison after the inertial estimator lock."""

    if str(config.params.get("policy_version", "v1-lower-confidence")) in {
        "r2-adaptive-verifier",
        "r2-f2-budgeted-verifier",
    }:
        return _run_p4_2_recovery2(config, writer)
    checks = common_checks(config, "p4_slack")
    params = config.params
    chunk_length = int(params.get("chunk_length", 3))
    total_steps = int(params.get("total_steps", 12))
    beta = float(params.get("beta", 0.1))
    margin = float(params.get("margin", 1.0))
    v_fail = float(params.get("v_fail", -1.0))
    minimum_gain = float(params.get("minimum_success_gain", 0.01))
    maximum_clean_loss = float(params.get("maximum_clean_progress_loss", 0.05))
    checks += [
        PrerequisiteCheck("positive P4.2 beta", beta > 0),
        PrerequisiteCheck(
            "episode divisible by chunk", total_steps % chunk_length == 0
        ),
        PrerequisiteCheck("meaningful gain threshold positive", minimum_gain > 0),
    ]
    if not all(check.passed for check in checks):
        return VerdictReport(
            question=config.preregistration.primary_hypothesis,
            status=Status.INCONCLUSIVE,
            evidence_level="E0",
            primary_result={"metric": "configuration_valid", "value": 0.0, "unit": "bool"},
            controls=checks,
            interpretation="P4.2 did not run because its frozen policy protocol is invalid.",
            non_claim=config.preregistration.non_claim,
            decision="Repair configuration before using fresh disturbance units.",
        )

    recovery_version = str(params.get("policy_version", "v1-lower-confidence"))
    if recovery_version == "v1-lower-confidence":
        obstacles = ((2, 0), (0, 2))
        use_lower_confidence = True
    elif recovery_version == "r1-point-estimate":
        obstacles = ((1, 0), (0, 2))
        use_lower_confidence = False
    else:
        raise ValueError(f"unknown P4.2 policy version {recovery_version!r}")
    slip = float(params.get("actuator_slip", 0.1))
    grid = inertial_grid(obstacles=obstacles, actuator_slip=slip)
    surrogate = locked_inertial_surrogate(v_fail=v_fail)
    training_impulses = (
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    )
    heldout_impulses = (
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
    )

    def train(name: str, length: int, method_beta: float) -> ChunkPolicy:
        return train_chunk_policy(
            grid,
            surrogate,
            layout="policy-heldout-layout",
            obstacles=obstacles,
            slip=slip,
            value_horizon=int(params.get("value_horizon", 12)),
            impulses=training_impulses,
            chunk_length=length,
            beta=method_beta,
            margin=margin,
            v_fail=v_fail,
            progress_weight=float(params.get("progress_weight", 2.0)),
            name=name,
            use_lower_confidence=use_lower_confidence,
        )

    policies = {
        "task_only": train("task_only", chunk_length, 0.0),
        "slack_regularized": train("slack_regularized", chunk_length, beta),
        "shorter_chunk": train("shorter_chunk", 2, 0.0),
        "periodic_replanning": train("periodic_replanning", 1, 0.0),
        "surrogate_verifier": train("surrogate_verifier", 1, beta),
        "pace_boundary": _pace_policy(grid),
    }
    rows: list[dict[str, object]] = []
    heldout: dict[str, dict[str, float]] = {}
    clean: dict[str, dict[str, float]] = {}
    for name, policy in policies.items():
        heldout[name] = evaluate_chunk_policy(
            grid, policy, impulses=heldout_impulses, total_steps=total_steps
        )
        clean[name] = evaluate_chunk_policy(
            grid, policy, impulses=((0, 0),), total_steps=total_steps
        )
        rows.extend(
            (
                {
                    "stage": "p4.2",
                    "split": "heldout_diagonal_impulses",
                    "policy": name,
                    **heldout[name],
                },
                {"stage": "p4.2", "split": "clean", "policy": name, **clean[name]},
            )
        )
    task = heldout["task_only"]
    slack = heldout["slack_regularized"]
    success_gain = slack["goal_probability"] - task["goal_probability"]
    clean_progress_loss = clean["task_only"]["progress"] - clean["slack_regularized"]["progress"]
    calls_matched = slack["replan_calls"] == task["replan_calls"]
    useful = success_gain >= minimum_gain and clean_progress_loss <= maximum_clean_loss
    checks += [
        PrerequisiteCheck("task/slack replanning calls matched", calls_matched),
        PrerequisiteCheck(
            "held-out diagonal impulse gain",
            success_gain >= minimum_gain,
            f"gain={success_gain:.6f}, required={minimum_gain:.6f}",
        ),
        PrerequisiteCheck(
            "clean progress retained",
            clean_progress_loss <= maximum_clean_loss,
            f"loss={clean_progress_loss:.6f}, maximum={maximum_clean_loss:.6f}",
        ),
        PrerequisiteCheck(
            "short and verification costs exposed",
            heldout["shorter_chunk"]["replan_calls"] > slack["replan_calls"]
            and heldout["surrogate_verifier"]["replan_calls"] > slack["replan_calls"],
        ),
    ]
    writer.write_metrics(rows)
    writer.event(
        stage="p4.2",
        heldout_success_gain=success_gain,
        clean_progress_loss=clean_progress_loss,
        slack_calls=slack["replan_calls"],
        task_calls=task["replan_calls"],
    )
    status = (
        Status.NARROW
        if useful and all(check.passed for check in checks)
        else Status.INCONCLUSIVE
    )
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=status,
        evidence_level="E1",
        primary_result={
            "metric": "heldout_goal_probability_gain_at_matched_replan_calls",
            "value": success_gain,
            "unit": "probability",
        },
        baselines_run=[
            "task-only equal chunk/calls",
            "shorter chunk",
            "periodic replanning",
            "post-hoc surrogate verifier",
            "PACE-like boundary braking",
        ],
        baselines_skipped={
            "recovery demonstrations": "none exist in the exact inertial benchmark",
            "entropy regularization": "no stochastic learned policy in this enumerated slice",
        },
        controls=checks,
        interpretation=(
            "This bounded open-loop study uses the locked in-domain estimator and fresh diagonal "
            "impulses. NARROW requires a meaningful same-call gain; full GO still requires an "
            "adaptive matched-average-call baseline and learned policy class."
        ),
        non_claim=config.preregistration.non_claim,
        decision=(
            "Preregister a learned P4.2 replication with adaptive matched-call controls."
            if status is Status.NARROW
            else "Do not scale P4.2; diagnose estimator regularization versus task-only chunks."
        ),
    )
