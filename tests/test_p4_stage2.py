from __future__ import annotations

import numpy as np

from instinct.p4_slack.inertial import inertial_grid
from instinct.p4_slack.stage2 import (
    ChunkPolicy,
    evaluate_adaptive_verifier,
    evaluate_budgeted_verifier,
    evaluate_chunk_policy,
    locked_inertial_surrogate,
    train_chunk_policy,
)


def test_chunk_policy_has_one_open_loop_sequence_per_launch_state() -> None:
    obstacles = ((0, 2), (2, 2))
    grid = inertial_grid(obstacles=obstacles, actuator_slip=0.1)
    surrogate = locked_inertial_surrogate(v_fail=-1.0)
    policy = train_chunk_policy(
        grid,
        surrogate,
        layout="test-layout",
        obstacles=obstacles,
        slip=0.1,
        value_horizon=12,
        impulses=((0, 0), (-1, 0), (1, 0)),
        chunk_length=2,
        beta=0.1,
        margin=1.0,
        v_fail=-1.0,
        progress_weight=1.0,
        name="slack",
        use_lower_confidence=True,
    )
    assert policy.actions.shape == (grid.mdp.n_states, 2)
    assert np.all((policy.actions >= 0) & (policy.actions < grid.mdp.n_actions))


def test_chunk_evaluation_reports_exact_call_count_and_probabilities() -> None:
    obstacles = ((0, 2), (2, 2))
    grid = inertial_grid(obstacles=obstacles, actuator_slip=0.1)
    surrogate = locked_inertial_surrogate(v_fail=-1.0)
    policy = train_chunk_policy(
        grid,
        surrogate,
        layout="test-layout",
        obstacles=obstacles,
        slip=0.1,
        value_horizon=12,
        impulses=((0, 0),),
        chunk_length=2,
        beta=0.0,
        margin=1.0,
        v_fail=-1.0,
        progress_weight=1.0,
        name="task",
        use_lower_confidence=True,
    )
    metrics = evaluate_chunk_policy(
        grid, policy, impulses=((0, 0),), total_steps=6
    )
    assert metrics["replan_calls"] == 3.0
    assert 0.0 <= metrics["goal_probability"] <= 1.0
    assert 0.0 <= metrics["failure_probability"] <= 1.0


def test_adaptive_verifier_counts_expected_interventions() -> None:
    obstacles = ((0, 2), (2, 2))
    grid = inertial_grid(obstacles=obstacles, actuator_slip=0.1)
    surrogate = locked_inertial_surrogate(v_fail=-1.0)
    base = train_chunk_policy(
        grid,
        surrogate,
        layout="adaptive-test",
        obstacles=obstacles,
        slip=0.1,
        value_horizon=12,
        impulses=((0, 0),),
        chunk_length=2,
        beta=0.0,
        margin=1.0,
        v_fail=-1.0,
        progress_weight=1.0,
        name="base",
    )
    verifier = train_chunk_policy(
        grid,
        surrogate,
        layout="adaptive-test",
        obstacles=obstacles,
        slip=0.1,
        value_horizon=12,
        impulses=((0, 0),),
        chunk_length=1,
        beta=0.1,
        margin=1.0,
        v_fail=-1.0,
        progress_weight=1.0,
        name="verifier",
    )
    metrics = evaluate_adaptive_verifier(
        grid,
        base,
        verifier,
        unsafe=np.zeros(grid.mdp.n_states, dtype=bool),
        impulses=((0, 0),),
        total_steps=4,
    )
    assert metrics["base_replan_calls"] == 2.0
    assert metrics["verifier_calls"] == 0.0
    assert metrics["replan_calls"] == 2.0


def test_budgeted_verifier_matches_calls_pathwise() -> None:
    obstacles = ((0, 2), (2, 2))
    grid = inertial_grid(obstacles=obstacles, actuator_slip=0.1)
    surrogate = locked_inertial_surrogate(v_fail=-1.0)
    base = train_chunk_policy(
        grid,
        surrogate,
        layout="budget-test",
        obstacles=obstacles,
        slip=0.1,
        value_horizon=12,
        impulses=((0, 0),),
        chunk_length=2,
        beta=0.0,
        margin=1.0,
        v_fail=-1.0,
        progress_weight=1.0,
        name="base",
    )
    verifier = ChunkPolicy("verifier", base.actions[:, :1], 1, 0.1)
    metrics = evaluate_budgeted_verifier(
        grid,
        base,
        verifier,
        unsafe=np.zeros(grid.mdp.n_states, dtype=bool),
        impulses=((0, 0),),
        total_steps=4,
        verifier_budget=2,
    )
    assert np.isclose(metrics["base_replan_calls"], 2.0)
    assert np.isclose(metrics["verifier_calls"], 2.0)
    assert np.isclose(metrics["replan_calls"], 4.0)
