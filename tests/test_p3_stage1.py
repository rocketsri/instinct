from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from instinct.cli import app
from instinct.core.envs.tabular import chase_chain
from instinct.core.rng import SeedScope
from instinct.p3_codesign.counterfactual import (
    exact_first_action_labels,
    planner_actions_for_reflex,
    sampled_first_action_labels,
)
from instinct.p3_codesign.learner import (
    TimeConditionedSoftmaxReflex,
    fixed_snapshot,
)
from instinct.p3_codesign.recovery import (
    CELLS,
    finite_planner_q,
    fit_advantage_regression,
    optionality_actions,
)
from instinct.p3_codesign.recovery2 import planned_actions, reflex_actions

ROOT = Path(__file__).parents[1]


def test_exact_and_crn_labels_agree_and_share_horizon_semantics() -> None:
    mdp = chase_chain(n_positions=3, gamma=0.9, drift=0.3)
    _, q, _ = mdp.value_iteration()
    hold = fixed_snapshot(
        "hold", np.ones(mdp.n_states, dtype=np.int64), 3, mdp.n_actions
    )
    pending = planner_actions_for_reflex(
        mdp, planner_q=q, simulated_reflex=hold, delay=3
    )
    exact = exact_first_action_labels(
        mdp,
        start_state=0,
        pending_action=int(pending[0]),
        planner_q=q,
        suffix_reflex=hold,
        remaining=3,
    )
    sampled = sampled_first_action_labels(
        mdp,
        start_state=0,
        pending_action=int(pending[0]),
        planner_q=q,
        suffix_reflex=hold,
        remaining=3,
        n_rollouts=4096,
        scope=SeedScope(7).child("test"),
    )
    assert np.allclose(exact, sampled.mean_returns, atol=0.12)
    assert not sampled.terminated.any()
    assert sampled.truncated.all()


def test_linear_reflex_freezes_distinct_reproducible_snapshot() -> None:
    learner = TimeConditionedSoftmaxReflex(2, 3, 4, learning_rate=0.2)
    before = learner.snapshot()
    learner.fit(
        np.array([0, 1, 0, 1]),
        np.array([1, 1, 2, 2]),
        np.array([2, 1, 2, 1]),
        seed=11,
        epochs=3,
    )
    after = learner.snapshot()
    assert before.fingerprint != after.fingerprint
    assert np.array_equal(after.actions(3), np.array([2, 1]))
    assert not after.logits.flags.writeable


def test_p3_1_cli_smoke_reports_heldout_and_crossplay(tmp_path: Path) -> None:
    runner = CliRunner()
    config = ROOT / "configs/p3/p3_1/smoke.yaml"
    output = tmp_path / "results"
    result = runner.invoke(
        app,
        ["run", "--proposal", "p3_codesign", "--config", str(config), "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    run_dir = next((output / "p3_codesign").iterdir())
    metrics = pd.read_parquet(run_dir / "metrics.parquet")
    assert {"p3.1", "p3.1-crossplay", "p3.1-label-audit"} <= set(metrics["stage"])
    heldout = metrics[metrics["stage"] == "p3.1"]
    assert heldout["planner_work"].nunique() == 1
    assert {"terminated_probability", "truncated_probability"} <= set(heldout)
    verdict = json.loads((run_dir / "verdict.json").read_text())
    assert verdict["status"] in {"NARROW", "INCONCLUSIVE"}
    assert verdict["evidence_level"] == "E1"


def test_recovery_advantage_model_is_delay_and_speed_conditioned() -> None:
    cell = CELLS[0]
    model, queries = fit_advantage_regression(
        cell,
        (0.1, 0.3, 0.5),
        (1, 2, 4),
        all_speeds=(0.1, 0.2, 0.3, 0.4, 0.5),
        max_delay=5,
        gamma=0.9,
        blocks=1,
        ridge=1e-4,
    )
    assert queries > 0
    assert model.snapshot(0.2).fingerprint != model.snapshot(0.4).fingerprint
    assert not np.array_equal(model.snapshot(0.2).logits[3], model.snapshot(0.2).logits[5])


def test_recovery_optionality_is_not_an_alias_for_hold() -> None:
    from instinct.core.envs.tabular import corridor_with_pit

    mdp = corridor_with_pit(length=5, gamma=0.9, slip=0.2)
    planner_q = finite_planner_q(mdp, 3)
    actions = optionality_actions(mdp, planner_q)
    assert np.any(actions[~mdp.terminal] != 1)


def test_p3_recovery_cli_has_fresh_cells_and_equal_compute_pg(tmp_path: Path) -> None:
    runner = CliRunner()
    config = ROOT / "configs/p3/p3_1_recovery/smoke.yaml"
    output = tmp_path / "recovery-results"
    result = runner.invoke(
        app,
        ["run", "--proposal", "p3_codesign", "--config", str(config), "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    run_dir = next((output / "p3_codesign").iterdir())
    metrics = pd.read_parquet(run_dir / "metrics.parquet")
    recovery = metrics[metrics["stage"] == "p3.1-recovery"]
    assert recovery["cell_id"].nunique() == 3
    assert set(recovery["speed"]) == {0.2, 0.4}
    assert set(recovery["delay"]) == {3, 5}
    queries = recovery.loc[
        recovery["baseline"].isin(
            ["delay_advantage_regression", "equal_compute_policy_gradient"]
        )
    ].groupby(["cell_id", "baseline"])["label_queries"].first().unstack()
    assert (queries["delay_advantage_regression"] == queries["equal_compute_policy_gradient"]).all()
    verdict = json.loads((run_dir / "verdict.json").read_text())
    assert verdict["status"] == "STOP"
    assert verdict["failure_code"] == "F4"
    mismatch = next(
        control
        for control in verdict["controls"]
        if control["name"] == "planner mismatch changes pending actions"
    )
    assert mismatch["passed"]


def test_inertial_reflex_has_structural_interior_action() -> None:
    from instinct.core.env import EnvState
    from instinct.core.envs.control import InertialIntervention

    env = InertialIntervention(noise_std=0.0)
    state = EnvState(
        lane_ids=np.array([1, 2]),
        fields={
            "position": np.array([0.2, 0.55]),
            "velocity": np.array([0.2, 1.0]),
            "failed": np.array([False, False]),
        },
    )
    interior = reflex_actions(env, state, policy="threshold", threshold=0.1)
    hold = reflex_actions(env, state, policy="hold")
    immediate = reflex_actions(env, state, policy="immediate")
    assert not np.array_equal(interior, hold)
    assert not np.array_equal(interior, immediate)


def test_inertial_planner_simulation_depends_on_reflex_model() -> None:
    from instinct.core.env import EnvState

    state = EnvState(
        lane_ids=np.array([1]),
        fields={
            "position": np.array([-0.8]),
            "velocity": np.array([-0.9]),
            "failed": np.array([False]),
        },
    )
    delays = np.array([1])
    matched = planned_actions(
        state,
        delays,
        budget=4,
        model_policy="threshold",
        model_threshold=0.1,
    )
    mismatch = planned_actions(
        state, delays, budget=4, model_policy="hold", model_threshold=0.0
    )
    assert np.any(matched != mismatch)
