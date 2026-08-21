from __future__ import annotations

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from instinct.cli import app
from instinct.p4_slack.inertial import (
    impulse_recovery_values,
    inertial_grid,
    one_step_failure_floor,
)
from instinct.p4_slack.slack import finite_horizon_replan_value

ROOT = Path(__file__).parents[1]


def test_inertial_grid_is_exact_stochastic_mdp_with_absorbing_terminals() -> None:
    grid = inertial_grid(actuator_slip=0.15)
    assert np.allclose(grid.mdp.P.sum(axis=2), 1.0)
    for terminal in (grid.goal_state, grid.failure_state):
        assert np.all(grid.mdp.P[terminal, :, terminal] == 1.0)
        assert np.count_nonzero(grid.mdp.P[terminal]) == grid.mdp.n_actions


def test_legal_states_include_recoverable_and_already_unrecoverable_motion() -> None:
    grid = inertial_grid(actuator_slip=0.0)
    floor = one_step_failure_floor(grid)
    nonterminal = ~grid.mdp.terminal
    assert np.any(floor[nonterminal] == 0.0)
    assert np.any(floor[nonterminal] == 1.0)


def test_impulse_labels_share_exact_value_origin_and_separate_failure() -> None:
    grid = inertial_grid(actuator_slip=0.1)
    value = finite_horizon_replan_value(grid.mdp, horizon=10, v_fail=-1.0)
    disturbed = impulse_recovery_values(
        grid,
        value,
        ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)),
    )
    assert disturbed.shape == (grid.mdp.n_states, 5)
    assert np.isfinite(disturbed).all()
    assert np.all(disturbed[grid.failure_state] == -1.0)
    assert np.all(disturbed[grid.goal_state] == 0.0)


def test_impulse_kernel_is_deterministic_and_preserves_position() -> None:
    grid = inertial_grid(actuator_slip=0.1)
    kernel = grid.impulse_kernel(((1, -1),))
    assert np.all(kernel.sum(axis=2) == 1.0)
    state = grid.state_index[(0, 0, 0, 0)]
    successor = int(np.argmax(kernel[0, state]))
    assert grid.states[successor] == (0, 0, 1, -1)


def test_inertial_exact_gate_cli_smoke(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--proposal",
            "p4_slack",
            "--config",
            str(ROOT / "configs/p4/p4_0_inertial/smoke.yaml"),
            "--output",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "GO" in result.output
