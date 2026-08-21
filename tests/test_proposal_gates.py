from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from instinct.cli import app
from instinct.p3_codesign.objective import constructed_case, enumerate_reflexes, option_objective
from instinct.p4_slack.slack import lower_quantile, slack_penalty
from instinct.p5_development.generator import CoordinateGenerator
from instinct.p6_certification.certificates import (
    BernoulliUpperProcess,
    longest_certified,
    polynomial_spend,
)
from instinct.p7_chunking.queue import Item, QueueConfig, simulate_queue

ROOT = Path(__file__).parents[1]
SMOKE_CONFIGS = {
    "p1_atlas": ROOT / "configs/p1/p1_0/smoke.yaml",
    "p2_harmful_write": ROOT / "configs/p2/p2_0/smoke.yaml",
    "p3_codesign": ROOT / "configs/p3/p3_0/smoke.yaml",
    "p4_slack": ROOT / "configs/p4/p4_0/smoke.yaml",
    "p5_development": ROOT / "configs/p5/p5_0_1/smoke.yaml",
    "p6_certification": ROOT / "configs/p6/p6_0_1/smoke.yaml",
    "p7_chunking": ROOT / "configs/p7/simulator/smoke.yaml",
}


def test_p3_frozen_objective_and_all_constructed_optima() -> None:
    for kind in ("hold", "greedy", "interior", "destructive"):
        mdp, action, q, expected = constructed_case(kind)
        rows = enumerate_reflexes(mdp, planner_action=action, planner_q=q, delay=1)
        best = max(rows, key=lambda row: row.value)
        assert best.policy[0] == expected
        reflex = np.zeros(mdp.n_states, dtype=np.int64)
        assert np.allclose(
            option_objective(mdp, reflex=reflex, planner_action=action, planner_q=q, delay=0),
            q[np.arange(mdp.n_states), action],
        )


def test_p4_lower_quantile_and_terminal_mask() -> None:
    assert lower_quantile(np.array([-2.0, 0.0, 0.0, 5.0]), 0.5) == 0.0
    assert slack_penalty(np.array([0.0, -100.0]), margin=1.0, active=np.array([True, False])) == 1.0


def test_p5_generator_and_update_are_permutation_equivariant() -> None:
    g = CoordinateGenerator()
    pre, post = np.linspace(0.1, 0.9, 5), np.linspace(0.2, 0.8, 4)
    pp, qp = np.array([2, 0, 4, 1, 3]), np.array([3, 1, 0, 2])
    W = g.weights(pre, post)
    assert np.allclose(g.weights(pre[pp], post[qp]), W[np.ix_(qp, pp)])


def test_p6_process_rejects_nonbernoulli_and_spending_is_bounded() -> None:
    process = BernoulliUpperProcess(0.05, 3)
    with pytest.raises(ValueError, match="zero or one"):
        process.update(2)
    for value in (0, 0, 1, 0):
        process.update(value)
    assert 0 <= process.upper(1) <= 1
    assert longest_certified({1: 0.1, 2: 0.3}, 0.2) == 1
    assert sum(polynomial_spend(0.2, t) for t in range(1, 20_000)) <= 0.2


def test_p7_queue_shape_owner_and_padding() -> None:
    items = [Item(str(i), i * 0.1, 1.0, f"s{i % 2}") for i in range(7)]
    trace = simulate_queue(items, QueueConfig(buffer_size=3, max_age=0.5), seed=0)
    assert trace.updates
    assert all(event.kernel_shape == (3,) for event in trace.updates)
    assert all(event.real_tokens + event.padding_tokens == 3 for event in trace.updates)
    assert {event.stream_id for event in trace.updates} <= {"s0", "s1"}


@pytest.mark.parametrize("proposal,config", SMOKE_CONFIGS.items())
def test_each_proposal_cli_smoke(proposal: str, config: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    validated = runner.invoke(app, ["validate", "--proposal", proposal, "--config", str(config)])
    assert validated.exit_code == 0, validated.output
    output = tmp_path / "results"
    result = runner.invoke(
        app, ["run", "--proposal", proposal, "--config", str(config), "--output", str(output)]
    )
    assert result.exit_code == 0, result.output
    run_dir = next((output / proposal).iterdir())
    metrics = pd.read_parquet(run_dir / "metrics.parquet")
    assert {"proposal_version", "theory_id", "amendment_id"} <= set(metrics)
    verdict = json.loads((run_dir / "verdict.json").read_text())
    assert verdict["proposal_version"] and verdict["theory_id"]
    if proposal == "p7_chunking":
        assert verdict["status"] == "INCONCLUSIVE"
        assert verdict["failure_code"] == "F1"
