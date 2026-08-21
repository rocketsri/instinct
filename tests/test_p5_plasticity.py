from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from instinct.cli import app
from instinct.p5_development.plasticity import (
    LocalPlasticityProgram,
    Topology,
    adaptation_curve,
    conventional_initialization,
    make_task,
)

ROOT = Path(__file__).parents[1]


def test_local_rule_commutes_with_endpoint_permutations() -> None:
    rng = np.random.default_rng(5)
    program = LocalPlasticityProgram(np.array([1.0, -0.2, 0.1, 0.01]))
    x, y, error = rng.normal(size=(9, 5)), rng.normal(size=(9, 4)), rng.normal(size=9)
    weights = rng.normal(size=(4, 5))
    coordinates = (np.arange(4) + 0.5) / 4
    reference = program.local_delta_layer(
        pre_activity=x,
        post_activity=y,
        broadcast_error=error,
        weights=weights,
        post_coordinates=coordinates,
        is_output=False,
        step=1,
    )
    p, q = rng.permutation(5), rng.permutation(4)
    got = program.local_delta_layer(
        pre_activity=x[:, p],
        post_activity=y[:, q],
        broadcast_error=error,
        weights=weights[np.ix_(q, p)],
        post_coordinates=coordinates[q],
        is_output=False,
        step=1,
    )
    assert np.allclose(got, reference[np.ix_(q, p)])


def test_remote_neurons_cannot_taint_selected_synapse() -> None:
    rng = np.random.default_rng(9)
    program = LocalPlasticityProgram(np.array([0.8, 0.1, -0.1, 0.02]))
    x, y, error = rng.normal(size=(8, 4)), rng.normal(size=(8, 3)), rng.normal(size=8)
    weights = rng.normal(size=(3, 4))
    coordinates = (np.arange(3) + 0.5) / 3
    reference = program.local_delta_layer(
        pre_activity=x,
        post_activity=y,
        broadcast_error=error,
        weights=weights,
        post_coordinates=coordinates,
        is_output=False,
        step=0,
    )
    changed_x, changed_y = rng.normal(size=x.shape), rng.normal(size=y.shape)
    changed_x[:, 2], changed_y[:, 1] = x[:, 2], y[:, 1]
    changed = program.local_delta_layer(
        pre_activity=changed_x,
        post_activity=changed_y,
        broadcast_error=error,
        weights=weights,
        post_coordinates=coordinates,
        is_output=False,
        step=0,
    )
    assert changed[1, 2] == reference[1, 2]


def test_plasticity_arms_share_zero_shot_initialization() -> None:
    topology = Topology(5, 2)
    initial = conventional_initialization(topology, seed=17)
    task = make_task(31, "composition")
    program = LocalPlasticityProgram(np.array([1.0, 0.0, 0.0, 0.0]))
    learned = adaptation_curve(initial, task, program=program, steps=3)
    frozen = adaptation_curve(initial, task, program=None, steps=3)
    direct = adaptation_curve(initial, task, program=None, steps=3, direct_backprop=True)
    assert learned[0] == frozen[0] == direct[0]
    assert program.serialization_manifest()["initializer_bytes"] == 0


def test_p5_2_cli_smoke_is_fail_closed_and_accounted(tmp_path: Path) -> None:
    runner = CliRunner()
    config = ROOT / "configs/p5/p5_2/smoke.yaml"
    output = tmp_path / "results"
    result = runner.invoke(
        app,
        ["run", "--proposal", "p5_development", "--config", str(config), "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    run_dir = next((output / "p5_development").iterdir())
    metrics = pd.read_parquet(run_dir / "metrics.parquet")
    heldout = metrics[metrics["stage"] == "p5.2"]
    assert set(heldout["control"]) == {
        "learned_local_rule",
        "frozen_local_rule",
        "random_local_rule",
        "no_plasticity",
        "direct_backprop",
    }
    assert heldout.groupby(["eval_seed", "width", "depth"])["zero_shot_loss"].nunique().max() == 1
    assert not heldout["initializer_owned_by_program"].any()
    verdict = json.loads((run_dir / "verdict.json").read_text())
    assert verdict["status"] in {"NARROW", "INCONCLUSIVE"}
    assert "P5.3" in verdict["decision"]
