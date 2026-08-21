from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from instinct.cli import app
from instinct.p5_development.feedback_basis import FeedbackBasisProgram

ROOT = Path(__file__).parents[1]


def test_feedback_basis_is_permutation_equivariant_and_time_dependent() -> None:
    rng = np.random.default_rng(4)
    coefficients = np.linspace(-0.4, 0.6, 12)
    program = FeedbackBasisProgram(coefficients)
    pre, post = rng.normal(size=(10, 5)), rng.normal(size=(10, 4))
    error, weights = rng.normal(size=10), rng.normal(size=(4, 5))
    coordinates = (np.arange(4) + 0.5) / 4
    reference = program.local_delta_layer(
        pre_activity=pre,
        post_activity=post,
        broadcast_error=error,
        weights=weights,
        post_coordinates=coordinates,
        layer_coordinate=0.4,
        is_output=False,
        step=2,
    )
    p, q = rng.permutation(5), rng.permutation(4)
    permuted = program.local_delta_layer(
        pre_activity=pre[:, p],
        post_activity=post[:, q],
        broadcast_error=error,
        weights=weights[np.ix_(q, p)],
        post_coordinates=coordinates[q],
        layer_coordinate=0.4,
        is_output=False,
        step=2,
    )
    assert np.allclose(permuted, reference[np.ix_(q, p)])
    early = program.feedback(coordinates, layer_coordinate=0.4, is_output=False, step=0)
    late = program.feedback(coordinates, layer_coordinate=0.4, is_output=False, step=12)
    assert not np.allclose(early, late)
    assert program.serialization_manifest()["total_bytes"] == 104


def test_final_recovery_smoke_archives_or_passes_without_reusing_units(tmp_path: Path) -> None:
    runner = CliRunner()
    config = ROOT / "configs/p5/p5_2_r2/smoke.yaml"
    output = tmp_path / "results"
    result = runner.invoke(
        app,
        ["run", "--proposal", "p5_development", "--config", str(config), "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    run_dir = next((output / "p5_development").iterdir())
    metrics = pd.read_parquet(run_dir / "metrics.parquet")
    heldout = metrics[metrics["stage"] == "p5.2-recovery-2"]
    assert not {591, 592, 691, 692, 693} & set(heldout["eval_seed"])
    assert len({name for name in heldout["control"] if name.startswith("random_feedback_")}) == 8
    prereg = metrics[metrics["stage"] == "p5.2-recovery-2-preregistration"].iloc[0]
    assert prereg["meaningful_effect_threshold"] > 0
    assert prereg["total_bytes"] == 104
    verdict = json.loads((run_dir / "verdict.json").read_text())
    assert verdict["status"] in {"NARROW", "STOP"}
    if verdict["status"] == "STOP":
        assert verdict["failure_code"] == "F3"
        assert "Archive P5.2" in verdict["decision"]
