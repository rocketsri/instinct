from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from instinct.cli import app
from instinct.p5_development.plasticity import LocalPlasticityProgram

ROOT = Path(__file__).parents[1]


def test_extended_program_has_fixed_complete_accounting() -> None:
    program = LocalPlasticityProgram(np.linspace(-0.2, 0.3, 8), step_size=0.04)
    manifest = program.serialization_manifest()
    assert manifest["coefficient_count"] == 8
    assert manifest["total_bytes"] == 72
    assert manifest["initializer_bytes"] == 0
    assert manifest["task_conditioning_bytes"] == 0


def test_recovery_smoke_uses_fresh_units_and_multiple_random_rules(tmp_path: Path) -> None:
    runner = CliRunner()
    config = ROOT / "configs/p5/p5_2_r1/smoke.yaml"
    output = tmp_path / "results"
    result = runner.invoke(
        app,
        ["run", "--proposal", "p5_development", "--config", str(config), "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    run_dir = next((output / "p5_development").iterdir())
    metrics = pd.read_parquet(run_dir / "metrics.parquet")
    heldout = metrics[metrics["stage"] == "p5.2-recovery-1"]
    assert not {591, 592} & set(heldout["eval_seed"])
    random_names = {name for name in heldout["control"] if name.startswith("random_rule_")}
    assert len(random_names) == 8
    prereg = metrics[metrics["stage"] == "p5.2-recovery-preregistration"].iloc[0]
    assert prereg["meaningful_effect_threshold"] > 0
    assert prereg["total_bytes"] == 72
    verdict = json.loads((run_dir / "verdict.json").read_text())
    assert verdict["status"] in {"NARROW", "INCONCLUSIVE"}
    assert "P5.3" in verdict["decision"]
