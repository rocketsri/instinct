"""Spec section 3.4: "serialization/resume equivalence."

Runs the dummy fixture proposal two ways at the same seed/config: straight
through, and forced to crash partway via ``abort_after`` then picked back up
with ``instinct resume``. The two must produce byte-identical ``metrics.
parquet`` — the whole point of checkpointing being state, not a best-effort
hint.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import instinct.cli as cli_mod
from tests._dummy_config import write_dummy_config
from tests.fixtures.dummy_proposal import run as dummy_run

runner = CliRunner()


@pytest.fixture(autouse=True)
def _dummy_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_mod, "load_plugin", lambda proposal_id: dummy_run)


def _run_metrics(output: Path, run_dir: Path) -> bytes:
    return (run_dir / "metrics.parquet").read_bytes()


def test_resumed_run_matches_an_uninterrupted_run(tmp_path: Path) -> None:
    # Uninterrupted: no abort_after, runs straight through in one process.
    clean_config = write_dummy_config(tmp_path / "clean.yaml")
    clean_output = tmp_path / "clean_results"
    result = runner.invoke(
        cli_mod.app,
        ["run", "--proposal", "dummy", "--config", str(clean_config), "--output", str(clean_output)],
    )
    assert result.exit_code == 0, result.output
    clean_run_dir = next((clean_output / "dummy").iterdir())
    clean_metrics = _run_metrics(clean_output, clean_run_dir)

    # Interrupted: same seed/total_steps, but crashes after step 2 and must be resumed.
    crash_config = write_dummy_config(tmp_path / "crash.yaml", abort_after=2)
    crash_output = tmp_path / "crash_results"
    crashed = runner.invoke(
        cli_mod.app,
        ["run", "--proposal", "dummy", "--config", str(crash_config), "--output", str(crash_output)],
    )
    assert crashed.exit_code != 0, "the fixture's simulated crash must actually propagate"
    crash_run_dir = next((crash_output / "dummy").iterdir())
    assert (crash_run_dir / "checkpoint.json").exists()

    import json

    manifest_before_resume = json.loads((crash_run_dir / "manifest.json").read_text())
    assert manifest_before_resume["status"] == "failed"

    resumed = runner.invoke(
        cli_mod.app,
        ["resume", "--run", str(crash_run_dir), "--output", str(crash_output)],
    )
    assert resumed.exit_code == 0, resumed.output
    resumed_run_dirs = [d for d in (crash_output / "dummy").iterdir() if d != crash_run_dir]
    assert len(resumed_run_dirs) == 1
    resumed_run_dir = resumed_run_dirs[0]

    resumed_metrics = _run_metrics(crash_output, resumed_run_dir)
    assert resumed_metrics == clean_metrics, "resumed metrics must byte-match an uninterrupted run"

    resumed_manifest = json.loads((resumed_run_dir / "manifest.json").read_text())
    assert resumed_manifest["parent_run"] == manifest_before_resume["run_id"]


def test_resume_of_an_already_completed_run_is_rejected(tmp_path: Path) -> None:
    config = write_dummy_config(tmp_path / "smoke.yaml")
    output = tmp_path / "results"
    result = runner.invoke(
        cli_mod.app,
        ["run", "--proposal", "dummy", "--config", str(config), "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    run_dir = next((output / "dummy").iterdir())

    resumed = runner.invoke(cli_mod.app, ["resume", "--run", str(run_dir), "--output", str(output)])
    assert resumed.exit_code != 0
