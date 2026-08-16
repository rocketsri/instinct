"""Spec section 3.4: "CLI smoke test per proposal."

The Stage-A-owned instance here exercises ``validate`` and ``run`` end to end
against the dummy fixture proposal (see ``tests/fixtures/dummy_proposal/``),
standing in for every real proposal until its Stage B branch lands. Each
proposal's own ``tests/pN_*/test_cli_smoke.py`` (added by its Stage B worker)
runs this same shape of test against its real ``configs/pN/smoke.yaml``, once
that config exists — this file intentionally does not try to enumerate p1..p7,
since failing on a proposal that has not been built yet would make Stage A's
own suite depend on Stage B's completion.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import instinct.cli as cli_mod
from tests._dummy_config import write_dummy_config
from tests.fixtures.dummy_proposal import run as dummy_run

runner = CliRunner()

REQUIRED_FILES = (
    "manifest.json",
    "config.lock.yaml",
    "metrics.parquet",
    "events.jsonl",
    "profile.json",
    "verdict.json",
    "report.md",
)


@pytest.fixture(autouse=True)
def _dummy_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every CLI command resolves 'dummy' to the fixture plugin, never a real proposal."""
    monkeypatch.setattr(cli_mod, "load_plugin", lambda proposal_id: dummy_run)


def test_validate_exits_zero_and_spends_no_compute(tmp_path: Path) -> None:
    config = write_dummy_config(tmp_path / "smoke.yaml")
    result = runner.invoke(cli_mod.app, ["validate", "--proposal", "dummy", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "results").exists(), "validate must never open a run directory"


def test_validate_fails_loudly_on_a_bad_config(tmp_path: Path) -> None:
    config = write_dummy_config(tmp_path / "smoke.yaml", total_steps=0)
    result = runner.invoke(cli_mod.app, ["validate", "--proposal", "dummy", "--config", str(config)])
    assert result.exit_code != 0


def test_run_produces_all_seven_required_files(tmp_path: Path) -> None:
    config = write_dummy_config(tmp_path / "smoke.yaml")
    output = tmp_path / "results"
    result = runner.invoke(
        cli_mod.app,
        ["run", "--proposal", "dummy", "--config", str(config), "--output", str(output)],
    )
    assert result.exit_code == 0, result.output

    run_dirs = list((output / "dummy").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    for name in REQUIRED_FILES:
        assert (run_dir / name).exists(), f"missing {name} in {run_dir}"


def test_report_re_renders_without_recomputation(tmp_path: Path) -> None:
    config = write_dummy_config(tmp_path / "smoke.yaml")
    output = tmp_path / "results"
    run_result = runner.invoke(
        cli_mod.app,
        ["run", "--proposal", "dummy", "--config", str(config), "--output", str(output)],
    )
    assert run_result.exit_code == 0, run_result.output
    run_dir = next((output / "dummy").iterdir())

    before = (run_dir / "metrics.parquet").read_bytes()
    report_result = runner.invoke(cli_mod.app, ["report", "--run", str(run_dir), "--output", str(output)])
    assert report_result.exit_code == 0, report_result.output
    after = (run_dir / "metrics.parquet").read_bytes()
    assert before == after, "report must never recompute metrics"
    assert "GO" in (run_dir / "report.md").read_text()


def test_compare_lists_every_requested_run(tmp_path: Path) -> None:
    config = write_dummy_config(tmp_path / "smoke.yaml")
    output = tmp_path / "results"
    run_dirs = []
    for _ in range(2):
        result = runner.invoke(
            cli_mod.app,
            ["run", "--proposal", "dummy", "--config", str(config), "--output", str(output)],
        )
        assert result.exit_code == 0, result.output
    run_dirs = sorted((output / "dummy").iterdir())
    assert len(run_dirs) == 2

    result = runner.invoke(
        cli_mod.app,
        ["compare", "--runs", str(run_dirs[0]), "--runs", str(run_dirs[1]), "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    for rd in run_dirs:
        assert rd.name in result.output
