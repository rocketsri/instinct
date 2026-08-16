"""The five commands spec section 3.3 requires: validate, run, resume, report, compare.

Each command is a thin driver over machinery that already lives in ``core/``
and does the real work — this file's job is only to wire them together in the
right order and produce the right exit codes. In particular:

* ``validate`` never opens a run directory. It loads the config, loads the
  proposal's plugin module, and calls ``plugin.validate(config)``. No
  :func:`~instinct.core.manifest.run_context`, no recorder, no compute spent —
  that boundary is what makes ``validate`` safe to run before every real job
  (spec section 2.5, rule 2: "run smoke configuration before changing
  scientific code").
* ``run`` writes ``config.lock.yaml`` before calling into the proposal, so a
  run that fails immediately still leaves proof of what it was configured to
  do, and wires a periodic checkpoint hook via
  :meth:`~instinct.core.results.ResultsWriter.checkpoint_if_due` — though the
  actual calls happen inside the proposal's own run loop, since only the
  proposal knows where a safe checkpoint boundary is.
* ``resume`` never re-derives a manifest from scratch. It loads the
  interrupted run's own manifest and checkpoint, and opens a *new* run context
  whose ``parent_run`` points at the interrupted run's id — so the resumed
  run's provenance chain is auditable back to the original.
* ``report`` re-renders ``report.md`` from an existing ``verdict.json`` and
  ``manifest.json`` without recomputing anything, which is what makes a
  template change retroactively applicable to old runs.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from instinct.core.checkpoint import CheckpointStore
from instinct.core.configschema import ProposalRunConfig, load_proposal_config
from instinct.core.device import capture_device_info
from instinct.core.manifest import (
    load_manifest,
    manifest_from_dict,
    resolve_run,
    run_context,
    write_config_lock,
)
from instinct.core.plugin import load_plugin
from instinct.core.profile import current_ram_mb, peak_vram_mb
from instinct.core.results import ResultsWriter
from instinct.core.verdict import Status, load_verdict_json, render_report_md

app = typer.Typer(no_args_is_help=True, add_completion=False)
# A fixed width rather than auto-detected: run ids carry a timestamp, a config
# hash, and a nonce (see core/manifest.py's make_run_id), and an
# auto-detected narrow width (common when stdout is piped or captured, as in
# a test runner) would wrap or truncate them mid-token in `compare`'s table.
console = Console(width=200)
err_console = Console(width=200, stderr=True)


@app.command()
def validate(
    proposal: str = typer.Option(..., "--proposal", help="Proposal id, e.g. p1_atlas"),
    config: Path = typer.Option(..., "--config", exists=True, help="Path to a proposal config"),
) -> None:
    """Dry-run: load config, run the plugin's own checks. Spends no compute."""
    cfg = load_proposal_config(config)
    plugin = load_plugin(proposal)
    checks = plugin.validate(cfg)

    table = Table(title=f"validate {proposal} :: {config}")
    table.add_column("check")
    table.add_column("passed")
    table.add_column("detail")
    for c in checks:
        table.add_row(c.name, "yes" if c.passed else "[bold red]no[/bold red]", c.detail)
    console.print(table)

    if not checks:
        err_console.print(f"[yellow]warning:[/yellow] {proposal} reported no prerequisite checks")
    if any(not c.passed for c in checks):
        raise typer.Exit(code=1)


@app.command()
def run(
    proposal: str = typer.Option(..., "--proposal", help="Proposal id, e.g. p1_atlas"),
    config: Path = typer.Option(..., "--config", exists=True, help="Path to a proposal config"),
    output: Path = typer.Option(
        Path("results"), "--output", help="Results root (a proposal/run_id dir is created under it)"
    ),
) -> None:
    """Execute a proposal's run and write spec section 3.3's seven result files."""
    cfg = load_proposal_config(config)
    plugin = load_plugin(proposal)
    device = capture_device_info()

    with run_context(
        proposal,
        cfg,
        output_root=output,
        seeds=list(cfg.seeds),
        device=device,
    ) as recorder:
        write_config_lock(recorder.run_dir / "config.lock.yaml", cfg)
        writer = ResultsWriter(recorder)
        report = plugin.run(cfg, writer)
        writer.write_verdict(report)
        writer.write_profile(ram_peak_mb=current_ram_mb(), vram_peak_mb=peak_vram_mb())

    console.print(f"[bold]{report.status.value}[/bold]  {recorder.run_dir}")
    if report.status in (Status.STOP, Status.INCONCLUSIVE):
        raise typer.Exit(code=0)  # a valid scientific outcome, not a CLI failure


@app.command()
def resume(
    run: str = typer.Option(..., "--run", help="Run id (or path) of the interrupted run"),
    output: Path = typer.Option(Path("results"), "--output", help="Results root"),
) -> None:
    """Pick up an interrupted run from its manifest and last checkpoint."""
    run_dir = resolve_run(run, output_root=output)
    old_manifest = manifest_from_dict(load_manifest(run_dir))
    if old_manifest.status == "ok":
        err_console.print(f"[yellow]{run_dir} already finished ok; nothing to resume[/yellow]")
        raise typer.Exit(code=1)

    cfg: ProposalRunConfig = load_proposal_config(run_dir / "config.lock.yaml")
    plugin = load_plugin(old_manifest.proposal)
    checkpoint_state = CheckpointStore(run_dir=run_dir).load() or {}

    with run_context(
        old_manifest.proposal,
        cfg,
        output_root=output,
        seeds=list(cfg.seeds),
        device=capture_device_info(),
        parent_run=old_manifest.run_id,
    ) as recorder:
        write_config_lock(recorder.run_dir / "config.lock.yaml", cfg)
        writer = ResultsWriter(recorder)
        report = plugin.resume(cfg, writer, checkpoint_state)
        writer.write_verdict(report)
        writer.write_profile(ram_peak_mb=current_ram_mb(), vram_peak_mb=peak_vram_mb())

    console.print(f"[bold]{report.status.value}[/bold]  {recorder.run_dir} (resumed {run_dir})")


@app.command()
def report(
    run: str = typer.Option(..., "--run", help="Run id (or path) to re-render report.md for"),
    output: Path = typer.Option(Path("results"), "--output", help="Results root"),
) -> None:
    """Re-render report.md from an existing run's verdict + manifest. No recomputation."""
    run_dir = resolve_run(run, output_root=output)
    manifest = manifest_from_dict(load_manifest(run_dir))
    verdict = load_verdict_json(run_dir / "verdict.json")
    text = render_report_md(verdict, manifest)
    (run_dir / "report.md").write_text(text)
    console.print(text)


@app.command()
def compare(
    runs: list[str] = typer.Option(..., "--runs", help="Run ids or paths to compare"),
    output: Path = typer.Option(Path("results"), "--output", help="Results root"),
) -> None:
    """Print a table of config hash, primary metric, and status across runs."""
    table = Table(title="instinct compare")
    table.add_column("run")
    table.add_column("config hash")
    table.add_column("status")
    table.add_column("primary metric")

    for r in runs:
        run_dir = resolve_run(r, output_root=output)
        manifest = manifest_from_dict(load_manifest(run_dir))
        try:
            verdict = load_verdict_json(run_dir / "verdict.json")
            status = verdict.status.value
            metric = verdict.primary_result.get("metric", "")
            value = verdict.primary_result.get("value", "")
            metric_s = f"{metric} = {value}" if metric else ""
        except FileNotFoundError:
            status, metric_s = manifest.status, ""
        table.add_row(manifest.run_id, manifest.config_hash[:12], status, metric_s)

    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
