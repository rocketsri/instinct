"""``ResultsWriter``: the one path that produces spec section 3.3's seven files.

Nothing here is new machinery — it composes :class:`~instinct.core.manifest.
RunRecorder`, :class:`~instinct.core.checkpoint.CheckpointStore`, and
:mod:`instinct.core.verdict`'s writers, which already exist and already know
how to write their own piece. What this module adds is the guarantee that a
proposal calling it cannot accidentally produce a *different* set of files: a
proposal's ``run()`` receives a :class:`ResultsWriter` instead of a bare
:class:`RunRecorder`, and its only path to ``metrics.parquet``,
``verdict.json``, and ``report.md`` is through this class's named methods —
never an arbitrary ``recorder.table("whatever_i_felt_like")``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from instinct.core.checkpoint import CheckpointStore
from instinct.core.manifest import RunRecorder
from instinct.core.profile import write_profile_json
from instinct.core.verdict import VerdictReport, render_report_md, write_verdict_json

__all__ = ["METRICS_TABLE_NAME", "ResultsWriter"]

#: The one metrics table name spec section 3.3 promises as ``metrics.parquet``.
METRICS_TABLE_NAME = "metrics"


class ResultsWriter:
    """Wraps a :class:`RunRecorder` so a run produces exactly the required files.

    ``events.jsonl`` is covered by ``recorder.event(...)`` (the default stream
    name is ``"events"``, matching the required filename directly);
    ``manifest.json``/``config.lock.yaml`` are written by ``cli.py`` around the
    ``run_context``/``write_config_lock`` calls before a proposal ever sees this
    writer. This class is what remains: the metrics table, the checkpoint
    store, and the two verdict-derived files.
    """

    def __init__(self, recorder: RunRecorder) -> None:
        self.recorder = recorder
        self.checkpoints = CheckpointStore(run_dir=recorder.run_dir)

    def _theory_fields(self) -> dict[str, str]:
        theory = getattr(self.recorder.manifest.config, "theory", None)
        if theory is None or not theory.proposal_version or not theory.theory_id:
            raise ValueError("run config is missing proposal/theory traceability")
        return {
            "proposal_version": str(theory.proposal_version),
            "theory_id": str(theory.theory_id),
            "amendment_id": str(theory.amendment_id),
        }

    def event(self, **fields: Any) -> None:
        """Append to ``events.jsonl`` — the required event stream name."""
        self.recorder.event("events", **fields)

    def write_metrics(self, rows: pd.DataFrame | Sequence[Mapping[str, Any]]) -> Path:
        """Write ``metrics.parquet``. The only sanctioned path to that filename."""
        frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame.from_records(rows)
        for key, value in self._theory_fields().items():
            if key in frame and not frame[key].astype(str).eq(value).all():
                raise ValueError(f"metric rows conflict with configured {key}={value!r}")
            frame[key] = value
        return self.recorder.table(METRICS_TABLE_NAME, frame)

    def write_verdict(
        self, report: VerdictReport, *, metrics_summary: Mapping[str, Any] | None = None
    ) -> None:
        """Write ``verdict.json`` and ``report.md`` together, from one report.

        Together because ``report.md`` is a rendering *of* the verdict, never a
        second source of truth for it — see ``core/verdict.py``'s module
        docstring on why every number in the rendered markdown must trace back
        to a field on the report that also went into ``verdict.json``.
        """
        trace = self._theory_fields()
        report.proposal_version = trace["proposal_version"]
        report.theory_id = trace["theory_id"]
        report.amendment_id = trace["amendment_id"]
        write_verdict_json(self.recorder.run_dir / "verdict.json", report)
        report_md = render_report_md(
            report, self.recorder.manifest, metrics_summary=metrics_summary
        )
        (self.recorder.run_dir / "report.md").write_text(report_md)
        if "verdict" not in self.recorder.manifest.artifacts:
            self.recorder.manifest.artifacts.append("verdict")

    def checkpoint_if_due(
        self, state: Mapping[str, Any], *, elapsed_s: float, every_s: float
    ) -> bool:
        """Save ``state`` if the checkpoint interval has elapsed; report whether it did.

        The one call a proposal's run loop needs for spec section 3.2's
        "checkpoint every 10-20 minutes" — it can call this every iteration
        without timing the interval itself; :class:`~instinct.core.checkpoint.
        CheckpointStore` tracks its own last-save time.
        """
        if self.checkpoints.should_checkpoint(elapsed_s, every_s):
            self.checkpoints.save(state)
            return True
        return False

    def write_profile(
        self,
        *,
        ram_peak_mb: float,
        vram_peak_mb: float | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Write ``profile.json`` from the recorder's own compute ledger."""
        write_profile_json(
            self.recorder.run_dir / "profile.json",
            self.recorder.ledger,
            ram_peak_mb=ram_peak_mb,
            vram_peak_mb=vram_peak_mb,
            extra=extra,
        )
