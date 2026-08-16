"""A tiny, deterministic plugin: five steps, one metric row each, then GO.

Exists to exercise ``instinct.cli`` and ``core/plugin.py``/``core/results.py``
end to end without depending on any real proposal's science. Two config
params drive it: ``total_steps`` (how many rows to produce) and the optional
``abort_after`` (raise immediately after checkpointing that step, to let
``tests/test_core_resume_equivalence.py`` simulate a crash and then resume).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from instinct.core.configschema import ProposalRunConfig
from instinct.core.results import ResultsWriter
from instinct.core.verdict import PrerequisiteCheck, Status, VerdictReport

__all__ = ["resume", "run", "validate"]


def validate(config: ProposalRunConfig) -> list[PrerequisiteCheck]:
    total_steps = config.params.get("total_steps", 0)
    return [
        PrerequisiteCheck(
            name="total_steps configured",
            passed=isinstance(total_steps, int) and total_steps > 0,
            detail=f"total_steps={total_steps!r}",
        )
    ]


def _advance(
    config: ProposalRunConfig, writer: ResultsWriter, *, start_step: int, rows: list[dict[str, Any]]
) -> VerdictReport:
    total_steps = int(config.params.get("total_steps", 5))
    abort_after = config.params.get("abort_after")

    for step in range(start_step, total_steps):
        value = float(config.seed * 1000 + step)
        rows.append({"step": step, "value": value})
        writer.event(step=step, value=value)
        writer.checkpoints.save({"next_step": step + 1, "rows": rows})
        if abort_after is not None and step == int(abort_after):
            raise RuntimeError(f"simulated crash after step {step}")

    writer.write_metrics(rows)
    final_value = rows[-1]["value"] if rows else 0.0
    return VerdictReport(
        question="does the dummy fixture proposal run to completion?",
        status=Status.GO,
        evidence_level="E0",
        primary_result={"metric": "final_value", "value": final_value, "unit": "units"},
        controls=[PrerequisiteCheck(name="rows written", passed=len(rows) == total_steps)],
        interpretation="fixture-only; not a scientific result",
        non_claim="this fixture proves nothing about any real proposal",
        decision="none; this is test infrastructure",
    )


def run(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    return _advance(config, writer, start_step=0, rows=[])


def resume(
    config: ProposalRunConfig, writer: ResultsWriter, checkpoint_state: Mapping[str, Any]
) -> VerdictReport:
    start_step = int(checkpoint_state.get("next_step", 0))
    rows = list(checkpoint_state.get("rows", []))
    return _advance(config, writer, start_step=start_step, rows=rows)
