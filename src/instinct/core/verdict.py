"""The portfolio-wide verdict: GO, NARROW, STOP, or INCONCLUSIVE, never upgraded.

Every proposal in the portfolio ends a run the same way (spec section 2.1):
never PASS, never KILL — those were P1's old, proposal-local vocabulary. The
four outcomes here are the only four a run may report, and the rule that
matters most is the one that is easiest to violate by accident: **an
INCONCLUSIVE result must never quietly become a GO because a downstream
consumer treated "no status" as success.** So :class:`Status` has no default,
:class:`VerdictReport` has no field that computes a status from the others, and
nothing in this module infers a verdict from a metric. A proposal's own
``verdict.py`` decides the status explicitly, by name, at the one call site
that has the domain knowledge to be right about it. This module only records
that decision faithfully and renders it the same way every time.

:class:`FailureCode` is the portfolio's shared taxonomy (spec section 2.4) for
*why* a run did not reach GO. It is optional — a clean STOP ("the baseline
won") does not need a failure code, and forcing one there would invite
choosing the nearest-sounding code instead of leaving the field empty.

:func:`render_report_md` is the only thing that turns a :class:`VerdictReport`
into prose, and it is deliberately mechanical: every number in the rendered
markdown comes from a field on the report, not from a second computation. That
is what makes the "report counts and units agree with raw data" shared test
(spec section 3.4) checkable at all — render, parse the numbers back out,
compare to the fields that produced them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from instinct.core.cache import to_canonical

if TYPE_CHECKING:
    from instinct.core.manifest import RunManifest

__all__ = [
    "VERDICT_NAME",
    "FailureCode",
    "PrerequisiteCheck",
    "Status",
    "VerdictReport",
    "load_verdict_json",
    "render_report_md",
    "write_verdict_json",
]

VERDICT_NAME = "verdict.json"


class Status(StrEnum):
    """The only four outcomes a run may report. See the module docstring."""

    GO = "GO"
    NARROW = "NARROW"
    STOP = "STOP"
    INCONCLUSIVE = "INCONCLUSIVE"


class FailureCode(StrEnum):
    """Spec section 2.4's taxonomy of why a run did not reach GO.

    Optional on a :class:`VerdictReport`: a STOP where a baseline simply won is
    a complete, correct result without one of these, and every code here names
    a *required response*, not just a symptom — see each member's docstring.
    """

    F0_CODE_INVARIANT = "F0"
    """Shape error, leakage, divergence, reference mismatch. Fix and rerun
    controls; no scientific interpretation of this run."""

    F1_UNIDENTIFIABLE = "F1"
    """Flat target, structurally collapsed axes, contaminated probes, too few
    events. Redesign the experiment; do not tune the method."""

    F2_ESTIMATOR_FAILURE = "F2"
    """Oracle/proxy misses a planted or exact effect. Repair or abandon the
    estimator before trusting anything it reports."""

    F3_MECHANISM_ABSENT = "F3"
    """A valid experiment finds no meaningful effect. Stop, or narrow the
    claim."""

    F4_BASELINE_DOMINANCE = "F4"
    """A simpler, matched baseline wins. Stop the method; the measurement
    itself is still a preserved result."""

    F5_SYSTEM_FAILURE = "F5"
    """The mechanism works but overhead/memory erases the gain. Reframe, or
    optimize only a profiled, removable bottleneck."""


@dataclass(frozen=True, slots=True)
class PrerequisiteCheck:
    """One named check a run's ``validate``/``run`` performed, and its outcome.

    This is the unit spec section 3.3 means by "the verdict lists every
    prerequisite check" — known-answer tests, control arms, invariant checks,
    and (for a dependency-gated proposal like P7) the upstream-proposal-passed
    check all report through this same shape.
    """

    name: str
    passed: bool
    detail: str = ""


@dataclass
class VerdictReport:
    """A run's complete scientific verdict: spec section 13's 11-point contract.

    Fields map to that contract point-for-point, except the four points spec
    13 asks for that a manifest already carries losslessly — commit/config,
    resources, and artifacts — which live on :class:`~instinct.core.manifest.
    RunManifest` instead of being duplicated here. :func:`render_report_md`
    takes both and stitches them into one document; nothing is lost, nothing
    is repeated in two places that could drift apart.
    """

    question: str
    status: Status
    evidence_level: str  # "E0".."E4", spec section 1.2
    # {"metric": ..., "value": ..., "ci_lo": ..., "ci_hi": ..., "unit": ...}
    primary_result: dict[str, Any]
    baselines_run: list[str] = field(default_factory=list)
    baselines_skipped: dict[str, str] = field(default_factory=dict)  # name -> reason
    controls: list[PrerequisiteCheck] = field(default_factory=list)
    failure_log: list[str] = field(default_factory=list)
    failure_code: FailureCode | None = None
    interpretation: str = ""
    non_claim: str = ""
    decision: str = ""  # exact next experiment, or explicit termination

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "status": self.status.value,
            "evidence_level": self.evidence_level,
            "primary_result": to_canonical(self.primary_result),
            "baselines_run": list(self.baselines_run),
            "baselines_skipped": dict(self.baselines_skipped),
            "controls": [
                {"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.controls
            ],
            "failure_log": list(self.failure_log),
            "failure_code": self.failure_code.value if self.failure_code is not None else None,
            "interpretation": self.interpretation,
            "non_claim": self.non_claim,
            "decision": self.decision,
        }

    @property
    def all_controls_passed(self) -> bool:
        return all(c.passed for c in self.controls)


def write_verdict_json(path: str | Path, report: VerdictReport) -> None:
    Path(path).write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True))


def load_verdict_json(path: str | Path) -> VerdictReport:
    """Reconstruct a :class:`VerdictReport` from a written ``verdict.json``.

    Used by ``instinct report`` to re-render without recomputation, and by a
    dependency-gated proposal (P7 on P2) to read an upstream run's status
    without importing that proposal's code.
    """
    p = Path(path)
    if p.is_dir():
        p = p / VERDICT_NAME
    data: Mapping[str, Any] = json.loads(p.read_text())
    return VerdictReport(
        question=str(data["question"]),
        status=Status(data["status"]),
        evidence_level=str(data["evidence_level"]),
        primary_result=dict(data["primary_result"]),
        baselines_run=list(data.get("baselines_run", [])),
        baselines_skipped=dict(data.get("baselines_skipped", {})),
        controls=[
            PrerequisiteCheck(name=c["name"], passed=bool(c["passed"]), detail=c.get("detail", ""))
            for c in data.get("controls", [])
        ],
        failure_log=list(data.get("failure_log", [])),
        failure_code=FailureCode(data["failure_code"]) if data.get("failure_code") else None,
        interpretation=str(data.get("interpretation", "")),
        non_claim=str(data.get("non_claim", "")),
        decision=str(data.get("decision", "")),
    )


def _fmt_primary_result(primary: Mapping[str, Any]) -> str:
    metric = primary.get("metric", "?")
    value = primary.get("value")
    unit = primary.get("unit", "")
    lo, hi = primary.get("ci_lo"), primary.get("ci_hi")
    value_s = f"{value:.6g}" if isinstance(value, int | float) else str(value)
    ci_s = (
        f" (95% CI [{lo:.6g}, {hi:.6g}])"
        if isinstance(lo, int | float) and isinstance(hi, int | float)
        else ""
    )
    return f"**{metric}** = {value_s}{f' {unit}' if unit else ''}{ci_s}"


def render_report_md(
    report: VerdictReport,
    manifest: RunManifest,
    *,
    metrics_summary: Mapping[str, Any] | None = None,
) -> str:
    """The section-13 template, as markdown. Every number here is a field lookup.

    No value in this function is recomputed from ``metrics_summary`` or
    anything else — it is included only as an optional appendix table, never
    as a source for a number that also appears in ``primary_result`` or the
    manifest. That is what makes report counts provably equal to the raw data
    that produced them, rather than merely usually equal.
    """
    lines: list[str] = [
        f"# {report.question}",
        "",
        f"**Status:** {report.status.value}"
        + (f" ({report.failure_code.value})" if report.failure_code is not None else ""),
        f"**Evidence level:** {report.evidence_level}",
        "",
        "## Commit / config",
        "",
        f"- run id: `{manifest.run_id}`",
        f"- git SHA: `{manifest.provenance.git_sha or '(unavailable)'}`"
        + (" (dirty)" if manifest.provenance.git_dirty else ""),
        f"- config hash: `{manifest.config_hash}`",
        f"- seeds: {manifest.seeds}",
        f"- parent run: `{manifest.parent_run}`" if manifest.parent_run else "- parent run: none",
        "",
        "## Resources",
        "",
        f"- device: {manifest.device or '(not captured)'}",
        f"- wall-clock: {manifest.wall_clock_s:.6g} s" if manifest.wall_clock_s is not None else "",
        f"- dependencies: {manifest.provenance.dep_versions}",
        "",
        "## Primary result",
        "",
        _fmt_primary_result(report.primary_result),
        "",
        "## Baselines",
        "",
        "Run: " + (", ".join(report.baselines_run) if report.baselines_run else "(none)"),
    ]
    if report.baselines_skipped:
        lines.append("")
        lines.append("Skipped:")
        for name, reason in report.baselines_skipped.items():
            lines.append(f"- **{name}**: {reason}")
    lines += ["", "## Controls", ""]
    if report.controls:
        lines.append("| Check | Passed | Detail |")
        lines.append("| --- | --- | --- |")
        for c in report.controls:
            lines.append(f"| {c.name} | {'yes' if c.passed else '**no**'} | {c.detail} |")
    else:
        lines.append("(none recorded)")
    lines += ["", "## Failure log", ""]
    lines += [f"- {entry}" for entry in report.failure_log] if report.failure_log else ["(none)"]
    lines += [
        "",
        "## Interpretation",
        "",
        report.interpretation or "(not recorded)",
        "",
        "## Non-claim",
        "",
        report.non_claim or "(not recorded)",
        "",
        "## Decision",
        "",
        report.decision or "(not recorded)",
    ]
    if metrics_summary:
        lines += ["", "## Metrics summary", ""]
        for k, v in metrics_summary.items():
            lines.append(f"- {k}: {v}")
    lines += ["", "## Artifacts", ""]
    lines += [f"- {a}" for a in manifest.artifacts] if manifest.artifacts else ["(none)"]
    return "\n".join(lines) + "\n"


#: A number literal, not a digit sequence embedded in an identifier. The
#: lookaround excludes both word characters *and* hyphens as neighbors, which
#: is what keeps this from misreading the ``-20260101`` inside a run id like
#: ``dummy-20260101T000000Z-abcd1234`` as a signed integer: the hyphen right
#: before the digits looks exactly like a sign unless hyphen-adjacency is
#: excluded too.
_NUMBER_RE = re.compile(r"(?<![\w-])[-+]?(?:\d*\.\d+|\d+)(?![\w-])")


def numbers_in(text: str) -> list[float]:
    """Every standalone number literal in rendered text, for the report-fidelity test.

    A small utility rather than a private helper: the shared test in
    ``tests/test_core_report.py`` needs exactly this to assert that what got
    rendered is a subset of what the report's fields actually contain. "Number
    literal" excludes digits that are part of an identifier (a run id, a
    config hash) — those are not counts or measurements, and flagging them
    would make this check useless noise rather than a real fidelity guard.
    """
    return [float(m) for m in _NUMBER_RE.findall(text)]
