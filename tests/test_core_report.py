"""Spec section 3.4: "report counts and units agree with raw data."

``render_report_md`` must never introduce a number that is not already a field
on the ``VerdictReport``/``RunManifest`` it was given. This checks that
property directly: every number rendered into the markdown is a member of the
set of numbers actually present on the two source objects.
"""

from __future__ import annotations

import re

from instinct.core.manifest import Provenance, RunManifest
from instinct.core.verdict import (
    PrerequisiteCheck,
    Status,
    VerdictReport,
    numbers_in,
    render_report_md,
)


def _manifest() -> RunManifest:
    return RunManifest(
        run_id="dummy-20260101T000000Z-abcd1234",
        proposal="dummy",
        config_hash="abcd1234ef",
        config={"seed": 7},
        provenance=Provenance(
            git_sha="deadbeef",
            git_branch="stage-a/substrate",
            git_dirty=False,
            host="test-host",
            platform="test-platform",
            python="3.11.0",
            dep_versions={"numpy": "1.26.0"},
        ),
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:12+00:00",
        wall_clock_s=12.345,
        status="ok",
        device={"cpu": "test-cpu", "cpu_count": 4},
        seeds=[7],
        artifacts=["metrics"],
    )


def _verdict() -> VerdictReport:
    return VerdictReport(
        question="does the fixture reach GO?",
        status=Status.GO,
        evidence_level="E0",
        primary_result={
            "metric": "final_value",
            "value": 7042.0,
            "ci_lo": 7000.0,
            "ci_hi": 7100.0,
            "unit": "units",
        },
        baselines_run=["baseline_a"],
        controls=[PrerequisiteCheck(name="rows written", passed=True, detail="5 of 5")],
        interpretation="fixture reached its target",
        non_claim="nothing about real proposals",
        decision="none",
    )


def _section(text: str, heading: str) -> str:
    """The body of one ``## heading`` section, up to the next ``##`` or EOF."""
    start = text.index(f"## {heading}")
    rest = text[start:]
    nxt = rest.find("\n## ", 1)
    return rest if nxt == -1 else rest[:nxt]


def test_every_number_in_the_primary_result_traces_to_its_source_field() -> None:
    """Spec section 3.4's actual target: the *quantitative claim* — value and
    CI — must render exactly, not the metadata sections (dependency versions,
    hashes, dates) that happen to also contain digits but carry no measured
    quantity for this rule to check."""
    manifest = _manifest()
    report = _verdict()
    text = render_report_md(report, manifest)
    primary_section = _section(text, "Primary result")

    source_numbers = {
        float(report.primary_result["value"]),
        float(report.primary_result["ci_lo"]),
        float(report.primary_result["ci_hi"]),
    }
    # "95% CI" is a fixed template label (the confidence level), not a value
    # drawn from any report field, so it is stripped before scanning rather
    # than added as a spurious extra "source number" to match against.
    found = numbers_in(re.sub(r"\d+%", "", primary_section))
    assert found, "primary result section rendered no numbers at all"
    for n in found:
        assert any(abs(n - s) < 1e-9 for s in source_numbers), f"{n} does not trace to a source field"
    for s in source_numbers:
        assert any(abs(n - s) < 1e-9 for n in found), f"source value {s} never appears in the report"


def test_wall_clock_renders_at_full_precision_not_rounded_away() -> None:
    manifest = _manifest()
    report = _verdict()
    text = render_report_md(report, manifest)
    resources_section = _section(text, "Resources")
    found = numbers_in(resources_section)
    assert any(abs(n - manifest.wall_clock_s) < 1e-9 for n in found), (
        f"wall_clock_s={manifest.wall_clock_s} does not appear at full precision in {found}"
    )


def test_primary_result_value_appears_verbatim_in_the_report() -> None:
    report = _verdict()
    text = render_report_md(report, _manifest())
    assert "7042" in text
    assert "final_value" in text


def test_status_and_failure_code_both_render() -> None:
    from instinct.core.verdict import FailureCode

    report = _verdict()
    report.status = Status.STOP
    report.failure_code = FailureCode.F4_BASELINE_DOMINANCE
    text = render_report_md(report, _manifest())
    assert "STOP" in text
    assert "F4" in text


def test_non_claim_and_decision_are_rendered_verbatim() -> None:
    report = _verdict()
    text = render_report_md(report, _manifest())
    assert report.non_claim in text
    assert report.interpretation in text
