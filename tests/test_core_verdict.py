"""Status/FailureCode/VerdictReport: round-trip and the "never upgraded" property."""

from __future__ import annotations

from instinct.core.verdict import (
    FailureCode,
    PrerequisiteCheck,
    Status,
    VerdictReport,
    load_verdict_json,
    write_verdict_json,
)


def _report(status: Status) -> VerdictReport:
    return VerdictReport(
        question="does the widget improve on baseline?",
        status=status,
        evidence_level="E2",
        primary_result={"metric": "regret", "value": 0.0421, "ci_lo": 0.01, "ci_hi": 0.07, "unit": "return units"},
        baselines_run=["fixed_budget", "lookup"],
        baselines_skipped={"learned_gate": "no pretrained checkpoint available on T4"},
        controls=[PrerequisiteCheck(name="known-answer test", passed=True, detail="matches exact solve")],
        failure_log=["shard 3 timed out once, retried ok"],
        failure_code=FailureCode.F3_MECHANISM_ABSENT if status == Status.STOP else None,
        interpretation="the effect is real but small",
        non_claim="does not establish transfer to unseen environments",
        decision="run the T4 pilot with 5 seeds",
    )


def test_all_four_statuses_are_distinct_values() -> None:
    values = {s.value for s in Status}
    assert values == {"GO", "NARROW", "STOP", "INCONCLUSIVE"}


def test_verdict_round_trips_through_json(tmp_path) -> None:
    report = _report(Status.GO)
    path = tmp_path / "verdict.json"
    write_verdict_json(path, report)
    loaded = load_verdict_json(path)

    assert loaded.status == report.status
    assert loaded.question == report.question
    assert loaded.evidence_level == report.evidence_level
    assert loaded.primary_result == report.primary_result
    assert loaded.baselines_run == report.baselines_run
    assert loaded.baselines_skipped == report.baselines_skipped
    assert [c.name for c in loaded.controls] == [c.name for c in report.controls]
    assert loaded.failure_code == report.failure_code
    assert loaded.interpretation == report.interpretation
    assert loaded.non_claim == report.non_claim
    assert loaded.decision == report.decision


def test_inconclusive_round_trips_as_inconclusive_never_upgraded(tmp_path) -> None:
    """The literal property spec section 2.1 asks for: no code path here can
    turn an INCONCLUSIVE report into anything else on its way to disk and back."""
    report = _report(Status.INCONCLUSIVE)
    path = tmp_path / "verdict.json"
    write_verdict_json(path, report)
    assert load_verdict_json(path).status == Status.INCONCLUSIVE


def test_stop_carries_a_failure_code_when_the_caller_supplies_one() -> None:
    report = _report(Status.STOP)
    assert report.failure_code == FailureCode.F3_MECHANISM_ABSENT


def test_go_has_no_failure_code_by_default() -> None:
    report = _report(Status.GO)
    assert report.failure_code is None


def test_all_controls_passed_reflects_every_check() -> None:
    passing = VerdictReport(
        question="q",
        status=Status.GO,
        evidence_level="E0",
        primary_result={},
        controls=[PrerequisiteCheck("a", True), PrerequisiteCheck("b", True)],
    )
    failing = VerdictReport(
        question="q",
        status=Status.STOP,
        evidence_level="E0",
        primary_result={},
        controls=[PrerequisiteCheck("a", True), PrerequisiteCheck("b", False)],
    )
    assert passing.all_controls_passed
    assert not failing.all_controls_passed
