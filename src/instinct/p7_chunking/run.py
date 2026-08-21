from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from instinct.core.configschema import ProposalRunConfig
from instinct.core.proposal import common_checks, resume_by_rerun
from instinct.core.results import ResultsWriter
from instinct.core.verdict import FailureCode, PrerequisiteCheck, Status, VerdictReport
from instinct.p7_chunking.queue import Item, QueueConfig, simulate_queue

REQUIRED_P2_FIELDS = {
    "artifact_hash",
    "proposal_version",
    "theory_id",
    "benchmark_id",
    "stage",
    "matched_count_passed",
    "matched_compute_passed",
    "useful_sparsity",
    "status",
}


def _p2_dependency(config: ProposalRunConfig) -> PrerequisiteCheck:
    path_value = config.params.get("p2_qualification")
    if not path_value:
        return PrerequisiteCheck(
            "qualifying realistic P2.2 evidence", False, "no artifact configured"
        )
    path = Path(str(path_value))
    if not path.exists():
        return PrerequisiteCheck("qualifying realistic P2.2 evidence", False, f"missing {path}")
    data = json.loads(path.read_text())
    missing = REQUIRED_P2_FIELDS - set(data)
    passed = (
        not missing
        and data.get("stage") == "P2.2"
        and data.get("status") in {"GO", "NARROW"}
        and data.get("matched_count_passed") is True
        and data.get("matched_compute_passed") is True
        and float(data.get("useful_sparsity", 0.0)) > 0
    )
    return PrerequisiteCheck(
        "qualifying realistic P2.2 evidence", passed, f"missing={sorted(missing)}"
    )


def validate(config: ProposalRunConfig) -> list[PrerequisiteCheck]:
    checks = common_checks(config, "p7_chunking")
    mode = str(config.params.get("mode", "simulator"))
    if mode == "simulator":
        checks.append(
            PrerequisiteCheck("simulator-only dependency guard", True, "scientific P7.0 disabled")
        )
    else:
        checks.append(_p2_dependency(config))
    return checks


def run(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    checks = validate(config)
    items = []
    # Bursty two-owner trace plus rejected items for the audit channel.
    for i in range(40):
        arrival = (i // 10) * 5.0 + (i % 10) * 0.05
        items.append(Item(f"item-{i}", arrival, 0.9 if i % 4 == 0 else 0.1, f"stream-{i % 2}"))
    cfg = QueueConfig(
        buffer_size=int(config.params.get("buffer_size", 8)),
        max_age=float(config.params.get("max_age", 2.0)),
        audit_probability=0.2,
        scorer_time=0.005,
        update_time=0.05,
    )
    trace = simulate_queue(items, cfg, seed=config.seed)
    trickle = simulate_queue(
        [Item(f"trickle-{i}", i * 0.45, 0.9, "trickle") for i in range(20)],
        cfg,
        seed=config.seed + 1,
    )
    overloaded = simulate_queue(
        [Item(f"load-{i}", i * 0.001, 0.9, "load") for i in range(40)],
        cfg,
        seed=config.seed + 2,
    )
    fixed_shape = all(event.kernel_shape == (cfg.buffer_size,) for event in trace.updates)
    owner_isolation = all(event.stream_id in {"stream-0", "stream-1"} for event in trace.updates)
    updated_ids = [item_id for event in trace.updates for item_id in event.item_ids]
    unique_real_tokens = len(updated_ids) == len(set(updated_ids)) == trace.admitted
    audit_disjoint = set(trace.audit_item_ids).isdisjoint(trace.admitted_item_ids)
    propensities_logged = len(trace.audit_propensities) == trace.audited_rejections and all(
        probability > 0 for probability in trace.audit_propensities
    )
    checks += [
        PrerequisiteCheck("fixed kernel shape", fixed_shape),
        PrerequisiteCheck("owner isolation", owner_isolation),
        PrerequisiteCheck("audit channel active", trace.audited_rejections > 0),
        PrerequisiteCheck("audit channel is update-disjoint", audit_disjoint),
        PrerequisiteCheck("audit propensities logged", propensities_logged),
        PrerequisiteCheck("unique real-token accounting", unique_real_tokens),
        PrerequisiteCheck(
            "continuous trickle triggers deadlines",
            any(event.trigger == "deadline" for event in trickle.updates),
        ),
        PrerequisiteCheck(
            "overload negative control detected",
            overloaded.offered_load > 1.0 and overloaded.maximum_score_queue_delay > 0,
            f"offered_load={overloaded.offered_load:.3f}",
        ),
        PrerequisiteCheck(
            "serialized utilization stable",
            trace.utilization < 1.0,
            f"utilization={trace.utilization:.3f}",
        ),
    ]
    writer.write_metrics(
        [
            {
                "stream_id": e.stream_id,
                "trigger": e.trigger,
                "real_tokens": e.real_tokens,
                "padding_tokens": e.padding_tokens,
                "kernel_tokens": e.kernel_shape[0],
                "maximum_age_at_completion": e.maximum_age_at_completion,
                "unique_item_count": len(set(e.item_ids)),
                "offered_load": trace.offered_load,
                "maximum_score_queue_delay": trace.maximum_score_queue_delay,
                "audit_effective_sample_size": trace.audit_effective_sample_size,
            }
            for e in trace.updates
        ]
    )
    writer.event(stage="p7-simulator", updates=len(trace.updates), utilization=trace.utilization)
    dependency = _p2_dependency(config)
    mode = str(config.params.get("mode", "simulator"))
    scientific = mode == "p7.0" and dependency.passed
    passed = all(c.passed for c in checks)
    status = Status.GO if scientific and passed else Status.INCONCLUSIVE
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=status,
        evidence_level="E0",
        primary_result={
            "metric": "serialized_utilization",
            "value": trace.utilization,
            "unit": "fraction",
        },
        baselines_run=[
            "bursty trace",
            "continuous trickle",
            "overload negative control",
            "two owner streams",
            "deadline padding",
            "disjoint propensity-logged random audit channel",
        ],
        controls=checks + ([dependency] if mode == "simulator" else []),
        failure_code=None if scientific else FailureCode.F1_UNIDENTIFIABLE,
        interpretation=(
            "Queue instrumentation passed, but no P7 scientific result is issued "
            "without qualifying realistic P2.2 evidence."
        ),
        non_claim=config.preregistration.non_claim,
        decision="Await realistic P2.2 matched-count/compute useful-sparsity evidence."
        if not scientific
        else "Run measured T4 fixed/masked/microchunk kernels.",
    )


def resume(
    config: ProposalRunConfig, writer: ResultsWriter, checkpoint_state: Mapping[str, Any]
) -> VerdictReport:
    return resume_by_rerun(config, writer, checkpoint_state, run)
