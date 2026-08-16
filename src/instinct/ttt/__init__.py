"""Proposal 2, phase zero: the exact-fork oracle for harmful test-time writes.

The question this package exists to answer is whether a test-time memory can
tell useful plasticity from destructive plasticity, before or shortly after
committing a fast-weight update. Nothing here learns a gate. Phase zero measures
what a gate would have to predict, exactly, by brute force:

    :mod:`~instinct.ttt.streams`        streams whose harmful writes are built
                                        by construction, so the oracle has a
                                        known answer to be checked against
    :mod:`~instinct.ttt.fastweight`     branch-major SwiGLU fast weights with
                                        bitwise fork and rollback
    :mod:`~instinct.ttt.chunker`        chunking, and assembling the aged set
    :mod:`~instinct.ttt.probes`         the use ledger: finite-budget holdouts
                                        that retire when spent
    :mod:`~instinct.ttt.oracle`         clone, apply, evaluate both branches,
                                        compute ``U_t``, roll back
    :mod:`~instinct.ttt.frontier`       retention vs plasticity at matched
                                        write count
    :mod:`~instinct.ttt.decision_tree`  the four-way rule that selects phase two

Torch is imported at module scope here, so this package sits behind the ``nn``
extra like everything else neural in the repo.
"""

from __future__ import annotations

from instinct.ttt.chunker import build_aged_set, build_ledger, chunk_to_probes, future_chunks
from instinct.ttt.decision_tree import (
    ClusterReport,
    DecisionReport,
    DecisionThresholds,
    Outcome,
    PredictReport,
    classify,
    damage_clustering,
    held_out_rank_score,
)
from instinct.ttt.fastweight import (
    FastWeightConfig,
    FastWeightDelta,
    FastWeights,
    FastWeightSnapshot,
    chunk_delta,
    row_losses,
)
from instinct.ttt.frontier import (
    FrontierPoint,
    FrontierTrace,
    default_scores,
    top_m_mask,
    trace_frontier,
)
from instinct.ttt.oracle import (
    POST_FEATURES,
    PRE_FEATURES,
    OracleConfig,
    OracleResult,
    TimestepPlan,
    build_plans,
    evaluate_plan_naive,
    evaluate_plans_batched,
    run_oracle,
)
from instinct.ttt.probes import Probe, ProbeExhaustedError, ProbeLedger, UsageReport
from instinct.ttt.streams import (
    Chunk,
    SyntheticStream,
    associative_recall_stream,
    drifting_regression_stream,
)

__all__ = [
    "POST_FEATURES",
    "PRE_FEATURES",
    "Chunk",
    "ClusterReport",
    "DecisionReport",
    "DecisionThresholds",
    "FastWeightConfig",
    "FastWeightDelta",
    "FastWeightSnapshot",
    "FastWeights",
    "FrontierPoint",
    "FrontierTrace",
    "OracleConfig",
    "OracleResult",
    "Outcome",
    "PredictReport",
    "Probe",
    "ProbeExhaustedError",
    "ProbeLedger",
    "SyntheticStream",
    "TimestepPlan",
    "UsageReport",
    "associative_recall_stream",
    "build_aged_set",
    "build_ledger",
    "build_plans",
    "chunk_delta",
    "chunk_to_probes",
    "classify",
    "damage_clustering",
    "default_scores",
    "drifting_regression_stream",
    "evaluate_plan_naive",
    "evaluate_plans_batched",
    "future_chunks",
    "held_out_rank_score",
    "row_losses",
    "run_oracle",
    "top_m_mask",
    "trace_frontier",
]
