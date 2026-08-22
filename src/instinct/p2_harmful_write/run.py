from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import numpy as np

from instinct.core.configschema import ProposalRunConfig
from instinct.core.optional import try_import_torch
from instinct.core.proposal import common_checks, resume_by_rerun
from instinct.core.results import ResultsWriter
from instinct.core.rng import SeedScope
from instinct.core.verdict import PrerequisiteCheck, Status, VerdictReport


def validate(config: ProposalRunConfig) -> list[PrerequisiteCheck]:
    checks = common_checks(config, "p2_harmful_write")
    available = try_import_torch() is not None
    stage = str(config.params.get("stage", "p2_0"))
    checks.append(
        PrerequisiteCheck(
            "P2 backend",
            True,
            "realistic Torch/TorchVision preflight"
            if stage == "p2_2_r2"
            else (
                "natural-image numpy pilot"
                if stage in {"p2_2", "p2_2_r1"}
                else (
                    "torch fast-weight oracle"
                    if available
                    else "numpy algebraic E0 fallback"
                )
            ),
        )
    )
    checks.append(
        PrerequisiteCheck(
            "known P2 stage",
            stage in {"p2_0", "p2_1", "p2_2", "p2_2_r1", "p2_2_r2"},
            stage,
        )
    )
    if stage == "p2_1":
        checks.append(
            PrerequisiteCheck(
                "P2.1 stateful torch backend",
                available,
                "required for masked fast-weight replay and iterative rescoring",
            )
        )
    return checks


def _run_p2_2(
    config: ProposalRunConfig,
    writer: ResultsWriter,
    checks: list[PrerequisiteCheck],
) -> VerdictReport:
    """Run the bounded natural-image pilot and fail closed on realistic scale."""

    from instinct.p2_harmful_write.realistic import P2PilotConfig, run_natural_image_pilot

    pilot_cfg = P2PilotConfig(
        n_steps=int(config.params.get("n_steps", 28)),
        patch_size=int(config.params.get("patch_size", 10)),
        horizons=tuple(int(value) for value in config.params.get("horizons", [1, 2, 4])),
        horizon_weights=tuple(
            float(value) for value in config.params.get("horizon_weights", [0.5, 0.3, 0.2])
        ),
        min_age=int(config.params.get("min_age", 4)),
        write_fraction=float(config.params.get("write_fraction", 0.35)),
        learning_rate=float(config.params.get("learning_rate", 0.18)),
        retention_lambda=float(config.params.get("retention_lambda", 1.0)),
        write_cost=float(config.params.get("write_cost", 0.0001)),
        random_replicates=int(config.params.get("random_replicates", 8)),
        correlation_threshold=float(config.params.get("correlation_threshold", 0.25)),
        utility_margin=float(config.params.get("utility_margin", 0.00001)),
    )
    return _finish_p2_2(
        config, writer, checks, pilot_cfg, run_natural_image_pilot
    )


def _run_p2_2_recovery(
    config: ProposalRunConfig,
    writer: ResultsWriter,
    checks: list[PrerequisiteCheck],
) -> VerdictReport:
    """Run recovery cycle 1 without revising the frozen P2 estimand."""

    from instinct.p2_harmful_write.recovery import RecoveryConfig, run_recovery

    recovery_cfg = RecoveryConfig(
        n_steps=int(config.params.get("n_steps", 36)),
        patch_size=int(config.params.get("patch_size", 8)),
        regime_length=int(config.params.get("regime_length", 9)),
        horizons=tuple(int(value) for value in config.params.get("horizons", [1, 2, 4])),
        horizon_weights=tuple(
            float(value) for value in config.params.get("horizon_weights", [0.5, 0.3, 0.2])
        ),
        min_age=int(config.params.get("min_age", 4)),
        m=int(config.params.get("m", 3)),
        learning_rate=float(config.params.get("learning_rate", 0.035)),
        retention_lambda=float(config.params.get("retention_lambda", 1.0)),
        write_cost=float(config.params.get("write_cost", 0.00001)),
        random_replicates=int(config.params.get("random_replicates", 8)),
        correlation_threshold=float(config.params.get("correlation_threshold", 0.25)),
        utility_margin=float(config.params.get("utility_margin", 0.00001)),
    )
    return _finish_p2_2_recovery1(
        config, writer, checks, recovery_cfg, run_recovery
    )


def _run_p2_2_recovery2(
    config: ProposalRunConfig,
    writer: ResultsWriter,
    checks: list[PrerequisiteCheck],
) -> VerdictReport:
    """Fail closed without consuming recovery units when prerequisites are unavailable."""

    from instinct.p2_harmful_write.recovery2 import inspect_recovery2

    scoring_mode = str(config.params.get("scoring_mode", "frozen_reference"))
    preflight = inspect_recovery2(dict(config.params))
    starts = [start for start, _ in preflight.fixed_partitions.values()]
    stops = [stop for _, stop in preflight.fixed_partitions.values()]
    partitions_disjoint = all(
        stops[index] <= starts[index + 1] for index in range(len(starts) - 1)
    )
    checks += [
        PrerequisiteCheck(
            "working torch runtime", preflight.torch_importable, preflight.runtime_detail
        ),
        PrerequisiteCheck("working torchvision runtime", preflight.torchvision_importable),
        PrerequisiteCheck("T4/CUDA runtime", preflight.cuda_available),
        *[
            PrerequisiteCheck(
                f"verified external artifact: {artifact.requirement.name}",
                artifact.passed,
                (
                    f"path={artifact.path!r}, bytes={artifact.observed_bytes}, "
                    f"checksum={artifact.observed_checksum or 'missing'}"
                ),
            )
            for artifact in preflight.artifacts
        ],
        PrerequisiteCheck("fresh fixed whole-image stream partitions", partitions_disjoint),
        PrerequisiteCheck("frozen P2.2 realistic contract", preflight.realistic_contract_passed),
        PrerequisiteCheck("P7 dependency remains locked", not preflight.qualifies_p7),
        PrerequisiteCheck(
            "known P2.2 recovery-2 scoring mode",
            scoring_mode in {"frozen_reference", "f2_repair"},
            scoring_mode,
        ),
    ]
    writer.write_metrics([{**preflight.metric_row(), "scoring_mode": scoring_mode}])
    writer.event(
        stage="p2.2-r2-preflight",
        scoring_mode=scoring_mode,
        realistic_contract_passed=preflight.realistic_contract_passed,
        qualifies_p7=preflight.qualifies_p7,
        runtime_detail=preflight.runtime_detail,
    )
    missing = [
        f"{artifact.requirement.name}: {artifact.requirement.url} "
        f"({artifact.requirement.checksum_algorithm}={artifact.requirement.checksum}, "
        f"bytes={artifact.requirement.expected_bytes})"
        for artifact in preflight.artifacts
        if not artifact.passed
    ]
    reason = "final realistic recovery did not execute: " + "; ".join(missing)
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=Status.INCONCLUSIVE,
        evidence_level="E0",
        primary_result={
            "metric": "realistic_recovery_contract_passed",
            "value": float(preflight.realistic_contract_passed),
            "unit": "boolean",
        },
        baselines_run=[],
        baselines_skipped={
            name: "external artifact/runtime preflight failed before evaluation units were opened"
            for name in (
                "dense adaptation",
                "never write",
                "random matched-count",
                "validation-only",
                "EATA-style filtering",
                "SAR-style reliable update",
                "reset/restore",
                "offline oracle ceiling",
            )
        },
        controls=checks,
        failure_log=[reason, preflight.runtime_detail],
        interpretation=(
            "Recovery 2 attempted the frozen realistic contract and refused a toy fallback. "
            "No candidate, probe, validation, or downstream evaluation unit was consumed."
        ),
        non_claim=config.preregistration.non_claim,
        decision=(
            "Do not count this F0/external preflight as a scientific recovery. Resume the frozen "
            "recovery-2 version only after the exact verified CIFAR-10 archive, official "
            "ResNet-18 weights, compatible Torch/TorchVision runtime, and T4 are available. "
            "Keep P7 locked."
        ),
    )
def _finish_p2_2_recovery1(
    config: ProposalRunConfig,
    writer: ResultsWriter,
    checks: list[PrerequisiteCheck],
    recovery_cfg: Any,
    runner: Any,
) -> VerdictReport:
    """Finish recovery 1 after constructing its config."""

    result = runner(config.seed, recovery_cfg)
    eligible = set(result.eligible_timesteps)
    selected_are_eligible = all(
        set(arm.selected_timesteps) <= eligible
        for arm in result.arms
        if arm.method in {"oracle_ceiling", "validation_only"}
        or arm.method.startswith("random-")
    )
    decomposes = all(
        np.isclose(arm.utility, arm.benefit - arm.damage - arm.cost)
        for arm in result.arms
    )
    checks += [
        PrerequisiteCheck("three packaged natural-image sources", bool(result.source_sha256)),
        PrerequisiteCheck(
            "candidate/oracle/validation/downstream streams disjoint",
            result.streams_disjoint,
        ),
        PrerequisiteCheck("full frozen horizons only", result.full_horizons_only),
        PrerequisiteCheck("benefit-damage-cost decomposition", decomposes),
        PrerequisiteCheck("candidate restriction before subset search", selected_are_eligible),
        PrerequisiteCheck(
            "exact offline oracle subset search",
            result.oracle_subsets_evaluated > 1,
            f"subsets={result.oracle_subsets_evaluated}",
        ),
        PrerequisiteCheck(
            "matched admitted-write count",
            result.matched_count_passed,
            f"m={result.m}",
        ),
        PrerequisiteCheck("matched deployable compute", result.matched_compute_passed),
        PrerequisiteCheck(
            "aged loss predicts downstream retention",
            result.identifiability_passed,
            f"Spearman={result.aged_downstream_spearman:.3f}",
        ),
        PrerequisiteCheck(
            "oracle beats matched-count selectors",
            result.oracle_advantage > recovery_cfg.utility_margin,
            f"advantage={result.oracle_advantage:.6f}",
        ),
        PrerequisiteCheck(
            "oracle improves over never-write MSE",
            result.oracle_gain_over_never_mse > recovery_cfg.utility_margin,
            f"MSE gain={result.oracle_gain_over_never_mse:.6f}",
        ),
        PrerequisiteCheck(
            "frozen P2.2 realistic scale",
            result.realistic_scale_passed,
            "three packaged images and nonlinear color head; CIFAR-C + pretrained model absent",
        ),
        PrerequisiteCheck("P7 dependency remains locked", not result.qualifies_p7),
    ]
    writer.write_metrics(result.metric_rows())
    writer.event(
        stage="p2.2-r1",
        benchmark_id=result.benchmark_id,
        oracle_subsets_evaluated=result.oracle_subsets_evaluated,
        oracle_advantage=result.oracle_advantage,
        oracle_gain_over_never_mse=result.oracle_gain_over_never_mse,
        useful_sparsity=result.useful_sparsity,
        aged_downstream_spearman=result.aged_downstream_spearman,
        qualifies_p7=result.qualifies_p7,
    )
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=Status.INCONCLUSIVE,
        evidence_level="E2",
        primary_result={
            "metric": "recovery_oracle_downstream_utility_advantage",
            "value": result.oracle_advantage,
            "unit": "MSE utility",
        },
        baselines_run=[
            "always write",
            "never write",
            f"random x{recovery_cfg.random_replicates}",
            "validation-only top-m",
            "exact offline oracle subset ceiling",
        ],
        controls=checks,
        interpretation=(
            "Recovery cycle 1 repairs the original identity-head and frozen-ranking "
            "degeneracies without changing U_t. It finds provisional useful sparsity on three "
            "packaged images, but remains below the frozen realistic benchmark contract."
        ),
        non_claim=config.preregistration.non_claim,
        decision=(
            "Preserve both runs and keep P7 locked. Recovery cycle 2, if authorized, must use "
            "a pretrained image model on whole held-out CIFAR-C streams and add EATA/SAR, "
            "reset/restore, and dense matched-wall-clock baselines."
        ),
    )
def _finish_p2_2(
    config: ProposalRunConfig,
    writer: ResultsWriter,
    checks: list[PrerequisiteCheck],
    pilot_cfg: Any,
    runner: Any,
) -> VerdictReport:
    """Finish the original bounded pilot after constructing its typed config."""

    result = runner(config.seed, pilot_cfg)
    selected_are_eligible = all(
        set(arm.selected_timesteps) <= set(result.eligible_timesteps)
        for arm in result.arms
        if arm.method == "oracle_ceiling"
        or arm.method == "validation_only"
        or arm.method.startswith("random-")
    )
    utility_decomposes = all(
        np.isclose(candidate.utility, candidate.benefit - candidate.damage - candidate.cost)
        for candidate in result.candidates
    ) and all(
        np.isclose(arm.utility, arm.benefit - arm.damage - arm.cost)
        for arm in result.arms
    )
    checks += [
        PrerequisiteCheck(
            "natural-image rather than algebraic source",
            bool(result.source_sha256),
            result.benchmark_id,
        ),
        PrerequisiteCheck(
            "candidate/oracle/validation/downstream streams disjoint",
            result.streams_disjoint,
        ),
        PrerequisiteCheck("full frozen horizons only", result.full_horizons_only),
        PrerequisiteCheck("benefit-damage-cost decomposition", utility_decomposes),
        PrerequisiteCheck("candidate restriction before top-m", selected_are_eligible),
        PrerequisiteCheck(
            "matched admitted-write count",
            result.matched_count_passed,
            f"m={result.m}",
        ),
        PrerequisiteCheck("matched deployable compute", result.matched_compute_passed),
        PrerequisiteCheck(
            "aged loss predicts downstream retention",
            result.identifiability_passed,
            f"Spearman={result.aged_downstream_spearman:.3f}",
        ),
        PrerequisiteCheck(
            "frozen P2.2 realistic scale",
            result.realistic_scale_passed,
            "single real photograph and affine head; "
            "pretrained ResNet/ViT benchmark still required",
        ),
        PrerequisiteCheck(
            "P7 dependency remains locked",
            not result.qualifies_p7,
            f"useful_sparsity={result.useful_sparsity:.3f}",
        ),
    ]
    writer.write_metrics(result.metric_rows())
    writer.event(
        stage="p2.2",
        benchmark_id=result.benchmark_id,
        candidates=len(result.candidates),
        eligible_candidates=len(result.eligible_timesteps),
        m=result.m,
        oracle_advantage=result.oracle_advantage,
        useful_sparsity=result.useful_sparsity,
        aged_downstream_spearman=result.aged_downstream_spearman,
        qualifies_p7=result.qualifies_p7,
    )
    status = Status.GO if all(check.passed for check in checks) else Status.INCONCLUSIVE
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=status,
        evidence_level="E2",
        primary_result={
            "metric": "oracle_downstream_utility_advantage",
            "value": result.oracle_advantage,
            "unit": "MSE utility",
        },
        baselines_run=[
            "always write",
            "never write",
            f"random x{pilot_cfg.random_replicates}",
            "validation-only top-m",
            "offline oracle ceiling",
        ],
        controls=checks,
        interpretation=(
            "A deterministic natural-image corruption pilot measured the frozen multi-horizon "
            "utility with separate benefit, damage, and cost. The oracle is evaluation-only. "
            "Any observed sparsity is provisional because the frozen realistic-scale and/or "
            "retention-identifiability gates did not all pass."
        ),
        non_claim=config.preregistration.non_claim,
        decision=(
            "Keep P7 locked. Repeat P2.2 with a pretrained small ResNet/ViT on whole held-out "
            "CIFAR-C streams and re-establish aged-loss/downstream correlation before issuing "
            "a qualification artifact."
        ),
    )


def _auc(utility: np.ndarray, truth: np.ndarray) -> float:
    score = -utility
    pos = score[truth]
    neg = score[~truth]
    return float(
        ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
        / (pos.size * neg.size)
    )


def _stateful_frontier(
    config: ProposalRunConfig,
    *,
    stream: Any,
    fw: Any,
    oracle_cfg: Any,
    frozen: Any,
    partitions: Any,
) -> tuple[Any, Any, Any, list[int]]:
    """Run P2.1 with real fast-weight state transitions for every arm."""

    from instinct.ttt.chunker import build_ledger
    from instinct.ttt.frontier import (
        default_scores,
        iterative_rescore_scores,
        ranking_staleness,
        trace_frontier,
    )

    n_holdout = int(config.params.get("n_holdout_chunks", 4))
    n_replay = len(stream) - n_holdout
    max_horizon = int(frozen.notes["max_horizon"])
    eligible_positions = np.flatnonzero(frozen.t + max_horizon < n_replay)
    eligible_t = frozen.t[eligible_positions]
    if eligible_t.size < 2:
        raise ValueError("P2.1 needs at least two eligible candidates after holdout restriction")

    fractions = [float(x) for x in config.params.get("m_fractions", [0.0, 0.5, 1.0])]
    m_grid = sorted(
        {
            min(eligible_t.size, max(0, round(fraction * eligible_t.size)))
            for fraction in fractions
        }
        | {0, int(eligible_t.size)}
    )
    scores = default_scores(
        frozen,
        n_random=int(config.params.get("n_random", 5)),
        seed=config.seed,
        periodic_every=int(config.params.get("periodic_every", 5)),
    )
    iterative_ledger = build_ledger(
        partitions.oracle_probe,
        rows_per_probe=1,
        aged_budget=int(config.params.get("iterative_probe_budget", 256)),
    )
    iterative = iterative_rescore_scores(
        stream,
        fw,
        oracle_cfg,
        frozen.t,
        eligible_t,
        ledger=iterative_ledger,
        scope=SeedScope(config.seed).child("p2", "iterative"),
        block_size=int(config.params.get("rescore_block_size", 2)),
    )
    scores["oracle_iterative"] = iterative.scores
    stale = ranking_staleness(
        scores["oracle_frozen"], scores["oracle_iterative"], eligible_positions, m_grid
    )

    frontier_ledger = build_ledger(
        partitions.downstream,
        rows_per_probe=1,
        aged_budget=2,
        with_audit=False,
    )
    trace = trace_frontier(
        stream,
        fw,
        frozen,
        scores,
        m_grid,
        ledger=frontier_ledger,
        scope=SeedScope(config.seed).child("p2", "frontier"),
        evaluation_stream=partitions.downstream,
        n_holdout_chunks=n_holdout,
        n_retention_probes=int(config.params.get("n_retention_probes", 12)),
        retention_min_age=int(config.params.get("min_age", 4)),
        rows_per_probe=1,
        retention_lambda=oracle_cfg.retention_lambda,
        write_cost=float(config.params.get("write_cost", 0.001)),
        score_cost=float(config.params.get("score_cost", 0.0)),
        anchor_strength=float(config.params.get("anchor_strength", 0.1)),
        anchor_cost=float(config.params.get("anchor_cost", 0.0001)),
        reset_every=int(config.params.get("reset_every", 5)),
        reset_cost=float(config.params.get("reset_cost", 0.0001)),
        oracle_forward_calls={
            "oracle_frozen": frozen.n_forward_calls,
            "oracle_iterative": iterative.oracle_forward_calls,
        },
    )
    return trace, iterative, stale, m_grid


def run(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    checks = validate(config)
    stage = str(config.params.get("stage", "p2_0"))
    if stage == "p2_2_r2":
        return _run_p2_2_recovery2(config, writer, checks)
    if stage == "p2_2_r1":
        return _run_p2_2_recovery(config, writer, checks)
    if stage == "p2_2":
        return _run_p2_2(config, writer, checks)
    if try_import_torch() is not None:
        from instinct.ttt.chunker import build_ledger
        from instinct.ttt.fastweight import FastWeightConfig
        from instinct.ttt.frontier import split_stream_rows
        from instinct.ttt.oracle import OracleConfig, run_oracle
        from instinct.ttt.streams import drifting_regression_stream

        source_stream = drifting_regression_stream(
            d_in=5,
            d_out=3,
            n_chunks=int(config.params.get("n_chunks", 24)),
            chunk_size=int(config.params.get("chunk_size", 9)),
            drift=0.05,
            conflict_every=int(config.params.get("conflict_every", 13)),
            conflict_offset=3,
            seed=config.seed,
        )
        partitions = split_stream_rows(source_stream)
        stream = partitions.candidate
        fw = FastWeightConfig(d_in=5, d_hidden=12, d_out=3, lr=0.1)
        oracle_cfg = OracleConfig(
            horizons=tuple(int(h) for h in config.params.get("horizons", [1, 2, 4])),
            min_age=int(config.params.get("min_age", 4)),
            n_aged_probes=int(config.params.get("n_aged_probes", 4)),
            retention_lambda=float(config.params.get("retention_lambda", 1.0)),
            cost=float(config.params.get("write_cost", 0.001)),
        )
        fast = run_oracle(
            stream,
            fw,
            oracle_cfg,
            SeedScope(config.seed).child("p2"),
            ledger=build_ledger(
                partitions.oracle_probe, rows_per_probe=1, aged_budget=8
            ),
        )
        naive = run_oracle(
            stream,
            fw,
            oracle_cfg,
            SeedScope(config.seed).child("p2"),
            ledger=build_ledger(
                partitions.oracle_probe, rows_per_probe=1, aged_budget=8
            ),
            naive=True,
        )
        arrays = {name: getattr(fast, name) for name in ("utility", "benefit", "damage", "cost")}
        truth, timesteps = fast.harmful_truth, fast.t
        equivalent = all(np.allclose(arrays[f], getattr(naive, f), atol=1e-9) for f in arrays)
    else:
        # CPU-only algebraic E0 control. Sign-flipped candidates have known
        # retention damage; the two expressions are deliberately independent.
        timesteps = np.arange(20, dtype=np.int64)
        truth = timesteps % 7 == 3
        benefit = np.where(truth, -0.4, 0.3 + 0.01 * timesteps)
        damage = np.where(truth, 1.2, 0.05)
        cost = np.full(timesteps.size, 0.01)
        utility = benefit - damage - cost
        naive_utility = np.array([benefit[i] - damage[i] - cost[i] for i in range(len(timesteps))])
        arrays = {"utility": utility, "benefit": benefit, "damage": damage, "cost": cost}
        equivalent = np.array_equal(utility, naive_utility)
        partitions = None
        fast = None
        fw = None
        oracle_cfg = None
        stream = None
    auc = _auc(arrays["utility"], truth)
    checks += [
        PrerequisiteCheck("planted positives and negatives", bool(truth.any() and (~truth).any())),
        PrerequisiteCheck("batched equals naive", equivalent),
        PrerequisiteCheck("planted-harm ranking", auc >= 0.8, f"AUC={auc:.3f}"),
    ]
    candidates = [
        {
            "record_type": "candidate",
            "t": int(t),
            "utility": float(u),
            "benefit": float(b),
            "damage": float(d),
            "cost": float(c),
            "harmful_truth": bool(h),
        }
        for t, u, b, d, c, h in zip(
            timesteps,
            arrays["utility"],
            arrays["benefit"],
            arrays["damage"],
            arrays["cost"],
            truth,
        )
    ]
    metrics: list[dict[str, Any]] = candidates
    oracle_advantage = float("nan")
    staleness_shift = float("nan")
    m_grid: list[int] = []
    if stage == "p2_1" and all(x is not None for x in (fast, fw, oracle_cfg, partitions, stream)):
        assert fast is not None
        assert fw is not None
        assert oracle_cfg is not None
        assert partitions is not None
        assert stream is not None
        trace, iterative, stale, m_grid = _stateful_frontier(
            config,
            stream=stream,
            fw=fw,
            oracle_cfg=oracle_cfg,
            frozen=fast,
            partitions=partitions,
        )
        required_methods = {
            "oracle_frozen",
            "oracle_iterative",
            "surprise",
            "update_norm",
            "gradient_alignment",
            "periodic",
            "elastic_anchor",
            "periodic_reset",
            "no_write",
            "dense_all",
        }
        method_set = set(trace.methods)
        random_present = any(name.startswith("random-") for name in method_set)
        matched = all(point.n_writes == point.m for point in trace.points)
        full_horizons = bool(fast.notes.get("full_horizons_only"))
        checks += [
            PrerequisiteCheck("P2 streams row-disjoint", True),
            PrerequisiteCheck("full frozen horizons only", full_horizons),
            PrerequisiteCheck("candidate restriction before top-m", bool(trace.eligible_timesteps)),
            PrerequisiteCheck("P2.1 exact matched write count", matched),
            PrerequisiteCheck(
                "all required stateful arms",
                required_methods <= method_set and random_present,
                ",".join(trace.methods),
            ),
            PrerequisiteCheck(
                "iterative oracle actually rescored",
                iterative.rescoring_rounds >= 2 and iterative.oracle_forward_calls > 0,
                f"rounds={iterative.rescoring_rounds}",
            ),
        ]
        staleness_shift = stale.mean_normalized_rank_shift
        interior = [m for m in m_grid if 0 < m < len(trace.eligible_timesteps)]
        if interior:
            chosen_m = interior[len(interior) // 2]
            at_m = [point for point in trace.points if point.m == chosen_m]
            oracle_utility = max(
                point.utility
                for point in at_m
                if point.method in {"oracle_frozen", "oracle_iterative"}
            )
            baseline_utility = max(
                point.utility
                for point in at_m
                if point.method not in {"oracle_frozen", "oracle_iterative"}
            )
            oracle_advantage = oracle_utility - baseline_utility
        checks.append(
            PrerequisiteCheck(
                "stateful interior write budget",
                bool(interior),
                f"m_grid={m_grid}",
            )
        )
        for point in trace.points:
            row = asdict(point)
            row["record_type"] = "stateful_frontier"
            row["selected_timesteps"] = ",".join(str(t) for t in point.selected_timesteps)
            metrics.append(row)
        metrics.extend(
            {
                "record_type": "ranking_staleness",
                "m": m,
                "top_m_gap": gap,
                "mean_normalized_rank_shift": staleness_shift,
                "rescoring_rounds": iterative.rescoring_rounds,
                "iterative_oracle_forward_calls": iterative.oracle_forward_calls,
                "iterative_probe_rows_spent": iterative.probe_rows_spent,
            }
            for m, gap in stale.top_m_gap
        )
        writer.event(
            stage="p2.1",
            candidates=len(timesteps),
            eligible_candidates=len(trace.eligible_timesteps),
            auc=auc,
            oracle_staleness_rank_shift=staleness_shift,
            oracle_advantage=oracle_advantage,
        )
    else:
        writer.event(stage="p2.0", candidates=len(timesteps), auc=auc)

    writer.write_metrics(metrics)
    passed = all(c.passed for c in checks)
    if not passed:
        status = Status.INCONCLUSIVE
    elif stage == "p2_0" or (np.isfinite(oracle_advantage) and oracle_advantage > 0):
        status = Status.GO
    else:
        # All instruments passed, but a required simple baseline matched or
        # beat the offline ceiling on the stateful replay: a valid mechanism
        # failure, not an estimator failure and not a deployable success.
        status = Status.STOP
    primary = (
        {"metric": "stateful_oracle_utility_advantage", "value": oracle_advantage, "unit": "loss"}
        if stage == "p2_1"
        else {"metric": "planted_harm_auc", "value": auc, "unit": "AUC"}
    )
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=status,
        evidence_level="E1",
        primary_result=primary,
        baselines_run=[
            "naive clone/apply/rollback",
            "batched fork",
            "surprise",
            "update norm",
            "gradient alignment",
            "periodic",
            "random x5",
            "dense/all",
            "no writes",
            "elastic anchoring",
            "periodic reset",
            "iteratively rescored oracle",
        ],
        controls=checks,
        interpretation=(
            f"Synthetic {stage} measurement and stateful frontier validation only; "
            "offline oracle costs are reported separately and no oracle is a deployable trigger. "
            "This is not evidence of realistic useful sparsity or matched-compute deployment gain."
        ),
        non_claim=config.preregistration.non_claim,
        decision=(
            "Run P2.2 on a realistic frozen stream before any deployable gate."
            if status == Status.GO
            else (
                "Archive this P2.1 experiment version: a required baseline matched the ceiling."
                if status == Status.STOP
                else "Repair the oracle estimator or an invariant before interpretation."
            )
        ),
    )


def resume(
    config: ProposalRunConfig, writer: ResultsWriter, checkpoint_state: Mapping[str, Any]
) -> VerdictReport:
    return resume_by_rerun(config, writer, checkpoint_state, run)
