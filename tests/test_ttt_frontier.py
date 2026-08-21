"""P2.1 stateful frontier invariants and adversarial selection controls."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from instinct.core.rng import SeedScope  # noqa: E402
from instinct.ttt.chunker import build_ledger  # noqa: E402
from instinct.ttt.fastweight import FastWeightConfig  # noqa: E402
from instinct.ttt.frontier import (  # noqa: E402
    default_scores,
    iterative_rescore_scores,
    ranking_staleness,
    split_stream_rows,
    trace_frontier,
)
from instinct.ttt.oracle import OracleConfig, run_oracle  # noqa: E402
from instinct.ttt.streams import drifting_regression_stream  # noqa: E402

pytestmark = pytest.mark.needs_torch


def _fixture(n_chunks: int = 24):
    source = drifting_regression_stream(
        d_in=5,
        d_out=3,
        n_chunks=n_chunks,
        chunk_size=9,
        drift=0.05,
        conflict_every=13,
        conflict_offset=3,
        seed=17,
    )
    parts = split_stream_rows(source)
    fw = FastWeightConfig(d_in=5, d_hidden=10, d_out=3, lr=0.1)
    cfg = OracleConfig(horizons=(1, 2), min_age=3, n_aged_probes=2, cost=0.001)
    result = run_oracle(
        parts.candidate,
        fw,
        cfg,
        SeedScope(17).child("frozen"),
        ledger=build_ledger(parts.oracle_probe, rows_per_probe=1, aged_budget=32),
    )
    return parts, fw, cfg, result


def _trace(parts, fw, result, scores, m_grid):
    return trace_frontier(
        parts.candidate,
        fw,
        result,
        scores,
        m_grid,
        ledger=build_ledger(parts.downstream, rows_per_probe=1, aged_budget=2, with_audit=False),
        scope=SeedScope(18).child("frontier"),
        evaluation_stream=parts.downstream,
        n_holdout_chunks=3,
        n_retention_probes=6,
        retention_min_age=3,
        rows_per_probe=1,
        write_cost=0.001,
        anchor_cost=0.0001,
        reset_every=4,
        reset_cost=0.0001,
    )


def test_stream_partitions_are_row_disjoint_and_time_aligned() -> None:
    parts, _, _, _ = _fixture()
    parts.assert_disjoint()
    assert not (parts.candidate_units & parts.oracle_probe_units)
    assert not (parts.candidate_units & parts.downstream_units)
    assert [c.t for c in parts.candidate.chunks] == [c.t for c in parts.downstream.chunks]
    assert all(c.n_rows == 3 for c in parts.candidate.chunks)


def test_oracle_candidates_always_have_the_full_frozen_horizon() -> None:
    parts, _, cfg, result = _fixture()
    assert bool(result.notes["full_horizons_only"])
    assert int(result.t.max()) + max(cfg.horizons) < len(parts.candidate)


def test_candidate_restriction_happens_before_top_m() -> None:
    parts, fw, _, result = _fixture()
    # Make the latest candidates overwhelmingly attractive. They are outside
    # the replay window once the downstream tail and full horizons are reserved.
    adversarial = np.arange(len(result), dtype=np.float64)
    trace = _trace(parts, fw, result, {"adversarial": adversarial}, (2,))
    point = trace.by_method("adversarial")[0]
    assert point.n_writes == 2
    assert set(point.selected_timesteps) <= set(trace.eligible_timesteps)
    excluded = set(map(int, result.t)) - set(trace.eligible_timesteps)
    assert excluded
    assert not (set(point.selected_timesteps) & excluded)


def test_stateful_frontier_has_required_arms_and_separate_utility_terms() -> None:
    parts, fw, _, result = _fixture()
    scores = default_scores(result, n_random=2, seed=19, periodic_every=4)
    trace = _trace(parts, fw, result, scores, (0, 2))
    required = {
        "oracle_frozen",
        "surprise",
        "update_norm",
        "gradient_alignment",
        "periodic",
        "elastic_anchor",
        "periodic_reset",
        "random-0",
        "no_write",
        "dense_all",
    }
    assert required <= set(trace.methods)
    assert all(point.n_writes == point.m for point in trace.points)
    for point in trace.points:
        assert point.utility == pytest.approx(point.benefit - point.damage - point.cost)
        assert point.grad_steps == trace.n_replay_chunks
        assert point.score_evaluations == len(trace.eligible_timesteps)
    no_write = trace.by_method("no_write")[0]
    assert no_write.benefit == pytest.approx(0.0)
    assert no_write.damage == pytest.approx(0.0)
    assert trace.by_method("elastic_anchor")[-1].anchor_ops == 2
    assert trace.by_method("periodic_reset")[-1].reset_ops > 0


def test_iterative_oracle_recomputes_after_selected_blocks() -> None:
    parts, fw, cfg, result = _fixture()
    n_replay, max_horizon = len(parts.candidate) - 3, max(cfg.horizons)
    eligible = result.t[result.t + max_horizon < n_replay]
    ranking = iterative_rescore_scores(
        parts.candidate,
        fw,
        cfg,
        result.t,
        eligible,
        ledger=build_ledger(parts.oracle_probe, rows_per_probe=1, aged_budget=128),
        scope=SeedScope(20).child("iterative"),
        block_size=3,
    )
    assert ranking.rescoring_rounds == int(np.ceil(len(eligible) / 3))
    assert ranking.oracle_forward_calls >= ranking.rescoring_rounds
    assert set(ranking.selected_timesteps) == set(map(int, eligible))
    eligible_positions = np.flatnonzero(np.isin(result.t, eligible))
    assert np.isfinite(ranking.scores[eligible_positions]).all()


def test_ranking_staleness_reports_rank_and_top_m_changes() -> None:
    frozen = np.array([4.0, 3.0, 2.0, 1.0])
    iterative = frozen[::-1]
    report = ranking_staleness(frozen, iterative, np.arange(4), (0, 2, 4))
    assert report.mean_normalized_rank_shift > 0.5
    assert dict(report.top_m_gap) == {0: 0.0, 2: 1.0, 4: 0.0}
