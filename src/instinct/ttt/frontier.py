"""The retention-plasticity frontier, traced at matched write count.

The claim phase zero has to support is a *quality* claim: that knowing which
writes are destructive buys retention you could not have bought by simply
writing less. Writing less is the confound, and it is a strong one — almost any
selection rule improves retention if it is allowed to admit fewer writes than
the baseline, and almost any rule improves plasticity if it is allowed to admit
more. A comparison that does not hold the write count fixed measures write rate
and reports it as write quality.

So every curve here is indexed by ``m``, the number of admitted writes, and at
each ``m`` the arms differ *only* in which ``m`` they picked:

``oracle``    the top ``m`` by the exact-fork ``U_t``. Not achievable online —
              it is the ceiling the whole program is trying to approach.
``surprise``  the top ``m`` by pre-write chunk loss. The obvious cheap rule, and
              the one a reader will ask about.
``random-k``  ``m`` picked uniformly. The floor that says whether ranking
              mattered at all, run at several seeds because a single random
              draw is a coin flip, not a baseline.

Compute is matched in the strong sense, not the counting sense. Every arm is a
branch of one replay: every branch computes the candidate write at *every*
chunk, and rejection is a masked add rather than a skipped step. So the arms
share their forwards and backwards exactly, and the only thing that differs is
whether a tensor got added to. :func:`FrontierTrace.assert_matched` checks it
afterwards rather than trusting the description.

The gap this design does not close, stated plainly: ``U_t`` was measured along
the write-all reference trajectory, and an arm that admits only ``m`` writes
follows a different trajectory. The oracle ranking is therefore exact for the
path it was measured on and approximate for the path it selects. Fixing that
needs a re-scored inner loop per ``m`` (quadratic in stream length), which is a
phase-zero-plus experiment, not this one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from instinct.core.rng import SeedScope
from instinct.ttt.chunker import build_aged_set, chunk_to_probes
from instinct.ttt.fastweight import FastWeightConfig, FastWeights, chunk_delta, row_losses
from instinct.ttt.oracle import OracleConfig, OracleResult, run_oracle
from instinct.ttt.probes import AGED, FRESH, ProbeLedger, stack_probes
from instinct.ttt.streams import Chunk, SyntheticStream

__all__ = [
    "FrontierPoint",
    "FrontierTrace",
    "IterativeRanking",
    "RankingStaleness",
    "StreamPartitions",
    "default_scores",
    "iterative_rescore_scores",
    "ranking_staleness",
    "split_stream_rows",
    "top_m_mask",
    "trace_frontier",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    """One arm at one write budget."""

    method: str
    m: int
    n_writes: int
    retention_loss: float
    plasticity_loss: float
    grad_steps: int
    admitted_harmful: int
    benefit: float
    damage: float
    cost: float
    utility: float
    selected_timesteps: tuple[int, ...]
    selection_tie_rate: float
    score_evaluations: int
    oracle_forward_calls: int
    write_ops: int
    anchor_ops: int
    reset_ops: int
    compute_units: int


@dataclass(frozen=True, slots=True)
class FrontierTrace:
    points: tuple[FrontierPoint, ...]
    m_grid: tuple[int, ...]
    methods: tuple[str, ...]
    n_replay_chunks: int
    n_retention_rows: int
    n_plasticity_rows: int
    eligible_timesteps: tuple[int, ...]

    def by_method(self, method: str) -> list[FrontierPoint]:
        pts = [p for p in self.points if p.method == method]
        return sorted(pts, key=lambda p: p.m)

    def curve(self, method: str, field_name: str) -> FloatArray:
        return np.array([getattr(p, field_name) for p in self.by_method(method)], dtype=np.float64)

    def assert_matched(self) -> None:
        """Fail unless every arm really did the same work and really wrote ``m`` times.

        Two separate checks, because they fail for different reasons. Unequal
        ``grad_steps`` means an arm skipped work and the compute is not matched.
        ``n_writes != m`` means the selection rule could not fill its budget —
        usually because ties or a short candidate list quietly shrank it — and
        the "matched write count" comparison is not matched.
        """
        steps = {p.grad_steps for p in self.points}
        if len(steps) != 1:
            raise AssertionError(f"arms did unequal work: grad_steps took values {sorted(steps)}")
        bad = [(p.method, p.m, p.n_writes) for p in self.points if p.n_writes != p.m]
        if bad:
            raise AssertionError(f"write counts are not matched to the budget: {bad}")

        # Every stateful arm computes every candidate update. At a fixed write
        # budget the ordinary ranking arms therefore have identical deployable
        # work. Elastic anchoring and reset deliberately carry explicit extra
        # operations and are excluded from this equality check, not hidden.
        ordinary = [p for p in self.points if p.anchor_ops == 0 and p.reset_ops == 0]
        for m in sorted({p.m for p in ordinary}):
            work = {
                (p.grad_steps, p.score_evaluations, p.write_ops)
                for p in ordinary
                if p.m == m
            }
            if len(work) > 1:
                raise AssertionError(f"ordinary arms at m={m} have unmatched work: {work}")


@dataclass(frozen=True, slots=True)
class StreamPartitions:
    """Row-disjoint candidate, oracle-probe, and downstream streams."""

    candidate: SyntheticStream
    oracle_probe: SyntheticStream
    downstream: SyntheticStream
    candidate_units: frozenset[str]
    oracle_probe_units: frozenset[str]
    downstream_units: frozenset[str]

    def assert_disjoint(self) -> None:
        groups = (self.candidate_units, self.oracle_probe_units, self.downstream_units)
        for i, left in enumerate(groups):
            for right in groups[i + 1 :]:
                if left & right:
                    raise AssertionError("P2 stream partitions overlap")


@dataclass(frozen=True, slots=True)
class IterativeRanking:
    scores: FloatArray
    selected_timesteps: tuple[int, ...]
    rescoring_rounds: int
    oracle_forward_calls: int
    probe_rows_spent: int


@dataclass(frozen=True, slots=True)
class RankingStaleness:
    mean_normalized_rank_shift: float
    top_m_gap: tuple[tuple[int, float], ...]


def split_stream_rows(stream: SyntheticStream) -> StreamPartitions:
    """Split every chunk by row before any oracle or frontier sees it.

    Candidate writes/future benefit, aged oracle damage probes, and downstream
    frontier evaluation must never reuse a row. Splitting within chunks keeps
    the three streams on the same task and time index without sharing units.
    """

    groups: list[list[Chunk]] = [[], [], []]
    units: list[set[str]] = [set(), set(), set()]
    suffixes = ("candidate", "oracle-probe", "downstream")
    for chunk in stream.chunks:
        if chunk.n_rows < 3:
            raise ValueError("P2 disjoint row split requires at least three rows per chunk")
        row_groups = np.array_split(np.arange(chunk.n_rows, dtype=np.int64), 3)
        for group_index, rows in enumerate(row_groups):
            if rows.size == 0:
                raise AssertionError("row split unexpectedly produced an empty partition")
            groups[group_index].append(
                Chunk(
                    t=chunk.t,
                    keys=chunk.keys[rows].clone(),
                    values=chunk.values[rows].clone(),
                    task_id=chunk.task_id,
                    harmful=chunk.harmful,
                    collides_with=chunk.collides_with,
                )
            )
            units[group_index].update(f"t{chunk.t}:r{int(row)}" for row in rows)

    def make(index: int) -> SyntheticStream:
        return SyntheticStream(
            name=f"{stream.name}:{suffixes[index]}",
            d_in=stream.d_in,
            d_out=stream.d_out,
            chunks=tuple(groups[index]),
            dtype=stream.dtype,
            notes=f"{stream.notes} Row-disjoint P2 partition: {suffixes[index]}.",
            _meta={**stream._meta, "partition": suffixes[index]},
        )

    out = StreamPartitions(
        candidate=make(0),
        oracle_probe=make(1),
        downstream=make(2),
        candidate_units=frozenset(units[0]),
        oracle_probe_units=frozenset(units[1]),
        downstream_units=frozenset(units[2]),
    )
    out.assert_disjoint()
    return out


def top_m_mask(scores: FloatArray, m: int) -> BoolArray:
    """The ``m`` highest-scoring candidates, ties broken by index.

    Ties are broken deterministically rather than randomly so that two arms with
    identical scores produce identical masks; a random tie-break would inject a
    difference the comparison is supposed to have excluded.
    """
    n = scores.shape[0]
    if not 0 <= m <= n:
        raise ValueError(f"m={m} outside [0, {n}]")
    mask = np.zeros(n, dtype=bool)
    if m:
        order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
        mask[order[:m]] = True
    return mask


def default_scores(
    result: OracleResult, *, n_random: int = 3, seed: int = 0, periodic_every: int = 5
) -> dict[str, FloatArray]:
    """The frozen oracle ceiling and every required cheap ranking baseline.

    The random arms are drawn from the repo's counter-based streams and indexed
    by replicate, so replicate ``k`` sees the same draws no matter how many
    replicates were requested — which keeps a sweep over ``n_random``
    comparable to itself.
    """
    if periodic_every < 1:
        raise ValueError("periodic_every must be positive")
    periodic = -(np.asarray(result.t, dtype=np.int64) % periodic_every).astype(np.float64)
    scores: dict[str, FloatArray] = {
        "oracle_frozen": result.utility.copy(),
        "surprise": result.features["surprise"].copy(),
        "update_norm": result.features["update_norm"].copy(),
        "gradient_alignment": result.features["grad_align"].copy(),
        "periodic": periodic,
        # These use the same candidate ordering as their simple parent but
        # change the state transition during replay below.
        "elastic_anchor": result.features["surprise"].copy(),
        "periodic_reset": periodic.copy(),
    }
    st = SeedScope(root_seed=seed).child("frontier").stream("random-rank")
    n = len(result)
    for k in range(n_random):
        scores[f"random-{k}"] = np.asarray(st.uniform(np.array([k], dtype=np.int64), n))[0]
    return scores


def _tie_rate(scores: FloatArray) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.size < 2:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    return float(counts[counts > 1].sum() / values.size)


def iterative_rescore_scores(
    candidate_stream: SyntheticStream,
    fw_cfg: FastWeightConfig,
    cfg: OracleConfig,
    candidate_t: IntArray,
    eligible_t: IntArray,
    *,
    ledger: ProbeLedger,
    scope: SeedScope,
    block_size: int = 2,
) -> IterativeRanking:
    """Greedily rank candidates while recomputing utility on the selected path.

    This is the expensive offline ceiling required by P2.1. It begins from a
    write-none trajectory, admits a fixed block, then rebuilds every remaining
    candidate delta and utility from the resulting memory state. Probe use is
    charged through one persistent ledger across all rounds.
    """

    if block_size < 1:
        raise ValueError("block_size must be positive")
    candidate_t = np.asarray(candidate_t, dtype=np.int64)
    eligible_t = np.asarray(eligible_t, dtype=np.int64)
    if not set(eligible_t).issubset(set(candidate_t)):
        raise ValueError("eligible timesteps must be present in candidate_t")

    reference_mask = np.zeros(len(candidate_stream), dtype=bool)
    remaining = [int(t) for t in eligible_t]
    selected: list[int] = []
    forward_calls = 0
    rounds = 0
    while remaining:
        result = run_oracle(
            candidate_stream,
            fw_cfg,
            cfg,
            scope.child("round", str(rounds)),
            ledger=ledger,
            reference_mask=reference_mask,
            candidate_timesteps=remaining,
        )
        by_t = {int(t): float(u) for t, u in zip(result.t, result.utility, strict=True)}
        missing = set(remaining) - set(by_t)
        if missing:
            raise ValueError(
                "iterative oracle exhausted its probe stream for candidates "
                f"{sorted(missing)[:5]}"
            )
        ranked = sorted(remaining, key=lambda t: (-by_t[t], t))
        chosen = ranked[: min(block_size, len(ranked))]
        for t in chosen:
            reference_mask[t] = True
            remaining.remove(t)
            selected.append(t)
        rounds += 1
        forward_calls += result.n_forward_calls

    # A rank-valued score lets trace_frontier consume the stateful ordering
    # through the same exact top-m machinery as every static baseline.
    scores = np.full(candidate_t.shape, -np.inf, dtype=np.float64)
    position = {int(t): i for i, t in enumerate(candidate_t)}
    for rank, t in enumerate(selected):
        scores[position[t]] = float(len(selected) - rank)
    return IterativeRanking(
        scores=scores,
        selected_timesteps=tuple(selected),
        rescoring_rounds=rounds,
        oracle_forward_calls=forward_calls,
        probe_rows_spent=ledger.usage_report().rows_evaluated,
    )


def ranking_staleness(
    frozen_scores: FloatArray,
    iterative_scores: FloatArray,
    eligible: IntArray,
    m_grid: Sequence[int],
) -> RankingStaleness:
    """How far the write-all ranking moves after stateful rescoring."""

    eligible = np.asarray(eligible, dtype=np.int64)
    frozen = np.asarray(frozen_scores, dtype=np.float64)[eligible]
    iterative = np.asarray(iterative_scores, dtype=np.float64)[eligible]
    n = eligible.size
    if n == 0:
        return RankingStaleness(float("nan"), ())
    frozen_order = np.argsort(-frozen, kind="stable")
    iterative_order = np.argsort(-iterative, kind="stable")
    frozen_rank = np.empty(n, dtype=np.int64)
    iterative_rank = np.empty(n, dtype=np.int64)
    frozen_rank[frozen_order] = np.arange(n)
    iterative_rank[iterative_order] = np.arange(n)
    rank_shift = float(np.mean(np.abs(frozen_rank - iterative_rank)) / max(n - 1, 1))
    gaps: list[tuple[int, float]] = []
    for raw_m in m_grid:
        m = int(raw_m)
        if not 0 <= m <= n:
            raise ValueError(f"m={m} outside [0, {n}]")
        if m == 0:
            gaps.append((m, 0.0))
            continue
        a, b = set(frozen_order[:m]), set(iterative_order[:m])
        gaps.append((m, 1.0 - len(a & b) / m))
    return RankingStaleness(rank_shift, tuple(gaps))


def _register_plasticity_probes(
    stream: SyntheticStream, ledger: ProbeLedger, holdout: Sequence[int], rows_per_probe: int
) -> list[str]:
    ids: list[str] = []
    for t in holdout:
        probes = chunk_to_probes(
            stream.chunks[t], rows_per_probe=rows_per_probe, kind=FRESH, prefix="frontier:"
        )
        ledger.register_all(probes, 1)
        ids.extend(p.probe_id for p in probes)
    return ids


def trace_frontier(
    stream: SyntheticStream,
    fw_cfg: FastWeightConfig,
    result: OracleResult,
    scores: dict[str, FloatArray],
    m_grid: Sequence[int],
    *,
    ledger: ProbeLedger,
    scope: SeedScope,
    evaluation_stream: SyntheticStream | None = None,
    n_holdout_chunks: int = 4,
    n_retention_probes: int = 12,
    retention_min_age: int = 6,
    rows_per_probe: int = 2,
    retention_lambda: float = 1.0,
    write_cost: float = 0.0,
    score_cost: float = 0.0,
    anchor_strength: float = 0.1,
    anchor_cost: float = 0.0,
    reset_every: int = 5,
    reset_cost: float = 0.0,
    oracle_forward_calls: dict[str, int] | None = None,
) -> FrontierTrace:
    """Replay every ``(method, m)`` arm as a branch of one batched rollout.

    The holdout tail is never written by any arm, which is what makes the
    plasticity axis mean "adapts to information it has not memorized" rather
    than "memorized the thing it is being scored on".

    Retention and plasticity are each evaluated in a single forward shared by
    all branches, so the holdout rows are charged to the ledger exactly once no
    matter how many arms there are. That is the only way a finite-budget holdout
    survives a sweep.
    """
    if retention_lambda < 0 or min(write_cost, score_cost, anchor_cost, reset_cost) < 0:
        raise ValueError("frontier costs and retention_lambda must be nonnegative")
    if not 0.0 <= anchor_strength <= 1.0:
        raise ValueError("anchor_strength must lie in [0, 1]")
    if reset_every < 1:
        raise ValueError("reset_every must be positive")
    evaluation_stream = evaluation_stream or stream
    if len(evaluation_stream) != len(stream):
        raise ValueError("evaluation stream must align with the candidate stream in time")

    n_cand = len(result)
    for name, s in scores.items():
        if s.shape[0] != n_cand:
            raise ValueError(f"scores[{name!r}] has {s.shape[0]} entries, expected {n_cand}")
    grid = tuple(dict.fromkeys(int(m) for m in m_grid))

    n_total = len(stream)
    if n_holdout_chunks >= n_total:
        raise ValueError("holdout tail cannot be the whole stream")
    n_replay = n_total - n_holdout_chunks
    holdout = list(range(n_replay, n_total))

    # An admitted write must be a candidate the oracle actually scored, and it
    # must land inside the replay window; anything else would let one arm write
    # at a timestep another arm never saw.
    cand_t = np.asarray(result.t, dtype=np.int64)
    # Restrict candidates *before* top-m. Also keep every frozen future horizon
    # outside the downstream evaluation tail; otherwise oracle selection reads
    # the final evaluation stream.
    raw_max_horizon = result.notes.get("max_horizon", 0)
    if not isinstance(raw_max_horizon, int | float):
        raise ValueError("oracle max_horizon note must be numeric")
    max_horizon = int(raw_max_horizon)
    in_window = cand_t + max_horizon < n_replay
    eligible = np.flatnonzero(in_window)
    for m in grid:
        if not 0 <= m <= eligible.size:
            raise ValueError(f"m={m} outside [0, {eligible.size}] after candidate restriction")

    ranked_methods = tuple(scores)
    arms = [(name, m) for name in ranked_methods for m in grid]
    arms.extend([("no_write", 0), ("dense_all", int(eligible.size))])
    methods = tuple(dict.fromkeys(name for name, _ in arms))
    n_arms = len(arms)
    admit = np.zeros((n_arms, n_total), dtype=bool)
    for a, (name, m) in enumerate(arms):
        if name == "no_write":
            chosen = np.empty(0, dtype=np.int64)
        elif name == "dense_all":
            chosen = eligible
        else:
            restricted = top_m_mask(scores[name][eligible], m)
            chosen = eligible[restricted]
        admit[a, cand_t[chosen]] = True

    # Replay. Every branch computes the write at every chunk; only the add is masked.
    state = FastWeights.init(fw_cfg, 1, scope.child("frontier-init")).fork(n_arms)
    initial = state.snapshot()
    admit_t = torch.from_numpy(admit)
    anchor_branches = np.array([name == "elastic_anchor" for name, _ in arms], dtype=bool)
    reset_branches = np.array([name == "periodic_reset" for name, _ in arms], dtype=bool)
    reset_counts = np.zeros(n_arms, dtype=np.int64)
    anchor_counts = np.zeros(n_arms, dtype=np.int64)
    for t in range(n_replay):
        reset_mask = reset_branches & (t > 0) & (t % reset_every == 0)
        if reset_mask.any():
            reset_index = torch.from_numpy(np.flatnonzero(reset_mask))
            state.w1[reset_index] = initial.w1[reset_index]
            state.w2[reset_index] = initial.w2[reset_index]
            state.w3[reset_index] = initial.w3[reset_index]
            reset_counts[reset_mask] += 1
        chunk = stream.chunks[t]
        delta = chunk_delta(state, chunk.keys, chunk.values)
        state.apply_(delta, admit_t[:, t])
        anchored = anchor_branches & admit[:, t]
        if anchored.any():
            anchor_index = torch.from_numpy(np.flatnonzero(anchored))
            keep = 1.0 - anchor_strength
            state.w1[anchor_index] = (
                keep * state.w1[anchor_index] + anchor_strength * initial.w1[anchor_index]
            )
            state.w2[anchor_index] = (
                keep * state.w2[anchor_index] + anchor_strength * initial.w2[anchor_index]
            )
            state.w3[anchor_index] = (
                keep * state.w3[anchor_index] + anchor_strength * initial.w3[anchor_index]
            )
            anchor_counts[anchored] += 1

    # Retention: row-disjoint evaluation probes from the replayed prefix.
    aged = build_aged_set(
        n_replay - 1,
        ledger,
        n_items=n_retention_probes,
        min_age=retention_min_age,
        scope=scope.child("frontier-retention"),
        kind=AGED,
    )
    if not aged:
        raise ValueError("the aged pool is exhausted; the frontier has no retention sample left")
    rx, ry = stack_probes(aged)
    retention = row_losses(state, rx, ry).mean(dim=1).numpy()

    # Plasticity: the row-disjoint, never-written evaluation tail.
    fresh_ids = _register_plasticity_probes(
        evaluation_stream, ledger, holdout, rows_per_probe
    )
    fresh = ledger.draw(fresh_ids)
    fx, fy = stack_probes(fresh)
    plasticity = row_losses(state, fx, fy).mean(dim=1).numpy()

    harmful_t = {int(t) for t in np.asarray(result.t)[result.harmful_truth]}
    no_write_index = arms.index(("no_write", 0))
    base_retention = float(retention[no_write_index])
    base_plasticity = float(plasticity[no_write_index])
    oracle_forward_calls = oracle_forward_calls or {}
    built: list[FrontierPoint] = []
    for a, (name, m) in enumerate(arms):
        selected_t = tuple(int(t) for t in np.flatnonzero(admit[a]))
        benefit = base_plasticity - float(plasticity[a])
        damage = float(retention[a]) - base_retention
        score_ops = int(eligible.size)
        operational_cost = (
            write_cost * m
            + score_cost * score_ops
            + anchor_cost * int(anchor_counts[a])
            + reset_cost * int(reset_counts[a])
        )
        method_ties = 0.0 if name in {"no_write", "dense_all"} else _tie_rate(
            scores[name][eligible]
        )
        compute_units = (
            n_replay + score_ops + m + int(anchor_counts[a]) + int(reset_counts[a])
        )
        built.append(
            FrontierPoint(
                method=name,
                m=m,
                n_writes=len(selected_t),
                retention_loss=float(retention[a]),
                plasticity_loss=float(plasticity[a]),
                grad_steps=n_replay,
                admitted_harmful=sum(1 for t in selected_t if t in harmful_t),
                benefit=benefit,
                damage=damage,
                cost=operational_cost,
                utility=benefit - retention_lambda * damage - operational_cost,
                selected_timesteps=selected_t,
                selection_tie_rate=method_ties,
                score_evaluations=score_ops,
                oracle_forward_calls=int(oracle_forward_calls.get(name, 0)),
                write_ops=m,
                anchor_ops=int(anchor_counts[a]),
                reset_ops=int(reset_counts[a]),
                compute_units=compute_units,
            )
        )
    points = tuple(built)
    trace = FrontierTrace(
        points=points,
        m_grid=grid,
        methods=methods,
        n_replay_chunks=n_replay,
        n_retention_rows=int(rx.shape[0]),
        n_plasticity_rows=int(fx.shape[0]),
        eligible_timesteps=tuple(int(cand_t[i]) for i in eligible),
    )
    trace.assert_matched()
    return trace


def sequential_replay_reference(
    stream: SyntheticStream,
    fw_cfg: FastWeightConfig,
    admit: BoolArray,
    scope: SeedScope,
    n_replay: int,
    eval_x: Tensor,
    eval_y: Tensor,
) -> FloatArray:
    """One arm at a time, no branch axis. The reference for the batched replay.

    Exists so the frontier's fast path is checkable the same way the oracle's
    is. It is not used in any reported run.
    """
    out = np.zeros(admit.shape[0])
    for a in range(admit.shape[0]):
        state = FastWeights.init(fw_cfg, 1, scope.child("frontier-init"))
        for t in range(n_replay):
            chunk = stream.chunks[t]
            delta = chunk_delta(state, chunk.keys, chunk.values)
            if admit[a, t]:
                state.apply_(delta)
        out[a] = float(row_losses(state, eval_x, eval_y).mean())
    return out
