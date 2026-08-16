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
from instinct.ttt.oracle import OracleResult
from instinct.ttt.probes import AGED, FRESH, ProbeLedger, stack_probes
from instinct.ttt.streams import SyntheticStream

__all__ = [
    "FrontierPoint",
    "FrontierTrace",
    "default_scores",
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


@dataclass(frozen=True, slots=True)
class FrontierTrace:
    points: tuple[FrontierPoint, ...]
    m_grid: tuple[int, ...]
    methods: tuple[str, ...]
    n_replay_chunks: int
    n_retention_rows: int
    n_plasticity_rows: int

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
    result: OracleResult, *, n_random: int = 3, seed: int = 0
) -> dict[str, FloatArray]:
    """Oracle, surprise and several paired random rankings.

    The random arms are drawn from the repo's counter-based streams and indexed
    by replicate, so replicate ``k`` sees the same draws no matter how many
    replicates were requested — which keeps a sweep over ``n_random``
    comparable to itself.
    """
    scores: dict[str, FloatArray] = {
        "oracle": result.utility.copy(),
        "surprise": result.features["surprise"].copy(),
    }
    st = SeedScope(root_seed=seed).child("frontier").stream("random-rank")
    n = len(result)
    for k in range(n_random):
        scores[f"random-{k}"] = np.asarray(st.uniform(np.array([k], dtype=np.int64), n))[0]
    return scores


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
    n_holdout_chunks: int = 4,
    n_retention_probes: int = 12,
    retention_min_age: int = 6,
    rows_per_probe: int = 2,
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
    n_cand = len(result)
    for name, s in scores.items():
        if s.shape[0] != n_cand:
            raise ValueError(f"scores[{name!r}] has {s.shape[0]} entries, expected {n_cand}")
    grid = tuple(int(m) for m in m_grid)
    for m in grid:
        if not 0 <= m <= n_cand:
            raise ValueError(f"m={m} outside [0, {n_cand}]")

    methods = tuple(scores)
    arms = [(name, m) for name in methods for m in grid]
    n_arms = len(arms)

    n_total = len(stream)
    if n_holdout_chunks >= n_total:
        raise ValueError("holdout tail cannot be the whole stream")
    n_replay = n_total - n_holdout_chunks
    holdout = list(range(n_replay, n_total))

    # An admitted write must be a candidate the oracle actually scored, and it
    # must land inside the replay window; anything else would let one arm write
    # at a timestep another arm never saw.
    cand_t = np.asarray(result.t, dtype=np.int64)
    in_window = cand_t < n_replay
    admit = np.zeros((n_arms, n_total), dtype=bool)
    for a, (name, m) in enumerate(arms):
        sel = top_m_mask(scores[name], m)
        admit[a, cand_t[sel & in_window]] = True

    # Replay. Every branch computes the write at every chunk; only the add is masked.
    state = FastWeights.init(fw_cfg, 1, scope.child("frontier-init")).fork(n_arms)
    admit_t = torch.from_numpy(admit)
    for t in range(n_replay):
        chunk = stream.chunks[t]
        delta = chunk_delta(state, chunk.keys, chunk.values)
        state.apply_(delta, admit_t[:, t])

    # Retention: aged probes drawn from the replayed prefix, charged once.
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

    # Plasticity: the never-written tail, also charged once.
    fresh_ids = _register_plasticity_probes(stream, ledger, holdout, rows_per_probe)
    fresh = ledger.draw(fresh_ids)
    fx, fy = stack_probes(fresh)
    plasticity = row_losses(state, fx, fy).mean(dim=1).numpy()

    harmful_t = {int(t) for t in np.asarray(result.t)[result.harmful_truth]}
    points = tuple(
        FrontierPoint(
            method=name,
            m=m,
            n_writes=int(admit[a].sum()),
            retention_loss=float(retention[a]),
            plasticity_loss=float(plasticity[a]),
            grad_steps=n_replay,
            admitted_harmful=int(sum(1 for t in np.flatnonzero(admit[a]) if int(t) in harmful_t)),
        )
        for a, (name, m) in enumerate(arms)
    )
    return FrontierTrace(
        points=points,
        m_grid=grid,
        methods=methods,
        n_replay_chunks=n_replay,
        n_retention_rows=int(rx.shape[0]),
        n_plasticity_rows=int(fx.shape[0]),
    )


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
