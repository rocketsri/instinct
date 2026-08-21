"""The exact-fork oracle: what a candidate write was actually worth.

This is phase zero. Before anything is gated, projected or learned, we need to
know whether the quantity a gate would have to predict is even well defined and
measurable. So for every candidate write we do the expensive, unambiguous thing:

    clone the fast-weight state -> apply the update on one branch -> evaluate
    both branches on future chunks and on aged probes -> compute U_t -> roll back

with

    U_t = sum_h w_h [ L(W_t; B_{t+h}) - L(W_t + dW; B_{t+h}) ]     future benefit
          - lambda * E_{z ~ A_t}[ L(W_t + dW; z) - L(W_t; z) ]     retention damage
          - c_t                                                    cost

Three design notes, in decreasing order of how easy they are to get wrong.

**Differences, never ratios.** Both terms are differences of losses between two
branches that saw identical data. A ratio would be scale-dependent in a quantity
(loss magnitude) that drifts along the stream for reasons unrelated to the
write, and the resulting ranking would silently track the drift.

**One baseline forward per timestep, shared everywhere.** ``L(W_t; ...)`` appears
in both terms and is the same number for every candidate at that timestep. It is
computed once, on one branch, and reused. This is not only a saving: recomputing
it would make the two terms differ in the last bits for no reason, and the
retention term is a small difference of large numbers where that shows.

**Fork wide.** ``reports/t4_hardware.md`` measures 8 forks costing the same as 1
and 14.2x at 128, so the oracle batches timesteps until it has at least
``min_forks`` branches in flight and evaluates them in a single ``bmm``. Phase
zero is offline analysis, so nothing here has to be causal in wall-clock; it only
has to be causal in *information*, which the plan builder enforces.

:func:`run_oracle` is the fast path. :func:`evaluate_plan_naive` is the literal
clone/apply/rollback reference required by rule 7, and ``tests/test_ttt.py``
asserts they agree bitwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from instinct.core.rng import SeedScope
from instinct.ttt.chunker import build_aged_set, future_chunks
from instinct.ttt.fastweight import (
    FastWeightConfig,
    FastWeightDelta,
    FastWeights,
    chunk_delta,
    row_losses,
)
from instinct.ttt.probes import AGED, AUDIT, Probe, ProbeLedger, probe_row_weights, stack_probes
from instinct.ttt.streams import SyntheticStream

__all__ = [
    "OracleConfig",
    "OracleResult",
    "TimestepPlan",
    "build_plans",
    "evaluate_plan_naive",
    "evaluate_plans_batched",
    "run_oracle",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

# Cheap features a real gate could compute before committing. Named here so the
# decision tree and the oracle cannot drift apart about what "cheap" meant.
PRE_FEATURES = ("update_norm", "surprise", "grad_align", "memory_age", "chunk_grad_cos")
POST_FEATURES = ("audit_delta",)


@dataclass(frozen=True, slots=True)
class OracleConfig:
    """Everything the utility definition leaves free.

    None of these have assumed functional forms attached: ``horizon_weights``
    is an explicit vector rather than a decay rate, and ``cost`` is a flat
    charge rather than a fitted model of one. If a shape matters it should be
    swept, not baked in.
    """

    horizons: tuple[int, ...] = (1, 2, 4, 8)
    horizon_weights: tuple[float, ...] | None = None
    retention_lambda: float = 1.0
    cost: float = 0.0
    min_age: int = 6
    n_aged_probes: int = 6
    n_audit_probes: int = 2
    min_forks: int = 8
    max_forks: int = 32
    write_all: bool = True

    def weights(self) -> tuple[float, ...]:
        if self.horizon_weights is None:
            return tuple(1.0 / len(self.horizons) for _ in self.horizons)
        if len(self.horizon_weights) != len(self.horizons):
            raise ValueError("horizon_weights must align with horizons")
        return self.horizon_weights

    def __post_init__(self) -> None:
        if not self.horizons or any(h <= 0 for h in self.horizons):
            raise ValueError("horizons must be nonempty and strictly positive")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be unique")
        weights = self.weights()
        if any(w < 0 for w in weights) or not np.isclose(sum(weights), 1.0):
            raise ValueError("horizon weights must be nonnegative and sum to one")
        if self.retention_lambda < 0 or self.cost < 0:
            raise ValueError("retention_lambda and cost must be nonnegative")
        if self.min_forks < 8:
            raise ValueError(
                f"min_forks={self.min_forks}: measured on the target GPU, 8 forks cost the same "
                "as 1, so forking fewer than 8 buys nothing and costs a launch"
            )
        if self.max_forks < self.min_forks:
            raise ValueError("max_forks must be >= min_forks")


@dataclass(frozen=True, slots=True)
class TimestepPlan:
    """Everything needed to score the candidate write at one timestep.

    Built once, before any evaluation, so that the fast and naive paths score
    *identical* data. If each path drew its own aged set they would disagree for
    a reason that has nothing to do with batching, and the equivalence test
    would be testing the sampler.

    ``x``/``y`` are the concatenation of future-chunk rows, aged probe rows and
    audit rows, in that order, so one forward serves all three terms. The three
    weight vectors select among them; every one of them is zero on the rows it
    does not own, which is why padding is harmless.
    """

    t: int
    x: Tensor
    y: Tensor
    w_benefit: Tensor
    w_damage: Tensor
    w_audit: Tensor
    n_future_rows: int
    n_probe_rows: int
    n_audit_rows: int
    probe_ids: tuple[str, ...]
    used_horizons: tuple[int, ...]

    @property
    def n_rows(self) -> int:
        return int(self.x.shape[0])


@dataclass(slots=True)
class OracleResult:
    """Per-candidate utilities plus the pieces they were built from.

    The decomposition is kept, not just the total. A candidate can be rejected
    because it damages retention or because it simply does not help, and a gate
    trained on the sum alone cannot tell those apart — nor can anyone reading
    the frontier later.
    """

    t: IntArray
    utility: FloatArray
    benefit: FloatArray
    damage: FloatArray
    cost: FloatArray
    harmful_truth: BoolArray
    features: dict[str, FloatArray]
    deltas_flat: FloatArray
    max_forks_used: int
    n_forward_calls: int
    probe_rows_spent: int
    notes: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.utility.shape[0])

    def ranked_ascending(self) -> IntArray:
        """Candidate indices from worst utility to best."""
        return np.argsort(self.utility, kind="stable").astype(np.int64)


# ----------------------------------------------------------------------------
# Plan construction
# ----------------------------------------------------------------------------


def _stack_plan_inputs(
    parts: list[tuple[Tensor, Tensor, FloatArray, FloatArray, FloatArray]],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    xs = torch.cat([p[0] for p in parts], dim=0)
    ys = torch.cat([p[1] for p in parts], dim=0)
    wb = torch.from_numpy(np.concatenate([p[2] for p in parts])).to(xs.dtype)
    wd = torch.from_numpy(np.concatenate([p[3] for p in parts])).to(xs.dtype)
    wa = torch.from_numpy(np.concatenate([p[4] for p in parts])).to(xs.dtype)
    return xs, ys, wb, wd, wa


def build_plans(
    stream: SyntheticStream,
    ledger: ProbeLedger,
    cfg: OracleConfig,
    scope: SeedScope,
    *,
    timesteps: Sequence[int] | None = None,
) -> list[TimestepPlan]:
    """Draw every timestep's evaluation data up front, charging the ledger once.

    Charging here rather than inside the evaluator is what makes the fast path
    and the naive reference comparable: they consume no budget at all, they just
    read a plan that was already paid for. It also means the ledger's usage
    report describes the *experiment*, not one implementation of it.
    """
    plans: list[TimestepPlan] = []
    steps = range(len(stream)) if timesteps is None else timesteps
    weights = cfg.weights()

    for t in steps:
        fut = future_chunks(stream.chunks, t, cfg.horizons)
        if len(fut) != len(cfg.horizons):
            # A partial horizon changes the frozen multi-horizon estimand. Late
            # candidates are excluded instead of silently renormalizing it.
            continue
        aged = build_aged_set(
            t, ledger, n_items=cfg.n_aged_probes, min_age=cfg.min_age, scope=scope, kind=AGED
        )
        if not aged:
            continue  # aged pool spent or not yet aged; the retention term has no sample
        audit = build_aged_set(
            t, ledger, n_items=cfg.n_audit_probes, min_age=cfg.min_age, scope=scope, kind=AUDIT
        )

        by_h = dict(zip(cfg.horizons, weights))
        used_h = tuple(h for h, _ in fut)
        parts: list[tuple[Tensor, Tensor, FloatArray, FloatArray, FloatArray]] = []
        n_future = 0
        for h, chunk in fut:
            n = chunk.n_rows
            n_future += n
            parts.append(
                (
                    chunk.keys,
                    chunk.values,
                    np.full(n, by_h[h] / n),
                    np.zeros(n),
                    np.zeros(n),
                )
            )
        px, py = stack_probes(aged)
        n_probe = int(px.shape[0])
        parts.append((px, py, np.zeros(n_probe), probe_row_weights(aged), np.zeros(n_probe)))
        n_audit = 0
        if audit:
            ax, ay = stack_probes(audit)
            n_audit = int(ax.shape[0])
            parts.append((ax, ay, np.zeros(n_audit), np.zeros(n_audit), probe_row_weights(audit)))

        xs, ys, wb, wd, wa = _stack_plan_inputs(parts)
        plans.append(
            TimestepPlan(
                t=t,
                x=xs,
                y=ys,
                w_benefit=wb,
                w_damage=wd,
                w_audit=wa,
                n_future_rows=n_future,
                n_probe_rows=n_probe,
                n_audit_rows=n_audit,
                probe_ids=tuple(p.probe_id for p in (*aged, *audit)),
                used_horizons=used_h,
            )
        )
    return plans


# ----------------------------------------------------------------------------
# Evaluation: fast path and naive reference
# ----------------------------------------------------------------------------


def _pad_to(x: Tensor, n: int) -> Tensor:
    if x.shape[0] == n:
        return x
    pad = torch.zeros((n - x.shape[0], *x.shape[1:]), dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=0)


def _group_plans(n_plans: int, min_forks: int, max_forks: int) -> list[tuple[int, int]]:
    """Slice plans into fork groups. Each plan costs two branches: base + candidate.

    The group size is the largest that fits ``max_forks``, floored at what
    ``min_forks`` demands, because a group of one would pay a launch to fork a
    single candidate and the hardware note says that is pure waste.
    """
    per_group = max(max_forks // 2, (min_forks + 1) // 2, 1)
    return [(s, min(s + per_group, n_plans)) for s in range(0, n_plans, per_group)]


@dataclass(frozen=True, slots=True)
class _Scored:
    benefit: FloatArray
    damage: FloatArray
    audit_delta: FloatArray
    n_forward_calls: int
    max_forks_used: int


def evaluate_plans_batched(
    plans: Sequence[TimestepPlan],
    states: Sequence[FastWeights],
    deltas: Sequence[FastWeightDelta],
    cfg: OracleConfig,
) -> _Scored:
    """Fast path: many forks, one forward per group.

    ``states[i]`` is the single-branch memory as it stood at ``plans[i].t``, and
    ``deltas[i]`` is the candidate write. Within a group the branch layout is
    ``[base_0 .. base_{G-1}, cand_0 .. cand_{G-1}]``, so a slice recovers the
    cached baseline losses and the same tensor serves the benefit term and the
    retention term.

    Rows are zero-padded to the group's widest plan. Padding is safe because
    every weight vector is zero there — the padded row is evaluated and then
    multiplied by nothing. Masking instead would branch, and branching is what
    we are paying batching to avoid.
    """
    if not (len(plans) == len(states) == len(deltas)):
        raise ValueError("plans, states and deltas must be the same length")
    n = len(plans)
    benefit = np.zeros(n)
    damage = np.zeros(n)
    audit = np.zeros(n)
    forwards = 0
    max_forks = 0

    for lo, hi in _group_plans(n, cfg.min_forks, cfg.max_forks):
        g = hi - lo
        width = max(plans[i].n_rows for i in range(lo, hi))
        xs = torch.stack([_pad_to(plans[i].x, width) for i in range(lo, hi)])
        ys = torch.stack([_pad_to(plans[i].y, width) for i in range(lo, hi)])
        wb = torch.stack([_pad_to(plans[i].w_benefit, width) for i in range(lo, hi)])
        wd = torch.stack([_pad_to(plans[i].w_damage, width) for i in range(lo, hi)])
        wa = torch.stack([_pad_to(plans[i].w_audit, width) for i in range(lo, hi)])

        base = FastWeights.cat([states[i] for i in range(lo, hi)])
        cand = base.clone()
        cand.apply_(FastWeightDelta.cat([deltas[i] for i in range(lo, hi)]))
        both = FastWeights.cat([base, cand])

        x_both = torch.cat([xs, xs], dim=0)
        y_both = torch.cat([ys, ys], dim=0)
        rows = row_losses(both, x_both, y_both)
        forwards += 1
        max_forks = max(max_forks, 2 * g)

        rb, ru = rows[:g], rows[g:]
        benefit[lo:hi] = ((rb - ru) * wb).sum(dim=1).numpy()
        damage[lo:hi] = ((ru - rb) * wd).sum(dim=1).numpy()
        audit[lo:hi] = ((ru - rb) * wa).sum(dim=1).numpy()

    return _Scored(benefit, damage, audit, forwards, max_forks)


def evaluate_plan_naive(
    plan: TimestepPlan,
    state: FastWeights,
    delta: FastWeightDelta,
) -> tuple[float, float, float]:
    """Reference: literally clone, apply, evaluate, roll back. One branch at a time.

    Slow and unclever on purpose. It also *asserts* the rollback rather than
    trusting it, because "the state came back exactly" is the property the whole
    oracle rests on and it costs one comparison to check.
    """
    if state.n_branches != 1:
        raise ValueError("the naive reference works one branch at a time")
    snap = state.snapshot()
    base_rows = row_losses(state, plan.x, plan.y)[0]

    work = state.clone()
    work.apply_(delta)
    upd_rows = row_losses(work, plan.x, plan.y)[0]

    state.restore(snap)
    if not (
        torch.equal(state.w1, snap.w1)
        and torch.equal(state.w2, snap.w2)
        and torch.equal(state.w3, snap.w3)
    ):
        raise AssertionError("rollback was not exact; every utility after this point is unmoored")

    d = upd_rows - base_rows
    return (
        float((-d * plan.w_benefit).sum()),
        float((d * plan.w_damage).sum()),
        float((d * plan.w_audit).sum()),
    )


# ----------------------------------------------------------------------------
# The reference roll, and the driver
# ----------------------------------------------------------------------------


def _cosine(a: Tensor, b: Tensor) -> float:
    na = float(torch.linalg.vector_norm(a))
    nb = float(torch.linalg.vector_norm(b))
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(torch.dot(a.reshape(-1), b.reshape(-1)) / (na * nb))


def run_oracle(
    stream: SyntheticStream,
    fw_cfg: FastWeightConfig,
    cfg: OracleConfig,
    scope: SeedScope,
    *,
    ledger: ProbeLedger,
    plans: Sequence[TimestepPlan] | None = None,
    naive: bool = False,
    reference_mask: BoolArray | None = None,
    candidate_timesteps: Sequence[int] | None = None,
) -> OracleResult:
    """Score every candidate write along one reference trajectory.

    The reference trajectory is ``write_all`` by default: the memory absorbs
    every chunk, and ``U_t`` answers "given the memory that writing everything
    produced, was *this* write worth it?". That is a real choice and it has a
    real consequence — the trajectory a selective policy would follow is
    different, so ``U_t`` is exact for the reference path and only an
    approximation for any other. The frontier in :mod:`instinct.ttt.frontier`
    inherits that gap and reports it.

    Set ``naive=True`` to route through the one-at-a-time reference
    implementation instead. Same plans, same numbers, ~2G times the forwards.
    """
    if reference_mask is not None:
        reference_mask = np.asarray(reference_mask, dtype=bool)
        if reference_mask.shape != (len(stream),):
            raise ValueError(
                f"reference_mask has shape {reference_mask.shape}, expected {(len(stream),)}"
            )
    scope_states = scope.child("reference-roll")
    state = FastWeights.init(fw_cfg, 1, scope_states)

    # 1. Roll the reference trajectory, snapshotting the state and the candidate
    #    write at every timestep. Nothing is evaluated here: the memory must not
    #    see a probe before the plan says it may.
    snapshots: list[FastWeights] = []
    deltas: list[FastWeightDelta] = []
    surprise: list[float] = []
    grad_align: list[float] = []
    chunk_cos: list[float] = []
    memory_age: list[float] = []
    ema: Tensor | None = None
    writes = 0

    for chunk in stream.chunks:
        pre = float(row_losses(state, chunk.keys, chunk.values).mean())
        delta = chunk_delta(state, chunk.keys, chunk.values)
        flat = delta.flat()[0]
        snapshots.append(state.clone())
        deltas.append(delta)
        surprise.append(pre)
        grad_align.append(0.0 if ema is None else _cosine(flat, ema))
        chunk_cos.append(0.0 if not deltas[:-1] else _cosine(flat, deltas[-2].flat()[0]))
        memory_age.append(float(writes))
        ema = flat.clone() if ema is None else 0.7 * ema + 0.3 * flat
        should_write = cfg.write_all if reference_mask is None else bool(reference_mask[chunk.t])
        if should_write:
            state.apply_(delta)
            writes += 1

    # 2. Draw the evaluation data. Charging happens exactly once, here.
    if plans is None:
        plans = build_plans(
            stream,
            ledger,
            cfg,
            scope.child("plans"),
            timesteps=candidate_timesteps,
        )
    if not plans:
        raise ValueError(
            "no scorable timesteps: the stream is too short for these horizons/min_age"
        )

    idx = [p.t for p in plans]
    sel_states = [snapshots[t] for t in idx]
    sel_deltas = [deltas[t] for t in idx]

    # 3. Score.
    if naive:
        b = np.zeros(len(plans))
        dmg = np.zeros(len(plans))
        aud = np.zeros(len(plans))
        for i, plan in enumerate(plans):
            b[i], dmg[i], aud[i] = evaluate_plan_naive(plan, sel_states[i], sel_deltas[i])
        scored = _Scored(b, dmg, aud, n_forward_calls=2 * len(plans), max_forks_used=1)
    else:
        scored = evaluate_plans_batched(plans, sel_states, sel_deltas, cfg)

    cost = np.full(len(plans), float(cfg.cost))
    utility = scored.benefit - cfg.retention_lambda * scored.damage - cost

    features: dict[str, FloatArray] = {
        "update_norm": np.array([float(deltas[t].norm()[0]) for t in idx]),
        "surprise": np.array([surprise[t] for t in idx]),
        "grad_align": np.array([grad_align[t] for t in idx]),
        "memory_age": np.array([memory_age[t] for t in idx]),
        "chunk_grad_cos": np.array([chunk_cos[t] for t in idx]),
        "audit_delta": scored.audit_delta,
    }

    return OracleResult(
        t=np.array(idx, dtype=np.int64),
        utility=utility,
        benefit=scored.benefit,
        damage=scored.damage,
        cost=cost,
        harmful_truth=np.array([stream.chunks[t].harmful for t in idx], dtype=bool),
        features=features,
        deltas_flat=np.stack([deltas[t].flat()[0].numpy() for t in idx]),
        max_forks_used=scored.max_forks_used,
        n_forward_calls=scored.n_forward_calls,
        probe_rows_spent=ledger.usage_report().rows_evaluated,
        notes={
            "reference_policy": (
                "custom_mask"
                if reference_mask is not None
                else ("write_all" if cfg.write_all else "write_none")
            ),
            "reference_writes": writes,
            "stream": stream.name,
            "n_timesteps_skipped": len(stream) - len(plans),
            "max_horizon": max(cfg.horizons),
            "horizons": cfg.horizons,
            "full_horizons_only": True,
        },
    )


def probes_of(plan: TimestepPlan, ledger: ProbeLedger) -> list[Probe]:
    """Identify (without charging) the probes a plan was built from.

    For reporting only. Reading these to *evaluate* anything would be spending
    budget the ledger never saw.
    """
    return [ledger.peek(pid) for pid in plan.probe_ids]
