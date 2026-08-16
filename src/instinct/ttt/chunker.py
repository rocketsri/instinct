"""Chunking the stream, and assembling the aged set ``A_t``.

Two jobs that look clerical and are not.

**Chunking** decides how consequential a single write is. LaCT's whole premise
is that writes are infrequent and large, so the chunk boundary is the unit at
which harm can be attributed at all. Nothing here re-chunks adaptively: the
chunk boundaries come from the stream generator so that a harmful chunk is a
harmful *chunk*, not a harmful fraction of one, and the ground-truth flags stay
aligned with the things being scored.

**The aged set** is the retention term's sample. Three properties are load
bearing:

*Aged.* A probe drawn from information written last tick is not testing
retention, it is testing the write that just happened. :func:`build_aged_set`
enforces a minimum age in chunks.

*Rotating.* Every draw is charged against :mod:`instinct.ttt.probes`' budgets,
so ``A_t`` cannot be the same rows at every ``t``. That is a constraint on the
experiment, not a nicety — see that module for why.

*Causal.* A probe may only be drawn from information that already exists at
``t``. Sampling the aged set from the whole stream, including the future, would
make the retention term partly a forecast, and the resulting utility would be
unattainable by any online gate.

The pool is deliberately built from *all* of the stream's history, including the
conflicting chunks in the drifting-regression stream. Restricting it to the
"good" task would be filtering the retention set using the same ground truth the
oracle is being tested against, and the known-answer check would stop being a
check. The cost is that ``A_t`` contains a minority of rows a harmful write
actually helps, which shrinks the measured damage; that is the honest version.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from instinct.core.rng import SeedScope
from instinct.ttt.probes import AGED, AUDIT, Probe, ProbeLedger
from instinct.ttt.streams import Chunk, SyntheticStream

__all__ = [
    "build_aged_set",
    "build_ledger",
    "chunk_to_probes",
    "future_chunks",
]


def chunk_to_probes(
    chunk: Chunk,
    *,
    rows_per_probe: int,
    kind: str = AGED,
    prefix: str = "",
) -> list[Probe]:
    """Split one chunk into probe-sized units.

    Splitting matters for budget granularity. A whole chunk as one probe would
    retire an entire timestep's worth of retention evidence on a single use,
    and the aged pool would run dry long before the stream did. The trailing
    remainder becomes a short probe rather than being dropped: dropping it would
    bias the pool toward whichever rows happen to sit at chunk starts.
    """
    if rows_per_probe < 1:
        raise ValueError(f"rows_per_probe={rows_per_probe} must be at least 1")
    out: list[Probe] = []
    n = chunk.n_rows
    for j, start in enumerate(range(0, n, rows_per_probe)):
        stop = min(start + rows_per_probe, n)
        out.append(
            Probe(
                probe_id=f"{prefix}{kind}:c{chunk.t}:r{j}",
                inputs=chunk.keys[start:stop].clone(),
                targets=chunk.values[start:stop].clone(),
                born=chunk.t,
                kind=kind,
                source_chunk=chunk.t,
            )
        )
    return out


def build_ledger(
    stream: SyntheticStream,
    *,
    rows_per_probe: int = 2,
    aged_budget: int = 2,
    audit_budget: int = 1,
    with_audit: bool = True,
) -> ProbeLedger:
    """Register the aged pool, and optionally a disjoint cheap-audit pool.

    The audit pool is what a *transactional* scheme would actually pay for: a
    small, cheap look at the memory immediately after committing a write, to
    decide whether to keep it. It is registered separately, from the same rows
    but under different ids and its own budget, because a post-write audit that
    spent the retention set's budget would make the retention measurement and
    the detector statistically dependent — and the decision tree's
    "not predictable beforehand, detectable afterwards" branch would then be
    reading its own answer.

    Note that this leaves the audit rows correlated with the aged rows (same
    underlying stream). That is a real limitation and is reported rather than
    papered over: it inflates how detectable damage looks after the fact.
    """
    ledger = ProbeLedger(default_budget=aged_budget)
    for chunk in stream.chunks:
        probes = chunk_to_probes(chunk, rows_per_probe=rows_per_probe, kind=AGED)
        ledger.register_all(probes, aged_budget)
        if with_audit:
            ledger.register_all(
                chunk_to_probes(chunk, rows_per_probe=rows_per_probe, kind=AUDIT),
                audit_budget,
            )
    return ledger


def build_aged_set(
    t: int,
    ledger: ProbeLedger,
    *,
    n_items: int,
    min_age: int,
    scope: SeedScope,
    kind: str = AGED,
    charge: bool = True,
) -> list[Probe]:
    """Draw ``A_t``: up to ``n_items`` aged, unspent, causally-available probes.

    Selection is uniform over the eligible pool via the repo's counter-based
    streams, so the draw at timestep ``t`` depends on ``t`` alone and is
    reproducible independently of what other timesteps did. It is *not*
    independent of the ledger's state, which is unavoidable: consumption is the
    mechanism.

    Returns fewer than ``n_items`` — possibly zero — when the pool is short.
    Callers must handle that instead of assuming a fixed sample size; late in a
    long stream the aged pool genuinely runs out, and silently topping it up
    from retired probes is the exact failure this design exists to prevent.
    """
    eligible = ledger.available(kind=kind, born_at_most=t - min_age)
    if not eligible:
        return []
    take = min(n_items, len(eligible))
    st = scope.stream(f"aged-set-{kind}")
    # Random keys, sort, take the smallest: a permutation without replacement
    # that needs no rejection loop and so consumes a fixed number of draws.
    keys = np.asarray(st.uniform(np.array([t], dtype=np.int64), len(eligible)))[0]
    picked = [eligible[i] for i in np.argsort(keys, kind="stable")[:take]]
    if not charge:
        return [ledger.peek(pid) for pid in picked]
    return ledger.draw(picked)


def future_chunks(
    chunks: Sequence[Chunk], t: int, horizons: Sequence[int]
) -> list[tuple[int, Chunk]]:
    """The chunks at ``t + h`` that actually exist, paired with their ``h``.

    Truncating rather than padding at the end of the stream: a padded horizon
    would contribute a zero benefit term and pull late-timestep utilities toward
    zero for a reason that has nothing to do with the write. The caller
    renormalizes the horizon weights over whatever survives.
    """
    out: list[tuple[int, Chunk]] = []
    for h in horizons:
        idx = t + h
        if 0 <= idx < len(chunks):
            out.append((h, chunks[idx]))
    return out
