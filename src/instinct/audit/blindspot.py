"""Selective-label blind spots: where a trigger's declines were never checked.

**This is evaluation infrastructure, not a contribution.** The selective-labels
problem is well known outside RL — it is the judge who denies bail and never
learns whether the defendant would have appeared (Lakkaraju et al. 2017,
*The Selective Labels Problem*) — and nothing in this module is new. It is here
because the audited-trigger arms can fail in exactly that shape, silently, and
because a detector is cheap while the failure is not.

The failure
-----------
A learned trigger decides when to spend compute, or when to accept a fast-weight
write. Whenever it *declines*, the outcome of accepting is never observed. If the
trigger declines a region early in training and keeps declining it, no data from
that region ever enters the training set, so the trigger's estimate there never
updates. It stays confident and it stays untested. The trigger then evaluates
beautifully — on the region it chose to be evaluated on.

The observable symptom is: a region of covariate space with many declines and no
audits. An audit is any record where the action was taken despite a decline —
forced exploration, a deliberate audit budget, an epsilon-floor. Audits are the
only counterfactual evidence that exists about the decline region, and if they
are absent, the trigger's behaviour there is a claim with no support.

What this module does *not* do
------------------------------
It does not test whether the trigger is right. It reports whether it *could have
been found wrong* — support, not correctness. A trigger flagged here may be
perfectly calibrated; the finding is that the experiment cannot tell. That is
still worth a red flag, because "we did not measure this region" is a result
under this repo's rules and "our trigger is accurate" is not, when they are
computed from the same data.

No functional form is assumed (rule 1). Coverage is measured two ways, both
nonparametric: an occupancy count on quantile bins, and a nearest-audit distance
in a standardized covariate space. They fail differently — binning is blind
inside a bin and suffers in high dimension, distances need a scale — so both are
reported and disagreement between them is itself informative.

Counts, not ratios (rule 2). A cell with 3 declines and 0 audits and a cell with
3,000 declines and 0 audits both have an uncovered fraction of 1.0, and they are
not the same finding. The report carries raw counts throughout, and the fractions
that appear alongside them always sit next to their denominators.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

__all__ = [
    "BlindspotReport",
    "CellCoverage",
    "detect_blindspots",
    "nearest_audit_distance",
    "quantile_cells",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

#: Chunk size for the pairwise distance computation. Big enough to amortize the
#: NumPy call, small enough that a 200k x 5k audit set does not materialize a
#: 4 GB matrix and take the process down.
_DIST_CHUNK = 512


@dataclass(frozen=True, slots=True)
class CellCoverage:
    """One region of covariate space, and what the audit budget saw there."""

    cell: tuple[int, ...]
    n_declined: int
    n_audited: int
    n_accepted: int

    @property
    def uncovered(self) -> bool:
        return self.n_declined > 0 and self.n_audited == 0


@dataclass
class BlindspotReport:
    """What the audit could and could not have discovered.

    ``flagged`` is the headline: at least one decline region carries no audit
    evidence. Read ``declines_without_audit`` (a count) before
    ``uncovered_fraction``.
    """

    n_records: int
    n_declined: int
    n_audited: int
    declines_without_audit: int
    n_cells: int
    n_cells_uncovered: int
    max_nearest_audit_distance: float
    isolated_declines: int
    radius: float
    min_audits_per_cell: int
    cells: list[CellCoverage] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return self.declines_without_audit > 0 or self.isolated_declines > 0

    @property
    def uncovered_fraction(self) -> float:
        """Share of declines in an uncovered cell. Always read with the counts.

        Returns 0.0 when there are no declines at all, which is the only sane
        reading: a trigger that never declines has no blind spot of this kind.
        """
        if self.n_declined == 0:
            return 0.0
        return self.declines_without_audit / self.n_declined

    def summary(self) -> str:
        head = "BLIND SPOT" if self.flagged else "covered"
        return (
            f"{head}: {self.declines_without_audit} of {self.n_declined} declines sit in "
            f"cells with fewer than {self.min_audits_per_cell} audits "
            f"({self.n_cells_uncovered} of {self.n_cells} occupied cells); "
            f"{self.isolated_declines} declines lie further than {self.radius:.3g} "
            f"from any audited record (worst {self.max_nearest_audit_distance:.3g}). "
            f"{self.n_audited} audits over {self.n_records} records."
        )


def quantile_cells(x: FloatArray, n_bins: int) -> IntArray:
    """Assign each record to a cell by per-column quantile bins.

    Quantile bins rather than equal-width bins because covariates here are
    things like planning budget, staleness and uncertainty, whose distributions
    are lumpy and long-tailed. Equal-width bins on such a covariate put 95% of
    the records in one bin and report near-perfect coverage while measuring
    nothing.

    Returns ``(n_records, n_features)`` bin indices. Degenerate columns (a
    single distinct value) collapse to bin 0, which is correct: a covariate that
    does not vary cannot partition anything.
    """
    arr = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if arr.shape[0] == 1 and np.asarray(x).ndim == 1:
        arr = arr.T
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    out = np.zeros(arr.shape, dtype=np.int64)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        # Interior quantile edges only; searchsorted supplies the outer bins.
        edges = np.unique(np.quantile(col, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
        if edges.size:
            out[:, j] = np.searchsorted(edges, col, side="right")
    return out


def nearest_audit_distance(
    declined: FloatArray, audited: FloatArray, *, scales: FloatArray | None = None
) -> FloatArray:
    """Distance from each declined record to the closest audited record.

    Standardized per column before measuring, because the covariates have
    incommensurable units — a budget of 512 and a staleness of 0.3 cannot be
    added under a Euclidean norm without one of them deciding the answer. The
    scale is the audited set's standard deviation by default, since that is the
    spread the audit budget actually explored.

    Returns ``inf`` for every record when there are no audits, which is the
    correct and loud answer rather than a zero that reads as "covered".

    This is the fast path: chunked ``(chunk, n_audited)`` distance blocks.
    :func:`_nearest_audit_distance_naive` is the reference, and
    ``tests/test_infra.py`` asserts they agree (rule 7).
    """
    d = _as_matrix(declined)
    a = _as_matrix(audited)
    if a.shape[0] == 0:
        return np.full(d.shape[0], np.inf)
    if d.shape[1] != a.shape[1]:
        raise ValueError(f"feature counts differ: declined {d.shape[1]}, audited {a.shape[1]}")

    scale = np.asarray(scales, dtype=np.float64) if scales is not None else a.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    d, a = d / scale, a / scale

    out = np.empty(d.shape[0], dtype=np.float64)
    for start in range(0, d.shape[0], _DIST_CHUNK):
        block = d[start : start + _DIST_CHUNK]
        # (chunk, n_audited, n_features) would blow memory on wide inputs; the
        # squared-norm expansion keeps it 2-D.
        sq = (
            np.sum(block**2, axis=1)[:, None]
            - 2.0 * block @ a.T
            + np.sum(a**2, axis=1)[None, :]
        )
        out[start : start + block.shape[0]] = np.sqrt(np.maximum(sq.min(axis=1), 0.0))
    return out


def _nearest_audit_distance_naive(
    declined: FloatArray, audited: FloatArray, *, scales: FloatArray | None = None
) -> FloatArray:
    """Reference: two loops and a norm. Tests only."""
    d = _as_matrix(declined)
    a = _as_matrix(audited)
    if a.shape[0] == 0:
        return np.full(d.shape[0], np.inf)
    scale = np.asarray(scales, dtype=np.float64) if scales is not None else a.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    out = np.empty(d.shape[0], dtype=np.float64)
    for i in range(d.shape[0]):
        best = np.inf
        for j in range(a.shape[0]):
            dist = float(np.sqrt(np.sum(((d[i] - a[j]) / scale) ** 2)))
            best = min(best, dist)
        out[i] = best
    return out


def _as_matrix(x: FloatArray) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"covariates must be 1-D or 2-D, got {arr.ndim}-D")
    return arr


def detect_blindspots(
    covariates: FloatArray,
    declined: BoolArray,
    audited: BoolArray,
    *,
    n_bins: int = 4,
    min_audits_per_cell: int = 1,
    radius: float = 1.0,
    cells: IntArray | None = None,
) -> BlindspotReport:
    """Flag decline regions with no audit coverage.

    ``declined[i]`` is whether the trigger said no for record ``i``.
    ``audited[i]`` is whether the action was taken anyway, so the counterfactual
    outcome *was* observed. An audit is only informative about the decline
    region if the trigger declined it, so records with ``audited & ~declined``
    are counted but do not provide decline coverage — they are ordinary
    accept-and-observe records.

    ``radius`` is in standardized units (see :func:`nearest_audit_distance`), so
    ``1.0`` means "further than one audited-set standard deviation from any
    audit". It is a knob, not a constant: pick it against the scale at which the
    trigger's decisions actually change and say what you picked.

    >>> import numpy as np
    >>> x = np.concatenate([np.zeros(50), np.ones(50) * 10.0])
    >>> dec = np.ones(100, dtype=bool)
    >>> aud = np.zeros(100, dtype=bool); aud[:5] = True   # only the low cluster audited
    >>> rep = detect_blindspots(x, dec, aud, n_bins=2)
    >>> rep.flagged
    True
    """
    x = _as_matrix(covariates)
    dec = np.asarray(declined, dtype=bool)
    aud = np.asarray(audited, dtype=bool)
    n = x.shape[0]
    if dec.shape != (n,) or aud.shape != (n,):
        raise ValueError(
            f"declined {dec.shape} and audited {aud.shape} must both be ({n},) "
            "to line up with the covariate rows"
        )

    cell_ids = quantile_cells(x, n_bins) if cells is None else _as_int_matrix(cells, n)

    # Audits only count as decline coverage where the trigger declined.
    decline_audit = dec & aud
    coverage: list[CellCoverage] = []
    declines_without_audit = 0
    keys, inverse = np.unique(cell_ids, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    for c in range(keys.shape[0]):
        member = inverse == c
        n_dec = int(np.sum(member & dec))
        n_aud = int(np.sum(member & decline_audit))
        entry = CellCoverage(
            cell=tuple(int(v) for v in keys[c]),
            n_declined=n_dec,
            n_audited=n_aud,
            n_accepted=int(np.sum(member & ~dec)),
        )
        coverage.append(entry)
        if n_dec > 0 and n_aud < min_audits_per_cell:
            declines_without_audit += n_dec

    dist = nearest_audit_distance(x[dec], x[decline_audit])
    isolated = int(np.sum(dist > radius))
    worst = float(dist.max()) if dist.size else 0.0

    return BlindspotReport(
        n_records=n,
        n_declined=int(dec.sum()),
        n_audited=int(decline_audit.sum()),
        declines_without_audit=declines_without_audit,
        n_cells=len(coverage),
        n_cells_uncovered=sum(1 for c in coverage if c.n_audited < min_audits_per_cell
                              and c.n_declined > 0),
        max_nearest_audit_distance=worst,
        isolated_declines=isolated,
        radius=radius,
        min_audits_per_cell=min_audits_per_cell,
        cells=coverage,
    )


def _as_int_matrix(cells: IntArray, n: int) -> IntArray:
    arr = np.asarray(cells, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] != n:
        raise ValueError(f"cells has {arr.shape[0]} rows, expected {n}")
    return arr
