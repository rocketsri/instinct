"""Model selection for the staleness curve, and the taxonomy it produces.

The output of this stage is deliberately *not* a single fitted law. P1's honest
prior is that the compute-freshness surface has several regimes — some cells
decay smoothly, some fall off a cliff, some lose the trajectory and never
recover — and that a universal collapse would be a surprise. So
:func:`select_families` returns a per-cell record of which candidate shape won
and by how much, and :func:`taxonomy` bins those records into named regimes. A
run that produced one regime everywhere would be evidence for a law; a run that
is coerced into reporting one regime is just a claim.

Three things keep the selection honest.

**The lookup table competes.** Every cell is also fitted by a saturated
per-staleness table, which assumes nothing at all. A parametric family that
cannot beat it on unseen points has not found structure, it has found a
convenient way to redescribe the training points. The share of cells where some
family beats the table is a headline number, and one of the kill conditions.

**Held-out points, and a finite number of looks at them.** Selection uses NLL on
``(start_state, budget)`` cells that no fit saw. :class:`HoldoutPlan` counts how
many times the held-out set has been consumed and refuses further looks past its
budget, because a holdout re-examined once per modelling idea is a training set
with extra steps.

**Discretization is measured, not assumed away.** Delays are integers, so a run
of budgets can share one delay and produce a plateau followed by a step. That
looks exactly like a threshold regime. :func:`discretization_report` measures how
much of a cell's explainable variance a table indexed *only by delay* already
accounts for, and the taxonomy will not award a threshold label to a step that
the delay grouping already explains.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import numpy.typing as npt

from instinct.atlas.curves import (
    FAMILIES,
    LOOKUP_BASELINE,
    CellKey,
    CurveData,
    CurveFit,
    fit_cells,
)

__all__ = [
    "REGIMES",
    "CellSelection",
    "DiscretizationReport",
    "FamilyScore",
    "HoldoutPlan",
    "aic",
    "bic",
    "discretization_report",
    "make_holdout",
    "select_families",
    "taxonomy",
    "taxonomy_rows",
]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

HoldoutScheme = Literal["interleave", "extrapolate", "random"]

REGIMES: tuple[str, ...] = (
    "flat",
    "smooth-decay",
    "threshold",
    "irreversible",
    "discretization-artifact",
    "unresolved",
)

_LOG_2PI = float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------
# Held-out splits, with a use budget
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Holdout:
    """Which points of one cell are visible to the fit."""

    train: BoolArray
    test: BoolArray

    def __post_init__(self) -> None:
        if self.train.shape != self.test.shape:
            raise ValueError("train and test masks must be aligned")
        if np.any(self.train & self.test):
            raise ValueError("a point cannot be both trained on and held out")


def make_holdout(
    cell: CurveData,
    *,
    scheme: HoldoutScheme = "interleave",
    frac: float = 0.3,
    seed: int = 0,
) -> Holdout:
    """Split one cell's budgets into a training and a held-out set.

    The schemes answer different questions and are not interchangeable.

    ``interleave``
        every ``k``-th budget in staleness order is held out. Tests
        *interpolation*: can the family fill in a budget between two it saw?
        This is what a controller choosing among measured budgets needs.
    ``extrapolate``
        the largest budgets are held out. Tests whether the shape is right
        beyond the measured range, which is where an exponential fitted to a
        plateau fails most spectacularly and where the lookup table is
        guaranteed to fail — hence not the default, since it flatters
        parametric families for a reason unrelated to their correctness.
    ``random``
        a random subset. Kept for a sanity check against ``interleave``, which
        is systematic and could in principle align with a periodic artifact of
        the budget grid.

    ``interleave`` never holds out the endpoints: with them gone every family is
    extrapolating and the comparison stops being about interpolation.
    """
    n = cell.n
    if n < 4:
        raise ValueError(f"cell {cell.key} has {n} points; need at least 4 to hold any out")
    order = np.argsort(cell.x, kind="stable")
    n_test = max(1, round(frac * n))
    n_test = min(n_test, n - 3)  # always leave enough to identify a 3-parameter shape

    test = np.zeros(n, dtype=bool)
    if scheme == "interleave":
        interior = order[1:-1]
        step = max(1, round(interior.size / n_test))
        picked = interior[::step][:n_test]
        test[picked] = True
    elif scheme == "extrapolate":
        test[order[-n_test:]] = True
    elif scheme == "random":
        rng = np.random.default_rng(seed)
        interior = order[1:-1]
        test[rng.choice(interior, size=min(n_test, interior.size), replace=False)] = True
    else:  # pragma: no cover - Literal keeps this unreachable from typed callers
        raise ValueError(f"unknown holdout scheme {scheme!r}")
    return Holdout(train=~test, test=test)


@dataclass(slots=True)
class HoldoutPlan:
    """A set of splits plus a hard cap on how often they may be scored.

    Rule 5 of the protocol. A held-out set that is consulted once per idea is a
    validation set, and by the tenth idea it is a training set that nobody is
    correcting for. The counter here does not make selection unbiased — nothing
    can, after the fact — but it makes the number of looks a recorded fact
    instead of an unknown, and it stops an interactive session from quietly
    running the twentieth variant against the same points.
    """

    splits: dict[tuple[str, str, float, float, int], Holdout]
    max_uses: int = 3
    uses: int = 0
    log: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.log is None:
            self.log = []

    def consume(self, label: str) -> None:
        if self.uses >= self.max_uses:
            raise RuntimeError(
                f"held-out set exhausted after {self.uses} uses ({self.log}); "
                f"refusing to score {label!r}. Re-split with a new seed and say so "
                "in the report, or stop tuning."
            )
        self.uses += 1
        self.log.append(label)

    def for_cell(self, key: CellKey) -> Holdout:
        return self.splits[key.as_tuple()]

    @classmethod
    def build(
        cls,
        cells: Sequence[CurveData],
        *,
        scheme: HoldoutScheme = "interleave",
        frac: float = 0.3,
        seed: int = 0,
        max_uses: int = 3,
    ) -> HoldoutPlan:
        splits = {
            c.key.as_tuple(): make_holdout(c, scheme=scheme, frac=frac, seed=seed + i)
            for i, c in enumerate(cells)
        }
        return cls(splits=splits, max_uses=max_uses)


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------


def _sse_floor(cell: CurveData) -> float:
    """A lower bound on residual scale, so log(0) never decides a comparison."""
    scale = max(cell.effect_size(), float(np.max(np.abs(cell.y))), 1.0)
    return (1e-9 * scale) ** 2


def aic(sse: float, n: int, k: float, *, floor: float = 0.0) -> float:
    """Gaussian AIC with the variance profiled out.

    ``k`` is a float because two families report effective rather than counted
    degrees of freedom; the formula is indifferent to that, the interpretation
    is what needs care.
    """
    s = max(sse, n * floor, 1e-300)
    return float(n * np.log(s / n) + 2.0 * k)


def bic(sse: float, n: int, k: float, *, floor: float = 0.0) -> float:
    s = max(sse, n * floor, 1e-300)
    return float(n * np.log(s / n) + k * np.log(max(n, 2)))


@dataclass(frozen=True, slots=True)
class FamilyScore:
    """One family's account of one cell. Residuals are part of the record, not a footnote."""

    family: str
    n_train: int
    n_test: int
    n_params: float
    train_sse: float
    aic: float
    bic: float
    test_sse: float
    test_nll: float  # per held-out point, nats
    test_rmse: float  # return units
    test_bias: float  # mean signed residual, return units
    test_max_abs: float


def _score_family(
    name: str,
    cell: CurveData,
    fit: CurveFit,
    hold: Holdout,
    *,
    noise_var: float,
) -> FamilyScore:
    floor = _sse_floor(cell)
    y_test = cell.y[hold.test]
    resid = y_test - fit.predict(cell.x[hold.test])
    n_test = int(y_test.size)
    sse_test = float(np.sum(resid**2))
    nll = 0.5 * (_LOG_2PI + float(np.log(noise_var)) + sse_test / (n_test * noise_var))
    n_train = int(hold.train.sum())
    return FamilyScore(
        family=name,
        n_train=n_train,
        n_test=n_test,
        n_params=fit.n_params,
        train_sse=fit.sse,
        aic=aic(fit.sse, n_train, fit.n_params, floor=floor),
        bic=bic(fit.sse, n_train, fit.n_params, floor=floor),
        test_sse=sse_test,
        test_nll=float(nll),
        test_rmse=float(np.sqrt(sse_test / max(n_test, 1))),
        test_bias=float(np.mean(resid)) if n_test else float("nan"),
        test_max_abs=float(np.max(np.abs(resid))) if n_test else float("nan"),
    )


# --------------------------------------------------------------------------
# Discretization diagnostics
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DiscretizationReport:
    """How much of a cell's structure is the integer delay rather than the environment.

    ``delay_explained`` is the share of the *explainable* variation (what the
    saturated per-budget table captures) that a table indexed only by delay
    already captures. A value near 1 means every budget that shares a delay also
    shares an outcome: the curve is a staircase in delay, and any "threshold" in
    it is the rounding rule. A value well below 1 means budgets buy something at
    fixed staleness — planner quality — and the curve has content the delay grid
    cannot produce.

    ``within_delay_spread`` is that same content in return units, which is the
    form the report should quote: it is the part of the budget effect that
    survives holding staleness exactly fixed.
    """

    n_points: int
    n_delay_groups: int
    tied_share: float  # fraction of points sharing their delay with another point
    delay_explained: float  # nan when there is nothing to explain
    within_delay_spread: float  # return units


def discretization_report(cell: CurveData) -> DiscretizationReport:
    """Separate "the delay changed" from "the planner got better" within one cell.

    Mirrors :meth:`instinct.core.env.Timing.distinct_delays`: budgets are grouped
    by the delay they actually produced, which the sweep recorded per row rather
    than recomputing here — recomputing would re-apply a rounding rule that the
    frame already resolved, and the two could disagree.
    """
    y = cell.y
    delays = cell.delay
    groups, inverse, counts = np.unique(delays, return_inverse=True, return_counts=True)
    grand = float(np.mean(y))
    sse_mean = float(np.sum((y - grand) ** 2))

    group_means = np.zeros(groups.size)
    np.add.at(group_means, inverse, y)
    group_means /= counts
    sse_delay = float(np.sum((y - group_means[inverse]) ** 2))

    # The saturated per-budget table has zero residual whenever budgets are
    # unique, which they are within a cell; keep the expression general anyway.
    sse_lookup = 0.0
    denom = sse_mean - sse_lookup
    explained = (sse_mean - sse_delay) / denom if denom > _sse_floor(cell) else float("nan")

    spreads = [
        float(np.max(y[inverse == g]) - np.min(y[inverse == g]))
        for g in range(groups.size)
        if counts[g] > 1
    ]
    tied = float(np.sum(counts[counts > 1]) / y.size) if y.size else 0.0
    return DiscretizationReport(
        n_points=int(y.size),
        n_delay_groups=int(groups.size),
        tied_share=tied,
        delay_explained=float(explained),
        within_delay_spread=float(np.mean(spreads)) if spreads else 0.0,
    )


# --------------------------------------------------------------------------
# Per-cell selection
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CellSelection:
    """Which family won this cell, by how much, and against what.

    ``margin_over_lookup`` is the number that matters most: held-out NLL of the
    lookup table minus that of the winning parametric family, in nats per
    held-out point. Positive means a functional form bought something. Negative
    or zero across the grid is a kill condition, not a disappointing detail.
    """

    key: CellKey
    n_points: int
    effect_size: float  # return units, peak to trough
    noise_scale: float  # return units, one standard error
    scores: dict[str, FamilyScore]
    winner: str
    runner_up: str
    margin_runner_up: float  # nats per point, positive = winner clearly ahead
    best_parametric: str
    margin_over_lookup: float  # nats per point
    beats_lookup: bool
    discretization: DiscretizationReport
    regime: str = "unclassified"
    regime_note: str = ""

    @property
    def winner_fit_note(self) -> str:
        s = self.scores[self.winner]
        return f"{self.winner}: rmse={s.test_rmse:.4g}, bias={s.test_bias:+.3g}, k={s.n_params:g}"


def _noise_variance(cell: CurveData) -> float:
    """The variance the held-out likelihood is measured against.

    Shared across families within a cell on purpose. That makes the NLL ranking
    inside a cell exactly the held-out-SSE ranking — the likelihood adds no
    information there — while putting cells with different response scales onto
    one additive scale so they can be pooled. Anything cleverer would be a noise
    model this sweep has not earned.

    The floor is a fraction of the cell's own effect size, so an exact tabular
    arm (which reports a degenerate interval) does not divide by zero and does
    not get infinite leverage over the pooled numbers.
    """
    reported = float(np.mean(cell.noise**2))
    floor = (0.01 * max(cell.effect_size(), 1e-9)) ** 2
    return max(reported, floor, 1e-300)


def select_families(
    cells: Sequence[CurveData],
    *,
    names: Sequence[str] | None = None,
    plan: HoldoutPlan | None = None,
    scheme: HoldoutScheme = "interleave",
    frac: float = 0.3,
    seed: int = 0,
    label: str = "select_families",
    batched: bool = True,
) -> list[CellSelection]:
    """Fit every family on training points and score them on held-out points.

    The training subsets are fitted through :func:`instinct.atlas.curves.fit_cells`,
    so cells that share a budget grid still share one factorization even after
    the split — the holdout does not cost the batching.
    """
    if not cells:
        return []
    family_names = list(FAMILIES) if names is None else list(names)
    if LOOKUP_BASELINE not in family_names:
        family_names = [*family_names, LOOKUP_BASELINE]

    plan = plan or HoldoutPlan.build(cells, scheme=scheme, frac=frac, seed=seed)
    plan.consume(label)

    holds = [plan.for_cell(c.key) for c in cells]
    train_cells = [c.subset(h.train) for c, h in zip(cells, holds)]
    fits = {n: fit_cells(n, train_cells, batched=batched) for n in family_names}

    out: list[CellSelection] = []
    for i, cell in enumerate(cells):
        nv = _noise_variance(cell)
        scores = {n: _score_family(n, cell, fits[n][i], holds[i], noise_var=nv) for n in fits}
        ranked = sorted(scores.values(), key=lambda s: s.test_nll)
        winner, runner_up = ranked[0], ranked[1] if len(ranked) > 1 else ranked[0]

        parametric = [s for s in scores.values() if FAMILIES[s.family].parametric]
        best_par = min(parametric, key=lambda s: s.test_nll)
        lookup = scores[LOOKUP_BASELINE]

        out.append(
            CellSelection(
                key=cell.key,
                n_points=cell.n,
                effect_size=cell.effect_size(),
                noise_scale=float(np.sqrt(nv)),
                scores=scores,
                winner=winner.family,
                runner_up=runner_up.family,
                margin_runner_up=float(runner_up.test_nll - winner.test_nll),
                best_parametric=best_par.family,
                margin_over_lookup=float(lookup.test_nll - best_par.test_nll),
                # KNOWN DEFECT -- this comparison is not yet a valid kill-condition
                # test, and tests/test_atlas_fit.py carries the strict xfail.
                #
                # Two problems compound. Held-out NLL is the sole criterion, with
                # no complexity penalty, so nothing charges a family for its
                # parameters. And under the `interleave` holdout the lookup table
                # is structurally handicapped: it is asked to predict at
                # staleness values it never saw, which a saturated per-x table
                # cannot do by construction, while any smooth family simply
                # interpolates between the neighbours it retained.
                #
                # Together those make `beats_lookup` clearable on pure noise --
                # measured at +0.36 nats/point for pwlinear3 on independent
                # random values. Since this comparison *is* P1's "no staleness
                # law" kill condition, it currently cannot detect the outcome it
                # exists to detect, and must not be reported as if it could.
                #
                # The fix is a design decision, not a tweak: either score
                # selection on held-out NLL plus an explicit complexity penalty,
                # or compare against the lookup baseline in-sample by AIC/BIC
                # where its degrees of freedom are counted honestly. Picking one
                # without checking it against the known-answer suite would just
                # move the bias somewhere less visible.
                beats_lookup=bool(best_par.test_nll < lookup.test_nll),
                discretization=discretization_report(cell),
            )
        )
    return taxonomy(cells, out)


# --------------------------------------------------------------------------
# Taxonomy
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaxonomyThresholds:
    """The dials the regime labels turn on. Stated here so a reader can disagree with them."""

    flat_snr: float = 2.0  # effect must exceed this many noise scales to be a curve at all
    sharp_transition: float = 4.0  # logistic k*span above this reads as a step, not a slope
    delay_explained: float = 0.98  # above this, a step is the rounding rule
    floor_flatness: float = 0.15  # last-third spread below this share of the drop is a floor
    floor_drop: float = 0.6  # ...and the drop must be this share of the total effect


def taxonomy(
    cells: Sequence[CurveData],
    selections: Sequence[CellSelection],
    *,
    thresholds: TaxonomyThresholds | None = None,
) -> list[CellSelection]:
    """Label each cell with a regime. A piecewise taxonomy, not a universal law.

    The order of the tests is the argument:

    1. **unresolved** — the lookup table was not beaten. Nothing about shape can
       be claimed here, so no shape label is given. Calling such a cell
       "smooth-decay" because an exponential happened to rank first among the
       losers is exactly the error this module exists to prevent.
    2. **flat** — the whole budget effect is within a couple of noise scales.
       No regime, and a warning that this cell contributes nothing to any fit.
    3. **discretization-artifact** — a step-like family won, but a table indexed
       only by the integer delay already explains the cell. The step is the
       rounding rule.
    4. **threshold** — a step-like family won and survived that check.
    5. **irreversible** — the curve drops and then floors: further staleness
       costs nothing more because the trajectory is already lost.
    6. **smooth-decay** — everything else that beat the table.
    """
    th = thresholds or TaxonomyThresholds()
    by_key = {c.key.as_tuple(): c for c in cells}
    labelled: list[CellSelection] = []
    for sel in selections:
        cell = by_key[sel.key.as_tuple()]
        regime, note = _classify(cell, sel, th)
        labelled.append(replace(sel, regime=regime, regime_note=note))
    return labelled


_STEP_LIKE = frozenset({"threshold", "hinge", "logistic"})


def _classify(
    cell: CurveData, sel: CellSelection, th: TaxonomyThresholds
) -> tuple[str, str]:
    snr = sel.effect_size / max(sel.noise_scale, 1e-300)
    if snr < th.flat_snr:
        return "flat", f"effect {sel.effect_size:.3g} is {snr:.1f} noise scales"
    if not sel.beats_lookup:
        return (
            "unresolved",
            f"no parametric family beat the lookup table "
            f"(best {sel.best_parametric}, margin {sel.margin_over_lookup:+.3g} nats)",
        )

    win = sel.best_parametric
    step_like = win in _STEP_LIKE
    if win == "logistic":
        # A logistic with a gentle slope is a smooth decay, whatever it is called.
        span = float(np.max(cell.x) - np.min(cell.x)) or 1.0
        fit_phi = sel.scores[win]
        del fit_phi  # phi lives on the fit, not the score; recovered below
        step_like = _logistic_is_sharp(cell, th.sharp_transition, span)

    if step_like:
        d = sel.discretization
        if np.isfinite(d.delay_explained) and d.delay_explained >= th.delay_explained:
            return (
                "discretization-artifact",
                f"delay table explains {d.delay_explained:.3f} of the explainable "
                f"variation across {d.n_delay_groups} delays; the step is rounding",
            )
        return (
            "threshold",
            f"{win} wins; delay table explains only {d.delay_explained:.3f}, "
            f"within-delay spread {d.within_delay_spread:.3g} return units",
        )

    if _has_floor(cell, th):
        return "irreversible", "curve drops and then floors: further staleness costs nothing"
    return "smooth-decay", f"{win} beats the table by {sel.margin_over_lookup:+.3g} nats/point"


def _logistic_is_sharp(cell: CurveData, sharp: float, span: float) -> bool:
    """Refit the logistic to read its transition width, in units of the x-span."""
    fit = FAMILIES["logistic"].fit_one(cell.x, cell.y, weights=cell.weight)
    return bool(np.isfinite(fit.phi[0]) and fit.phi[0] * span >= sharp)


def _has_floor(cell: CurveData, th: TaxonomyThresholds) -> bool:
    """Does the response fall and then stop changing?

    Read off the data rather than a fitted family, so the label does not depend
    on which of two near-tied shapes happened to win.
    """
    order = np.argsort(cell.x, kind="stable")
    y = cell.y[order]
    if y.size < 6:
        return False
    total = float(np.max(y) - np.min(y))
    if total <= 0:
        return False
    tail = y[(2 * y.size) // 3 :]
    head_drop = float(y[0] - np.min(y[: (2 * y.size) // 3]))
    tail_spread = float(np.max(tail) - np.min(tail))
    return tail_spread <= th.floor_flatness * total and head_drop >= th.floor_drop * total


def taxonomy_rows(selections: Sequence[CellSelection]) -> list[dict[str, object]]:
    """Regime counts, ready for a markdown table in the report header."""
    counts: dict[str, int] = dict.fromkeys(REGIMES, 0)
    for s in selections:
        counts[s.regime] = counts.get(s.regime, 0) + 1
    total = max(len(selections), 1)
    return [
        {"regime": r, "cells": counts.get(r, 0), "share": f"{counts.get(r, 0) / total:.1%}"}
        for r in REGIMES
        if counts.get(r, 0) or r in ("unresolved", "threshold")
    ]
