"""Candidate shapes for the staleness curve, and the batched machinery to fit them.

The methodological point of this module is what it *refuses* to do. The tempting
move in P1 is to fit ``sigma ~ exp(-h * tau)``, report ``h`` per environment and
call it a law. That is close to circular: an exponential is the unique shape with
a constant hazard, so imposing it assumes precisely the memorylessness the study
is supposed to test, and it will report a plausible ``h`` even for a surface that
is a step, a plateau, or pure noise. So the exponential is one entry in a family
list here, never the family list.

The list spans genuinely different mechanisms:

======================  ====================================================
``constant``            budget does not matter. The null that must be beaten.
``exponential``         constant-hazard decay: memoryless staleness.
``power``               heavy-tailed decay: no characteristic timescale.
``threshold``           a discontinuous step: the plan is fine until it isn't.
``hinge``               flat, then linear: a deadline with graceful decay.
``pwlinear2/3``         one or two knots: piecewise regimes with slopes.
``logistic``            a smooth threshold with a finite transition width.
``isotonic``            monotone but otherwise unconstrained (nonparametric).
``gp``                  smooth but otherwise unconstrained (kernel ridge).
``lookup``              per-cell table of the distinct staleness values.
======================  ====================================================

``lookup`` is the one that decides whether there is a paper. It is the honest
saturated baseline: one free value per distinct staleness, no structure assumed.
If no parametric family beats it out of sample, then the surface has no
compressible shape and "the staleness curve" does not exist as a law — which is
one of P1's stated kill conditions, not a nuisance to be tuned away.

Performance
-----------
Every family here except ``isotonic`` and ``gp`` is *linear in its coefficients
given its nonlinear parameters* (a knot location, a decay rate). That structure
is what makes the batched path worthwhile: for one candidate nonlinear parameter
the design matrix depends only on ``x``, and all cells that share a budget grid
share that matrix. So a whole grid of cells is one :func:`numpy.linalg.lstsq`
with many right-hand sides per candidate, instead of one solve per cell per
candidate. :func:`fit_cells` does the grouping; ``batched=False`` runs the same
fits one cell at a time and exists so the equivalence test in
``tests/test_atlas_curves.py`` can hold the fast path to the slow one.

The two genuinely nonlinear families (``exponential``, ``logistic``, and
``power``'s exponent) additionally warm-start each cell's local refinement from
the neighbouring grid cell's solution, since adjacent cells of the sweep differ
by one step in ``nu_e`` or ``nu_h`` and their optima are close.

Isotonic regression is batched a different way: the min-max representation
``yhat_i = max_{k<=i} min_{j>=i} avg(k..j)`` is a pure array reduction, so all
cells are solved at once with no per-cell pool-adjacent-violators loop and no
data-dependent control flow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import optimize

__all__ = [
    "FAMILIES",
    "CellKey",
    "ConstantFamily",
    "CurveData",
    "CurveFamily",
    "CurveFit",
    "ExponentialFamily",
    "GPFamily",
    "HingeFamily",
    "IsotonicFamily",
    "LogisticFamily",
    "LookupFamily",
    "PiecewiseLinearFamily",
    "PowerFamily",
    "ThresholdFamily",
    "batched_lstsq",
    "cells_from_frame",
    "family",
    "fit_cells",
    "parametric_family_names",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

_TINY = 1e-12


# --------------------------------------------------------------------------
# The unit of fitting: one cell's budget frontier
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CellKey:
    """Everything held fixed while the budget is swept.

    ``nu_e`` and ``nu_h`` stay separate here for the same reason
    :class:`instinct.core.env.Timing` keeps them separate: whether the surface
    collapses onto their product is a hypothesis
    (:mod:`instinct.atlas.transfer` tests it), and a key that folded them
    together would make the hypothesis unfalsifiable by construction.
    """

    env: str
    reflex: str
    nu_e: float
    nu_h: float
    start_state: int = -1

    def as_tuple(self) -> tuple[str, str, float, float, int]:
        return (self.env, self.reflex, self.nu_e, self.nu_h, self.start_state)

    def __str__(self) -> str:
        return (
            f"{self.env}/{self.reflex}/nu_e={self.nu_e:g}/nu_h={self.nu_h:g}"
            f"/s0={self.start_state}"
        )


@dataclass(frozen=True, slots=True)
class CurveData:
    """One cell's frontier: a response sampled along a budget axis.

    ``x`` is whatever axis the caller chose to fit against (continuous staleness
    by default, but budget or integer delay are equally valid and the transfer
    tests use all three). ``budget`` and ``delay`` are carried alongside
    regardless, because the discretization confound can only be diagnosed by
    knowing which points collapsed onto the same delay.
    """

    key: CellKey
    x: FloatArray
    y: FloatArray
    budget: IntArray
    delay: IntArray
    weight: FloatArray
    x_name: str = "staleness"
    y_name: str = "sigma"

    def __post_init__(self) -> None:
        n = self.x.shape[0]
        for name in ("y", "budget", "delay", "weight"):
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(f"{name} has shape {arr.shape}, expected ({n},)")
        if n == 0:
            raise ValueError("a curve needs at least one point")

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    def effect_size(self) -> float:
        """Peak-to-trough spread of the response, in return units.

        The scale every error in this module is judged against. Deliberately a
        difference and not a ratio: ``sigma`` legitimately passes through zero,
        so a relative error would explode exactly where the surface is flattest.
        """
        return float(np.max(self.y) - np.min(self.y)) if self.n else 0.0

    def subset(self, mask: npt.NDArray[np.bool_]) -> CurveData:
        return CurveData(
            key=self.key,
            x=self.x[mask],
            y=self.y[mask],
            budget=self.budget[mask],
            delay=self.delay[mask],
            weight=self.weight[mask],
            x_name=self.x_name,
            y_name=self.y_name,
        )


def cells_from_frame(
    df: pd.DataFrame,
    *,
    x: str = "staleness",
    y: str = "sigma",
    weight: str | None = None,
) -> list[CurveData]:
    """Group atlas rows into per-cell budget frontiers.

    Cells come back in a deterministic grid order (env, reflex, ``nu_e``,
    ``nu_h``, start state) and points within a cell in ascending ``x``. Both
    orderings are load-bearing: the first is what makes "warm-start from the
    neighbouring grid cell" mean anything, and the second is what lets the
    isotonic and knot-search fits assume a sorted axis.

    ``weight`` names a column of per-point weights; the default weights every
    point equally. Weighting by the reported CI width is available but is *not*
    the default, because the exact tabular arm reports a degenerate interval and
    silently upweighting it to infinity would let one arm of the sweep decide
    the model selection for all of them.
    """
    for col in (x, y, "env", "reflex", "nu_e", "nu_h", "budget", "delay"):
        if col not in df.columns:
            raise ValueError(f"frame is missing column {col!r}")
    if weight is not None and weight not in df.columns:
        raise ValueError(f"frame is missing weight column {weight!r}")

    group_cols = ["env", "reflex", "nu_e", "nu_h", "start_state"]
    if "start_state" not in df.columns:
        raise ValueError("frame is missing column 'start_state'")

    cells: list[CurveData] = []
    for keys, sub in df.groupby(group_cols, sort=True):
        sub = sub.sort_values(x, kind="stable")
        env, reflex, nu_e, nu_h, start_state = keys  # type: ignore[misc]
        w = (
            np.asarray(sub[weight], dtype=np.float64)
            if weight is not None
            else np.ones(len(sub), dtype=np.float64)
        )
        cells.append(
            CurveData(
                key=CellKey(str(env), str(reflex), float(nu_e), float(nu_h), int(start_state)),
                x=np.asarray(sub[x], dtype=np.float64),
                y=np.asarray(sub[y], dtype=np.float64),
                budget=np.asarray(sub["budget"], dtype=np.int64),
                delay=np.asarray(sub["delay"], dtype=np.int64),
                weight=w,
                x_name=x,
                y_name=y,
            )
        )
    return cells


# --------------------------------------------------------------------------
# The result of a fit
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CurveFit:
    """A fitted curve, carrying enough state to predict at unseen ``x``.

    ``n_params`` is a float because two of the families do not have an integer
    count: the smoother's effective degrees of freedom is the trace of its hat
    matrix, and isotonic's is its number of level sets. Rounding either one up
    to "nonparametric, so infinite" would make the information criteria refuse
    to compare them; rounding down to 1 would let them win everything.
    """

    family: str
    beta: FloatArray  # linear coefficients, or per-knot levels for the tabular fits
    phi: FloatArray  # nonlinear parameters: rates, knots, lengthscales
    sse: float  # weighted in-sample sum of squares
    n_points: int
    n_params: float
    x_train: FloatArray
    fitted: FloatArray

    def predict(self, x: FloatArray) -> FloatArray:
        return FAMILIES[self.family].predict(self, np.asarray(x, dtype=np.float64))

    def residuals(self, y: FloatArray) -> FloatArray:
        return np.asarray(y, dtype=np.float64) - self.fitted

    @property
    def rmse(self) -> float:
        return float(np.sqrt(self.sse / max(self.n_points, 1)))


# --------------------------------------------------------------------------
# Batched linear algebra
# --------------------------------------------------------------------------


def batched_lstsq(A: FloatArray, Y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Least squares for one design and many responses.

    ``A`` is ``(m, p)``, ``Y`` is ``(m, n)``; returns coefficients ``(p, n)``
    and the per-column sum of squared residuals ``(n,)``. This is the whole
    batching trick: LAPACK factorizes ``A`` once and back-substitutes ``n``
    times, so the marginal cost of another cell is a triangular solve rather
    than another factorization.

    The residual is recomputed explicitly rather than read from ``lstsq``'s
    third return value, which is empty whenever the design is rank-deficient —
    exactly the case a knot search hits when a candidate knot falls outside the
    data and two basis columns coincide.
    """
    coef, _res, _rank, _sv = np.linalg.lstsq(A, Y, rcond=None)
    resid = Y - A @ coef
    sse = np.einsum("mn,mn->n", resid, resid)
    return np.asarray(coef, dtype=np.float64), np.asarray(sse, dtype=np.float64)


# --------------------------------------------------------------------------
# Family protocol
# --------------------------------------------------------------------------


class CurveFamily(ABC):
    """A candidate shape, fittable to many cells at once."""

    name: ClassVar[str] = "abstract"
    parametric: ClassVar[bool] = True

    @abstractmethod
    def fit_batch(
        self,
        x: FloatArray,
        Y: FloatArray,
        *,
        weights: FloatArray | None = None,
        warm_start: FloatArray | None = None,
    ) -> list[CurveFit]:
        """Fit every column of ``Y`` against the shared axis ``x``."""

    @abstractmethod
    def predict(self, fit: CurveFit, x: FloatArray) -> FloatArray:
        """Evaluate a fit at arbitrary ``x``, including outside the training range."""

    def fit_one(
        self,
        x: FloatArray,
        y: FloatArray,
        *,
        weights: FloatArray | None = None,
        warm_start: FloatArray | None = None,
    ) -> CurveFit:
        ws = None if warm_start is None else np.atleast_2d(warm_start)
        return self.fit_batch(x, y[:, None], weights=weights, warm_start=ws)[0]


def _weights_or_ones(weights: FloatArray | None, m: int) -> FloatArray:
    if weights is None:
        return np.ones(m, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (m,):
        raise ValueError(f"weights have shape {w.shape}, expected ({m},)")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    return w


# --------------------------------------------------------------------------
# Separable families: linear in beta given phi
# --------------------------------------------------------------------------


class SeparableFamily(CurveFamily):
    """Base for families that are linear once their nonlinear parameters are fixed.

    Fitting is variable projection: enumerate candidate ``phi``, solve the linear
    problem exactly for each, keep the best. That is more robust than throwing
    the whole parameter vector at a general optimizer, which for a knot model
    lands in a local minimum whenever the initial knot sits in the wrong segment
    — a failure that looks like "the piecewise model didn't help", i.e. like a
    scientific result rather than the numerical accident it is.
    """

    n_phi: ClassVar[int] = 0
    refine: ClassVar[bool] = False

    @abstractmethod
    def phi_candidates(self, x: FloatArray) -> FloatArray:
        """Candidate nonlinear parameters, shape ``(n_candidates, n_phi)``."""

    @abstractmethod
    def design(self, phi: FloatArray, x: FloatArray) -> FloatArray:
        """Design matrix ``(m, p)`` for one candidate."""

    def phi_bounds(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Box constraints for the local refinement; unused when ``refine`` is False."""
        raise NotImplementedError

    def n_params_for(self, phi: FloatArray, beta: FloatArray) -> float:
        return float(beta.shape[0] + self.n_phi)

    # -- fitting ----------------------------------------------------------

    def fit_batch(
        self,
        x: FloatArray,
        Y: FloatArray,
        *,
        weights: FloatArray | None = None,
        warm_start: FloatArray | None = None,
    ) -> list[CurveFit]:
        x = np.asarray(x, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        m, n = Y.shape
        w = _weights_or_ones(weights, m)
        sw = np.sqrt(w)[:, None]
        Yw = Y * sw

        cands = np.atleast_2d(np.asarray(self.phi_candidates(x), dtype=np.float64))
        if cands.size == 0:
            cands = np.zeros((1, self.n_phi))

        best_sse = np.full(n, np.inf)
        best_phi = np.zeros((n, max(self.n_phi, 1)))
        best_beta: list[FloatArray] = [np.zeros(1) for _ in range(n)]

        feasible = 0
        for phi in cands:
            A = self.design(phi, x)
            if A.shape[1] > m:
                continue
            feasible += 1
            coef, sse = batched_lstsq(A * sw, Yw)
            better = sse < best_sse
            if np.any(better):
                best_sse = np.where(better, sse, best_sse)
                idx = np.flatnonzero(better)
                for j in idx:
                    best_phi[j, : self.n_phi] = phi
                    best_beta[j] = coef[:, j]
        if feasible == 0:
            # Not enough distinct points to identify this shape. Fall back to the
            # cell mean rather than inventing a fit; selection will drop it.
            A = np.ones((m, 1))
            coef, sse = batched_lstsq(A * sw, Yw)
            return [
                CurveFit(
                    family=self.name,
                    beta=coef[:, j],
                    phi=np.zeros(self.n_phi),
                    sse=float(sse[j]),
                    n_points=m,
                    n_params=1.0,
                    x_train=x,
                    fitted=A @ coef[:, j],
                )
                for j in range(n)
            ]

        if self.refine:
            self._refine_columns(x, Yw, sw, best_phi, best_beta, best_sse, warm_start)

        fits: list[CurveFit] = []
        for j in range(n):
            phi_j = best_phi[j, : self.n_phi]
            A = self.design(phi_j, x)
            fits.append(
                CurveFit(
                    family=self.name,
                    beta=best_beta[j],
                    phi=phi_j.copy(),
                    sse=float(best_sse[j]),
                    n_points=m,
                    n_params=self.n_params_for(phi_j, best_beta[j]),
                    x_train=x,
                    fitted=A @ best_beta[j],
                )
            )
        return fits

    def _solve_at(
        self, phi: FloatArray, x: FloatArray, Yw_col: FloatArray, sw: FloatArray
    ) -> tuple[FloatArray, float]:
        A = self.design(phi, x) * sw
        coef, sse = batched_lstsq(A, Yw_col[:, None])
        return coef[:, 0], float(sse[0])

    def _refine_columns(
        self,
        x: FloatArray,
        Yw: FloatArray,
        sw: FloatArray,
        best_phi: FloatArray,
        best_beta: list[FloatArray],
        best_sse: FloatArray,
        warm_start: FloatArray | None,
    ) -> None:
        """Polish each column's nonlinear parameters, warm-started from its neighbour.

        Columns arrive in grid order, so column ``j-1`` is one step away in
        ``nu_e`` or ``nu_h`` and its optimum is the cheapest good guess
        available. The grid solution is kept as a fallback start, so a warm start
        that happens to be bad costs a few evaluations and never a worse fit:
        the refinement is only accepted when it lowers the weighted SSE.
        """
        lo, hi = self.phi_bounds(x)
        n = Yw.shape[1]
        previous: FloatArray | None = None
        for j in range(n):
            starts: list[FloatArray] = [best_phi[j, : self.n_phi].copy()]
            if previous is not None:
                starts.append(previous.copy())
            if warm_start is not None and warm_start.shape[0] > j:
                starts.append(np.asarray(warm_start[j, : self.n_phi], dtype=np.float64))

            scored = []
            for s in starts:
                s = np.clip(s, lo, hi)
                _beta, sse = self._solve_at(s, x, Yw[:, j], sw)
                scored.append((sse, s))
            scored.sort(key=lambda t: t[0])
            phi0 = scored[0][1]

            def residual(p: FloatArray, col: int = j) -> FloatArray:
                A = self.design(p, x) * sw
                coef, _ = batched_lstsq(A, Yw[:, col][:, None])
                return np.asarray(A @ coef[:, 0] - Yw[:, col], dtype=np.float64)

            try:
                sol = optimize.least_squares(
                    residual, phi0, bounds=(lo, hi), max_nfev=60, xtol=1e-10, ftol=1e-12
                )
            except (ValueError, np.linalg.LinAlgError):
                previous = phi0
                continue

            phi_new = np.asarray(sol.x, dtype=np.float64)
            beta_new, sse_new = self._solve_at(phi_new, x, Yw[:, j], sw)
            if sse_new < best_sse[j]:
                best_sse[j] = sse_new
                best_phi[j, : self.n_phi] = phi_new
                best_beta[j] = beta_new
            previous = best_phi[j, : self.n_phi].copy()

    def predict(self, fit: CurveFit, x: FloatArray) -> FloatArray:
        A = self.design(fit.phi, np.asarray(x, dtype=np.float64))
        if A.shape[1] != fit.beta.shape[0]:
            # Degenerate fallback fit (cell mean) evaluated at new points.
            return np.full(x.shape[0], float(fit.beta[0]))
        return np.asarray(A @ fit.beta, dtype=np.float64)


def _interior_knots(x: FloatArray, max_knots: int = 24) -> FloatArray:
    """Candidate knot locations: midpoints between consecutive distinct ``x``.

    Midpoints rather than data points so that a step model's discontinuity never
    lands *on* an observation, where its value would be decided by a tie-break
    rather than by the data.
    """
    u = np.unique(x)
    if u.size < 3:
        return np.zeros(0)
    mids = 0.5 * (u[:-1] + u[1:])
    if mids.size > max_knots:
        idx = np.linspace(0, mids.size - 1, max_knots).round().astype(int)
        mids = mids[np.unique(idx)]
    return mids


def _span(x: FloatArray) -> float:
    s = float(np.max(x) - np.min(x))
    return s if s > _TINY else 1.0


class ConstantFamily(SeparableFamily):
    """``y = c``. Budget buys nothing.

    Present because "the surface is flat within noise" is a real and reportable
    outcome, and because every other family's improvement has to be measured
    against something.
    """

    name: ClassVar[str] = "constant"
    n_phi: ClassVar[int] = 0

    def phi_candidates(self, x: FloatArray) -> FloatArray:
        return np.zeros((1, 0))

    def design(self, phi: FloatArray, x: FloatArray) -> FloatArray:
        return np.ones((x.shape[0], 1))


class ExponentialFamily(SeparableFamily):
    """``y = a + b * exp(-h * x)``: decay with a constant hazard.

    ``h`` is constrained positive and the sign of ``b`` carries the direction,
    so this covers saturating growth as well as decay but never runaway growth —
    a staleness curve that diverges with staleness is not a hypothesis worth
    keeping in the pool.
    """

    name: ClassVar[str] = "exponential"
    n_phi: ClassVar[int] = 1
    refine: ClassVar[bool] = True

    def phi_candidates(self, x: FloatArray) -> FloatArray:
        s = _span(x)
        return np.geomspace(0.02 / s, 20.0 / s, 24)[:, None]

    def phi_bounds(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        s = _span(x)
        return np.array([1e-4 / s]), np.array([1e3 / s])

    def design(self, phi: FloatArray, x: FloatArray) -> FloatArray:
        h = float(phi[0])
        return np.column_stack([np.ones_like(x), np.exp(-h * (x - float(np.min(x))))])


class PowerFamily(SeparableFamily):
    """``y = a + b * (1 + x/s)**(-p)``: decay with no characteristic timescale.

    The offset ``1 +`` keeps the basis finite at ``x = 0``, which matters because
    the smallest budget in every sweep sits at or near zero staleness and an
    unshifted power law would put an asymptote exactly there.
    """

    name: ClassVar[str] = "power"
    n_phi: ClassVar[int] = 1
    refine: ClassVar[bool] = True

    def phi_candidates(self, x: FloatArray) -> FloatArray:
        return np.geomspace(0.05, 12.0, 24)[:, None]

    def phi_bounds(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        return np.array([1e-3]), np.array([60.0])

    def design(self, phi: FloatArray, x: FloatArray) -> FloatArray:
        p = float(phi[0])
        s = _span(x)
        z = np.maximum(1.0 + (x - float(np.min(x))) / s, _TINY)
        return np.column_stack([np.ones_like(x), z**(-p)])


class ThresholdFamily(SeparableFamily):
    """``y = a + b * 1[x > t]``: a cliff.

    The family most at risk of being an artifact. Delays are integers, so a run
    of budgets can share one delay and produce a step that belongs to the
    rounding rule, not to the environment. Winning here is therefore never
    enough to claim a threshold regime; :func:`instinct.atlas.fit.taxonomy`
    cross-checks against the delay grouping before it will use the label.
    """

    name: ClassVar[str] = "threshold"
    n_phi: ClassVar[int] = 1

    def phi_candidates(self, x: FloatArray) -> FloatArray:
        return _interior_knots(x)[:, None]

    def design(self, phi: FloatArray, x: FloatArray) -> FloatArray:
        t = float(phi[0])
        return np.column_stack([np.ones_like(x), (x > t).astype(np.float64)])


class HingeFamily(SeparableFamily):
    """``y = a + b * max(x - t, 0)``: flat, then linear.

    A deadline model. Distinct from :class:`PiecewiseLinearFamily` with one knot
    because it forces the first segment to be exactly flat, which is a stronger
    and cheaper claim than "two slopes".
    """

    name: ClassVar[str] = "hinge"
    n_phi: ClassVar[int] = 1

    def phi_candidates(self, x: FloatArray) -> FloatArray:
        return _interior_knots(x)[:, None]

    def design(self, phi: FloatArray, x: FloatArray) -> FloatArray:
        t = float(phi[0])
        return np.column_stack([np.ones_like(x), np.maximum(x - t, 0.0)])


class PiecewiseLinearFamily(SeparableFamily):
    """Continuous piecewise linear with ``n_knots`` interior breakpoints.

    Two knots is where the taxonomy's "irreversible" shape lives: an initial
    slope, a steeper drop once the plan goes stale, and a floor where the
    trajectory has already been lost and further staleness costs nothing more.
    Fitting the knots by exhaustive search over midpoint pairs is affordable
    (a few hundred designs) and, unlike gradient descent on the knots, cannot
    stall in the wrong segment.
    """

    n_phi: ClassVar[int] = 0  # set per instance below

    def __init__(self, n_knots: int = 1) -> None:
        if n_knots not in (1, 2):
            raise ValueError("only 1- and 2-knot piecewise linear fits are in the pool")
        self.n_knots = n_knots
        self.name = f"pwlinear{n_knots + 1}"  # type: ignore[misc]
        self.n_phi = n_knots  # type: ignore[misc]

    def phi_candidates(self, x: FloatArray) -> FloatArray:
        mids = _interior_knots(x)
        if mids.size == 0:
            return np.zeros((0, self.n_knots))
        if self.n_knots == 1:
            return mids[:, None]
        pairs = [(a, b) for i, a in enumerate(mids) for b in mids[i + 1 :]]
        if not pairs:
            return np.zeros((0, 2))
        return np.asarray(pairs, dtype=np.float64)

    def design(self, phi: FloatArray, x: FloatArray) -> FloatArray:
        cols = [np.ones_like(x), x]
        cols.extend(np.maximum(x - float(t), 0.0) for t in phi[: self.n_knots])
        return np.column_stack(cols)


class LogisticFamily(SeparableFamily):
    """``y = a + b / (1 + exp(k * (x - x0)))``: a threshold with a finite width.

    Sits between :class:`ThresholdFamily` and the smooth decays, and earns its
    place by being the only family that can report *how sharp* a transition is.
    A surface where logistic wins with a large ``k`` is a threshold; one where it
    wins with a small ``k`` is a smooth decay wearing a different hat, and the
    taxonomy reads ``k`` rather than the family name.
    """

    name: ClassVar[str] = "logistic"
    n_phi: ClassVar[int] = 2
    refine: ClassVar[bool] = True

    def phi_candidates(self, x: FloatArray) -> FloatArray:
        s = _span(x)
        ks = np.geomspace(0.5 / s, 60.0 / s, 8)
        x0s = _interior_knots(x, max_knots=12)
        if x0s.size == 0:
            x0s = np.array([float(np.mean(x))])
        return np.asarray([(k, t) for k in ks for t in x0s], dtype=np.float64)

    def phi_bounds(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        s = _span(x)
        lo = np.array([1e-3 / s, float(np.min(x)) - s])
        hi = np.array([1e4 / s, float(np.max(x)) + s])
        return lo, hi

    def design(self, phi: FloatArray, x: FloatArray) -> FloatArray:
        k, x0 = float(phi[0]), float(phi[1])
        z = np.clip(k * (x - x0), -60.0, 60.0)
        return np.column_stack([np.ones_like(x), 1.0 / (1.0 + np.exp(z))])


class LookupFamily(SeparableFamily):
    """One free value per distinct ``x``. The baseline that decides whether P1 has a law.

    Deliberately the saturated model: it can reproduce any surface exactly
    in-sample, so it only loses out of sample, and only to a family that has
    found real structure. Prediction at an unseen ``x`` falls back to the nearest
    trained value, which is the strongest honest thing a table can do — a table
    that interpolated would be a smoothness assumption smuggled into the
    baseline, and the baseline is precisely where no assumptions are allowed.
    """

    name: ClassVar[str] = "lookup"
    parametric: ClassVar[bool] = False
    n_phi: ClassVar[int] = 0

    def phi_candidates(self, x: FloatArray) -> FloatArray:
        return np.zeros((1, 0))

    def design(self, phi: FloatArray, x: FloatArray) -> FloatArray:
        levels = np.unique(x)
        return (x[:, None] == levels[None, :]).astype(np.float64)

    def n_params_for(self, phi: FloatArray, beta: FloatArray) -> float:
        return float(beta.shape[0])

    def predict(self, fit: CurveFit, x: FloatArray) -> FloatArray:
        levels = np.unique(fit.x_train)
        if fit.beta.shape[0] != levels.shape[0]:  # degenerate fallback
            return np.full(x.shape[0], float(fit.beta[0]))
        idx = np.abs(np.asarray(x, dtype=np.float64)[:, None] - levels[None, :]).argmin(axis=1)
        return np.asarray(fit.beta[idx], dtype=np.float64)


# --------------------------------------------------------------------------
# Shape-constrained and nonparametric families
# --------------------------------------------------------------------------


def _pava_batch(Y: FloatArray, w: FloatArray) -> FloatArray:
    """Weighted nondecreasing isotonic regression for every column at once.

    Uses the min-max characterisation

        ``yhat_i = max_{k<=i} min_{j>=i} Av(k, j)``

    where ``Av(k, j)`` is the weighted mean of block ``k..j``. Pool-adjacent-
    violators is asymptotically cheaper, but its control flow depends on the
    data, so it cannot be run across cells in lockstep — and with the ~10-40
    budgets a sweep actually uses, the O(m^2) reduction over all cells at once
    is much faster in practice than a Python PAVA loop per cell. The test suite
    checks it against ``scipy.optimize.isotonic_regression``.
    """
    m, n = Y.shape
    cs = np.zeros((m + 1, n))
    np.cumsum(Y * w[:, None], axis=0, out=cs[1:])
    cw = np.concatenate([[0.0], np.cumsum(w)])

    # Av[k, j, :] = weighted mean over rows k..j
    num = cs[None, 1:, :] - cs[:-1, None, :]  # (k, j, n)
    den = cw[None, 1:] - cw[:-1, None]  # (k, j)
    with np.errstate(divide="ignore", invalid="ignore"):
        av = num / den[:, :, None]
    upper = np.triu(np.ones((m, m), dtype=bool))  # k <= j
    av = np.where(upper[:, :, None], av, np.inf)
    av = np.where(np.isfinite(av), av, np.inf)

    # S[k, i] = min over j >= i (and j >= k) of Av[k, j]
    suffix_min = np.minimum.accumulate(av[:, ::-1, :], axis=1)[:, ::-1, :]
    # restrict the outer max to k <= i
    lower = np.tril(np.ones((m, m), dtype=bool))  # k <= i  (rows k, cols i)
    masked = np.where(upper[:, :, None], suffix_min, -np.inf)
    masked = np.where(lower.T[:, :, None] | upper[:, :, None], masked, -np.inf)
    return np.asarray(np.max(masked, axis=0), dtype=np.float64)


class IsotonicFamily(CurveFamily):
    """Monotone in ``x``, otherwise unconstrained.

    The right nonparametric competitor for a decay hypothesis: it grants
    monotonicity — which every parametric decay family also assumes — and grants
    nothing else. A parametric family that cannot beat isotonic out of sample has
    contributed a functional form and no information.

    ``direction`` is fixed by the caller rather than fitted. Choosing the better
    of increasing and decreasing per cell would spend a parameter that the
    information criteria here do not charge for, and would let noise-only cells
    look structured.
    """

    name: ClassVar[str] = "isotonic"
    parametric: ClassVar[bool] = False

    def __init__(self, direction: str = "decreasing") -> None:
        if direction not in ("increasing", "decreasing"):
            raise ValueError("direction must be 'increasing' or 'decreasing'")
        self.direction = direction

    def fit_batch(
        self,
        x: FloatArray,
        Y: FloatArray,
        *,
        weights: FloatArray | None = None,
        warm_start: FloatArray | None = None,
    ) -> list[CurveFit]:
        x = np.asarray(x, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        m, n = Y.shape
        w = _weights_or_ones(weights, m)
        order = np.argsort(x, kind="stable")
        sign = -1.0 if self.direction == "decreasing" else 1.0
        yhat_sorted = _pava_batch(sign * Y[order], w[order])
        fitted = np.empty_like(Y)
        fitted[order] = sign * yhat_sorted

        resid = Y - fitted
        sse = np.einsum("mn,mn->n", resid * w[:, None], resid)
        fits: list[CurveFit] = []
        for j in range(n):
            blocks = float(np.unique(np.round(fitted[:, j], 12)).size)
            fits.append(
                CurveFit(
                    family=self.name,
                    beta=fitted[:, j].copy(),
                    phi=np.array([1.0 if sign > 0 else -1.0]),
                    sse=float(sse[j]),
                    n_points=m,
                    n_params=blocks,
                    x_train=x,
                    fitted=fitted[:, j].copy(),
                )
            )
        return fits

    def predict(self, fit: CurveFit, x: FloatArray) -> FloatArray:
        order = np.argsort(fit.x_train, kind="stable")
        xs, ys = fit.x_train[order], fit.beta[order]
        idx = np.searchsorted(xs, np.asarray(x, dtype=np.float64), side="right") - 1
        idx = np.clip(idx, 0, xs.shape[0] - 1)
        return np.asarray(ys[idx], dtype=np.float64)


class GPFamily(CurveFamily):
    """Kernel ridge / GP posterior mean with an RBF kernel.

    Smooth but shape-free: the counterpart to isotonic for the hypothesis "the
    curve is smooth" without committing to *which* smooth curve. Hyperparameters
    come from generalized cross-validation rather than marginal likelihood
    because GCV is a single closed-form expression on a shared hat matrix, so a
    whole grid of cells is scored in one matrix product — and because the
    likelihood here would be conditioned on a noise model the sweep has not
    established.

    Effective degrees of freedom is ``trace(H)``, which is what makes it
    comparable to the parametric families under AIC/BIC.
    """

    name: ClassVar[str] = "gp"
    parametric: ClassVar[bool] = False

    def __init__(
        self,
        lengthscale_factors: Sequence[float] = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5),
        ridges: Sequence[float] = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    ) -> None:
        self.lengthscale_factors = tuple(lengthscale_factors)
        self.ridges = tuple(ridges)

    @staticmethod
    def _kernel(xa: FloatArray, xb: FloatArray, ell: float) -> FloatArray:
        d = (xa[:, None] - xb[None, :]) / ell
        return np.asarray(np.exp(-0.5 * d * d), dtype=np.float64)

    def fit_batch(
        self,
        x: FloatArray,
        Y: FloatArray,
        *,
        weights: FloatArray | None = None,
        warm_start: FloatArray | None = None,
    ) -> list[CurveFit]:
        x = np.asarray(x, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        m, n = Y.shape
        w = _weights_or_ones(weights, m)
        span = _span(x)
        y_scale = np.maximum(Y.std(axis=0), _TINY)

        best_gcv = np.full(n, np.inf)
        best_alpha = np.zeros((m, n))
        best_fitted = np.zeros((m, n))
        best_phi = np.zeros((n, 2))
        best_df = np.zeros(n)

        for f in self.lengthscale_factors:
            ell = max(f * span, _TINY)
            K = self._kernel(x, x, ell)
            for lam in self.ridges:
                try:
                    inv = np.linalg.inv(K + lam * np.eye(m))
                except np.linalg.LinAlgError:
                    continue
                H = K @ inv
                df = float(np.trace(H))
                if df > m - 0.5:
                    continue
                fitted = H @ Y
                resid = Y - fitted
                sse = np.einsum("mn,mn->n", resid * w[:, None], resid)
                gcv = m * sse / ((m - df) ** 2 * y_scale**2)
                better = gcv < best_gcv
                if np.any(better):
                    best_gcv = np.where(better, gcv, best_gcv)
                    best_fitted[:, better] = fitted[:, better]
                    best_alpha[:, better] = (inv @ Y)[:, better]
                    best_phi[better] = (ell, lam)
                    best_df[better] = df

        resid = Y - best_fitted
        sse = np.einsum("mn,mn->n", resid * w[:, None], resid)
        return [
            CurveFit(
                family=self.name,
                beta=best_alpha[:, j].copy(),
                phi=best_phi[j].copy(),
                sse=float(sse[j]),
                n_points=m,
                n_params=float(max(best_df[j], 1.0)),
                x_train=x,
                fitted=best_fitted[:, j].copy(),
            )
            for j in range(n)
        ]

    def predict(self, fit: CurveFit, x: FloatArray) -> FloatArray:
        ell = float(fit.phi[0]) if fit.phi[0] > 0 else 1.0
        k = self._kernel(np.asarray(x, dtype=np.float64), fit.x_train, ell)
        return np.asarray(k @ fit.beta, dtype=np.float64)


# --------------------------------------------------------------------------
# Registry and orchestration
# --------------------------------------------------------------------------


def _build_registry() -> dict[str, CurveFamily]:
    members: list[CurveFamily] = [
        ConstantFamily(),
        ExponentialFamily(),
        PowerFamily(),
        ThresholdFamily(),
        HingeFamily(),
        PiecewiseLinearFamily(1),
        PiecewiseLinearFamily(2),
        LogisticFamily(),
        IsotonicFamily(),
        GPFamily(),
        LookupFamily(),
    ]
    return {m.name: m for m in members}


FAMILIES: dict[str, CurveFamily] = _build_registry()
LOOKUP_BASELINE = "lookup"


def family(name: str) -> CurveFamily:
    try:
        return FAMILIES[name]
    except KeyError:
        raise KeyError(f"unknown curve family {name!r}; have {sorted(FAMILIES)}") from None


def parametric_family_names() -> list[str]:
    """Families that claim a functional form, i.e. everything the lookup must beat."""
    return [n for n, f in FAMILIES.items() if f.parametric and n != "constant"]


@dataclass(frozen=True, slots=True)
class BatchReport:
    """How much work the batched path actually shared."""

    n_cells: int
    n_groups: int
    group_sizes: tuple[int, ...] = field(default_factory=tuple)


def _group_key(x: FloatArray) -> bytes:
    return np.round(x, 12).tobytes()


def fit_cells(
    name: str,
    cells: Sequence[CurveData],
    *,
    batched: bool = True,
) -> list[CurveFit]:
    """Fit one family to many cells.

    ``batched=True`` groups cells by their shared ``x`` grid and solves each
    group in one pass; ``batched=False`` fits them one at a time. The two must
    agree to numerical tolerance, and ``tests/test_atlas_curves.py`` asserts it
    — a fast path that quietly disagreed with the reference would corrupt every
    downstream verdict while looking like a speedup.
    """
    fam = family(name)
    if not cells:
        return []
    if not batched:
        return [fam.fit_one(c.x, c.y, weights=c.weight) for c in cells]

    groups: dict[bytes, list[int]] = {}
    for i, c in enumerate(cells):
        groups.setdefault(_group_key(c.x) + _group_key(c.weight), []).append(i)

    out: list[CurveFit | None] = [None] * len(cells)
    for idxs in groups.values():
        ref = cells[idxs[0]]
        Y = np.column_stack([cells[i].y for i in idxs])
        fits = fam.fit_batch(ref.x, Y, weights=ref.weight)
        for slot, fit in zip(idxs, fits):
            out[slot] = fit
    return [f for f in out if f is not None]


def fit_all_families(
    cells: Sequence[CurveData],
    *,
    names: Sequence[str] | None = None,
    batched: bool = True,
) -> dict[str, list[CurveFit]]:
    """Fit every family to every cell; the outer key is the family name."""
    chosen = list(FAMILIES) if names is None else list(names)
    return {n: fit_cells(n, cells, batched=batched) for n in chosen}
