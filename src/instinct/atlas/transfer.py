"""Does the atlas describe the environment, or only the experiment that produced it?

This is P1's actual deliverable. Fitting a staleness curve is easy and proves
nothing: any of ten families will produce a plausible curve for any cell. The
claim worth making is that a curve measured under one condition predicts the
right planning budget under a *different* one. That is what separates a law from
a description of one experiment.

Five transfers, each holding out a different axis:

======================  ====================================================
``speed``               fit at one ``nu_e``, predict at another
``hardware``            fit at one ``nu_h``, predict at another
``reflex``              fit under one reflex, predict under a different one
``environment``         fit on one environment, predict on another
``collapse``            does the surface reduce to ``delta = nu_e*nu_h*k``?
======================  ====================================================

The last is a *hypothesis*, never an assumption. It is tempting to treat
staleness as the only thing that matters and fold the two axes into their
product; :class:`instinct.core.env.Timing` keeps them apart precisely so this can
be falsified rather than baked in.

The metric
----------
Everything is scored by **budget regret**: the return actually lost by acting on
the budget the transferred curve recommends instead of the one the oracle knows
is best, in return units.

Not R-squared, and the distinction is not pedantic. A curve can track a frontier
closely and still order the budgets wrongly, and ordering is the only thing a
controller consumes — it takes an argmax, it does not read off a value. A fit
with excellent R-squared that inverts two adjacent budgets is worse, for the only
use anyone has, than a crude fit that ranks them correctly. Kendall tau is
reported beside it for the same reason.

Budget regret is also bounded below by zero, is expressed on the same scale as
``sigma``, and needs no denominator — so it stays finite exactly where the
surface is flattest and a relative error would blow up.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from instinct.atlas.curves import CurveData, CurveFit, cells_from_frame, fit_cells
from instinct.core.stats import kendall_tau

FloatArray = npt.NDArray[np.float64]

__all__ = [
    "AXES",
    "TransferResult",
    "budget_regret",
    "collapse_test",
    "run_transfer",
    "transfer_suite",
]

# Which column each transfer holds out. `collapse` is handled separately.
AXES: dict[str, str] = {
    "speed": "nu_e",
    "hardware": "nu_h",
    "reflex": "reflex",
    "environment": "env",
}


def budget_regret(
    predicted_sigma: FloatArray,
    true_sigma: FloatArray,
) -> float:
    """Return lost by taking the predicted argmax instead of the true one.

    Zero when the prediction picks a budget that is actually optimal, even if
    every predicted *value* is wrong — which is the point. A controller acts on
    the argmax and never sees the fitted values.
    """
    pred = np.asarray(predicted_sigma, dtype=np.float64)
    true = np.asarray(true_sigma, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: predicted {pred.shape}, true {true.shape}")
    if pred.size == 0:
        return float("nan")
    chosen = int(np.argmax(pred))
    return float(np.max(true) - true[chosen])


@dataclass(frozen=True, slots=True)
class TransferResult:
    """One held-out transfer, scored the way a controller would experience it."""

    axis: str
    family: str
    fit_on: str
    tested_on: str
    n_cells: int
    mean_budget_regret: float  # return units, 0 is perfect
    max_budget_regret: float
    mean_kendall_tau: float  # 1 is a perfect ordering
    exact_argmax_rate: float  # share of cells whose optimal budget was hit
    within_cell_regret: float  # regret of the best curve fitted ON the test cell
    target_effect: float = float("nan")  # peak-to-trough of the target frontier
    per_cell: list[float] = field(default_factory=list, repr=False)

    @property
    def excess_regret(self) -> float:
        """Regret above what a curve fitted directly on the test cell achieves.

        The honest baseline. A test cell whose own best fit already mis-ranks
        budgets is measuring the fitter's limits, not the transfer's — so the
        transfer should be judged on what it adds, not on the total.
        """
        return self.mean_budget_regret - self.within_cell_regret

    def summary(self) -> str:
        return (
            f"{self.axis}: {self.fit_on} -> {self.tested_on} | "
            f"regret {self.mean_budget_regret:+.4f} (excess {self.excess_regret:+.4f}), "
            f"tau {self.mean_kendall_tau:+.2f}, argmax hit {self.exact_argmax_rate:.0%}"
        )


def _cell_group(cell: CurveData, axis: str) -> object:
    key = cell.key
    return {
        "nu_e": key.nu_e,
        "nu_h": key.nu_h,
        "reflex": key.reflex,
        "env": key.env,
    }[axis]


def _score_against(fit_curve: CurveFit, cell: CurveData) -> tuple[float, float, bool]:
    """Score one fitted curve against one held-out cell."""
    predicted = fit_curve.predict(cell.x)
    regret = budget_regret(predicted, cell.y)
    tau = kendall_tau(predicted, cell.y) if cell.n > 1 else float("nan")
    hit = bool(np.argmax(predicted) == np.argmax(cell.y))
    return regret, tau, hit


def run_transfer(
    cells: Sequence[CurveData],
    *,
    axis: str,
    family: str = "exponential",
    fit_on: object | None = None,
    tested_on: object | None = None,
) -> TransferResult | None:
    """Fit on one slice of ``axis`` and score the prediction on another.

    Returns ``None`` when the axis has fewer than two distinct values, which is
    a legitimate outcome for a sweep that did not vary it — better than
    inventing a transfer that was never measured.
    """
    if axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}; expected one of {sorted(AXES)}")
    column = AXES[axis]

    groups: dict[object, list[CurveData]] = {}
    for cell in cells:
        groups.setdefault(_cell_group(cell, column), []).append(cell)
    if len(groups) < 2:
        return None

    keys = sorted(groups, key=str)
    source = keys[0] if fit_on is None else fit_on
    target = keys[1] if tested_on is None else tested_on
    if source not in groups or target not in groups:
        raise KeyError(f"axis {axis} has values {keys}, not {source!r}/{target!r}")

    train, test = groups[source], groups[target]

    # Pair cells by CONTEXT -- everything held fixed other than the transfer
    # axis. Without this the "speed transfer" asks a curve measured on
    # (chase_chain, greedy, state 0) to predict (corridor_with_pit, hold,
    # state 5), which is not a speed transfer at all but every axis at once.
    # That is what produced a 0% argmax hit rate on every transfer including
    # the ones that passed: a passing transfer that never picks the right
    # budget was the signal that the pairing, not the surface, was wrong.
    def context(cell: CurveData) -> tuple[object, ...]:
        k = cell.key
        full = {"env": k.env, "reflex": k.reflex, "nu_e": k.nu_e, "nu_h": k.nu_h}
        del full[column]
        return (*full.values(), k.start_state)

    by_context_train: dict[tuple[object, ...], CurveData] = {context(c): c for c in train}
    paired = [(by_context_train[context(c)], c) for c in test if context(c) in by_context_train]
    if not paired:
        return None

    source_cells = [pair[0] for pair in paired]
    target_cells = [pair[1] for pair in paired]
    source_fits = fit_cells(family, source_cells)

    regrets, taus, hits = [], [], []
    for fit, target_cell in zip(source_fits, target_cells):
        regret, tau, hit = _score_against(fit, target_cell)
        regrets.append(regret)
        taus.append(tau)
        hits.append(hit)
    test = target_cells

    # The floor: a curve fitted directly on each test cell.
    # A target whose frontier is flat cannot test anything: every prediction
    # scores zero regret against a constant, so the transfer "passes"
    # regardless. Measured on corridor_with_pit, whose effect size is exactly
    # 0.0000 -- depth-limited lookahead already finds the optimal action at
    # depth 1 there, so no budget buys anything and the environment contributes
    # no signal to the atlas at all.
    target_effect = float(np.median([c.effect_size() for c in test]))

    own_fits = fit_cells(family, list(test))
    within = float(np.mean([_score_against(f, c)[0] for f, c in zip(own_fits, test)]))

    return TransferResult(
        axis=axis,
        target_effect=target_effect,
        family=family,
        fit_on=str(source),
        tested_on=str(target),
        n_cells=len(test),
        mean_budget_regret=float(np.mean(regrets)),
        max_budget_regret=float(np.max(regrets)),
        mean_kendall_tau=float(np.nanmean(taus)) if taus else float("nan"),
        exact_argmax_rate=float(np.mean(hits)),
        within_cell_regret=within,
        per_cell=regrets,
    )


def collapse_test(df: pd.DataFrame, *, family: str = "exponential") -> TransferResult | None:
    """Does the surface depend only on ``delta = nu_e * nu_h * budget``?

    If it does, a curve fitted against ``staleness`` transfers across ``(nu_e,
    nu_h)`` pairs that share a product, and the two axes were never independent.
    If it does not, the collapse is false and the atlas genuinely needs both.

    Implemented as a transfer between distinct ``(nu_e, nu_h)`` pairs with equal
    products. Returns ``None`` when the sweep contains no such pair, which is a
    statement about the grid rather than about the hypothesis — a grid that
    cannot separate the two axes cannot test the collapse, and saying so is
    better than reporting a verdict it did not earn.
    """
    cells = cells_from_frame(df, x="staleness")
    by_product: dict[float, list[CurveData]] = {}
    for cell in cells:
        product = round(cell.key.nu_e * cell.key.nu_h, 9)
        by_product.setdefault(product, []).append(cell)

    regrets: list[float] = []
    within_regrets: list[float] = []
    taus: list[float] = []
    hits: list[bool] = []
    effects: list[float] = []
    comparisons: list[str] = []

    def context(cell: CurveData) -> tuple[str, str, int]:
        return cell.key.env, cell.key.reflex, cell.key.start_state

    for product, group in sorted(by_product.items()):
        pairs = {(c.key.nu_e, c.key.nu_h) for c in group}
        if len(pairs) < 2 or product == 0.0:
            continue
        ordered = sorted(pairs)
        for source_pair in ordered:
            source = {context(c): c for c in group if (c.key.nu_e, c.key.nu_h) == source_pair}
            for target_pair in ordered:
                if target_pair == source_pair:
                    continue
                target = {context(c): c for c in group if (c.key.nu_e, c.key.nu_h) == target_pair}
                shared = sorted(source.keys() & target.keys())
                for key in shared:
                    source_cell, target_cell = source[key], target[key]
                    fit = fit_cells(family, [source_cell])[0]
                    regret, tau, hit = _score_against(fit, target_cell)
                    own_fit = fit_cells(family, [target_cell])[0]
                    within, _, _ = _score_against(own_fit, target_cell)
                    regrets.append(regret)
                    within_regrets.append(within)
                    taus.append(tau)
                    hits.append(hit)
                    effects.append(target_cell.effect_size())
                    comparisons.append(f"{source_pair}->{target_pair}")

    if not regrets:
        return None
    return TransferResult(
        axis="collapse",
        family=family,
        fit_on="all equal-product pairs",
        tested_on=", ".join(sorted(set(comparisons))),
        n_cells=len(regrets),
        mean_budget_regret=float(np.mean(regrets)),
        max_budget_regret=float(np.max(regrets)),
        mean_kendall_tau=float(np.nanmean(taus)),
        exact_argmax_rate=float(np.mean(hits)),
        within_cell_regret=float(np.mean(within_regrets)),
        target_effect=float(np.median(effects)),
        per_cell=regrets,
    )


def transfer_suite(
    df: pd.DataFrame, *, family: str = "exponential", x: str = "budget"
) -> dict[str, TransferResult | None]:
    """Run every transfer the sweep's grid can support.

    ``x`` chooses the axis the curve is a function of, and the choice is not
    cosmetic — it changes what the speed transfer *means*.

    Against ``budget`` the question is the proposal's: a curve measured at speed
    ``v`` is asked to name the best budget at ``2v``, with both curves living on
    the same budget grid. Against ``staleness`` the curve is asked to predict at
    ``nu_e * nu_h * k``, which at a different speed is a stretched axis — so a
    successful transfer there is really the *collapse* hypothesis in disguise,
    since it says the surface is a function of the product alone.

    Default ``budget``, because conflating the two would let the collapse
    hypothesis be assumed by the very test meant to check it. :func:`collapse_test`
    remains the place that hypothesis is put on trial.

    Axes the sweep did not vary come back as ``None`` rather than as a passing
    result. An untested transfer is not a successful one, and a report that
    silently omitted it would read as if the atlas had generalized.
    """
    cells = cells_from_frame(df, x=x)
    out: dict[str, TransferResult | None] = {
        axis: run_transfer(cells, axis=axis, family=family) for axis in AXES
    }
    out["collapse"] = collapse_test(df, family=family)
    return out
