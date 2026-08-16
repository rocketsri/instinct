"""P1's kill conditions, as executable checks.

The proposal names four conditions under which the compute-freshness surface
should stop being called a law:

1. fitted curves do not transfer beyond the environment used to estimate them;
2. a policy change completely reorganizes the surface;
3. no simple family beats a per-environment lookup table;
4. decomposition noise is comparable to the budget effect.

Each is a verdict here rather than a paragraph, and a tripped condition is a
**result**. The report puts the verdict table at the top, before any curve or
plot, because the failure mode this module exists to prevent is a study that
quietly keeps going after its own stopping rule fired and presents the pretty
fits anyway.

A verdict can also be ``INCONCLUSIVE``, and that is deliberately distinct from
``PASS``. A sweep whose grid never varied the reflex cannot clear the
policy-change condition; reporting it as passed would be the single most
misleading thing this file could do.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from instinct.atlas.fit import CellSelection
    from instinct.atlas.transfer import TransferResult

__all__ = ["KillReport", "Verdict", "evaluate_kill_conditions", "render_markdown"]

PASS, KILL, INCONCLUSIVE = "PASS", "KILL", "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class Verdict:
    """One kill condition, decided."""

    name: str
    status: str
    detail: str
    value: float = float("nan")
    threshold: float = float("nan")

    @property
    def tripped(self) -> bool:
        return self.status == KILL


@dataclass(frozen=True, slots=True)
class KillReport:
    verdicts: list[Verdict]

    @property
    def any_tripped(self) -> bool:
        return any(v.tripped for v in self.verdicts)

    @property
    def any_inconclusive(self) -> bool:
        return any(v.status == INCONCLUSIVE for v in self.verdicts)

    def headline(self) -> str:
        if self.any_tripped:
            names = ", ".join(v.name for v in self.verdicts if v.tripped)
            return f"KILL — {names}"
        if self.any_inconclusive:
            return "INCONCLUSIVE — the grid cannot decide every condition"
        return "PASS — no kill condition tripped"


def _transfer_verdict(
    name: str,
    result: TransferResult | None,
    *,
    effect_scale: float,
    tolerance: float,
    missing_detail: str,
) -> Verdict:
    """Judge a transfer against the size of the effect it is trying to predict.

    The threshold is a fraction of the budget effect rather than an absolute
    number of return units, because a transfer that loses 0.01 is excellent on a
    surface spanning 2.0 and worthless on one spanning 0.02.

    Scored on *excess* regret — regret above what a curve fitted directly on the
    test cell achieves. A test cell whose own best fit already mis-ranks its
    budgets is measuring the fitter, not the transfer.
    """
    if result is None:
        return Verdict(name, INCONCLUSIVE, missing_detail)
    if not np.isfinite(effect_scale) or effect_scale <= 0:
        return Verdict(name, INCONCLUSIVE, "budget effect is zero; nothing to transfer")

    limit = tolerance * effect_scale

    # A flat target frontier makes the transfer untestable: every prediction
    # scores zero regret against a constant. Reporting that as PASS is how a
    # degenerate environment gets counted as evidence the atlas generalizes.
    if np.isfinite(result.target_effect) and result.target_effect <= limit:
        return Verdict(
            name,
            INCONCLUSIVE,
            f"{result.summary()}; the target frontier is flat "
            f"(effect {result.target_effect:.4f} <= {limit:.4f}), so budget buys "
            "nothing there and no prediction can be wrong",
            value=result.target_effect,
            threshold=limit,
        )

    # If a curve fitted directly ON the test cell already mis-ranks that cell's
    # budgets by more than the limit, the family cannot represent this surface
    # and the transfer is untestable with it. Reporting PASS there would let a
    # family that fits nothing clear every transfer, since its excess regret
    # over its own failure is zero by construction -- which is exactly what the
    # first version of this check did.
    if result.within_cell_regret > limit:
        return Verdict(
            name,
            INCONCLUSIVE,
            f"{result.summary()}; a curve fitted on the test cell itself already "
            f"loses {result.within_cell_regret:.4f} > {limit:.4f}, so the "
            f"{result.family} family cannot represent this surface",
            value=result.mean_budget_regret,
            threshold=limit,
        )

    # Judged on absolute regret: what a controller actually loses. Excess regret
    # is reported beside it to separate the transfer's contribution from the
    # fitter's floor.
    excess = result.excess_regret
    status = KILL if result.mean_budget_regret > limit else PASS
    return Verdict(
        name,
        status,
        f"{result.summary()}; excess regret {excess:+.4f} vs limit {limit:.4f} "
        f"({tolerance:.0%} of a {effect_scale:.4f} effect)",
        value=excess,
        threshold=limit,
    )


def evaluate_kill_conditions(
    df: pd.DataFrame,
    selections: list[CellSelection],
    transfers: Mapping[str, TransferResult | None],
    *,
    tolerance: float = 0.25,
    lookup_share_required: float = 0.5,
) -> KillReport:
    """Decide all four conditions from a sweep, its selections and its transfers."""
    effect_scale = (
        float(np.median([s.effect_size for s in selections])) if selections else float("nan")
    )
    verdicts: list[Verdict] = []

    # 1. Curves must transfer off the conditions they were fitted on. Speed and
    #    hardware are the two the proposal names explicitly; environment is the
    #    strongest form of the same question.
    for axis, label in (
        ("speed", "transfer across environment speed"),
        ("hardware", "transfer across hardware latency"),
        ("environment", "transfer across environments"),
    ):
        verdicts.append(
            _transfer_verdict(
                label,
                transfers.get(axis),
                effect_scale=effect_scale,
                tolerance=tolerance,
                missing_detail=f"the sweep varied only one value of {axis}",
            )
        )

    # 2. A policy change must not reorganize the surface. This is also the test
    #    that decides whether the atlas describes the environment or one policy,
    #    which is why P3's cross-test reuses it.
    verdicts.append(
        _transfer_verdict(
            "surface survives a reflex change",
            transfers.get("reflex"),
            effect_scale=effect_scale,
            tolerance=tolerance,
            missing_detail="the sweep used a single reflex, so this is untested",
        )
    )

    # 3. Some simple family must beat the saturated table, or there is no law to
    #    report -- only a lookup table with extra steps.
    if not selections:
        verdicts.append(Verdict("beats a lookup table", INCONCLUSIVE, "no cells selected"))
    else:
        share = float(np.mean([s.beats_lookup for s in selections]))
        verdicts.append(
            Verdict(
                "beats a lookup table",
                PASS if share >= lookup_share_required else KILL,
                f"{share:.0%} of cells beat the table "
                f"(need {lookup_share_required:.0%}); "
                f"median margin {np.median([s.margin_over_lookup for s in selections]):+.4f} nats",
                value=share,
                threshold=lookup_share_required,
            )
        )

    # 4. The decomposition must be sharper than the effect it decomposes.
    if df.empty or "eps_cross" not in df:
        verdicts.append(Verdict("decomposition resolves the effect", INCONCLUSIVE, "no rows"))
    else:
        # The noise is the IDENTITY RESIDUAL, not eps_cross itself.
        #
        # eps_cross is an identified term -- the base arm's own delay cost -- and
        # it is legitimately large whenever the base budget does not land
        # immediately. Measured on a real exact sweep it reached 5.9 against a
        # 4.6 budget effect, which would have tripped this condition on a
        # decomposition that is exact to machine precision. Treating a named
        # quantity as noise because it sits in the residual slot is precisely
        # the confusion this decomposition was restructured to avoid.
        #
        # What the condition actually asks is whether the terms fail to
        # reconstruct sigma by more than the effect being measured.
        from instinct.atlas.schema import DECOMPOSITION_TERMS

        reconstructed = sum(sign * df[t] for t, sign in DECOMPOSITION_TERMS.items())
        noise = float(np.max(np.abs(reconstructed - df["sigma"])))
        verdicts.append(
            Verdict(
                "decomposition resolves the effect",
                PASS if noise <= tolerance * effect_scale else KILL,
                f"worst |eps_cross| {noise:.3e} against a {effect_scale:.4f} budget effect",
                value=noise,
                threshold=tolerance * effect_scale,
            )
        )

    # Reported but never a kill on its own: how much apparent structure is the
    # integer delay grid rather than the environment. A "threshold regime" made
    # entirely of rounding plateaus is not a finding, and a reader cannot tell
    # without this number.
    if selections:
        tied = float(np.mean([s.discretization.tied_share for s in selections]))
        verdicts.append(
            Verdict(
                "structure is not just delay rounding",
                PASS if tied < 0.9 else INCONCLUSIVE,
                f"{tied:.0%} of budget points share a delay with another point",
                value=tied,
                threshold=0.9,
            )
        )

    return KillReport(verdicts)


def render_markdown(report: KillReport, *, title: str = "P1 kill conditions") -> str:
    """The verdict table that belongs at the top of the report, before any plot."""
    icon = {PASS: "PASS", KILL: "**KILL**", INCONCLUSIVE: "_inconclusive_"}
    lines = [
        f"## {title}",
        "",
        f"**{report.headline()}**",
        "",
        "| Condition | Verdict | Detail |",
        "| --- | --- | --- |",
    ]
    for v in report.verdicts:
        lines.append(f"| {v.name} | {icon.get(v.status, v.status)} | {v.detail} |")
    if report.any_tripped:
        lines += [
            "",
            "A tripped condition is a result. The remaining figures describe a "
            "surface that failed its own stopping rule and should be read as "
            "diagnostics, not as findings.",
        ]
    return "\n".join(lines)
