"""The driver that actually produces an atlas.

Everything else in :mod:`instinct.atlas` is machinery. This runs it: a grid over
environments, reflexes, environment speeds and hardware latencies, one exact
solve per cell, then selection, transfer and the kill verdicts, ending in a
markdown report that leads with the verdict table.

Why the exact arm carries the grid
----------------------------------
Every cell here is a closed-form linear solve rather than a batch of rollouts,
which is what makes a grid of this size a few seconds of CPU instead of an
overnight job. It also removes the one confound that would otherwise dominate a
first result: with no sampling error, ``epsilon_id`` is machine epsilon, so the
"decomposition noise swamps the budget effect" kill condition is cleared by
construction and any transfer failure is a fact about the surface rather than
about how many seeds were spent.

The sampled arm exists (``core/rollout.py``) and is validated against this one;
it is what a later run over the gridworld and arcade environments would use.
Starting there would have meant debugging estimator variance and transfer at the
same time.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from instinct.atlas.curves import cells_from_frame
from instinct.atlas.decomposition import exact_atlas_rows
from instinct.atlas.fit import CellSelection, select_families, taxonomy
from instinct.atlas.kill import KillReport, evaluate_kill_conditions, render_markdown
from instinct.atlas.schema import to_frame, validate_frame
from instinct.atlas.transfer import TransferResult, transfer_suite
from instinct.core.env import Timing
from instinct.core.envs.tabular import chase_chain, corridor_with_pit
from instinct.core.mdp import TabularMDP

__all__ = ["SweepConfig", "SweepResult", "run_sweep", "write_report"]

DEFAULT_BUDGETS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def _default_envs() -> dict[str, TabularMDP]:
    """Two environments chosen to differ in the way that matters.

    ``chase_chain`` has no absorbing states, so every mistake is recoverable and
    ``L_irreversible`` should measure ~0. ``corridor_with_pit`` absorbs, so it
    should not. An atlas built on one kind alone could not tell the two apart,
    and the environment transfer would be untestable.
    """
    return {
        "chase_chain": chase_chain(n_positions=5, gamma=0.9, drift=0.4),
        "corridor_with_pit": corridor_with_pit(length=6, gamma=0.9, slip=0.25),
    }


def _reflexes(mdp: TabularMDP) -> dict[str, np.ndarray]:
    """Fixed reflexes spanning the freeze-to-greedy range.

    ``hold`` is the freeze endpoint and ``greedy`` the myopic one; P3's whole
    question is whether anything interesting lives between them, so P1 measures
    both and the reflex transfer asks whether the surface survives the swap.
    """
    n = mdp.n_states
    return {
        "hold": np.full(n, min(1, mdp.n_actions - 1), dtype=np.int64),
        "greedy": mdp.R.argmax(axis=1).astype(np.int64),
    }


@dataclass(frozen=True, slots=True)
class SweepConfig:
    budgets: tuple[int, ...] = DEFAULT_BUDGETS
    nu_e_values: tuple[float, ...] = (0.5, 1.0, 2.0)
    nu_h_values: tuple[float, ...] = (0.25, 0.5, 1.0)
    commit: int = 1
    cost_per_simulation: float = 0.0
    n_states: int = 6  # states sampled per environment, for report size
    family: str = "pwlinear3"
    label: str = "p1-exact"


@dataclass(slots=True)
class SweepResult:
    frame: pd.DataFrame
    selections: list[CellSelection] = field(default_factory=list)
    transfers: dict[str, TransferResult | None] = field(default_factory=dict)
    report: KillReport | None = None
    wall_clock_s: float = 0.0


def run_sweep(
    cfg: SweepConfig | None = None,
    *,
    envs: dict[str, TabularMDP] | None = None,
) -> SweepResult:
    """Measure the grid, fit it, test transfer, and decide the kill conditions."""
    cfg = cfg or SweepConfig()
    envs = envs or _default_envs()
    started = time.perf_counter()

    rows = []
    for env_name, mdp in envs.items():
        # A spread of start states rather than all of them: the solve returns
        # every state anyway, but the report and the fits do not need hundreds
        # of near-duplicate cells.
        states = list(np.linspace(0, mdp.n_states - 1, cfg.n_states, dtype=int))
        for reflex_name, reflex in _reflexes(mdp).items():
            for nu_e in cfg.nu_e_values:
                for nu_h in cfg.nu_h_values:
                    rows += exact_atlas_rows(
                        mdp,
                        env_name=env_name,
                        reflex=reflex,
                        reflex_name=reflex_name,
                        budgets=list(cfg.budgets),
                        timing=Timing(nu_e=nu_e, nu_h=nu_h),
                        states=states,
                        commit=cfg.commit,
                        cost_per_simulation=cfg.cost_per_simulation,
                    )

    frame = to_frame(rows)
    validate_frame(frame)  # the identity is not optional

    cells = cells_from_frame(frame, x="budget")
    selections = taxonomy(cells, select_families(cells))
    transfers = transfer_suite(frame, family=cfg.family, x="budget")
    report = evaluate_kill_conditions(frame, selections, transfers)

    return SweepResult(
        frame=frame,
        selections=selections,
        transfers=transfers,
        report=report,
        wall_clock_s=time.perf_counter() - started,
    )


def _regime_table(selections: Sequence[CellSelection]) -> str:
    counts: dict[str, int] = {}
    for sel in selections:
        counts[sel.regime] = counts.get(sel.regime, 0) + 1
    total = max(sum(counts.values()), 1)
    lines = ["| Regime | Cells | Share |", "| --- | --- | --- |"]
    for regime, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {regime} | {n} | {n / total:.0%} |")
    return "\n".join(lines)


def _transfer_table(transfers: dict[str, TransferResult | None]) -> str:
    lines = [
        "| Transfer | Budget regret | Excess | Kendall tau | Argmax hit |",
        "| --- | --- | --- | --- | --- |",
    ]
    for axis, res in transfers.items():
        if res is None:
            lines.append(f"| {axis} | _not varied in this sweep_ | | | |")
            continue
        lines.append(
            f"| {axis} ({res.fit_on} → {res.tested_on}) | {res.mean_budget_regret:+.4f} | "
            f"{res.excess_regret:+.4f} | {res.mean_kendall_tau:+.2f} | "
            f"{res.exact_argmax_rate:.0%} |"
        )
    return "\n".join(lines)


def write_report(result: SweepResult, path: str | Path) -> Path:
    """Write the markdown report, verdict table first."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = result.frame

    if result.report is None:
        raise ValueError("sweep result has no kill report; run_sweep did not complete")
    body = [
        render_markdown(result.report),
        "",
        "## What was measured",
        "",
        f"- {len(df)} cells: {df['env'].nunique()} environments x "
        f"{df['reflex'].nunique()} reflexes x {df['nu_e'].nunique()} speeds x "
        f"{df['nu_h'].nunique()} latencies x {df['budget'].nunique()} budgets x "
        f"{df['start_state'].nunique()} start states",
        f"- Exact closed-form solves, no sampling. Wall clock {result.wall_clock_s:.1f}s.",
        f"- Worst |epsilon_id| residual: identity holds to "
        f"{float(np.abs(df['epsilon_id']).max()):.3g} in return units.",
        "",
        "## Regime taxonomy",
        "",
        _regime_table(result.selections),
        "",
        "## Transfer",
        "",
        _transfer_table(result.transfers),
        "",
        "## Decomposition, averaged over cells",
        "",
        "| Term | Mean | Note |",
        "| --- | --- | --- |",
        f"| G_plan | {df['G_plan'].mean():+.4f} | benefit of the better decision |",
        f"| R_intermediate | {df['R_intermediate'].mean():+.4f} | banked by the reflex |",
        f"| L_arrival | {df['L_arrival'].mean():+.4f} | arrival regret: the decision went stale |",
        f"| L_wait | {df['L_wait'].mean():+.4f} | whole cost of waiting, incl. discounting |",
        f"| L_irreversible | {df['L_irreversible'].mean():+.4f} | what planning cannot undo |",
        f"| sigma | {df['sigma'].mean():+.4f} | net gain over the base budget |",
        "",
        "`L_irreversible` by environment, which is the term that should separate "
        "an absorbing environment from a recoverable one:",
        "",
        "| Environment | mean | min | one-signed? |",
        "| --- | --- | --- | --- |",
    ]
    for env_name, sub in df.groupby("env"):
        one_signed = bool(sub["L_irreversible"].min() >= -1e-9)
        body.append(
            f"| {env_name} | {sub['L_irreversible'].mean():+.4f} | "
            f"{sub['L_irreversible'].min():+.4f} | {'yes' if one_signed else 'no'} |"
        )

    out.write_text("\n".join(body) + "\n")
    return out
