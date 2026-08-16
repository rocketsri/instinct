"""Transfer tests and the kill harness, on surfaces with known answers.

The two that carry weight are
:func:`test_a_surface_that_ignores_speed_transfers_across_it` and
:func:`test_a_surface_that_depends_on_speed_does_not`. Together they show the
transfer test can both pass and fail for the right reason, which a test that
only checked one direction would not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from instinct.atlas.curves import cells_from_frame
from instinct.atlas.decomposition import exact_atlas_rows
from instinct.atlas.fit import select_families
from instinct.atlas.kill import INCONCLUSIVE, KILL, PASS, evaluate_kill_conditions, render_markdown
from instinct.atlas.schema import ATLAS_COLUMNS, to_frame
from instinct.atlas.transfer import (
    budget_regret,
    collapse_test,
    run_transfer,
    transfer_suite,
)
from instinct.core.env import Timing
from instinct.envs.tabular import chase_chain

BUDGETS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def _synthetic_frame(speed_dependent: bool, *, nu_es=(1.0, 6.0), nu_hs=(1.0,)) -> pd.DataFrame:
    """A surface whose dependence on speed is known by construction."""
    rows = []
    for nu_e in nu_es:
        for nu_h in nu_hs:
            for k in BUDGETS:
                # In-family by construction: a + b*exp(-x/tau). Speed enters
                # only through tau when the surface is meant to depend on it,
                # which is what moves the argmax between speeds.
                tau = (12.0 / (nu_e * nu_h)) if speed_dependent else 6.0
                gain = 1.0 - np.exp(-k / tau)
                # A faster world also charges more per unit of thinking, which is
                # what actually relocates the optimum rather than just rescaling.
                loss = (0.02 * nu_e * nu_h * k) if speed_dependent else 0.02 * k
                sigma = gain - loss
                rows.append(
                    dict(
                        env="synth", reflex="mu0", nu_e=nu_e, nu_h=nu_h, budget=k,
                        delay=int(nu_e * nu_h * k), staleness=nu_e * nu_h * k,
                        start_state=0,
                        J_actual=sigma, J_instant=gain, J_fresh=sigma, J_base=0.0,
                        G_plan=gain, R_intermediate=0.0, L_arrival=loss, L_wait=0.0,
                        L_irreversible=0.0, C_hw=0.0, eps_cross=0.0, sigma=sigma,
                        n_seeds=1, ci_lo=sigma, ci_hi=sigma, exact=True,
                        simulations=k, wall_clock_s=0.0,
                    )
                )
    return pd.DataFrame(rows, columns=ATLAS_COLUMNS)


# -- budget regret --------------------------------------------------------


def test_budget_regret_is_zero_when_the_argmax_is_right() -> None:
    """Wrong values, right choice, zero regret. That is the whole point."""
    true = np.array([0.0, 1.0, 0.5])
    wildly_wrong_scale = np.array([-100.0, -50.0, -75.0])  # same ordering
    assert budget_regret(wildly_wrong_scale, true) == 0.0


def test_budget_regret_charges_the_return_actually_lost() -> None:
    true = np.array([0.0, 1.0, 0.4])
    predicted = np.array([0.0, 0.1, 0.9])  # picks index 2, worth 0.4, best is 1.0
    assert budget_regret(predicted, true) == pytest.approx(0.6)


def test_budget_regret_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        budget_regret(np.zeros(3), np.zeros(4))


# -- the transfer tests ---------------------------------------------------


def test_a_surface_that_ignores_speed_transfers_across_it() -> None:
    """Built so the optimal budget does not depend on nu_e; transfer must succeed."""
    cells = cells_from_frame(_synthetic_frame(speed_dependent=False), x="budget")
    result = run_transfer(cells, axis="speed", family="pwlinear3")
    assert result is not None
    assert result.mean_budget_regret == pytest.approx(0.0, abs=1e-6), result.summary()


def test_a_surface_that_depends_on_speed_does_not() -> None:
    """The negative control: a real speed dependence must show up as regret.

    Without this, a transfer test that always passed would look like a law.
    """
    frame = _synthetic_frame(speed_dependent=True)
    cells = cells_from_frame(frame, x="budget")
    result = run_transfer(cells, axis="speed", family="pwlinear3")
    assert result is not None
    # The optimal budgets genuinely differ between the two speeds.
    per_speed = {
        c.key.nu_e: int(c.budget[np.argmax(c.y)]) for c in cells
    }
    assert len(set(per_speed.values())) > 1, f"precondition failed: {per_speed}"
    assert result.mean_budget_regret > 0.0, result.summary()


def test_an_unvaried_axis_reports_none_rather_than_success() -> None:
    """An untested transfer is not a passed one."""
    cells = cells_from_frame(_synthetic_frame(False, nu_hs=(1.0,)), x="budget")
    assert run_transfer(cells, axis="hardware") is None
    assert run_transfer(cells, axis="reflex") is None


def test_unknown_axis_is_rejected() -> None:
    cells = cells_from_frame(_synthetic_frame(False), x="budget")
    with pytest.raises(ValueError, match="unknown axis"):
        run_transfer(cells, axis="wishful")


def test_collapse_returns_none_when_the_grid_cannot_test_it() -> None:
    """No two (nu_e, nu_h) pairs share a product, so the hypothesis is untestable.

    Reporting a verdict here would be claiming something the grid never measured.
    """
    frame = _synthetic_frame(False, nu_es=(1.0, 6.0), nu_hs=(1.0,))
    assert collapse_test(frame) is None


def test_collapse_is_testable_when_products_coincide() -> None:
    """(1.0, 2.0) and (2.0, 1.0) share a staleness product, so the axes separate."""
    rows = []
    for nu_e, nu_h in ((1.0, 2.0), (2.0, 1.0)):
        for k in BUDGETS:
            sigma = 1.0 - np.exp(-k / 6.0) - 0.02 * nu_e * nu_h * k
            rows.append(
                dict(
                    env="synth", reflex="mu0", nu_e=nu_e, nu_h=nu_h, budget=k,
                    delay=int(nu_e * nu_h * k), staleness=nu_e * nu_h * k, start_state=0,
                    J_actual=sigma, J_instant=0.0, J_fresh=sigma, J_base=0.0,
                    G_plan=0.0, R_intermediate=0.0, L_arrival=0.0, L_wait=0.0,
                    L_irreversible=0.0, C_hw=0.0, eps_cross=0.0, sigma=sigma,
                    n_seeds=1, ci_lo=sigma, ci_hi=sigma, exact=True,
                    simulations=k, wall_clock_s=0.0,
                )
            )
    result = collapse_test(pd.DataFrame(rows, columns=ATLAS_COLUMNS), family="pwlinear3")
    assert result is not None
    # This surface depends only on the product, so the collapse should hold.
    # pwlinear3 is not exact on this shape, so judge against the effect size.
    effect = float(max(c.effect_size() for c in cells_from_frame(pd.DataFrame(rows, columns=ATLAS_COLUMNS), x='budget')))
    assert result.mean_budget_regret < 0.1 * effect, result.summary()


def test_suite_covers_every_axis_and_marks_the_untested_ones() -> None:
    suite = transfer_suite(_synthetic_frame(False), family="pwlinear3")
    assert set(suite) == {"speed", "hardware", "reflex", "environment", "collapse"}
    assert suite["speed"] is not None
    assert suite["reflex"] is None  # a single reflex was swept


# -- the kill harness -----------------------------------------------------


def test_a_good_surface_passes_and_a_bad_one_kills() -> None:
    good = _synthetic_frame(speed_dependent=False)
    cells = cells_from_frame(good, x="budget")
    report = evaluate_kill_conditions(good, select_families(cells), transfer_suite(good, family="pwlinear3"))
    speed = next(v for v in report.verdicts if "speed" in v.name)
    assert speed.status == PASS, speed.detail

    bad = _synthetic_frame(speed_dependent=True)
    bad_cells = cells_from_frame(bad, x="budget")
    bad_report = evaluate_kill_conditions(bad, select_families(bad_cells), transfer_suite(bad, family="pwlinear3"))
    bad_speed = next(v for v in bad_report.verdicts if "speed" in v.name)
    assert bad_speed.status == KILL, bad_speed.detail
    assert bad_report.any_tripped
    assert "KILL" in bad_report.headline()


def test_an_untested_condition_is_inconclusive_not_passed() -> None:
    """The single most misleading thing this module could do is call it a pass."""
    frame = _synthetic_frame(False)
    cells = cells_from_frame(frame, x="budget")
    report = evaluate_kill_conditions(frame, select_families(cells), transfer_suite(frame, family="pwlinear3"))
    reflex = next(v for v in report.verdicts if "reflex" in v.name)
    assert reflex.status == INCONCLUSIVE
    assert not reflex.tripped
    assert report.any_inconclusive


def test_the_report_leads_with_the_verdict_table() -> None:
    frame = _synthetic_frame(speed_dependent=True)
    cells = cells_from_frame(frame, x="budget")
    md = render_markdown(
        evaluate_kill_conditions(frame, select_families(cells), transfer_suite(frame, family="pwlinear3"))
    )
    assert md.index("KILL") < md.index("| Condition |")
    assert "failed its own stopping rule" in md


def test_kill_harness_runs_on_a_real_exact_sweep() -> None:
    """End to end on the tabular arm, where eps_cross is machine epsilon."""
    mdp = chase_chain(n_positions=5, gamma=0.9, drift=0.4)
    reflex = np.ones(mdp.n_states, dtype=np.int64)
    rows = []
    for nu_e in (1.0, 2.0):
        rows += exact_atlas_rows(
            mdp, env_name="chase", reflex=reflex, reflex_name="stay",
            budgets=BUDGETS, timing=Timing(nu_e=nu_e, nu_h=0.5), states=[0, 7],
        )
    frame = to_frame(rows)
    cells = cells_from_frame(frame, x="budget")
    report = evaluate_kill_conditions(frame, select_families(cells), transfer_suite(frame, family="pwlinear3"))

    noise = next(v for v in report.verdicts if "decomposition" in v.name)
    assert noise.status == PASS, noise.detail
    assert render_markdown(report).startswith("## P1 kill conditions")
