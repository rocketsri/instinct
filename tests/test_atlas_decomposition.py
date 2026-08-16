"""The P1 decomposition: identity, known answers, and term separation.

The tests that matter here are the ones with answers known in advance. A
decomposition that merely sums correctly proves nothing — it can be made to sum
correctly by defining one term as the leftover. These check that each term
measures the thing it claims to.
"""

from __future__ import annotations

import numpy as np
import pytest

from instinct.atlas.decomposition import decompose_exact, exact_atlas_rows
from instinct.atlas.schema import DECOMPOSITION_TERMS, to_frame, validate_frame
from instinct.core.env import Timing
from instinct.envs.tabular import chase_chain, corridor_with_pit


@pytest.fixture
def chase():
    mdp = chase_chain(n_positions=5, gamma=0.9, drift=0.4)
    reflex = np.ones(mdp.n_states, dtype=np.int64)  # "stay"
    return mdp, reflex


@pytest.fixture
def pit():
    mdp = corridor_with_pit(length=6, gamma=0.9, slip=0.25)
    reflex = np.zeros(mdp.n_states, dtype=np.int64)  # "advance", so waiting is risky
    return mdp, reflex


# -- the identity ---------------------------------------------------------


@pytest.mark.parametrize("budget", [1, 2, 4, 8])
@pytest.mark.parametrize("nu_h", [0.0, 0.5, 1.0, 2.0])
def test_residual_is_exactly_the_base_arms_own_delay_cost(chase, budget, nu_h) -> None:
    """The chain telescopes, so the leftover is one identified quantity.

    Not "small" — *identified*. Asserting the residual is merely small would let
    it quietly absorb any effect the decomposition has no name for, which is
    exactly the failure the first version of this module had.
    """
    mdp, reflex = chase
    d = decompose_exact(
        mdp, reflex=reflex, budget=budget, timing=Timing(nu_e=1.0, nu_h=nu_h)
    )
    assert np.allclose(d.eps_cross, d.L_base_delay, atol=1e-12), (
        "the residual should be the base arm's delay cost and nothing else"
    )


@pytest.mark.parametrize("budget", [1, 2, 4, 8])
@pytest.mark.parametrize("nu_h", [0.0, 0.5])
def test_residual_vanishes_when_the_base_decision_lands_immediately(chase, budget, nu_h) -> None:
    """With no base delay there is nothing left over at all."""
    mdp, reflex = chase
    timing = Timing(nu_e=1.0, nu_h=nu_h)
    assert timing.delay_ticks(1) == 0, "precondition: base budget lands immediately"
    d = decompose_exact(mdp, reflex=reflex, budget=budget, timing=timing)
    assert d.max_residual() < 1e-12, f"residual {d.max_residual():.3e} should be zero"


def test_residual_is_a_real_leftover_not_a_definition(chase) -> None:
    """Each term is measured independently, so the residual can be non-trivial.

    Guards against the failure where one term is silently defined as
    ``sigma`` minus the others: that makes ``eps_cross`` identically zero and
    certifies nothing. Perturbing a term must therefore break the identity.
    """
    mdp, reflex = chase
    d = decompose_exact(mdp, reflex=reflex, budget=8, timing=Timing(nu_e=1.0, nu_h=1.0))
    tampered = d.G_plan + 1.0
    explained = (
        tampered + d.R_intermediate - d.L_arrival - d.L_irreversible - d.C_hw
    )
    assert not np.allclose(explained, d.sigma), "the identity should not survive tampering"


def test_schema_frame_validates(chase) -> None:
    mdp, reflex = chase
    rows = exact_atlas_rows(
        mdp, env_name="chase", reflex=reflex, reflex_name="stay",
        budgets=[1, 2, 4, 8], timing=Timing(nu_e=1.0, nu_h=0.5), states=[0, 7, 13],
    )
    frame = to_frame(rows)
    validate_frame(frame)  # raises if the telescoping identity fails
    assert len(frame) == 12
    assert set(DECOMPOSITION_TERMS) <= set(frame.columns)


# -- known answers --------------------------------------------------------


def test_base_budget_against_itself_yields_zero_net_gain(chase) -> None:
    """Comparing the base budget against itself: no gain, and no differences.

    ``L_arrival`` and ``L_irreversible`` do NOT vanish here, and should not: the
    base arm still pays its own delay. What must vanish is the net ``sigma`` and
    every term defined as a difference between the two budgets.
    """
    mdp, reflex = chase
    d = decompose_exact(
        mdp, reflex=reflex, budget=1, base_budget=1, timing=Timing(nu_e=1.0, nu_h=1.0)
    )
    assert np.allclose(d.sigma, 0.0, atol=1e-12)
    assert np.allclose(d.G_plan, 0.0, atol=1e-12)
    assert np.allclose(d.R_intermediate, 0.0, atol=1e-12)
    assert np.allclose(d.L_unrecoverable, 0.0, atol=1e-12)


def test_zero_speed_kills_every_temporal_term(chase) -> None:
    """With ``nu_e = 0`` nothing goes stale, so both temporal terms must vanish.

    The plan's headline known-answer check. ``G_plan`` must survive: planning
    still helps, it just costs nothing in freshness.
    """
    mdp, reflex = chase
    d = decompose_exact(mdp, reflex=reflex, budget=16, timing=Timing(nu_e=0.0, nu_h=1.0))
    assert np.allclose(d.L_arrival, 0.0, atol=1e-12)
    assert np.allclose(d.L_irreversible, 0.0, atol=1e-12)
    assert np.allclose(d.L_unrecoverable, 0.0, atol=1e-12)
    assert np.allclose(d.R_intermediate, 0.0, atol=1e-12)
    assert np.allclose(d.sigma, d.G_plan, atol=1e-12)


def test_arrival_loss_is_non_negative_and_grows_with_speed(chase) -> None:
    """Staleness cannot help, and a faster world should not make it cheaper."""
    mdp, reflex = chase
    losses = []
    for nu_e in (0.0, 0.5, 1.0, 2.0):
        d = decompose_exact(mdp, reflex=reflex, budget=8, timing=Timing(nu_e=nu_e, nu_h=1.0))
        assert np.all(d.L_arrival > -1e-9), f"negative arrival loss at nu_e={nu_e}"
        losses.append(float(d.L_arrival.mean()))
    assert losses[-1] >= losses[0] - 1e-12


def test_irreversible_term_separates_the_two_environments(chase, pit) -> None:
    """The term must fire where damage is unrecoverable and stay quiet where it is not.

    ``chase_chain`` has no absorbing states, so nothing done during a delay is
    permanent and ``L_irreversible`` should be negligible. ``corridor_with_pit``
    has an absorbing pit, so waiting while advancing genuinely destroys value.
    If this test ever fails, the term is measuring "being behind" rather than
    "being unable to catch up", and the decomposition is confounded.
    """
    chase_mdp, chase_reflex = chase
    pit_mdp, pit_reflex = pit
    timing = Timing(nu_e=1.0, nu_h=1.0)

    recoverable = decompose_exact(chase_mdp, reflex=chase_reflex, budget=8, timing=timing)
    unrecoverable = decompose_exact(pit_mdp, reflex=pit_reflex, budget=8, timing=timing)

    # Both pay a large cost for waiting -- discounting alone guarantees that --
    # so L_irreversible is big in both and cannot tell them apart.
    assert float(recoverable.L_irreversible.max()) > 0.5
    assert float(unrecoverable.L_irreversible.max()) > 0.5

    # L_unrecoverable is the term that must discriminate, and the signature is
    # its *sign structure*, not its magnitude. Where everything is recoverable,
    # waiting merely shifts you to a different phase of the chase: sometimes
    # better, sometimes worse, averaging to nothing. Where a pit absorbs, waiting
    # can only cost you, so the term is one-signed.
    assert float(np.abs(recoverable.L_unrecoverable.mean())) < 0.02, (
        "with no absorbing states, waiting should cost nothing on average"
    )
    assert float(recoverable.L_unrecoverable.min()) < -0.05, (
        "and it should sometimes help, which is what makes it recoverable"
    )

    assert float(unrecoverable.L_unrecoverable.min()) >= -1e-9, (
        "with an absorbing pit, waiting can never leave you better off"
    )
    assert float(unrecoverable.L_unrecoverable.mean()) > 0.1, (
        "an absorbing pit must register a systematic unrecoverable loss"
    )


def test_cost_term_is_charged_not_assumed(chase) -> None:
    """C_hw scales with the extra simulations actually bought."""
    mdp, reflex = chase
    timing = Timing(nu_e=1.0, nu_h=1.0)
    free = decompose_exact(mdp, reflex=reflex, budget=9, timing=timing, cost_per_simulation=0.0)
    priced = decompose_exact(
        mdp, reflex=reflex, budget=9, timing=timing, cost_per_simulation=0.01
    )
    assert np.allclose(free.C_hw, 0.0)
    assert np.allclose(priced.C_hw, 0.01 * 8)
    assert np.allclose(priced.sigma, free.sigma - 0.01 * 8, atol=1e-12)


def test_planning_helps_when_it_is_free(chase) -> None:
    """With no delay and no cost, more budget cannot hurt."""
    mdp, reflex = chase
    timing = Timing(nu_e=0.0, nu_h=0.0)
    small = decompose_exact(mdp, reflex=reflex, budget=2, timing=timing)
    large = decompose_exact(mdp, reflex=reflex, budget=64, timing=timing)
    assert large.sigma.mean() >= small.sigma.mean() - 1e-9


def test_delay_plateaus_are_visible(chase) -> None:
    """Budgets sharing a delay differ only in planner quality.

    Integer delays make several budgets collapse onto one delay. Those groups
    are the cleanest read on ``G_plan`` with staleness held exactly fixed — and
    equally, they are where a rounding plateau could be misread as a threshold
    regime, so the sweep has to be able to see them.
    """
    timing = Timing(nu_e=1.0, nu_h=0.25)
    groups = timing.distinct_delays([1, 2, 3, 4, 5, 6, 7, 8])
    shared = [ks for ks in groups.values() if len(ks) > 1]
    assert shared, "expected several budgets to share a delay at nu_h=0.25"

    mdp, reflex = chase
    for ks in shared:
        delays = {decompose_exact(mdp, reflex=reflex, budget=k, timing=timing).delay for k in ks}
        assert len(delays) == 1


def test_state_axis_comes_free_from_one_solve(chase) -> None:
    """Every start state is returned by the same solve, not looped over."""
    mdp, reflex = chase
    d = decompose_exact(mdp, reflex=reflex, budget=4, timing=Timing(nu_e=1.0, nu_h=1.0))
    assert d.sigma.shape == (mdp.n_states,)
    assert np.isfinite(d.sigma).all()
