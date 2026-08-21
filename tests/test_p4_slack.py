from __future__ import annotations

import numpy as np
import pytest

from instinct.core.envs.tabular import corridor_with_pit
from instinct.p4_slack.slack import (
    disturbed_replan_values,
    finite_horizon_replan_value,
    lower_quantile,
    pareto_dominated,
    recovery_slack,
    reward_rescaling_consistent,
)


def test_lower_quantile_exposes_rare_catastrophic_mass() -> None:
    values = np.array([-10.0, 2.0, 2.0, 2.0])
    assert lower_quantile(values, 0.25) == -10.0
    assert lower_quantile(values, 0.5) == 2.0


def test_exact_replan_preserves_failure_origin() -> None:
    mdp = corridor_with_pit(length=5, gamma=0.9, slip=0.2)
    value = finite_horizon_replan_value(mdp, horizon=6, v_fail=-3.0)
    assert value[mdp.failure][0] == -3.0
    assert np.isfinite(value).all()


def test_disturbance_kernel_validation_and_exact_projection() -> None:
    value = np.array([1.0, -1.0])
    kernel = np.array([[[0.75, 0.25], [0.0, 1.0]]])
    assert np.allclose(disturbed_replan_values(value, kernel), [[0.5, -1.0]])
    with pytest.raises(ValueError, match="probability distribution"):
        disturbed_replan_values(value, kernel * 2)


def test_reward_rescaling_requires_margin_rescaling() -> None:
    values = np.array([[0.0, 1.0, 2.0], [-1.0, 0.5, 1.5]])
    assert reward_rescaling_consistent(
        values,
        v_fail=-2.0,
        margin=1.5,
        q=0.25,
        scale=4.0,
        offset=9.0,
    )
    assert np.allclose(
        recovery_slack(4 * values + 9, v_fail=1.0, q=0.25),
        4 * recovery_slack(values, v_fail=-2.0, q=0.25),
    )


def test_matched_baseline_dominance_is_two_dimensional() -> None:
    baselines = [(0.6, 0.8), (0.8, 0.6)]
    assert pareto_dominated((0.5, 0.7), baselines)
    assert not pareto_dominated((0.7, 0.7), baselines)
