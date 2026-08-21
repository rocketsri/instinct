from __future__ import annotations

import numpy as np
import pytest

from instinct.p3_codesign.objective import constructed_case, crossplay_matrix


def test_crossplay_separates_planner_model_from_execution_reflex() -> None:
    mdp, _, _, _ = constructed_case("interior")
    hold = np.array([0, 0, 0, 0])
    setup = np.array([2, 0, 0, 0])
    plan_hold = np.zeros(4, dtype=np.int64)
    plan_setup = np.ones(4, dtype=np.int64)
    q = np.zeros((4, 3))
    q[1, 0], q[3, 1] = 3.0, 3.0
    values = crossplay_matrix(
        mdp,
        execution_reflexes=(hold, setup),
        planned_actions=(plan_hold, plan_setup),
        planner_q=q,
        delay=1,
    )
    assert values[0, 0] > values[0, 1]
    assert values[1, 1] > values[1, 0]


def test_crossplay_rejects_unpaired_variant_sets() -> None:
    mdp, action, q, _ = constructed_case("hold")
    with pytest.raises(ValueError, match="equally many"):
        crossplay_matrix(
            mdp,
            execution_reflexes=(np.zeros(4, dtype=np.int64),),
            planned_actions=(action, action),
            planner_q=q,
            delay=1,
        )
