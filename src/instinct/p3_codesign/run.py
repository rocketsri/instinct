from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from instinct.core.configschema import ProposalRunConfig
from instinct.core.envs.tabular import chase_chain
from instinct.core.proposal import common_checks, resume_by_rerun
from instinct.core.results import ResultsWriter
from instinct.core.verdict import PrerequisiteCheck, Status, VerdictReport
from instinct.p3_codesign.objective import (
    constructed_case,
    crossplay_matrix,
    enumerate_reflexes,
    option_objective,
    policy_fingerprint,
)


def validate(config: ProposalRunConfig) -> list[PrerequisiteCheck]:
    if config.params.get("stage") == "p3.1":
        if config.params.get("recovery_version") == "v5-inertial-latency":
            from instinct.p3_codesign.recovery2 import validate_recovery2

            return validate_recovery2(config)
        if config.params.get("recovery_version") == "v4-advantage-regression":
            from instinct.p3_codesign.recovery import validate_recovery

            return validate_recovery(config)
        from instinct.p3_codesign.stage1 import validate_p3_1

        return validate_p3_1(config)
    return common_checks(config, "p3_codesign")


def run(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    if config.params.get("stage") == "p3.1":
        if config.params.get("recovery_version") == "v5-inertial-latency":
            from instinct.p3_codesign.recovery2 import run_recovery2

            return run_recovery2(config, writer)
        if config.params.get("recovery_version") == "v4-advantage-regression":
            from instinct.p3_codesign.recovery import run_recovery

            return run_recovery(config, writer)
        from instinct.p3_codesign.stage1 import run_p3_1

        return run_p3_1(config, writer)
    checks = validate(config)
    metrics: list[dict[str, Any]] = []
    margins = []
    for kind in ("hold", "greedy", "interior", "destructive"):
        mdp, action, q, expected = constructed_case(kind)
        rows = enumerate_reflexes(mdp, planner_action=action, planner_q=q, delay=1)
        by_first = {a: max(r.value for r in rows if r.policy[0] == a) for a in range(mdp.n_actions)}
        ranking = sorted(by_first, key=lambda item: by_first[item], reverse=True)
        margin = by_first[ranking[0]] - by_first[ranking[1]]
        margins.append(margin)
        checks.append(
            PrerequisiteCheck(
                f"{kind} strict optimum",
                ranking[0] == expected and margin > 0,
                f"action={ranking[0]}, margin={margin:.3g}",
            )
        )
        metrics.extend(
            {"case": kind, "reflex_action": a, "objective": v, "expected_action": expected}
            for a, v in by_first.items()
        )
    mdp, action, q, _ = constructed_case("interior")
    reflex = np.array([2, 0, 0, 0], dtype=np.int64)
    at_zero = option_objective(mdp, reflex=reflex, planner_action=action, planner_q=q, delay=0)
    expected_zero = q[np.arange(mdp.n_states), action]
    at_one = option_objective(mdp, reflex=reflex, planner_action=action, planner_q=q, delay=1)
    brute_one = mdp.R[np.arange(mdp.n_states), reflex] + mdp.gamma * np.array(
        [mdp.P[s, reflex[s]] @ q[:, action[s]] for s in range(mdp.n_states)]
    )
    checks += [
        PrerequisiteCheck("d=0 arrival indexing", bool(np.allclose(at_zero, expected_zero))),
        PrerequisiteCheck("d=1 brute-force trajectory", bool(np.allclose(at_one, brute_one))),
    ]
    stochastic = chase_chain(n_positions=3, gamma=0.85, drift=0.4)
    _, _, stochastic_pi = stochastic.value_iteration()
    stochastic_q = stochastic.R + stochastic.gamma * np.einsum(
        "sat,t->sa", stochastic.P, stochastic.value_iteration()[0]
    )
    stochastic_reflex = np.ones(stochastic.n_states, dtype=np.int64)
    stochastic_one = option_objective(
        stochastic,
        reflex=stochastic_reflex,
        planner_action=stochastic_pi,
        planner_q=stochastic_q,
        delay=1,
    )
    stochastic_brute = stochastic.R[
        np.arange(stochastic.n_states), stochastic_reflex
    ] + stochastic.gamma * np.array(
        [
            stochastic.P[s, stochastic_reflex[s]] @ stochastic_q[:, stochastic_pi[s]]
            for s in range(stochastic.n_states)
        ]
    )

    # A known-answer cross-play construction: each planner is correct for the
    # arrival distribution induced by the reflex it simulated, and mismatch is
    # strictly worse. Fingerprints make accidental identity impossible.
    cross_mdp, _, _, _ = constructed_case("interior")
    hold = np.array([0, 0, 0, 0], dtype=np.int64)
    setup = np.array([2, 0, 0, 0], dtype=np.int64)
    plan_hold = np.zeros(4, dtype=np.int64)
    plan_setup = np.ones(4, dtype=np.int64)
    cross_q = np.zeros((4, 3))
    cross_q[1, 0] = 3.0
    cross_q[3, 1] = 3.0
    cross = crossplay_matrix(
        cross_mdp,
        execution_reflexes=(hold, setup),
        planned_actions=(plan_hold, plan_setup),
        planner_q=cross_q,
        delay=1,
    )
    diagonal_margin = float(min(cross[0, 0] - cross[0, 1], cross[1, 1] - cross[1, 0]))
    identity_distinct = policy_fingerprint(hold) != policy_fingerprint(setup)
    checks += [
        PrerequisiteCheck(
            "stochastic d=1 brute-force trajectory",
            bool(np.allclose(stochastic_one, stochastic_brute)),
        ),
        PrerequisiteCheck(
            "planner-reflex cross-play diagonal",
            diagonal_margin > 0,
            f"minimum diagonal margin={diagonal_margin:.3g}",
        ),
        PrerequisiteCheck("planner/reflex identity is explicit", identity_distinct),
    ]
    metrics.extend(
        {
            "case": "crossplay",
            "reflex_action": i,
            "objective": float(cross[i, j]),
            "expected_action": i,
            "planner_variant": j,
        }
        for i in range(2)
        for j in range(2)
    )
    writer.write_metrics(metrics)
    writer.event(stage="p3.0", cases=4, minimum_margin=min(margins))
    passed = all(c.passed for c in checks)
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=Status.GO if passed else Status.INCONCLUSIVE,
        evidence_level="E0",
        primary_result={
            "metric": "minimum_exact_optimum_margin",
            "value": min(margins),
            "unit": "return",
        },
        baselines_run=[
            "hold",
            "greedy",
            "interior setup",
            "destructive temptation",
            "full deterministic enumeration",
            "planner/reflex mismatch cross-play",
            "stochastic-transition brute force",
        ],
        controls=checks,
        interpretation=(
            "The frozen one-interval objective can express all four strict optima; "
            "this is not learned co-design evidence."
        ),
        non_claim=config.preregistration.non_claim,
        decision="Add exact concurrent-control/when-to-plan baselines before P3.1."
        if passed
        else "Repair the objective before neural work.",
    )


def resume(
    config: ProposalRunConfig, writer: ResultsWriter, checkpoint_state: Mapping[str, Any]
) -> VerdictReport:
    return resume_by_rerun(config, writer, checkpoint_state, run)
