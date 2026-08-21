"""Small shared helpers for proposal plugins."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from instinct.core.configschema import ProposalRunConfig
from instinct.core.verdict import PrerequisiteCheck


def common_checks(config: ProposalRunConfig, expected_experiment: str) -> list[PrerequisiteCheck]:
    return [
        PrerequisiteCheck(
            "experiment id",
            config.experiment == expected_experiment,
            f"configured={config.experiment!r}",
        ),
        PrerequisiteCheck(
            "preregistration complete",
            all(
                (
                    config.preregistration.estimand,
                    config.preregistration.primary_hypothesis,
                    config.preregistration.non_claim,
                )
            ),
        ),
        PrerequisiteCheck(
            "theory traceability",
            bool(config.theory.proposal_version and config.theory.theory_id),
            f"{config.theory.proposal_version}/{config.theory.theory_id}",
        ),
        PrerequisiteCheck(
            "seed split disjoint",
            not set(config.pilot_seeds) & set(config.eval_seeds),
        ),
    ]


def param(config: ProposalRunConfig, name: str, default: Any, expected: type) -> Any:
    value = config.params.get(name, default)
    if not isinstance(value, expected):
        raise ValueError(f"params.{name} must be {expected.__name__}, got {value!r}")
    return value


def resume_by_rerun(
    config: ProposalRunConfig,
    writer: Any,
    checkpoint_state: Mapping[str, Any],
    run_fn: Any,
) -> Any:
    """Exact CPU gates are deterministic and cheap; restart them on resume."""
    del checkpoint_state
    return run_fn(config, writer)
