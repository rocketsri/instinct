"""Bounded P5.2 plasticity-only regression pilot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from instinct.core.configschema import ProposalRunConfig
from instinct.core.proposal import common_checks
from instinct.core.results import ResultsWriter
from instinct.core.verdict import PrerequisiteCheck, Status, VerdictReport
from instinct.p5_development.generator import CoordinateGenerator
from instinct.p5_development.plasticity import (
    LocalPlasticityProgram,
    Topology,
    adaptation_curve,
    conventional_initialization,
    make_task,
)


def _ints(raw: object, name: str) -> tuple[int, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise ValueError(f"{name} must be a sequence")
    values = tuple(int(value) for value in raw)
    if not values or min(values) < 1:
        raise ValueError(f"{name} must contain positive integers")
    return values


def validate_p5_2(config: ProposalRunConfig) -> list[PrerequisiteCheck]:
    checks = common_checks(config, "p5_development")
    try:
        pilot_widths = _ints(config.params.get("pilot_widths", [4, 6]), "pilot_widths")
        eval_widths = _ints(config.params.get("eval_widths", [8]), "eval_widths")
        pilot_depths = _ints(config.params.get("pilot_depths", [1]), "pilot_depths")
        eval_depths = _ints(config.params.get("eval_depths", [2]), "eval_depths")
        checks += [
            PrerequisiteCheck(
                "width holdout disjoint",
                not set(pilot_widths) & set(eval_widths),
                f"pilot={pilot_widths}, eval={eval_widths}",
            ),
            PrerequisiteCheck(
                "depth holdout disjoint",
                not set(pilot_depths) & set(eval_depths),
                f"pilot={pilot_depths}, eval={eval_depths}",
            ),
        ]
    except ValueError as exc:
        checks.append(PrerequisiteCheck("valid topology splits", False, str(exc)))
    checks.append(
        PrerequisiteCheck(
            "pilot and evaluation seeds configured",
            bool(config.pilot_seeds) and bool(config.eval_seeds),
        )
    )
    return checks


def _pilot_score(
    coefficients: np.ndarray,
    *,
    seeds: list[int],
    topologies: tuple[Topology, ...],
    steps: int,
) -> float:
    program = LocalPlasticityProgram(coefficients)
    scores = []
    for seed in seeds:
        for family_index, family in enumerate(("sinusoid", "polynomial")):
            task = make_task(seed + 101 * family_index, family)
            for topology in topologies:
                initial = conventional_initialization(topology, seed=17)
                curve = adaptation_curve(initial, task, program=program, steps=steps)
                scores.append(float(curve.mean() / max(curve[0], 1e-12)))
    return float(np.mean(scores))


def _learn_program(
    *,
    seeds: list[int],
    topologies: tuple[Topology, ...],
    steps: int,
) -> tuple[LocalPlasticityProgram, float, int]:
    candidates = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.7, -0.3, 0.2, 0.05]),
        np.array([0.5, 0.2, 0.0, 0.0]),
        np.array([1.2, -0.1, 0.0, 0.02]),
    ]
    for seed in seeds:
        rng = np.random.default_rng(seed)
        candidates.extend(rng.normal(0.0, 0.55, size=(4, 4)))
    scores = [
        _pilot_score(candidate, seeds=seeds, topologies=topologies, steps=steps)
        for candidate in candidates
    ]
    best = int(np.argmin(scores))
    return LocalPlasticityProgram(candidates[best]), float(scores[best]), len(candidates)


def _equivariance_audit(program: LocalPlasticityProgram, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    pre, post, batch = 5, 4, 12
    x = rng.normal(size=(batch, pre))
    y = rng.normal(size=(batch, post))
    error = rng.normal(size=batch)
    weights = rng.normal(size=(post, pre))
    coordinates = (np.arange(post) + 0.5) / post
    reference = program.local_delta_layer(
        pre_activity=x,
        post_activity=y,
        broadcast_error=error,
        weights=weights,
        post_coordinates=coordinates,
        is_output=False,
        step=2,
    )
    pp, qp = rng.permutation(pre), rng.permutation(post)
    permuted = program.local_delta_layer(
        pre_activity=x[:, pp],
        post_activity=y[:, qp],
        broadcast_error=error,
        weights=weights[np.ix_(qp, pp)],
        post_coordinates=coordinates[qp],
        is_output=False,
        step=2,
    )
    equivariance_error = float(np.max(np.abs(permuted - reference[np.ix_(qp, pp)])))

    remote_x, remote_y = rng.normal(size=x.shape), rng.normal(size=y.shape)
    target_pre, target_post = 2, 1
    remote_x[:, target_pre] = x[:, target_pre]
    remote_y[:, target_post] = y[:, target_post]
    remote = program.local_delta_layer(
        pre_activity=remote_x,
        post_activity=remote_y,
        broadcast_error=error,
        weights=weights,
        post_coordinates=coordinates,
        is_output=False,
        step=2,
    )
    locality_error = abs(
        float(remote[target_post, target_pre] - reference[target_post, target_pre])
    )
    return equivariance_error, locality_error


def run_p5_2(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    checks = validate_p5_2(config)
    if not all(check.passed for check in checks):
        return VerdictReport(
            question=config.preregistration.primary_hypothesis,
            status=Status.INCONCLUSIVE,
            evidence_level="E0",
            primary_result={"metric": "configuration_valid", "value": 0.0, "unit": "bool"},
            controls=checks,
            interpretation="P5.2 did not run because its topology/seed split was invalid.",
            non_claim=config.preregistration.non_claim,
            decision="Repair the preregistered split before plasticity training.",
        )

    params: Mapping[str, Any] = config.params
    pilot_widths = _ints(params.get("pilot_widths", [4, 6]), "pilot_widths")
    eval_widths = _ints(params.get("eval_widths", [8]), "eval_widths")
    pilot_depths = _ints(params.get("pilot_depths", [1]), "pilot_depths")
    eval_depths = _ints(params.get("eval_depths", [2]), "eval_depths")
    pilot_topologies = tuple(
        Topology(width, depth) for width in pilot_widths for depth in pilot_depths
    )
    eval_topologies = tuple(
        Topology(width, depth) for width in eval_widths for depth in eval_depths
    )
    pilot_steps = int(params.get("pilot_steps", 5))
    eval_steps = int(params.get("eval_steps", 12))
    learned, pilot_score, candidate_count = _learn_program(
        seeds=config.pilot_seeds,
        topologies=pilot_topologies,
        steps=pilot_steps,
    )
    frozen = LocalPlasticityProgram(CoordinateGenerator().coefficients)
    random_coefficients = np.random.default_rng(config.seed).normal(0.0, 0.55, size=4)
    random = LocalPlasticityProgram(random_coefficients)
    manifest = learned.serialization_manifest()
    equivariance_error, locality_error = _equivariance_audit(learned, config.seed)

    controls: dict[str, tuple[LocalPlasticityProgram | None, bool]] = {
        "learned_local_rule": (learned, False),
        "frozen_local_rule": (frozen, False),
        "random_local_rule": (random, False),
        "no_plasticity": (None, False),
        "direct_backprop": (None, True),
    }
    rows: list[dict[str, Any]] = []
    auc_by_control: dict[str, list[float]] = {name: [] for name in controls}
    zero_loss_spreads = []
    for seed in config.eval_seeds:
        task = make_task(seed, "composition")
        for topology in eval_topologies:
            initial = conventional_initialization(topology, seed=17)
            cell_zero = []
            for name, (program, direct) in controls.items():
                curve = adaptation_curve(
                    initial,
                    task,
                    program=program,
                    steps=eval_steps,
                    direct_backprop=direct,
                )
                normalized_auc = float(curve.mean() / max(curve[0], 1e-12))
                auc_by_control[name].append(normalized_auc)
                cell_zero.append(float(curve[0]))
                program_parameters = 0 if program is None else 5
                program_bytes = (
                    0
                    if program is None
                    else int(program.serialization_manifest()["total_bytes"])
                )
                rows.append(
                    {
                        "stage": "p5.2",
                        "split": "joint_task_topology_holdout",
                        "eval_seed": seed,
                        "task_family": task.family,
                        "width": topology.width,
                        "depth": topology.depth,
                        "control": name,
                        "zero_shot_loss": float(curve[0]),
                        "normalized_adaptation_auc": normalized_auc,
                        "final_loss": float(curve[-1]),
                        "adaptation_steps": eval_steps,
                        "plasticity_program_parameters": program_parameters,
                        "plasticity_program_bytes": program_bytes,
                        "instantiated_parameters": topology.parameter_count,
                        "initializer_parameters": topology.parameter_count,
                        "initializer_owned_by_program": False,
                        "local_rule_compute_units": (
                            0
                            if name == "no_plasticity"
                            else topology.parameter_count * 24 * eval_steps
                        ),
                    }
                )
            zero_loss_spreads.append(max(cell_zero) - min(cell_zero))

    mean_auc = {name: float(np.mean(values)) for name, values in auc_by_control.items()}
    comparator_auc = min(
        mean_auc["frozen_local_rule"],
        mean_auc["random_local_rule"],
        mean_auc["no_plasticity"],
    )
    relative_gain = (comparator_auc - mean_auc["learned_local_rule"]) / max(
        comparator_auc, 1e-12
    )
    rows.append(
        {
            "stage": "p5.2-program-audit",
            "split": "pilot_only",
            "control": "learned_local_rule",
            "pilot_normalized_auc": pilot_score,
            "candidate_count": candidate_count,
            **manifest,
        }
    )
    writer.write_metrics(rows)
    checks += [
        PrerequisiteCheck(
            "local-rule permutation equivariance",
            equivariance_error < 1e-12,
            f"error={equivariance_error:.2e}",
        ),
        PrerequisiteCheck(
            "strict local-information taint",
            locality_error < 1e-12,
            f"error={locality_error:.2e}",
        ),
        PrerequisiteCheck(
            "initialization separated across controls",
            max(zero_loss_spreads) < 1e-12,
            f"max zero-shot spread={max(zero_loss_spreads):.2e}",
        ),
        PrerequisiteCheck(
            "fixed complete program description",
            int(manifest["total_bytes"]) == 40
            and int(manifest["initializer_bytes"]) == 0
            and int(manifest["task_conditioning_bytes"]) == 0,
        ),
        PrerequisiteCheck(
            "held-out losses finite",
            all(np.isfinite(value) for values in auc_by_control.values() for value in values),
        ),
    ]
    minimum_gain = float(params.get("minimum_relative_gain", 0.0))
    plasticity_passes = relative_gain > minimum_gain
    status = (
        Status.NARROW
        if plasticity_passes and all(check.passed for check in checks)
        else Status.INCONCLUSIVE
    )
    writer.event(
        stage="p5.2",
        learned_coefficients=np.asarray(learned.coefficients).tolist(),
        pilot_score=pilot_score,
        heldout_relative_gain=relative_gain,
        program_bytes=manifest["total_bytes"],
    )
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=status,
        evidence_level="E1",
        primary_result={
            "metric": "heldout_normalized_auc_relative_gain_vs_best_nonbackprop_control",
            "value": relative_gain,
            "unit": "fraction",
        },
        baselines_run=[
            "frozen local rule",
            "scale-matched random local rule",
            "no plasticity",
            "direct backprop with same examples and steps",
        ],
        baselines_skipped={
            "NNiT/full hypernetwork": "not comparable in the bounded CPU plasticity-only slice",
            "P5.3 joint generator": "blocked until P5.2 passes independent review",
        },
        controls=checks,
        interpretation=(
            "The fixed local rule improves early adaptation on a joint task/topology holdout, "
            "but this plasticity-only result is not joint developmental-program evidence."
            if status is Status.NARROW
            else (
                "The plasticity-only rule did not beat its cheap held-out controls "
                "after accounting."
            )
        ),
        non_claim=config.preregistration.non_claim,
        decision=(
            "P5.2 passes narrowly; require independent review before preregistering P5.3."
            if status is Status.NARROW
            else (
                "Do not begin P5.3; diagnose the local rule against the direct "
                "and frozen controls."
            )
        ),
    )
