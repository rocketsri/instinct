"""P5.2 recovery 1: fresh holdouts, richer rule, and pilot-derived effect size."""

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
from instinct.p5_development.stage2 import _equivariance_audit


def _ints(raw: object, name: str) -> tuple[int, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise ValueError(f"{name} must be a sequence")
    values = tuple(int(value) for value in raw)
    if not values or min(values) < 1:
        raise ValueError(f"{name} must contain positive integers")
    return values


def validate_recovery(config: ProposalRunConfig) -> list[PrerequisiteCheck]:
    checks = common_checks(config, "p5_development")
    old_eval_seeds = {591, 592}
    checks += [
        PrerequisiteCheck(
            "recovery evaluation units are fresh",
            not old_eval_seeds & set(config.seeds),
            f"sealed prior eval seeds={sorted(old_eval_seeds)}",
        ),
        PrerequisiteCheck(
            "pilot selection and threshold calibration separated",
            len(config.pilot_seeds) >= 3,
            "last pilot seed is calibration-only",
        ),
        PrerequisiteCheck(
            "fresh evaluation seeds configured",
            bool(config.eval_seeds),
        ),
    ]
    try:
        pilot_widths = _ints(config.params.get("pilot_widths", [5, 7]), "pilot_widths")
        eval_widths = _ints(config.params.get("eval_widths", [9]), "eval_widths")
        pilot_depths = _ints(config.params.get("pilot_depths", [1, 2]), "pilot_depths")
        eval_depths = _ints(config.params.get("eval_depths", [3]), "eval_depths")
        checks += [
            PrerequisiteCheck(
                "recovery width holdout disjoint",
                not set(pilot_widths) & set(eval_widths),
            ),
            PrerequisiteCheck(
                "recovery depth holdout disjoint",
                not set(pilot_depths) & set(eval_depths),
            ),
        ]
    except ValueError as exc:
        checks.append(PrerequisiteCheck("valid recovery topology split", False, str(exc)))
    return checks


def _score(
    program: LocalPlasticityProgram,
    *,
    seeds: Sequence[int],
    topologies: Sequence[Topology],
    steps: int,
) -> float:
    values = []
    for seed in seeds:
        for family_index, family in enumerate(("sinusoid", "polynomial", "composition")):
            task = make_task(seed + 103 * family_index, family)
            for topology in topologies:
                initial = conventional_initialization(topology, seed=29)
                curve = adaptation_curve(initial, task, program=program, steps=steps)
                values.append(float(curve.mean() / max(curve[0], 1e-12)))
    return float(np.mean(values))


def _search(
    *,
    selection_seeds: Sequence[int],
    topologies: Sequence[Topology],
    steps: int,
) -> tuple[LocalPlasticityProgram, float, int]:
    zero_tail = np.zeros(4)
    candidates: list[LocalPlasticityProgram] = [
        LocalPlasticityProgram(np.r_[np.array([1.0, 0.0, 0.0, 0.0]), zero_tail]),
        LocalPlasticityProgram(np.r_[np.array([1.2, -0.1, 0.0, 0.02]), zero_tail]),
        LocalPlasticityProgram(np.array([1.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0])),
        LocalPlasticityProgram(np.array([0.8, 0.1, 0.0, 0.01, 0.2, 0.1, 0.0, 0.0])),
    ]
    for seed in selection_seeds:
        rng = np.random.default_rng(seed)
        for coefficients in rng.normal(0.0, 0.45, size=(8, 8)):
            candidates.append(LocalPlasticityProgram(coefficients))
    scores = [
        _score(candidate, seeds=selection_seeds, topologies=topologies, steps=steps)
        for candidate in candidates
    ]
    best = candidates[int(np.argmin(scores))]
    rng = np.random.default_rng(sum(selection_seeds) + 17)
    refinements = [
        LocalPlasticityProgram(
            np.asarray(best.coefficients) + rng.normal(0.0, 0.08, size=8),
            step_size=step_size,
        )
        for step_size in (0.025, 0.04, 0.06)
        for _ in range(4)
    ]
    refined_scores = [
        _score(candidate, seeds=selection_seeds, topologies=topologies, steps=steps)
        for candidate in refinements
    ]
    candidates.extend(refinements)
    scores.extend(refined_scores)
    winner = int(np.argmin(scores))
    return candidates[winner], float(scores[winner]), len(candidates)


def _random_programs(seed: int, count: int) -> tuple[LocalPlasticityProgram, ...]:
    rng = np.random.default_rng(seed)
    return tuple(
        LocalPlasticityProgram(rng.normal(0.0, 0.45, size=8)) for _ in range(count)
    )


def _pilot_threshold(
    learned: LocalPlasticityProgram,
    *,
    calibration_seed: int,
    topologies: Sequence[Topology],
    random_rules: Sequence[LocalPlasticityProgram],
    steps: int,
) -> tuple[float, np.ndarray]:
    frozen = LocalPlasticityProgram(CoordinateGenerator().coefficients)
    gains = []
    for family_index, family in enumerate(("sinusoid", "polynomial", "composition")):
        task = make_task(calibration_seed + 103 * family_index, family)
        for topology in topologies:
            initial = conventional_initialization(topology, seed=29)
            learned_curve = adaptation_curve(initial, task, program=learned, steps=steps)
            learned_auc = float(learned_curve.mean() / max(learned_curve[0], 1e-12))
            comparator_auc = [1.0]
            for program in (frozen, *random_rules):
                curve = adaptation_curve(initial, task, program=program, steps=steps)
                comparator_auc.append(float(curve.mean() / max(curve[0], 1e-12)))
            strongest = min(comparator_auc)
            gains.append((strongest - learned_auc) / max(strongest, 1e-12))
    gain_array = np.asarray(gains, dtype=np.float64)
    standard_error = float(gain_array.std(ddof=1) / np.sqrt(gain_array.size))
    threshold = max(float(np.nextafter(0.0, 1.0)), 1.96 * standard_error)
    return threshold, gain_array


def run_recovery(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    checks = validate_recovery(config)
    if not all(check.passed for check in checks):
        return VerdictReport(
            question=config.preregistration.primary_hypothesis,
            status=Status.INCONCLUSIVE,
            evidence_level="E0",
            primary_result={"metric": "recovery_config_valid", "value": 0.0, "unit": "bool"},
            controls=checks,
            interpretation="Recovery did not run because fresh-unit controls failed.",
            non_claim=config.preregistration.non_claim,
            decision="Repair recovery registration without opening sealed evaluation units.",
        )

    params: Mapping[str, Any] = config.params
    pilot_topologies = tuple(
        Topology(width, depth)
        for width in _ints(params.get("pilot_widths", [5, 7]), "pilot_widths")
        for depth in _ints(params.get("pilot_depths", [1, 2]), "pilot_depths")
    )
    eval_topologies = tuple(
        Topology(width, depth)
        for width in _ints(params.get("eval_widths", [9, 11]), "eval_widths")
        for depth in _ints(params.get("eval_depths", [3]), "eval_depths")
    )
    selection_seeds = config.pilot_seeds[:-1]
    calibration_seed = config.pilot_seeds[-1]
    pilot_steps = int(params.get("pilot_steps", 6))
    eval_steps = int(params.get("eval_steps", 15))
    random_count = int(params.get("random_rule_count", 8))
    learned, pilot_score, candidate_count = _search(
        selection_seeds=selection_seeds,
        topologies=pilot_topologies,
        steps=pilot_steps,
    )
    random_rules = _random_programs(config.seed + 700, random_count)
    threshold, calibration_gains = _pilot_threshold(
        learned,
        calibration_seed=calibration_seed,
        topologies=pilot_topologies,
        random_rules=random_rules,
        steps=pilot_steps,
    )
    frozen = LocalPlasticityProgram(CoordinateGenerator().coefficients)
    programs: dict[str, tuple[LocalPlasticityProgram | None, bool]] = {
        "learned_extended_rule": (learned, False),
        "frozen_local_rule": (frozen, False),
        "no_plasticity": (None, False),
        "direct_backprop": (None, True),
    }
    programs.update(
        {f"random_rule_{index:02d}": (program, False) for index, program in enumerate(random_rules)}
    )
    rows: list[dict[str, Any]] = []
    aucs: dict[str, list[float]] = {name: [] for name in programs}
    zero_spreads = []
    for seed in config.eval_seeds:
        task = make_task(seed, "warped")
        for topology in eval_topologies:
            initial = conventional_initialization(topology, seed=29)
            zero_losses = []
            for name, (program, direct) in programs.items():
                curve = adaptation_curve(
                    initial,
                    task,
                    program=program,
                    steps=eval_steps,
                    direct_backprop=direct,
                )
                auc = float(curve.mean() / max(curve[0], 1e-12))
                aucs[name].append(auc)
                zero_losses.append(float(curve[0]))
                manifest = program.serialization_manifest() if program is not None else None
                rows.append(
                    {
                        "stage": "p5.2-recovery-1",
                        "split": "fresh_joint_holdout",
                        "eval_seed": seed,
                        "task_family": task.family,
                        "width": topology.width,
                        "depth": topology.depth,
                        "control": name,
                        "zero_shot_loss": float(curve[0]),
                        "normalized_adaptation_auc": auc,
                        "final_loss": float(curve[-1]),
                        "adaptation_steps": eval_steps,
                        "plasticity_program_parameters": (
                            0 if program is None else len(program.coefficients) + 1
                        ),
                        "plasticity_program_bytes": (
                            0 if manifest is None else int(manifest["total_bytes"])
                        ),
                        "instantiated_parameters": topology.parameter_count,
                        "initializer_owned_by_program": False,
                    }
                )
            zero_spreads.append(max(zero_losses) - min(zero_losses))

    mean_auc = {name: float(np.mean(values)) for name, values in aucs.items()}
    random_names = [name for name in mean_auc if name.startswith("random_rule_")]
    comparator_names = ["frozen_local_rule", "no_plasticity", *random_names]
    strongest_name = min(comparator_names, key=mean_auc.__getitem__)
    strongest_auc = mean_auc[strongest_name]
    heldout_gain = (strongest_auc - mean_auc["learned_extended_rule"]) / max(
        strongest_auc, 1e-12
    )
    random_values = np.asarray([mean_auc[name] for name in random_names])
    manifest = learned.serialization_manifest()
    equivariance_error, locality_error = _equivariance_audit(learned, config.seed)
    rows.append(
        {
            "stage": "p5.2-recovery-preregistration",
            "split": "pilot_calibration_only",
            "control": "pilot_effect_threshold",
            "meaningful_effect_threshold": threshold,
            "calibration_gain_mean": float(calibration_gains.mean()),
            "calibration_gain_std": float(calibration_gains.std(ddof=1)),
            "pilot_score": pilot_score,
            "candidate_count": candidate_count,
            "random_rule_count": random_count,
            **manifest,
        }
    )
    writer.write_metrics(rows)
    checks += [
        PrerequisiteCheck("pilot-derived threshold is nonzero", threshold > 0.0),
        PrerequisiteCheck(
            "extended-rule permutation equivariance",
            equivariance_error < 1e-12,
            f"error={equivariance_error:.2e}",
        ),
        PrerequisiteCheck(
            "extended-rule strict locality",
            locality_error < 1e-12,
            f"error={locality_error:.2e}",
        ),
        PrerequisiteCheck(
            "recovery initialization separated",
            max(zero_spreads) < 1e-12,
            f"max zero-shot spread={max(zero_spreads):.2e}",
        ),
        PrerequisiteCheck(
            "fixed extended program accounting",
            int(manifest["total_bytes"]) == 72
            and int(manifest["initializer_bytes"]) == 0
            and int(manifest["task_conditioning_bytes"]) == 0,
        ),
        PrerequisiteCheck(
            "multiple random controls executed",
            len(random_names) >= 8 and bool(np.isfinite(random_values).all()),
        ),
    ]
    convincingly_useful = (
        heldout_gain > threshold
        and mean_auc["learned_extended_rule"] < float(np.median(random_values))
        and mean_auc["learned_extended_rule"] < mean_auc["no_plasticity"]
    )
    status = (
        Status.NARROW
        if convincingly_useful and all(check.passed for check in checks)
        else Status.INCONCLUSIVE
    )
    writer.event(
        stage="p5.2-recovery-1",
        coefficients=np.asarray(learned.coefficients).tolist(),
        step_size=learned.step_size,
        pilot_effect_threshold=threshold,
        heldout_gain=heldout_gain,
        strongest_comparator=strongest_name,
    )
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=status,
        evidence_level="E1",
        primary_result={
            "metric": "fresh_holdout_gain_minus_pilot_effect_threshold",
            "value": heldout_gain - threshold,
            "unit": "fraction",
        },
        baselines_run=[
            "frozen four-term local rule",
            f"{random_count} scale-matched random eight-term rules",
            "no plasticity",
            "same-step direct backprop",
        ],
        controls=checks,
        interpretation=(
            "Recovery clears a pilot-derived effect threshold on fresh joint holdouts, but "
            "remains plasticity-only evidence."
            if status is Status.NARROW
            else (
                "Recovery does not establish a meaningful plasticity advantage after multiple "
                "random controls and a pilot-derived effect threshold."
            )
        ),
        non_claim=config.preregistration.non_claim,
        decision=(
            "Require independent theory review before considering a P5.3 preregistration."
            if status is Status.NARROW
            else "Do not start P5.3; preserve this as recovery-cycle evidence."
        ),
    )
