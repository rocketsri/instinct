"""Final P5.2 recovery: coordinate/type feedback basis with time modulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from instinct.core.configschema import ProposalRunConfig
from instinct.core.proposal import common_checks
from instinct.core.results import ResultsWriter
from instinct.core.verdict import FailureCode, PrerequisiteCheck, Status, VerdictReport
from instinct.p5_development.feedback_basis import FeedbackBasisProgram
from instinct.p5_development.plasticity import (
    LocalPlasticityProgram,
    PlasticityProgram,
    Topology,
    adaptation_curve,
    conventional_initialization,
    make_task,
)

RECOVERY1_COEFFICIENTS = np.array(
    [
        0.8427267247198754,
        0.6507922106183255,
        -0.3738448532281268,
        0.4023928491670297,
        0.3357648323091478,
        0.733357383904119,
        -0.000504736523074506,
        -0.3200694537037858,
    ]
)


def _ints(raw: object, name: str) -> tuple[int, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise ValueError(f"{name} must be a sequence")
    values = tuple(int(value) for value in raw)
    if not values or min(values) < 1:
        raise ValueError(f"{name} must contain positive integers")
    return values


def validate_recovery2(config: ProposalRunConfig) -> list[PrerequisiteCheck]:
    checks = common_checks(config, "p5_development")
    sealed_eval = {591, 592, 691, 692, 693}
    checks += [
        PrerequisiteCheck(
            "all prior evaluation units sealed",
            not sealed_eval & set(config.seeds),
            f"sealed={sorted(sealed_eval)}",
        ),
        PrerequisiteCheck(
            "selection and threshold pilot units separated",
            len(config.pilot_seeds) >= 3,
            "last pilot seed is threshold-only",
        ),
        PrerequisiteCheck("fresh final evaluation units configured", bool(config.eval_seeds)),
    ]
    try:
        pilot_widths = _ints(config.params.get("pilot_widths", [6, 8]), "pilot_widths")
        eval_widths = _ints(config.params.get("eval_widths", [10, 12]), "eval_widths")
        pilot_depths = _ints(config.params.get("pilot_depths", [1, 2]), "pilot_depths")
        eval_depths = _ints(config.params.get("eval_depths", [3, 4]), "eval_depths")
        checks += [
            PrerequisiteCheck(
                "final recovery width holdout disjoint",
                not set(pilot_widths) & set(eval_widths),
            ),
            PrerequisiteCheck(
                "final recovery depth holdout disjoint",
                not set(pilot_depths) & set(eval_depths),
            ),
        ]
    except ValueError as exc:
        checks.append(PrerequisiteCheck("valid final recovery topology split", False, str(exc)))
    return checks


def _score(
    program: PlasticityProgram,
    *,
    seeds: Sequence[int],
    topologies: Sequence[Topology],
    steps: int,
) -> float:
    values = []
    for seed in seeds:
        for family_index, family in enumerate(("polynomial", "composition", "warped")):
            task = make_task(seed + 107 * family_index, family)
            for topology in topologies:
                initial = conventional_initialization(topology, seed=41)
                curve = adaptation_curve(initial, task, program=program, steps=steps)
                value = float(curve.mean() / max(curve[0], 1e-12))
                values.append(value if np.isfinite(value) else float("inf"))
    return float(np.mean(values))


def _search(
    *,
    selection_seeds: Sequence[int],
    topologies: Sequence[Topology],
    steps: int,
) -> tuple[FeedbackBasisProgram, float, int]:
    candidates = [
        FeedbackBasisProgram(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)),
        FeedbackBasisProgram(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)),
        FeedbackBasisProgram(np.array([0.2, 0.8, 0, 0, 0.2, 0, 0.2, 0, 0, 0, 0, 0])),
        FeedbackBasisProgram(np.zeros(12)),
    ]
    for seed in selection_seeds:
        rng = np.random.default_rng(seed)
        candidates.extend(
            FeedbackBasisProgram(coefficients)
            for coefficients in rng.normal(0.0, 0.4, size=(8, 12))
        )
    scores = [
        _score(candidate, seeds=selection_seeds, topologies=topologies, steps=steps)
        for candidate in candidates
    ]
    best = candidates[int(np.argmin(scores))]
    rng = np.random.default_rng(sum(selection_seeds) + 29)
    refinements = [
        FeedbackBasisProgram(
            np.asarray(best.coefficients) + rng.normal(0.0, 0.07, size=12),
            step_size=step_size,
        )
        for step_size in (0.025, 0.04, 0.06)
        for _ in range(4)
    ]
    candidates.extend(refinements)
    scores.extend(
        _score(candidate, seeds=selection_seeds, topologies=topologies, steps=steps)
        for candidate in refinements
    )
    winner = int(np.argmin(scores))
    return candidates[winner], float(scores[winner]), len(candidates)


def _random_programs(seed: int, count: int) -> tuple[FeedbackBasisProgram, ...]:
    rng = np.random.default_rng(seed)
    return tuple(FeedbackBasisProgram(rng.normal(0.0, 0.4, size=12)) for _ in range(count))


def _comparators(
    random_programs: Sequence[FeedbackBasisProgram],
) -> dict[str, PlasticityProgram | None]:
    frozen = FeedbackBasisProgram(
        np.array([0.2, 0.8, 0, 0, 0.2, 0, 0.2, 0, 0, 0, 0, 0], dtype=float)
    )
    recovery1 = LocalPlasticityProgram(RECOVERY1_COEFFICIENTS, step_size=0.06)
    result: dict[str, PlasticityProgram | None] = {
        "frozen_feedback_basis": frozen,
        "recovery1_frozen_rule": recovery1,
        "no_plasticity": None,
    }
    result.update(
        {f"random_feedback_{index:02d}": program for index, program in enumerate(random_programs)}
    )
    return result


def _pilot_threshold(
    learned: FeedbackBasisProgram,
    *,
    calibration_seed: int,
    topologies: Sequence[Topology],
    comparators: Mapping[str, PlasticityProgram | None],
    steps: int,
) -> tuple[float, np.ndarray]:
    gains = []
    for family_index, family in enumerate(("polynomial", "composition", "warped")):
        task = make_task(calibration_seed + 107 * family_index, family)
        for topology in topologies:
            initial = conventional_initialization(topology, seed=41)
            curve = adaptation_curve(initial, task, program=learned, steps=steps)
            learned_auc = float(curve.mean() / max(curve[0], 1e-12))
            comparator_aucs = []
            for program in comparators.values():
                other = adaptation_curve(initial, task, program=program, steps=steps)
                comparator_aucs.append(float(other.mean() / max(other[0], 1e-12)))
            strongest = min(comparator_aucs)
            gains.append((strongest - learned_auc) / max(strongest, 1e-12))
    values = np.asarray(gains, dtype=np.float64)
    threshold = max(
        float(np.nextafter(0.0, 1.0)),
        1.96 * float(values.std(ddof=1) / np.sqrt(values.size)),
    )
    return threshold, values


def _audit(program: FeedbackBasisProgram, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    batch, n_pre, n_post = 11, 5, 4
    pre = rng.normal(size=(batch, n_pre))
    post = rng.normal(size=(batch, n_post))
    error = rng.normal(size=batch)
    weights = rng.normal(size=(n_post, n_pre))
    coordinates = (np.arange(n_post) + 0.5) / n_post
    reference = program.local_delta_layer(
        pre_activity=pre,
        post_activity=post,
        broadcast_error=error,
        weights=weights,
        post_coordinates=coordinates,
        layer_coordinate=0.35,
        is_output=False,
        step=3,
    )
    pp, qp = rng.permutation(n_pre), rng.permutation(n_post)
    permuted = program.local_delta_layer(
        pre_activity=pre[:, pp],
        post_activity=post[:, qp],
        broadcast_error=error,
        weights=weights[np.ix_(qp, pp)],
        post_coordinates=coordinates[qp],
        layer_coordinate=0.35,
        is_output=False,
        step=3,
    )
    permutation_error = float(np.max(np.abs(permuted - reference[np.ix_(qp, pp)])))
    remote_pre = rng.normal(size=pre.shape)
    remote_post = rng.normal(size=post.shape)
    remote_pre[:, 2] = pre[:, 2]
    remote_post[:, 1] = post[:, 1]
    remote = program.local_delta_layer(
        pre_activity=remote_pre,
        post_activity=remote_post,
        broadcast_error=error,
        weights=weights,
        post_coordinates=coordinates,
        layer_coordinate=0.35,
        is_output=False,
        step=3,
    )
    locality_error = abs(float(remote[1, 2] - reference[1, 2]))
    time_effect = float(
        np.max(
            np.abs(
                program.feedback(coordinates, layer_coordinate=0.35, is_output=False, step=0)
                - program.feedback(
                    coordinates, layer_coordinate=0.35, is_output=False, step=12
                )
            )
        )
    )
    return permutation_error, locality_error, time_effect


def run_recovery2(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    checks = validate_recovery2(config)
    if not all(check.passed for check in checks):
        return VerdictReport(
            question=config.preregistration.primary_hypothesis,
            status=Status.INCONCLUSIVE,
            evidence_level="E0",
            primary_result={"metric": "recovery_config_valid", "value": 0.0, "unit": "bool"},
            controls=checks,
            interpretation="Final recovery did not open because sealing controls failed.",
            non_claim=config.preregistration.non_claim,
            decision="Repair only F0 configuration errors; never reuse sealed units.",
        )
    params: Mapping[str, Any] = config.params
    pilot_topologies = tuple(
        Topology(width, depth)
        for width in _ints(params.get("pilot_widths", [6, 8]), "pilot_widths")
        for depth in _ints(params.get("pilot_depths", [1, 2]), "pilot_depths")
    )
    eval_topologies = tuple(
        Topology(width, depth)
        for width in _ints(params.get("eval_widths", [10, 12]), "eval_widths")
        for depth in _ints(params.get("eval_depths", [3, 4]), "eval_depths")
    )
    pilot_steps = int(params.get("pilot_steps", 7))
    eval_steps = int(params.get("eval_steps", 18))
    random_count = int(params.get("random_rule_count", 8))
    learned, pilot_score, candidate_count = _search(
        selection_seeds=config.pilot_seeds[:-1],
        topologies=pilot_topologies,
        steps=pilot_steps,
    )
    random_programs = _random_programs(config.seed + 900, random_count)
    comparator_programs = _comparators(random_programs)
    threshold, calibration_gains = _pilot_threshold(
        learned,
        calibration_seed=config.pilot_seeds[-1],
        topologies=pilot_topologies,
        comparators=comparator_programs,
        steps=pilot_steps,
    )
    programs: dict[str, tuple[PlasticityProgram | None, bool]] = {
        "learned_feedback_basis": (learned, False),
        **{name: (program, False) for name, program in comparator_programs.items()},
        "direct_backprop": (None, True),
    }
    rows: list[dict[str, Any]] = []
    aucs: dict[str, list[float]] = {name: [] for name in programs}
    zero_spreads = []
    for seed in config.eval_seeds:
        task = make_task(seed, "chirp")
        for topology in eval_topologies:
            initial = conventional_initialization(topology, seed=41)
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
                        "stage": "p5.2-recovery-2",
                        "split": "final_fresh_joint_holdout",
                        "eval_seed": seed,
                        "task_family": task.family,
                        "width": topology.width,
                        "depth": topology.depth,
                        "control": name,
                        "zero_shot_loss": float(curve[0]),
                        "normalized_adaptation_auc": auc,
                        "final_loss": float(curve[-1]),
                        "plasticity_program_parameters": (
                            0 if manifest is None else int(manifest["coefficient_count"]) + 1
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
    comparator_names = list(comparator_programs)
    strongest_name = min(comparator_names, key=mean_auc.__getitem__)
    strongest_auc = mean_auc[strongest_name]
    learned_auc = mean_auc["learned_feedback_basis"]
    heldout_gain = (strongest_auc - learned_auc) / max(strongest_auc, 1e-12)
    random_values = np.asarray(
        [value for name, value in mean_auc.items() if name.startswith("random_feedback_")]
    )
    manifest = learned.serialization_manifest()
    permutation_error, locality_error, time_effect = _audit(learned, config.seed)
    rows.append(
        {
            "stage": "p5.2-recovery-2-preregistration",
            "split": "new_pilot_calibration_only",
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
        PrerequisiteCheck("new pilot-derived threshold is nonzero", threshold > 0.0),
        PrerequisiteCheck(
            "feedback-basis permutation equivariance",
            permutation_error < 1e-12,
            f"error={permutation_error:.2e}",
        ),
        PrerequisiteCheck(
            "feedback-basis strict locality",
            locality_error < 1e-12,
            f"error={locality_error:.2e}",
        ),
        PrerequisiteCheck(
            "time modulation is active",
            time_effect > 1e-6,
            f"effect={time_effect:.2e}",
        ),
        PrerequisiteCheck(
            "final recovery initialization separated",
            max(zero_spreads) < 1e-12,
            f"max zero-shot spread={max(zero_spreads):.2e}",
        ),
        PrerequisiteCheck(
            "fixed feedback-program accounting",
            int(manifest["total_bytes"]) == 104
            and int(manifest["initializer_bytes"]) == 0
            and int(manifest["persistent_synapse_state_bytes"]) == 0,
        ),
        PrerequisiteCheck(
            "multiple final random mechanisms executed",
            random_values.size >= 8 and bool(np.isfinite(random_values).all()),
        ),
    ]
    convincingly_useful = (
        heldout_gain > threshold
        and learned_auc < float(np.median(random_values))
        and learned_auc < mean_auc["no_plasticity"]
        and learned_auc < mean_auc["recovery1_frozen_rule"]
    )
    passed_controls = all(check.passed for check in checks)
    status = Status.NARROW if convincingly_useful and passed_controls else Status.STOP
    writer.event(
        stage="p5.2-recovery-2",
        coefficients=np.asarray(learned.coefficients).tolist(),
        step_size=learned.step_size,
        pilot_effect_threshold=threshold,
        heldout_gain=heldout_gain,
        strongest_comparator=strongest_name,
        direct_backprop_auc=mean_auc["direct_backprop"],
    )
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=status,
        evidence_level="E1",
        failure_code=None if status is Status.NARROW else FailureCode.F3_MECHANISM_ABSENT,
        primary_result={
            "metric": "final_fresh_gain_minus_pilot_effect_threshold",
            "value": heldout_gain - threshold,
            "unit": "fraction",
        },
        baselines_run=[
            "frozen coordinate/type feedback basis",
            "sealed recovery-1 rule as frozen frontier",
            f"{random_count} random feedback-basis programs",
            "no plasticity",
            "same-step direct backprop",
        ],
        controls=checks,
        interpretation=(
            "The distinct feedback-basis mechanism clears the final meaningful-effect gate, "
            "but remains plasticity-only evidence."
            if status is Status.NARROW
            else (
                "Two valid recovery mechanisms fail to establish a meaningful local-plasticity "
                "advantage on fresh joint holdouts."
            )
        ),
        non_claim=config.preregistration.non_claim,
        decision=(
            "Independent review is required before any P5.3 preregistration."
            if status is Status.NARROW
            else "Archive P5.2 after its final allowed recovery; do not start P5.3."
        ),
    )
