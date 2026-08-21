from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from instinct.core.configschema import ProposalRunConfig
from instinct.core.proposal import common_checks, resume_by_rerun
from instinct.core.results import ResultsWriter
from instinct.core.verdict import PrerequisiteCheck, Status, VerdictReport
from instinct.p5_development.generator import CoordinateGenerator


def validate(config: ProposalRunConfig) -> list[PrerequisiteCheck]:
    if config.params.get("stage") == "p5.2-r2":
        from instinct.p5_development.recovery2 import validate_recovery2

        return validate_recovery2(config)
    if config.params.get("stage") == "p5.2-r1":
        from instinct.p5_development.recovery1 import validate_recovery

        return validate_recovery(config)
    if config.params.get("stage") == "p5.2":
        from instinct.p5_development.stage2 import validate_p5_2

        return validate_p5_2(config)
    return common_checks(config, "p5_development")


def run(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    if config.params.get("stage") == "p5.2-r2":
        from instinct.p5_development.recovery2 import run_recovery2

        return run_recovery2(config, writer)
    if config.params.get("stage") == "p5.2-r1":
        from instinct.p5_development.recovery1 import run_recovery

        return run_recovery(config, writer)
    if config.params.get("stage") == "p5.2":
        from instinct.p5_development.stage2 import run_p5_2

        return run_p5_2(config, writer)
    checks = validate(config)
    g = CoordinateGenerator()
    pre = (np.arange(8) + 0.5) / 8
    post = (np.arange(6) + 0.5) / 6
    rng = np.random.default_rng(config.seed)
    pp, qp = rng.permutation(len(pre)), rng.permutation(len(post))
    W = g.weights(pre, post)
    Wp = g.weights(pre[pp], post[qp])
    weight_error = float(np.max(np.abs(Wp - W[np.ix_(qp, pp)])))
    x, y, e = rng.normal(size=8), rng.normal(size=6), rng.normal(size=6)
    D = g.local_delta(x, y, e, pre, post, step=3)
    Dp = g.local_delta(x[pp], y[qp], e[qp], pre[pp], post[qp], step=3)
    update_error = float(np.max(np.abs(Dp - D[np.ix_(qp, pp)])))
    function = g.function_output(x, pre, post)
    permuted_function = g.function_output(x[pp], pre[pp], post[qp])
    function_error = float(np.max(np.abs(permuted_function - function[qp])))

    # Remote-neuron taint: preserve the selected synapse's declared local
    # inputs while changing every other activity/error. Its update must not move.
    remote_x, remote_y, remote_e = rng.normal(size=8), rng.normal(size=6), rng.normal(size=6)
    target_pre, target_post = 3, 2
    remote_x[target_pre] = x[target_pre]
    remote_y[target_post] = y[target_post]
    remote_e[target_post] = e[target_post]
    remote_D = g.local_delta(remote_x, remote_y, remote_e, pre, post, step=3)
    locality_error = abs(float(remote_D[target_post, target_pre] - D[target_post, target_pre]))

    # Coordinates are semantic inputs, not hidden neuron IDs. Permuting activity
    # without its coordinates is therefore a deliberate negative control.
    coordinate_sensitivity = float(np.max(np.abs(g.function_output(x[pp], pre, post) - function)))

    duplicated = g.function_output(np.repeat(x, 2), np.repeat(pre, 2), post)
    duplication_error = float(np.max(np.abs(duplicated - function)))
    probe = np.array([0.2, 0.5, 0.8])
    reference = g.response(256, probe)
    metrics = []
    transfer_errors = []
    for width in (8, 16, 32, 64):
        response = g.response(width, probe)
        error = float(np.max(np.abs(response - reference)))
        transfer_errors.append(error)
        metrics.append(
            {
                "width": width,
                "response_error_vs_256": error,
                "code_bytes": g.code_bytes,
                "instantiated_parameters": width * len(probe),
                "generator_parameters": len(g.coefficients),
                "record_type": "width_transfer",
            }
        )
    manifest = g.serialization_manifest()
    metrics.append(
        {
            "width": 0,
            "response_error_vs_256": np.nan,
            "code_bytes": manifest["total_bytes"],
            "instantiated_parameters": 0,
            "generator_parameters": manifest["coefficient_count"],
            "record_type": "serialization_audit",
        }
    )
    nontrivial = np.linalg.matrix_rank(W) >= 2 and float(W.std()) > 1e-3
    checks += [
        PrerequisiteCheck(
            "weight permutation equivariance", weight_error < 1e-12, f"error={weight_error:.2e}"
        ),
        PrerequisiteCheck(
            "local-update permutation equivariance",
            update_error < 1e-12,
            f"error={update_error:.2e}",
        ),
        PrerequisiteCheck(
            "generated-function permutation equivariance",
            function_error < 1e-12,
            f"error={function_error:.2e}",
        ),
        PrerequisiteCheck(
            "strict-locality taint test",
            locality_error < 1e-12,
            f"error={locality_error:.2e}",
        ),
        PrerequisiteCheck(
            "duplicate-width normalization",
            duplication_error < 1e-12,
            f"error={duplication_error:.2e}",
        ),
        PrerequisiteCheck(
            "coordinate-order negative control",
            coordinate_sensitivity > 1e-3,
            f"effect={coordinate_sensitivity:.2e}",
        ),
        PrerequisiteCheck("nontrivial generator", nontrivial),
        PrerequisiteCheck(
            "fixed serialized length", all(m["code_bytes"] == g.code_bytes for m in metrics)
        ),
        PrerequisiteCheck(
            "complete serialization manifest",
            int(manifest["total_bytes"]) == g.code_bytes and manifest["dtype"] == "float64",
        ),
        PrerequisiteCheck("width transfer improves", transfer_errors[-1] < transfer_errors[0]),
    ]
    writer.write_metrics(metrics)
    writer.event(
        stage="p5.0-p5.1",
        permutation_error=max(weight_error, update_error, function_error),
        locality_error=locality_error,
        duplication_error=duplication_error,
        code_bytes=g.code_bytes,
    )
    passed = all(c.passed for c in checks)
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=Status.GO if passed else Status.INCONCLUSIVE,
        evidence_level="E0",
        primary_result={
            "metric": "max_permutation_error",
            "value": max(weight_error, update_error, function_error),
            "unit": "absolute",
        },
        baselines_run=[
            "permuted coordinates",
            "coordinate-order negative control",
            "strict-locality taint",
            "duplicate-width normalization",
            "fixed-byte audit",
            "width quadrature transfer",
        ],
        baselines_skipped={
            "learned-local-rule baselines": (
                "required before P5.2; this run is generator instrumentation"
            )
        },
        controls=checks,
        interpretation=(
            "The fixed program is equivariant and width-scaled on a constructed "
            "function; it is not a learned developmental advantage."
        ),
        non_claim=config.preregistration.non_claim,
        decision=(
            "Add frozen Cartesian tasks and direct plasticity/meta-learning baselines before P5.2."
        )
        if passed
        else "Repair symmetry/scaling before training.",
    )


def resume(
    config: ProposalRunConfig, writer: ResultsWriter, checkpoint_state: Mapping[str, Any]
) -> VerdictReport:
    return resume_by_rerun(config, writer, checkpoint_state, run)
