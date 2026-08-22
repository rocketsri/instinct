"""Colab-compatible entry point for the frozen P2.2 recovery-2 benchmark.

Pilot (official train split only; can never qualify P7):

    python benchmarks/p2_recovery2_colab.py --mode pilot \
      --artifact-dir /content/instinct-p2-artifacts \
      --output /content/p2-r2-pilot.json

Full reserved evaluation (exact confirmation is intentionally required):

    python benchmarks/p2_recovery2_colab.py --mode full \
      --confirm OPEN_FROZEN_P2_R2_TEST_PARTITION \
      --artifact-dir /content/instinct-p2-artifacts \
      --output /content/p2-r2-full-seed241.json

F2 estimator repair (pilot-only; see docs/reviews/p2/v7/f2_repair_preregistration.md):

    python benchmarks/p2_recovery2_colab.py --mode pilot --scoring-mode f2_repair \
      --artifact-dir /content/instinct-p2-artifacts \
      --output /content/p2-r2-f2repair.json

The script downloads only the two official, checksum-pinned inputs, then writes
one strict JSON artifact.  It does not invoke the repository CLI or overwrite
any prior recovery result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# A Colab invocation commonly runs this file from a repository checkout without
# installing the editable package first.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from instinct.p2_harmful_write.recovery2_benchmark import (  # noqa: E402
    F2REPAIR_ARTIFACT_SCHEMA,
    F2_REPAIR_SOURCE_OFFSET,
    FULL_EVAL_CONFIRMATION,
    BenchmarkConfig,
    F2RepairConfig,
    ensure_official_artifacts,
    run_recovery2_benchmark,
    run_recovery2_f2_repair,
    write_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--confirm", default="", help="required exact token for full mode")
    parser.add_argument(
        "--scoring-mode",
        choices=("frozen_reference", "f2_repair"),
        default="frozen_reference",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("/content/instinct-p2-artifacts"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("/content/instinct-p2-data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=241)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--source-epochs", type=int, default=None)
    parser.add_argument("--pilot-source-images", type=int, default=2048)
    parser.add_argument("--pilot-stream-images", type=int, default=96)
    parser.add_argument("--adaptation-lr", type=float, default=1e-3)
    parser.add_argument("--cost-loss-per-second", type=float, default=1e-3)
    # f2_repair-only knobs; ignored under --scoring-mode frozen_reference.
    parser.add_argument("--repair-offset", type=int, default=F2_REPAIR_SOURCE_OFFSET)
    parser.add_argument("--audit-pool-size", type=int, default=4096)
    parser.add_argument("--downstream-label-pool-size", type=int, default=1536)
    parser.add_argument("--rescoring-block-size", type=int, default=2)
    parser.add_argument("--audit-probe-budget", type=int, default=3)
    parser.add_argument("--audit-ages", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--spearman-gate", type=float, default=0.25)
    parser.add_argument("--minimum-accuracy-gain", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "full" and args.confirm != FULL_EVAL_CONFIRMATION:
        raise SystemExit(
            "Refusing to open reserved CIFAR-10 test units. Pass "
            f"--confirm {FULL_EVAL_CONFIRMATION} exactly."
        )
    if args.scoring_mode == "f2_repair" and args.mode != "pilot":
        raise SystemExit("the F2 repair only ever runs in pilot mode")
    defaults = {
        "pilot": {"n_steps": 12, "source_epochs": 1},
        "full": {"n_steps": 50, "source_epochs": 5},
    }[args.mode]
    config = BenchmarkConfig(
        mode=args.mode,
        seed=args.seed,
        n_steps=args.n_steps or defaults["n_steps"],
        source_epochs=args.source_epochs or defaults["source_epochs"],
        pilot_source_images=args.pilot_source_images,
        pilot_stream_images=args.pilot_stream_images,
        adaptation_lr=args.adaptation_lr,
        cost_loss_per_second=args.cost_loss_per_second,
        full_eval_confirmation=args.confirm,
    )
    cifar_archive, weights, checks = ensure_official_artifacts(args.artifact_dir)

    if args.scoring_mode == "f2_repair":
        repair_config = F2RepairConfig(
            base=config,
            repair_offset=args.repair_offset,
            audit_pool_size=args.audit_pool_size,
            downstream_label_pool_size=args.downstream_label_pool_size,
            rescoring_block_size=args.rescoring_block_size,
            audit_probe_budget=args.audit_probe_budget,
            audit_ages=tuple(args.audit_ages),
            spearman_gate=args.spearman_gate,
            minimum_accuracy_gain=args.minimum_accuracy_gain,
        )
        result = run_recovery2_f2_repair(
            repair_config,
            cifar_archive=cifar_archive,
            resnet18_weights=weights,
            data_root=args.data_root,
            artifact_checks=checks,
        )
        output = write_artifact(result, args.output, expected_schema=F2REPAIR_ARTIFACT_SCHEMA)
        print(
            json.dumps(
                {
                    "artifact": str(output),
                    "schema": result.artifact()["schema"],
                    "mode": config.mode,
                    "scoring_mode": args.scoring_mode,
                    "archival_condition": result.archival_condition,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    result = run_recovery2_benchmark(
        config,
        cifar_archive=cifar_archive,
        resnet18_weights=weights,
        data_root=args.data_root,
        artifact_checks=checks,
    )
    output = write_artifact(result, args.output)
    print(
        json.dumps(
            {
                "artifact": str(output),
                "schema": result.artifact()["schema"],
                "mode": config.mode,
                "scoring_mode": args.scoring_mode,
                "qualifies_p7": result.gates["qualifies_p7"],
                "gate_checks": result.gates["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
