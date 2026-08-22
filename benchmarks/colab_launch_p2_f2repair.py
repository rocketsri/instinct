"""Launch the preregistered F2 estimator repair pilot on Colab.

See docs/reviews/p2/v7/f2_repair_preregistration.md for the locked field
values reproduced as CLI flags below.
"""

import runpy
import sys

sys.argv = [
    "/content/benchmarks/p2_recovery2_colab.py",
    "--mode",
    "pilot",
    "--scoring-mode",
    "f2_repair",
    "--artifact-dir",
    "/content/instinct-p2-artifacts",
    "--data-root",
    "/content/instinct-p2-data",
    "--output",
    "/content/p2-r2-f2repair.json",
    "--seed",
    "251",
    "--n-steps",
    "32",
    "--source-epochs",
    "2",
    "--pilot-source-images",
    "4096",
    "--pilot-stream-images",
    "256",
    "--repair-offset",
    "4096",
    "--audit-pool-size",
    "4096",
    "--downstream-label-pool-size",
    "1536",
    "--rescoring-block-size",
    "2",
    "--audit-probe-budget",
    "3",
    "--audit-ages",
    "4",
    "8",
    "12",
    "--spearman-gate",
    "0.25",
    "--minimum-accuracy-gain",
    "0.005",
]
runpy.run_path(sys.argv[0], run_name="__main__")
