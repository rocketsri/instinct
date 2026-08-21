"""Launch the guarded train-split-only P2 pilot in an initialized Colab session."""

import runpy
import sys

sys.argv = [
    "/content/benchmarks/p2_recovery2_colab.py",
    "--mode",
    "pilot",
    "--artifact-dir",
    "/content/instinct-p2-artifacts",
    "--data-root",
    "/content/instinct-p2-data",
    "--output",
    "/content/p2-r2-pilot.json",
]
runpy.run_path(sys.argv[0], run_name="__main__")
