"""Launch the preregistered extended train-only P2 pilot on Colab."""

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
    "/content/p2-r2-pilot2.json",
    "--n-steps",
    "32",
    "--source-epochs",
    "2",
    "--pilot-source-images",
    "4096",
    "--pilot-stream-images",
    "256",
]
runpy.run_path(sys.argv[0], run_name="__main__")
