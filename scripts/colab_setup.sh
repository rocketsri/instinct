#!/usr/bin/env bash
# One-time Colab CLI setup, plus a T4 entitlement check.
#
# The entitlement check matters more than it looks. Accelerator availability is
# tier-gated and most accounts only get CPU, so whether a T4 actually allocates
# is a fact about the account, not about this code. Stage 2's LM arm assumes a
# T4; if none allocates, that arm needs rescoping. Better to learn that now than
# after the code is written.
set -uo pipefail

echo "== 1. install =="
if command -v colab >/dev/null 2>&1; then
    echo "colab CLI already present: $(command -v colab)"
else
    uv tool install google-colab-cli
fi

echo
echo "== 2. authenticate =="
# The CLI prints a consent URL and blocks reading a code on stdin, and the PKCE
# verifier lives in that process -- so the URL and the code cannot be handled by
# two separate invocations. Interactively, just run `colab sessions` and paste
# the code. Headless (agent-driven), use the FIFO pair:
#
#   bash scripts/colab_auth_start.sh          # prints the consent URL
#   bash scripts/colab_auth_finish.sh <CODE>  # feeds the code back
#
# `colab auth` is a different thing entirely: it injects GCP credentials into
# the *VM's* kernel for in-notebook BigQuery/GCS calls. It never fixes a CLI
# 401/403 -- those are scope problems.
if colab sessions </dev/null 2>&1 | grep -qi "accounts.google.com"; then
    echo "NOT AUTHENTICATED. Run: bash scripts/colab_auth_start.sh"
    exit 1
fi
echo "authenticated."

echo
echo "== 3. T4 entitlement check =="
# `colab run` rather than `colab new`: it provisions, runs, and tears down in one
# command, and self-cleans even when the script raises. An unstopped `colab new`
# session burns compute units until the 24h keep-alive cap -- which is exactly
# the failure mode an unattended agent would cause.
#
# Gotcha: an unrecognized --gpu value silently falls back to A100, which then
# fails at allocation. A 400 here means no entitlement for that accelerator.
cat > /tmp/instinct_gpu_probe.py <<'PY'
import subprocess, sys
try:
    import torch
except ImportError:
    print("torch missing on the VM"); sys.exit(1)
print("torch          :", torch.__version__)
print("cuda available :", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("no CUDA device -- accelerator did not attach")
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
print("device         :", name)
print("capability     :", f"{major}.{minor}")
print("memory (GiB)   :", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
# T4 is sm_75: fp16 tensor cores, but no bf16 and no TF32, and no FlashAttention-2.
print("bf16 supported :", torch.cuda.is_bf16_supported())
print("-> expect False for bf16 on a T4; the LM arm uses fp16 + mem-efficient SDPA")
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader"], check=False)
PY

colab run --gpu T4 /tmp/instinct_gpu_probe.py
echo
echo "If the above printed a Tesla T4 at capability 7.5, the GPU path is good."
