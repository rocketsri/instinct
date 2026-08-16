#!/usr/bin/env bash
# Complete the flow started by colab_auth_start.sh by feeding it the code.
#   bash scripts/colab_auth_finish.sh 4/0AVGz...
set -uo pipefail

RUNDIR="${COLAB_AUTH_RUNDIR:-/tmp/colab-auth}"
FIFO="$RUNDIR/stdin.fifo"
LOG="$RUNDIR/auth.log"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <authorization-code>" >&2
    exit 2
fi
if [[ ! -p "$FIFO" ]]; then
    echo "no pending auth flow; run scripts/colab_auth_start.sh first" >&2
    exit 1
fi

printf '%s\n' "$1" > "$FIFO"

for _ in $(seq 1 60); do
    if ! kill -0 "$(cat "$RUNDIR/colab.pid" 2>/dev/null)" 2>/dev/null; then break; fi
    sleep 1
done

tr -d '\r' < "$LOG" | tail -20
kill "$(cat "$RUNDIR/holder.pid" 2>/dev/null)" 2>/dev/null || true
