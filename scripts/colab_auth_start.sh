#!/usr/bin/env bash
# Start the Colab CLI OAuth flow in a way a headless agent can complete.
#
# The CLI prints a consent URL and then blocks reading an authorization code on
# stdin. The PKCE verifier lives in that process, so the code must go back to
# *the same* invocation -- you cannot print the URL in one command and paste the
# code into another. This script keeps the process alive behind a FIFO so the
# URL and the code can be handled in separate steps.
#
#   bash scripts/colab_auth_start.sh          # prints the consent URL
#   bash scripts/colab_auth_finish.sh <CODE>  # completes it
set -uo pipefail

RUNDIR="${COLAB_AUTH_RUNDIR:-/tmp/colab-auth}"
mkdir -p "$RUNDIR"
FIFO="$RUNDIR/stdin.fifo"
LOG="$RUNDIR/auth.log"

pkill -f "colab sessions" 2>/dev/null || true
rm -f "$FIFO" "$LOG" "$RUNDIR/holder.pid"
mkfifo "$FIFO"

# A writer must hold the FIFO open, otherwise the reader's open() blocks and the
# CLI never gets far enough to print anything.
sleep 86400 > "$FIFO" &
echo $! > "$RUNDIR/holder.pid"

setsid colab sessions < "$FIFO" > "$LOG" 2>&1 &
echo $! > "$RUNDIR/colab.pid"

for _ in $(seq 1 40); do
    if grep -q "accounts.google.com" "$LOG" 2>/dev/null; then break; fi
    sleep 1
done

tr -d '\r' < "$LOG"
