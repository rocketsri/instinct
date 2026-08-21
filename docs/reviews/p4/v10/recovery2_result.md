# P4.2 recovery-2 result v10

Artifact: `/tmp/instinct-p4-stage2-r2/p4_slack/p4_slack-20260817T063419Z-ae6e8d7f-d461ff`

The frozen adaptive verifier improved held-out goal probability by `0.050642`,
while same-call slack training improved it by only `0.000221`. The verifier
used `3.698005` expected calls, outside the preregistered `4.00 +/- 0.25`
window. The recorded verdict is therefore **STOP, E1**. This unit is preserved
and its threshold is not relaxed.

The failed condition is cost matching, so amendment `p4-1.1` permits a fresh F2
instrumentation repair. The result is not evidence for slack training and does
not resolve the v11 expectation-versus-quantile blocker.
