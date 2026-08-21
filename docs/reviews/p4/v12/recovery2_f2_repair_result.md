# P4.2 recovery-2 F2 repair result v12

Artifact: `/tmp/instinct-p4-stage2-r2-f2/p4_slack/p4_slack-20260817T064132Z-5e77cfdd-052c83`

The exact evaluator delivered the frozen resource target: two long-chunk calls
plus two verifier calls on every trajectory, for exactly `4.000000` total.
On the fresh layout, however, the recovery gate reduced goal probability by
`0.069322` relative to task-only. Slack regularization gained only `0.001105`,
below the frozen `0.01` threshold; the six-call shorter-chunk control remained
stronger than either four-call chunk policy.

Verdict: **STOP, E1**. Bounded P4.2 is archived after two scientific recovery
cycles and the preserved F2 repair. P4.0's exact recovery environment and
P4.1's in-domain estimator GO remain valid instruments, but no learned-policy,
training, robotics, scale, or safety claim follows. The v11 lower-quantile
objective blocker remains mandatory if a future theory version revisits slack
training.
