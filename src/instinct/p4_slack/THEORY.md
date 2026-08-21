# P4 theory map

- Question/hypothesis: frozen section 7.1.
- Estimand: lower-tail quantile of exact `V_replan - V_fail`; loss is `L_task + beta sum_j [m-R_j]_+`.
- Assumptions: frozen disturbance distribution/timing, common reward origin and units, exact replanning policy in P4.0, terminal prefixes masked.
- Controls: inactivity, reckless progress, strict interior frontier, exact reachability/viability DP, recovery switch, shorter chunks and periodic verification at matched calls/latency.
- Verdict: section 7.5. P4.0 is an exact-label mechanism gate, not a safety certificate.
- P4.1 estimator repair v4: whole length/slip/horizon conditions are split before label generation; the failed v3 held-out condition remains excluded. Known terminal values are structural constraints, and nested condition-balanced split calibration prevents longer corridors from dominating the residual distribution. Passing is an estimator gate only and does not change the P4 policy estimand.
- P4.0 inertial completion: the exact finite MDP now includes 2-D position, bounded velocity, obstacles, actuator slip, absorbing collision failure, and post-prefix velocity impulses. Legal states include both recoverable and already-unrecoverable motion. This closes the environment blocker but deliberately does not reuse the corridor-only v4 estimator out of domain.
- P4.1 inertial extension: complete obstacle layouts, not individual states, are assigned to train, calibration, and test. The surrogate uses state/dynamics features with exact goal/failure constraints and condition-balanced calibration. It is frozen before generating the held-out-layout labels.
- P4.2 verifier branch (`p4-1.1`): the original slack-training estimand is unchanged. After two bounded training attempts failed its effect threshold, recovery 2 tests recovery estimates only as an inference-time gate. The F2 repair matches four calls pathwise on fresh units; even a pass is NARROW and cannot establish slack training. The v11 audit's expectation-versus-quantile blocker must be repaired before any later training claim.
- Clarification: section 7 controls P4.0 where section 11's shorthand differs. No estimand amendment.
- Source: proposal version 2.1; portfolio amendment 2.1 changes scheduling only.
