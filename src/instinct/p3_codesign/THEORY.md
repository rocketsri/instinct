# P3 theory map

- Question/hypothesis: frozen sections 6.1 and 6.3.
- Estimand: the one-interval objective `g_mu^d(s) + gamma^d P_mu^d Q_Pi(s_d, A_Pi(s_0,k,mu))`; it is not recurring real-time value and never selects a fresh action from `s_d`.
- Assumptions: finite MDP in P3.0, stale/open-loop planner action, planner simulates the same reflex. Mismatch is a baseline.
- Controls: exact enumeration, hold/greedy/interior/destructive optima, shuffled planner values, cross-play, concurrent-control baselines.
- Verdict: P3.0 is E0 objective validation only. Learned/novelty claims remain blocked pending direct concurrent-planning baselines.
- P3.1 implementation: exact first-action counterfactual labels hold the pending
  plan fixed and compose the remaining time-conditioned reflex. A sampled audit
  shares transition draws across action branches and records task termination
  separately from prefix-horizon truncation.
- P3.1 scope: pilot seeds train a linear time-conditioned softmax reflex on
  delays disjoint from evaluation cells. Frozen fingerprints, shuffled planner
  values, match/mismatch cross-play, and equal planner work are recorded.
- P3.1 verdict: at most NARROW for this single exact tabular planner; P3.2 and
  cross-planner claims remain blocked even when the held-out-delay gain is positive.
- P3.1 recovery cycle 1: the estimand is unchanged. A fresh three-cell registry crosses unseen speed and delay values, delay-conditioned action-advantage regression is compared with an equal-label-query policy-gradient optimizer, optionality differs from hold, and mismatch must change pending planner actions. P1-style finite-planner-horizon curves are diagnostic proxies, not measured wall-clock frontiers.
- P3.1 recovery cycle 2: the estimand remains unchanged in continuous inertial control with observed CPU planning latency. A recovery-margin reflex supplies structural interior behavior and is simulated by the matched planner; fresh event-rate/work cells test whether that interior survives learning. The final STOP/F3 endpoint collapse archives learned P3 co-design after both permitted recoveries while preserving P3.0 and mismatch diagnostics.
- Source: proposal version 2.1; portfolio amendment 2.1 changes scheduling only.
