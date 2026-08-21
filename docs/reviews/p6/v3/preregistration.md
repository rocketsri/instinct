# P6.3 scale/frontier preregistration

## Frozen stage

P6.2 remains preserved. P6.3 uses pilot seeds 631/632 and one untouched evaluation seed, 641. The learned rollout model, four shift families, 16-prefix family, trajectory scales, 8,192-exit schedule, confidence/risk budgets, and verdict thresholds are frozen before seed 641 is used.

The primary method is `shift_robust_csa_specialization`: a Bernoulli e-process for each declared adjusted-risk threshold, Bonferroni across the finite prefix grid, and the longest-prefix/max-certified rule. A fixed predictable alternative 0.02 below each null threshold was selected on pilot units and then locked. Its confidence budget is split from the independent paired-discordance envelope.

The primary largest-scale gate requires all of:

1. one-sided 95% upper confidence bound on true false certification at most 0.05 in every shift family;
2. positive nonvacuity in every family;
3. average selected-prefix gain of at least 0.3 actions over `shift_robust_union`;
4. at least 10x certificate-only latency speedup over that union.

Shift-calibration latency is common to robust methods and is reported separately. It cannot be omitted from end-to-end scaling claims. Exact learned-model and true-environment risks remain post-selection evaluation quantities.

## Pilot record

Pilot seed 632 was used to map the regime, never for the final estimate. With five and 64 exits the exact finite union was essentially tied with the e-process. Dense exit schedules exposed the expected horizon dependence of the union and horizon independence of Ville certification. On the final 8,192-exit pilot protocol, the observed mean prefix gain was 0.398 actions with zero primary false certifications. This supported locking the smaller 0.3-action threshold; no evaluation outcomes informed it.

## Literature scope

The CSA comparator is a faithful specialization only under the present IID Bernoulli filtration, predictable fixed bet, finite Bonferroni grid, and monotone prefix risks. The AVCRC comparator implements the paper's Theorem 4.1 stitched bounded-loss correction on the learned-model loss family. It is composed with a separate paired-discordance envelope because the benchmark does not supply the density ratios required by AVCRC's distribution-shift theorem. Neither implementation is presented as a general reproduction.
