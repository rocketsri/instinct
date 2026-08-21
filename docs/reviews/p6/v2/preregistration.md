# P6.2 preregistration — model mismatch

## Frozen question

Does a prefix certificate built from a learned rollout model retain validity for the true environment under preregistered dynamics shifts? Model-relative and true-environment validity are different estimands and will be reported separately.

## Protocol

- Fit one nested Bernoulli prefix model on pilot seed 611 only.
- Lock computation exits, candidate prefixes, risk budget, confidence budget, shift cells, and all thresholds before evaluation seed 621 is used.
- In each evaluation repetition, use disjoint samples for paired shift calibration, model-rollout certification, and execution.
- The certificate may consume model failure counts. The robust certificate may additionally consume paired binary discordance counts. Exact model and environment risks are evaluation-only.
- Split robust confidence error equally between the model upper process and the fixed-time shift envelope; Bonferroni-correct prefixes and use declared-exit union or summable look spending as stated by the method.

The primary metric is the worst-cell true-environment false-certification rate of `shift_robust_anytime`. It must be at most 0.05 and have positive nonvacuity in every cell. Failure of true validity is a deployment STOP. Validity with vacuity or without material gain over the union comparator remains NARROW.

## Controls and bounded literature analogues

Controls are fixed-time pointwise, the current conservative declared-exit union, the current anytime max-certified process, always abstain, and always execute. `avcrc_style_cp_analogue` is only the finite Bernoulli/growing-calibration specialization of the anytime-valid idea. `csa_style_eprocess_analogue` is only a one-threshold Bernoulli e-process with Ville/Bonferroni selection. Neither name asserts reproduction of a paper's general conformal algorithm.

The held-out result cannot revise this experiment version. A later change to the risk target, shift estimand, or validity claim requires a versioned amendment and fresh evaluation units.
