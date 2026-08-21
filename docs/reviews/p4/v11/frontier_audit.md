# P4 frontier audit v11

Date: 2026-08-17  
Scope: frozen P4 theory, exact inertial pilot, inertial surrogate, P4.2 stage-two
training, and recovery-2 preregistration/code. This is an evidence audit only;
existing results are preserved and no experiment was run.

## Executive verdict

The exact inertial environment is a useful E0 mechanism test: it contains
inertia, obstacles, actuator slip, disturbances, delayed replanning, exact
failure/goal states, and exact recovery values. The whole-layout surrogate split
and structural terminal constraints are also materially better than random-state
validation.

P4.2 cannot yet support the frozen **prefix recovery-slack training** claim. The
implemented stage-two objective is not the specified lower-disturbance-quantile
objective, and recovery2 changes the mechanism to execution-time switching.
Recovery2 is scientifically distinct enough to be a second recovery cycle, but a
positive result would only justify NARROW to an adaptive verifier and archival of
slack-policy training, exactly as v10 states. No versioned recovery2 result was
found; the repository still describes P4.2 as pending, so no empirical recovery2
verdict is auditable.

## Findings

### BLOCKER — B1: stage two does not implement the frozen estimand

The frozen loss applies a hinge to a lower disturbance quantile,

\[
[m-\operatorname{Quantile}_{q,\delta}
  (V_{replan}(s_j^\delta)-V_{fail})]_+.
\]

`train_chunk_policy` instead forms one mean impulse transition and propagates
`transition[action] @ (state_penalty + slack_cost)`. This is an expectation of
state-wise hinge penalties after an averaged disturbance kernel. In general,
\(E[(m-X)_+]\neq[m-Q_q(X)]_+\); the tail level \(q\) is absent. P4.0's exact
quantile calculation does not repair this P4.2 objective drift. Before any
slack-training claim, implement the frozen quantile over disturbance-specific
prefix states or add a reviewed theory amendment and a new experiment version.

### BLOCKER — B2: there is no recovery2 evidence artifact

The v10 document is a preregistration and `stage2.py` contains an executable
protocol, but no versioned metrics, verdict, or reproduction record for
`p4.2-recovery2` is present. Therefore neither NARROW nor STOP can currently be
accepted. This blocker concerns the claim, not permission to run the already
preregistered bounded experiment.

### MAJOR — M1: recovery2 is distinct, but it changes the question

Recovery1 retained the chunk-training mechanism while replacing an almost
uniform confidence-bound penalty with the frozen surrogate point estimate.
Recovery2 instead uses a long task chunk, observes the current state at every
prefix, gates on predicted slack, and sometimes substitutes a one-step recovery
action. It also uses a fresh layout and magnitude-two axial disturbances. Those
are mechanistically and distributionally distinct changes, so recovery2 is a
legitimate second recovery rather than another beta sweep.

It nevertheless tests whether recovery value is useful for **inference-time
intervention**, not whether the stated training loss improves chunk policies.
This distinction is important because task/recovery switching is established by
[Recovery RL](https://arxiv.org/abs/2010.15920), and intervention-based safe RL
is represented by
[SAILR](https://proceedings.mlr.press/v139/wagener21a.html). A verifier win may
retain a useful engineering direction, but it does not rescue P4's training
novelty.

### MAJOR — M2: “matched calls” is not matched cost or matched feedback

The task-only control makes four fixed chunk launches. The adaptive method makes
two long-chunk launches plus roughly two expected action substitutions, but it
checks the fresh state and evaluates the slack gate at all twelve prefixes.
Moreover, every policy here is a precomputed table lookup: counting a verifier
lookup as a “replan call” does not demonstrate equal planning compute, latency,
observation cost, or model invocations. The shorter-chunk control has six calls
and therefore only exposes a higher-cost frontier.

A fair test must report, separately, policy/model queries, gate/scorer queries,
observations, action substitutions, wall-clock latency, and total compute. It
must include periodic and randomized intervention schedules with the same
feedback opportunities and expected intervention count, plus an adaptive
non-recovery score control. Current frontier work makes this especially
important: [PACE](https://arxiv.org/abs/2606.00537) selects replanning boundaries
from low-speed phases, while
[CheckVLA](https://arxiv.org/abs/2607.26789) reports matched invocation budgets,
false-alarm controls, and latency-aware execution-time verification. These 2026
papers are preprints, so they delimit the current comparison frontier rather
than establish settled results.

### MAJOR — M3: inactivity and clean-task loss are not adequately audited

`evaluate_chunk_policy` reports the unweighted fraction of hold actions across
the entire state-by-prefix policy table, not the hold rate on visited states;
the adaptive verifier reports `NaN`. Either can hide freezing in high-probability
states. Recovery2 also does not place clean return, clean success/progress, or a
clean-loss constraint in its verdict.

Required diagnostics are occupancy-weighted hold rate, speed, displacement and
progress by prefix; clean success/return; disturbed success-progress curves;
and an explicit minimum-progress constraint. If inactivity appears, follow the
frozen fallback and use constrained minimum progress rather than further beta
tuning. An entropy/viable-action control is still necessary; the recent
[Viability of Future Actions](https://arxiv.org/abs/2506.10871) is a direct
learned robustness baseline.

### MAJOR — M4: the surrogate can be gamed and is not calibrated for the gate

Recovery2 thresholds the point prediction, although the surrogate audit
calibrated symmetric residual intervals. Marginal/condition coverage does not
establish false-safe control near the intervention threshold or under the
policy-induced state distribution. Endpoint goal probability alone cannot show
whether the gate ranks recoverability rather than correlates with the chosen
layout.

The feature set also includes model-derived one-step failure floors and all
action-specific collision probabilities. These are defensible in an exact toy
diagnostic, but they are privileged dynamics/risk features for a learned-system
claim and can reduce the surrogate to a near-oracle safety lookup. Report
false-safe/false-unsafe rates, threshold reliability, subgroup calibration and
policy-shift coverage using exact truth for evaluation only; add feature
ablations and account for feature-computation cost. Adaptive shielding under
model mismatch, as studied by
[Lu et al. (2025)](https://proceedings.mlr.press/v283/lu25a.html), is the relevant
robustness control.

### MAJOR — M5: stage two is exact enumeration, not a frontier learned prototype

The policy is selected by exhaustive enumeration of constant open-loop action
sequences in a small tabular MDP. That is appropriate for bounded diagnosis and
at most NARROW, but it does not satisfy the frozen P4.2 learned-policy stage or
support scaling claims. It also lacks several frozen baselines at matched cost:
periodic replanning, PACE-like boundaries, entropy/viability regularization,
recovery demonstrations where available, and a post-hoc verifier with fully
matched resources.

### MINOR — N1: exact-pilot strengths should remain explicit

Preserve the exact inertial result as E0 evidence even if P4.2 is archived. Its
absorbing failure state, inertia/braking dynamics, disturbance kernels and exact
replanning values are useful for identifying legal-but-unrecoverable prefixes
and for unit-testing quantile signs. Do not promote its surrogate coverage or
interior frontier to evidence of a learned policy effect.

## What a frontier learned P4 prototype would need next

Only proceed after B1 is repaired or amended and a bounded recovery2 result is
published without revising its locked threshold.

1. Train an actual chunk policy in a continuous-control or robot-learning
   environment with the correct lower-quantile prefix objective, a frozen
   recovery estimator, disjoint layouts/tasks, and qualitatively held-out
   disturbance types and magnitudes.
2. Compare task-only, fixed shorter chunks, periodic/random matched-budget
   replanning, Recovery-RL-style switching, entropy/viability regularization,
   PACE, and a same-estimator post-hoc verifier. Closed-loop act-or-repeat methods
   such as
   [SDAR](https://openreview.net/forum?id=PDgZ3rvqHn) are also necessary controls;
   they directly exploit current-state feedback that fixed open-loop chunks lack.
3. Match the full resource vector—not a scalar expected-call count—and publish
   success-progress-compute frontiers with clean-task loss, latency,
   observations, scorer/policy queries, interventions, and variance over seeds.
4. Audit occupancy-weighted inactivity and estimator gaming; calibrate on fresh
   policy-induced states and report exact-evaluation false-safe rates without
   feeding truth into training or gating.
5. Demonstrate that slack **training** adds benefit beyond the strongest
   inference-time verifier. If verification matches or exceeds it more cheaply,
   the frozen STOP condition is met and the training proposal should be archived.

## Classification summary

- **BLOCKER:** B1 estimand mismatch; B2 absent recovery2 evidence artifact.
- **MAJOR:** M1 mechanism/claim change; M2 unmatched cost and feedback; M3
  inactivity/clean-loss gap; M4 gate calibration and privileged-feature gaming;
  M5 no learned frontier prototype and incomplete baselines.
- **MINOR:** N1 preserve, but tightly bound, the exact-pilot contribution.

