# P3.1 recovery-cycle-1 multi-cell review v4

## Preserved evidence and unchanged estimand

The v3 held-out-delay artifact remains unchanged and reproduces
**INCONCLUSIVE/E1** with gain `-0.695912`. Recovery v4 does not reuse its delay
cell, labels, fitted snapshot, or thresholds. The mathematical target remains
the frozen one-interval planner/reflex objective from section 6.2, so no theory
amendment is required.

Version `2.1-p3.1-recovery-v4` preregisters three environment/planner cells:
`chase3-h2`, `chase4-h4`, and irreversible `corridor5-h3`. It fits on speeds
`{0.10, 0.30, 0.50}` and delays `{1,2,4}`, then evaluates only unseen speeds
`{0.20,0.40}` crossed with unseen delays `{3,5}`. The proposed optimizer fits
action advantages using a frozen polynomial of speed and remaining delay. A
policy-gradient baseline receives exactly the same counterfactual label vectors
and one update per vector; evaluation planner work is identical within each
cell.

Optionality is no longer an alias for hold. It chooses the action maximizing
the worst planner value among its supported next states and differs from hold
on `71.7%` of active state/cell combinations. A diagnostic-only F0 repair chose
hold as the mismatched planner model based on training-cell action disagreement
(`44.4%` for hold versus `0%` for immediate). This changed no fitted model,
deployable baseline, evaluation label, threshold, or primary return. On fresh
cells the repaired diagnostic changes pending planner actions `47.2%` of the
time, and matched execution exceeds mismatch by `0.114045` mean return.

## Frozen result

Recovery cycle 1 returns **STOP/F4 baseline dominance**. The preregistered
primary gain is `-1.009984` against the best deployable arm; the required gain
was `+0.05`. No cell clears the margin:

| Cell | Mean gain versus best deployable |
| --- | ---: |
| `chase3-h2` | `-1.429501` |
| `chase4-h4` | `-1.354243` |
| `corridor5-h3` | `-0.246209` |

Mean returns across the twelve fresh cells are:

| Arm | Mean return |
| --- | ---: |
| immediate | `3.350326` |
| planner distilled | `3.350326` |
| optionality-only | `2.426118` |
| delay-advantage regression | `2.340342` |
| equal-compute policy gradient | `2.211703` |
| hold | `1.500873` |
| random | `1.429879` |

Immediate and planner-distilled actions agree on every active fresh state, so
their equal returns are structural rather than an accounting accident. The
learned reflex agrees with immediate on `90.8%` of scheduled actions; its
minority deviations are harmful enough to dominate the aggregate result. The
stronger regression does beat equal-compute policy gradient by `0.128639`, but
that optimizer improvement is much smaller than the gap to the simple greedy
arms.

The finite-compute before/after proxy frontier shows the learned reflex beats
hold by roughly `0.83--0.85` return over planner horizons `{1,2,4,8}`. That is a
real improvement over inactivity, not a victory over the relevant baselines;
the gain slightly decreases as planner horizon grows. It must not be presented
as a measured wall-clock P1 frontier.

## Current primary-literature and code boundary

- [Learning to Act While Thinking](https://openreview.net/pdf?id=btbT4Lzzqj)
  formalizes a real-time meta-reasoning MDP in which an RL agent jointly chooses
  control actions and whether to invoke a delayed asynchronous planner. It
  directly occupies learned waiting behavior and planner-invocation baselines;
  this bounded one-interval study is narrower.
- [When to Plan: Learning to Select Between Reactive Control and Deliberative
  Planning](https://arxiv.org/abs/2607.16421) conditions a learned
  meta-reasoning policy on reactive-policy uncertainty. Any future P3.2 claim
  needs an uncertainty-triggered reactive/planning selector baseline.
- [Planning and Acting While the Clock
  Ticks](https://arxiv.org/abs/2403.14796) studies concurrent planning and
  dispatch under wall-clock deadlines and measured planning speed. It occupies
  the systems/metareasoning boundary even though it is not the same RL
  objective.
- [CO-PILOT](https://openreview.net/forum?id=mTmddcaOSwz) jointly trains a
  planner and RL agent through a subtask curriculum; its
  [public implementation](https://github.com/Shuang-AO/CO-PILOT) is a direct
  co-adaptation precedent, though not an acting-during-delay method.

No public implementation for *Learning to Act While Thinking* was located in
the authors' primary paper/project pages as of 2026-08-17. A future large-scale
claim must either reproduce its described fixed and heuristic waiting
baselines or document why an exact implementation cannot be compared.

## Disposition

P3.2 remains locked. This valid fresh result shows that delay conditioning,
actual planner disagreement, and stronger optimization do not overcome simple
immediate/distilled policies in these cells. Recovery cycle 2, if authorized,
must be scientifically distinct and motivated without modifying this version's
fresh cells. Otherwise P3 should be narrowed to the exact objective and
mismatch diagnostic rather than a learned co-design claim.

Source: frozen specification sections 6.1--6.8, proposal version 2.1;
portfolio amendment 2.1 changes scheduling only.
