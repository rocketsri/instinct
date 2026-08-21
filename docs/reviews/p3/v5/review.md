# P3.1 final recovery and archive review v5

## Sealed predecessors and unchanged theory

The v3 held-out-delay result (`-0.695912`, INCONCLUSIVE) and recovery-cycle-1
result (`-1.009984`, STOP/F4) remain preserved. Their evaluation units are not
loaded by this version. Recovery cycle 2 retains the frozen one-interval
planner/reflex objective, so no estimand amendment is needed.

This final recovery is scientifically distinct from both tabular action
classifiers. It imports the existing continuous `InertialIntervention`
environment read-only and measures per-unit CPU planner latency. Pilot cells use
event rates `{15000,60000}`, work scales `{1,4}`, and seeds `{317,318}`.
Evaluation uses unseen event rates `{30000,120000}`, unseen work scales `{2,8}`,
and seeds `{397,398,399,400}`. Observed mean planner latency was approximately
`13--57` microseconds, producing fresh mean delay cells from `1` to `7.25`
environment ticks.

The structural reflex pursues the target while recovery margin is ample and
brakes when deterministic stopping distance approaches the boundary. The
planner simulates that same reflex before choosing its pending action. A
known-answer test exhibits an interior action distinct from both hold and
immediate pursuit, and a separate known state makes matched and hold-model
planners choose different pending actions.

Pilot threshold returns are reused by two optimizers without extra environment
queries: condition regression predicts a threshold from event rate and observed
delay, while exponentiated policy gradient learns a global threshold arm. The
run also includes hold, immediate, distilled, reflex-only, a local
acting-while-thinking-style waiting policy, and a local uncertainty-gated
when-to-plan analogue. These last two are explicitly local structural analogues,
not reproductions of the external neural systems.

## F0 repair record

The first v5 artifact was INCONCLUSIVE because regression coefficients were
incorrectly clipped in threshold units. Coefficients have different units;
only predicted thresholds may be clipped. Correcting that line was an F0
estimator implementation repair and changed no registry, label, feature,
threshold grid, baseline, or success criterion. After repair, the same endpoint
solution persisted, establishing that collapse is scientific rather than the
original unit bug.

## Frozen final result

The final verdict is **STOP/F3 mechanism absent**. Primary gain against the best
deployable baseline is `0.0`, below the preregistered `+0.05`; zero of four
fresh cells clears the margin. Predicted thresholds are approximately
`[-0.100000, -0.099996, -0.100000, -0.099980]`, at the lower grid endpoint,
so learned interior behavior collapses to target pursuit.

Mean fresh-cell returns are:

| Arm | Mean return |
| --- | ---: |
| conditioned reflex | `0.221197` |
| immediate | `0.221197` |
| distilled | `0.221197` |
| local AWT-style waiting | `0.213723` |
| equal-compute policy gradient | `0.213723` |
| reflex-only | `0.213723` |
| local WTP uncertainty gate | `0.213723` |
| hold | `-0.148289` |

The hold-model mismatch changes `25%` of pending planner actions, satisfying the
informativeness control, but matched execution is `0.008677` worse on average.
Thus mismatch is real without supporting a universal claim that matching always
helps.

The observed-latency P1 before/after frontier shows learned-versus-hold gains of
`0.089945`, `0.160431`, `0.394326`, and `0.541563` at planner budgets
`{1,2,4,8}`. This confirms improvement over inactivity while simultaneously
showing that the learned program merely recovers immediate/distilled behavior.
It is a bounded CPU frontier, not GPU or deployment transfer.

## Archive disposition

Both allowed learned P3 recovery cycles are exhausted. P3.2 is locked and the
learned co-design direction is archived: recovery 1 was dominated by simple
baselines, while recovery 2 removed the tabular mechanism limitation but
collapsed to an immediate reflex. The exact P3.0 objective, cross-play
diagnostics, and bounded mismatch findings remain usable. Reopening learned P3
would require new user authorization and a new scientific claim, not tuning any
sealed evaluation version.

The primary-literature boundary remains that documented in v4: concurrent
planning/execution, learned acting while waiting, uncertainty-triggered
planning, and planner/RL mutual training are occupied. No novelty claim follows
from the archived prototypes.

Source: frozen specification sections 6.1--6.8, proposal version 2.1;
portfolio amendment 2.1 changes scheduling only.
