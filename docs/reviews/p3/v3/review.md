# P3.1 bounded tabular implementation review v3

- **Theory:** the pending planner action is fixed before reflex-action branches,
  and the same frozen reflex schedule is used by the matched planner model.
- **Heuristics:** hold, immediate, random, distilled, optionality-only, shuffled
  values, exact-label oracle, mismatch, and cross-play arms are retained at the
  same declared planner work.
- **Instrumentation:** exact labels are audited by common-random-number sampled
  branches. Task termination and fixed-prefix truncation are separate fields.
- **Boundary:** a positive result is NARROW because only one exact tabular
  planner/environment family is tested. Direct large reference harnesses are
  not reproduced and no P3.2 or novelty claim follows.

## Smoke result and disposition

The frozen smoke was **INCONCLUSIVE/E1**. The planner-aware reflex returned
`8.671917` on the held-out delay, versus `9.367829` for the best conventional
arm (gain `-0.695912`). All split, accounting, snapshot, and label-audit
controls passed. This is an initial mechanism/transfer failure, not permission
to tune against the held-out delay and not a P3.2 result.

The mismatch diagnostic is non-informative in this cell: the compared planner
models choose the same pending action. A recovery must preregister a fresh
environment/delay split that actually induces planner disagreement.

## Open blockers before a full P3.1 verdict

- Add unseen speed as well as delay cells and more than one planner/environment.
- Add the frozen-spec equal-compute policy-gradient baseline.
- Measure the P1 computation--freshness frontier before and after reflex
  learning.
- Replace the duplicate hold/optionality construction with an environment in
  which recovery-only behavior is distinguishable from inactivity.
- Run a fresh preregistered recovery based on delay-distribution conditioning;
  the completed held-out cell remains sealed.
