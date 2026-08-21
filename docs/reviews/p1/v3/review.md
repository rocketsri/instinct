# P1.1 two-family CPU implementation review v3

## Implemented controls

- Stage-specific `p1.1` dispatch leaves the frozen P1.0 path unchanged.
- Pursuit and Tetris-lite cover smooth/recoverable and threshold/failure CPU
  families across three event rates, three measured CPU workload regimes,
  eight budgets, and hold/greedy/distilled reflexes.
- Every planner latency is an observed per-episode CPU duration. Random launch
  phase maps it to event delay under amendment P1 1.1; recurring work is charged
  per live episode and pending work is not free.
- Whole speed/runtime contexts, never budget points, are assigned to frozen
  train/validation/test combinations. Distilled-reflex test contexts are not
  used to fit M0--M5.
- Oracle-identification and regret-evaluation episode identifiers are disjoint.
  Regret is evaluated as a paired difference on the latter units, with a
  conservative one-sided t bound over whole test contexts.
- The operational recurring frontier and matched one-handoff decomposition are
  distinct record types. All five handoff arms are required and the identity is
  checked episodewise. Irreversibility remains a matched-time probability.
- M0--M5, smallest/largest budgets, a speed schedule, and an FTT-lite
  uncertainty gate are emitted in the same artifact.

## Frozen limitation and verdict ceiling

The continuous/discretized delayed-control family is not implemented. The
runner therefore cannot return GO. It returns NARROW only if every implemented
statistical criterion passes; otherwise it returns INCONCLUSIVE. The first
frozen CPU pilot is INCONCLUSIVE/F1: M4's absolute bound is nonvacuous, but M4
does not beat the required baselines on independent regret units, and the
identified-oracle comparisons expose appreciable selection uncertainty.

## Open blockers before P1.1 GO

1. Add the third delayed-control family with recoverable and unrecoverable
   regions, preserving the frozen context/unit split.
2. Establish a stronger oracle-identification protocol; negative evaluation
   differences show that the small oracle sample can misidentify the frontier.
3. Replace FTT-lite with the closest lightweight official/reference-equivalent
   implementation before a comparative literature claim.
4. Add measured T4 timing and asynchronous resource-interference controls for
   E4. CPU timing replay is not deployment transfer.
5. Add a separately preregistered planner/reflex match-mismatch diagnostic if a
   claim is made about planner/reflex coordination rather than conditioning.
