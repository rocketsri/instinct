# P1.1 three-family continuous-control completion review v4

## Theory traceability

This implementation preserves the frozen P1 estimand and P1 amendment 1.1.
Recurring plan/act return remains the operational frontier; the named
`Sigma` identity remains a separate matched one-handoff measurement, and
irreversibility remains a matched-time counterfactual. No objective, baseline,
or verdict threshold changed.

The added `InertialIntervention` family is a continuous one-dimensional point
mass sampled at event ticks. Its state is position and velocity, its action is
left/zero/right acceleration, and crossing either boundary is absorbing
failure. A legal state is dynamically unrecoverable when

`boundary margin - velocity**2 / (2 * maximum acceleration) < 0`.

That recovery margin is a diagnostic, not an input to the planner or learned
models. Noise is counter-addressed by episode, tick, and persistent lane id so
matched arms see the same exogenous acceleration disturbance.

## Controls and mechanism coverage

- The exact required family set is pursuit, Tetris-lite, and continuous
  control; an arbitrary third family cannot satisfy the coverage check.
- Hold, safe maximum braking, greedy target pursuit, and distilled
  deterministic lookahead are distinct control-family policies.
- The planner's budget changes lookahead horizon while preserving the fixed
  three-action set.
- A known-state gate verifies one recoverable and one already-unrecoverable
  legal state before a run may pass instrumentation controls.
- Scalar-reference equivalence, batch subset/reorder common randomness,
  absorbing failure, sampled identity, transfer, and rollout regression tests
  pass.

## Executed smoke result

The three-family smoke completed with 81 contexts (27 per family), all
prerequisite controls passing, and maximum handoff identity error
`4.44e-16`. Its honest verdict is **INCONCLUSIVE/F1**, not GO:

- M4 context-regret UCB95: `0.321329`;
- median held-out target range: `0.594420`;
- M5 context-regret UCB95: `0.099624`;
- M4's lower-bound improvements over M0, M2, M5, and FTT-lite are all
  non-positive.

Thus the third family removes the implementation coverage ceiling but does not
rescue the preregistered conditional-atlas hypothesis. In this short smoke no
control episode failed, even though both recovery regions exist and are gated
by constructed known states. Threshold/irreversibility claims therefore still
require a longer, freshly preregistered pilot with demonstrated natural
occupancy near the recovery frontier; this smoke must not be tuned after
inspection.

## Remaining blockers before a P1.1 GO or scale claim

1. Establish a stronger, independent oracle-identification protocol; the small
   oracle sample still permits frontier selection uncertainty.
2. Demonstrate held-out visitation and failures near the continuous recovery
   frontier without changing this completed experiment version.
3. Replace FTT-lite with the closest official/reference-equivalent baseline
   before comparative literature claims.
4. Add measured T4 timing and asynchronous resource-interference controls.
5. Keep planner/reflex match-mismatch as a separately preregistered diagnostic
   if coordination claims are made.

Source: frozen specification sections 4.1--4.9, proposal version 2.1;
`docs/amendments/p1/1.1.md`.
