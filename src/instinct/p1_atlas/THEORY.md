# P1 theory map

- Question/hypothesis: frozen specification sections 4.1 and 4.6.
- Estimand: `Sigma = G_plan + R_intermediate - L_arrival - L_wait - C_hw + L_base_delay + epsilon_id`.
- Assumptions: exact finite MDP for P1.0; matched contexts for transfer; failure states explicitly distinguished from successful terminal states.
- Controls/baselines: M0–M5, planted product collapse, planted separate hardware-quality term, flat-frontier inconclusive control.
- Verdict: section 4.8. P1.0 is a known-answer E0 gate; sampled real-time transfer begins at P1.1.
- P1.1 amendment: operational recurring return and the matched one-handoff decomposition are separate outputs; observed event-time latency and cumulative per-episode hardware cost are mandatory.
- P1.1 delayed-control family: a continuous one-dimensional inertial state is discretized only at environment ticks. Legal states can already be unrecoverable when maximum-braking distance exceeds the remaining boundary margin. Hold, safe braking, greedy target pursuit, and deterministic lookahead reflexes remain distinct; failure is absorbing.
- Novelty boundary: adaptive variable-delay real-time RL and acting while planning are occupied; P1 is an offline decomposition, collapse-identifiability, and whole-context transfer atlas. Finding-the-Time-to-Think-style gating and planner/reflex match/mismatch are mandatory P1.1 baselines.
- Source: `docs/AI_Instinct_Seven_Proposal_Execution_Plan.md`, proposal version 2.1; portfolio amendment 2.1 changes scheduling only; `docs/amendments/p1/1.1.md` applies to P1.1.

## P1.1 recovery-1 mapping

- The recovery changes only the fresh benchmark distribution: continuous starts
  sample both sides of the deterministic recovery margin and use new oracle and
  regret seeds. The Sigma estimand, signs, matched arms, and threshold are unchanged.
- `M5` is the deployable nearest-training-context lookup. A separate
  `LOOKUP_ORACLE_EVAL_ONLY` is the held-out surface argmax and cannot support a
  transfer claim.
- `FTT_LITE` is not official Finding-the-Time-to-Think equivalence. The adapter
  freezes the released finite-budget gate boundary, while source/checkpoint
  equivalence and measured T4 cells remain explicit prerequisites.

## P1.1 recovery cycle 2 mapping (v6 preregistration)

- Original question and estimand: unchanged. Recovery2 continues to estimate
  recurring budget regret and the original matched-handoff Sigma identity; it
  does not alter any sign, arm, cost, or irreversibility definition.
- Mechanism: `M4S` is an additional uncertainty-gated convex stack of original
  M4 and deployable M5. Its weight and support attenuation are estimated only
  by leave-one-whole-training-context-out predictions. M0--M5 remain intact.
- Assumptions/identifiability: descriptor proximity must transfer frontier
  shape, the held-out target range must exceed `0.01`, and whole context/seed
  registries must be disjoint. Test Sigma is forbidden from fitting the gate.
- Controls/baselines: M0--M5, recovery1's strongest non-registry `SPEED`
  baseline, smallest/largest, FTT-lite, exact identity, natural failures, and
  the evaluation-only lookup ceiling. Official FTT and T4 stay fail-closed.
- Verdict: positive one-sided 95% paired M5-minus-M4S improvement and M4S UCB95
  below `0.2 * target_range`, with all CPU instrumentation passing. Thresholds
  are unchanged from recovery1.
- Source: frozen spec sections 4.1--4.9, amendment P1-1.1, and
  `docs/reviews/p1/v6/recovery2_preregistration.md`.
