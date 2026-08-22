# Decision log

## 2026-08-22 — P2 F2 repair archived; P6.5 realistic-simulator NARROW

- Diagnosed P2's recovery-2 pilot failure as an F2 estimator/identifiability
  problem (reference-path staleness: candidates scored on a write-all
  trajectory but replayed on a sparse one) rather than a mechanism failure, per
  `docs/reviews/p2/v6/theory_literature_synthesis.md`. Implemented the one
  permissible pilot-only repair (`docs/reviews/p2/v7/f2_repair_preregistration.md`):
  masked live scoring, isolated-write downstream damage labels, iterative
  rescoring, and corrected dense-arm step accounting. Ran the repaired
  estimator on real T4/CIFAR-10 at the preregistered full pilot scale (24
  candidates, 20 isolated labels). Both archival gates fail at every write
  fraction (`0.1`/`0.25`/`0.5`): identifiability and mechanism-ceiling are both
  `false` throughout, and oracle_ceiling downstream accuracy never beats
  never-write (`0.543` vs `0.563`, `0.484` vs `0.563`, `0.395` vs `0.563`).
  P2 is archived as **unidentifiable under the frozen utility and benchmark**.
  The estimator bug was real and is fixed, but fixing it did not change the
  underlying finding. P2's scientific recovery remains fully unconsumed, the
  reserved seed-241 test partition was never opened, and P7 stays locked.
- P6.4's verdict text called for "a fresh realistic simulator or logged-system
  integration" without a preregistered scope for it; wrote a continuation memo
  (`docs/reviews/p6/v5/continuation_memo.md`) treating that as a declared,
  budgeted scope expansion rather than an unexamined continuation, per the
  portfolio's own stopping-budget rule. Discretized `InertialIntervention`
  into a closed-form-exact `TabularMDP` (verified independently: the
  transition kernel is degenerate/1-D, driven by a single Gaussian disturbance
  term, and the implementation correctly integrates over it rather than
  treating position and velocity as an independent bivariate pair) and paired
  it with the certificate/comparator/mismatch machinery unmodified. Ran
  P6.5 (`configs/p6/p6_5/cpu.yaml`) to a verdict: worst one-sided 95% upper
  bound `0.046`, minimum nonvacuity `0.375`, model/true validity gap `1.000` —
  all pass — but gain over robust union is `-0.082` against a `+0.200` target.
  The certificate machinery's *validity* generalizes unmodified to a real
  continuous stochastic environment (confirmed zero-diff on
  `certificates.py`/`comparators.py`/`mismatch.py`); its *utility* advantage
  over the simpler robust-union baseline does not transfer from the toy
  3-state chain. P6 closes **NARROW** on this axis; no further P6 stage is
  planned.

## 2026-08-17 — recovery synthesis and successful-direction continuation

- P1 recovery 2 assigned zero weight to unstable M4 and fell back exactly to
  deployable M5. The conditional-atlas mechanism is STOP/E2/F4 and archived;
  exact Sigma, M5, and SPEED evidence remains reusable.
- P3 recovery 2 collapsed to immediate/distilled behavior and P5 recovery 2
  lost to the sealed recovery-1 rule. Both learned directions are archived
  after their two allowed scientific recoveries.
- P4's repaired in-domain estimator passed, but neither slack training nor the
  final exactly four-call verifier transferred. P4.2 is STOP/E1 and archived;
  the exact environment and estimator remain instruments only.
- P6.2 established a useful NARROW shift-robust certificate. P6.3 then reached
  bounded CPU GO: zero observed true false certifications, worst 95% upper
  bound `0.0307`, minimum nonvacuity `0.5853`, and a `0.4375`-action prefix gain
  over robust union. Calibration cost remains separate from certificate latency.
- P2 recovery-2 infrastructure now has checksum-verified official CIFAR-10 and
  ResNet-18 artifacts plus a guarded T4 runner. Two official-train-only pilots
  consumed no reserved test units. The extended pilot found positive and
  negative write utilities but an oracle accuracy-frontier gap of `-0.035156`,
  so the full test stream was not opened. P2 recovery 2 remains unconsumed and
  P7 remains locked.

## 2026-08-17 — parallel learned-pilot tranche

- Completed P1.1's continuous delayed-control family. The three-family smoke is
  preserved as INCONCLUSIVE/F1: all decomposition controls pass, but M4 does
  not beat the strongest lookup baseline and the short control cells contain
  no natural failures.
- Froze a bounded P3.1 counterfactual-label pilot as INCONCLUSIVE/E1. The
  planner-aware reflex loses `0.695912` return to the best conventional arm on
  the held-out delay, so P3.2 remains blocked.
- Repaired the P4.1 F2 estimator on a new whole-condition split. Its fresh
  estimator gate is GO/E1 at `259/260` coverage; the old 70% failure remains
  preserved, and only a separately preregistered P4.2 policy test is unlocked.
- Froze the bounded P2.2 natural-image pilot as INCONCLUSIVE. Useful sparsity is
  zero, never-write beats the oracle, the realistic-scale prerequisite fails,
  and P7 remains locked.
- Froze P5.2 as NARROW/E1: the local rule improves normalized adaptation AUC by
  only `0.2207%` over its best non-backprop control, while direct backprop is
  substantially stronger. P5.3 requires independent review and a meaningful
  effect threshold.

## 2026-08-17 — RL harness and P4.1 estimator pilot

- Adopted Gymnasium-compatible termination/truncation concepts, separate deterministic evaluation, stable counter-based lane identities, complete checkpoint state, and CleanRL-style auditable reference learners without adding a mandatory external RL framework.
- P4.1's original frozen recovery surrogate achieved modest held-out mean error
  but only 70% exact-label coverage versus the 90% requirement. It remains
  preserved as INCONCLUSIVE/F2; the later v4 repair above uses fresh conditions.

## 2026-08-16 — parallel initial-gate expansion

- Ran independent P1.1, P2.1, and P6 workstreams while expanding P3, P4, P5, and P7 controls locally.
- P1.1 returned INCONCLUSIVE/F1 on the honest two-family CPU pilot; M4 did not dominate every required baseline and the continuous delayed-control family remains absent.
- P2.1 returned GO at E1 on the stateful synthetic frontier. This is not realistic P2.2 evidence and does not unlock P7.0.
- P3.0, P4.0, and P5.0/P5.1 remain bounded E0 GO results after stronger cross-play, exact-label, locality, scaling, and accounting controls.
- P6 remains NARROW at E1: validity/nonvacuity are measurable, but the anytime method did not materially beat the declared-exit union comparator.
- P7 remains simulator-only and INCONCLUSIVE while its audit channel, overload detection, trickle deadlines, and real-token accounting are strengthened.

## 2026-08-16 — portfolio execution amendment 2.1

- Removed the cap of two paper-scale directions at the user's direction.
- Every successful proposal may continue; GPU execution is scheduled by capacity rather than an arbitrary direction count.
- This changes portfolio resource policy only. Scientific estimands and proposal-level gates remain unchanged.

## 2026-08-16 — P1 sampled-semantics amendment 1.1

- Preserved recurring P1 policy value while versioning matched one-handoff decomposition and observed event-time semantics after independent P1.1 theory, heuristics, and literature review.
- Pending plans are charged at launch, per-lane work excludes terminated episodes, and the sampled decomposition closes episodewise.
- No P1.1 scientific verdict is authorized until the remaining v2 review blockers are resolved.

## 2026-08-16 — pre-implementation review v1

- The frozen scientific source is `docs/AI_Instinct_Seven_Proposal_Execution_Plan.md` version 2.0.
- P1's existing `eps_cross = L_base_delay` implementation was classified F0 and repaired to the frozen identity. This is not an amendment: the code had diverged from the specification.
- P1's signed unequal-time handoff-value diagnostic was replaced by a matched-time excess failure-probability counterfactual. Historical artifacts remain under `docs/history/` and are not relabeled.
- P2 now restricts full-horizon candidates before top-m and excludes oracle horizons overlapping the downstream tail. Complete P2.1 baselines remain future work.
- P3.0, P4.0, and P5.0/P5.1 may run as E0 instruments. Their outputs cannot support learned or novelty claims.
- P6 uses a separate finite-family Bernoulli process; `core/confseq.py` is not used for action-prefix certification. The initial verdict is NARROW until direct anytime-risk-control baselines establish nonvacuity.
- P7 code is simulator-only. Scientific P7.0 fails closed without a structured realistic P2.2 qualification artifact.
- No estimand, objective, or verdict criterion was amended in this implementation version.
