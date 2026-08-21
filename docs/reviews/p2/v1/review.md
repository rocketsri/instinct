# P2 pre-implementation review v1

- **Theory — BLOCKER resolved for P2.0:** candidates must share the complete frozen horizon set.
- **Heuristics — BLOCKER resolved for replay:** restrict before top-m; oracle horizons cannot touch downstream evaluation. P2.1 still lacks reset, anchor, periodic, dense/no-write, update-norm, gradient-alignment, and iterative-rescoring arms.
- **Literature — MAJOR:** VANE/AURA/FSM/reset methods constrain any deployable novelty claim.
- **Synthesis:** utility unchanged; only synthetic E1 measurement may run. No realistic useful-sparsity claim exists.
- **Falsifier:** planted harm AUC fails, naive/fast forks disagree, matched-write count fails, or reset/dense matched-compute dominates.
