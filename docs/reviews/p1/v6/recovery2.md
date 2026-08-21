# P1.1 recovery cycle 2 result

## Verdict

`STOP`, evidence `E2`, failure code `F4_BASELINE_DOMINANCE`.

Run:
`/private/tmp/instinct-p1-r2-final/p1_atlas/p1_atlas-20260817T064805Z-c1a08cc3-051b53`

The preregistration in `recovery2_preregistration.md` was frozen before these
fresh units were consumed. Recovery1 remains unchanged under `v5/`.

## Mechanism result

The leave-one-whole-training-context-out convex fit estimated raw M4 weight
`0.039160`. Its cross-context surface-MSE improvement LCB95 was `-0.034568`,
so the preregistered uncertainty gate set the active M4 weight to zero. M4S
therefore fell back exactly to deployable M5 on every evaluation context. This
is the intended safe behavior of the gate, but it provides no new mechanism
gain:

- M4S mean regret: `0.176798`; UCB95: `0.262483`;
- M5 mean regret: `0.176798`; UCB95: `0.262483`;
- paired M5-minus-M4S improvement LCB95: `0.000000`;
- held-out target range: `0.202038`;
- unchanged 20%-of-range tolerance: `0.040408`.

Both frozen success conditions fail. The simpler M5 exactly matches M4S, and
M4S exceeds the absolute tolerance by over sixfold. Original M4 also remains
poor (mean `0.162465`, UCB95 `0.280550`), so shrinkage did not conceal a viable
conditional surface.

Recovery1's strongest non-registry baseline was carried forward explicitly as
`R1_SPEED_CHAMPION`. It has mean regret `0.108463` and UCB95 `0.202075`, lower
than M4S/M5 in this run. Its paired advantage is not itself decisive at 95%
(SPEED-minus-M4S mean `-0.068335`, one-sided 95% upper bound `0.018796`),
but the primary M5 tie and absolute-tolerance failure already trigger the
preregistered archive rule.

## Controls and traceability

- M0--M5 are unchanged and all reported; smallest/largest, FTT-lite, SPEED,
  and the evaluation-only lookup ceiling are also present.
- Training and evaluation context identifiers are disjoint. M4S fitting uses
  only training-context Sigma; perturbing all evaluation Sigma values leaves
  its predictions and choices unchanged in the leakage test.
- Pilot seeds `501`--`508` and evaluation seeds `601`--`612` are mutually
  disjoint and fresh relative to the initial run and recovery1.
- Sampled Sigma identity residual maximum: `1.332268e-15`.
- Natural continuous-control failed/survived rows: `2234 / 2086`.
- Initial recovery-margin range: `[-0.119017, 0.239880]`.
- All internal instrumentation controls passed.
- Official FTT source/checkpoint equivalence remains false because the external
  checkout/checkpoint is absent. Measured T4 timing remains false because the
  timing-cell artifact is absent. Neither is relabeled as a local equivalent.

Static and integration verification passed: Ruff, mypy over all nine P1 source
files, and 26 scoped P1 tests.

## Final disposition

Archive the conditional-atlas mechanism after two valid theory-informed
recoveries. Preserve the exact Sigma decomposition, continuous failure-rich
benchmark, and empirical M5/SPEED evidence. Do not claim a universal product
collapse or revive M4/M4S through threshold adjustment. External official FTT
or T4 evidence may extend baseline/hardware coverage, but cannot reverse this
locked recovery2 mechanism verdict.
