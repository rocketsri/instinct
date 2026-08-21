# P1.1 recovery cycle 2 preregistration

Status: frozen before any recovery2 simulation or timing unit is consumed.

## Preserved evidence and diagnosis

Recovery1 remains preserved at `docs/reviews/p1/v5/recovery1.md`. Its sampled
Sigma identity was exact and its continuous-control distribution produced
natural failures, but the original M4 conditional ridge was dominated by the
deployable M5 nearest-context surface: M4 regret UCB95 was `0.124188`, M5 was
`0.040792`, and M5-minus-M4 paired-improvement LCB95 was `-0.126633`. The
frozen absolute M4 tolerance also failed (`0.124188 > 0.2 * 0.389729`).

The mechanism diagnosis is sparse-context extrapolation, not a threshold or
estimand defect. M4 estimates many budget/descriptor interactions with almost
no shrinkage, whereas M5 copies a complete observed frontier from the nearest
training context. Recovery2 therefore adds a new method, `M4S`; it does not
rename, refit, or replace M4 or M5.

## Frozen mechanism

For each training context, fit M4 and M5 without that entire context and form
out-of-fold surface predictions. Estimate the convex stacking coefficient

\[
 a^*=\operatorname{clip}_{[0,1]}
 \frac{\langle \hat\Sigma_{M4}-\hat\Sigma_{M5},
 \Sigma-\hat\Sigma_{M5}\rangle}
 {\|\hat\Sigma_{M4}-\hat\Sigma_{M5}\|_2^2}.
\]

Compute one surface-MSE improvement per omitted context for the resulting
stack. Activate `a*` only if the one-sided 95% t lower bound across whole
contexts is positive; otherwise set it to zero and fall back to M5. For an
admitted test context, attenuate its M4 weight outside descriptor support as

\[
 a(s)=a^*\exp\{-\max[d(s)/d_{0.9}-1,0]^2\},
\]

where distance uses training-only standardized descriptors and `d_0.9` is the
90th percentile leave-one-context-out nearest-neighbor distance. The selected
budget maximizes `a(s) M4 + (1-a(s)) M5`. No recovery2 validation or evaluation
outcome may tune the coefficient, confidence level, distance quantile, or
attenuation rule.

## Fresh units and controls

- Pilot seeds `501`--`508` and locked evaluation seeds `601`--`612` are fresh,
  mutually disjoint, and disjoint from the initial run and recovery1.
- The three event-rate cells (`12000`, `45000`, `160000` Hz) and three measured
  CPU work cells (`3`, `10`, `30`) differ from recovery1.
- Atlas fitting, leave-one-context-out gating, and the speed baseline use only
  whole training contexts. Complete test contexts enter only oracle-frontier
  identification and paired evaluation with separate seed streams.
- The unchanged Sigma identity, matched-time irreversibility, failure-rich
  continuous control, M0--M5, smallest/largest, FTT-lite, and `SPEED` are
  retained. `SPEED` is explicitly the strongest non-registry recovery1
  baseline by regret UCB95.
- Official Finding-the-Time-to-Think equivalence and measured T4 timing remain
  fail-closed external controls. Their absence cannot be relabeled a pass.

## Frozen verdict and archival rule

M4S passes the bounded CPU mechanism gate only if all internal controls pass,
the target range exceeds `0.01`, M4S regret UCB95 is strictly below `20%` of
the target range, and paired `M5 regret - M4S regret` has one-sided 95% LCB
strictly above zero. M0--M5 and the recovery1 `SPEED` champion remain reported,
but do not replace the primary comparison.

Official-equivalence and T4 absence yield `INCONCLUSIVE` only after the CPU
mechanism gate passes. If the valid CPU mechanism gate fails, the conditional-
atlas mechanism is archived after this second and final theory-informed
recovery; the exact decomposition and empirical M5/SPEED baselines remain
usable evidence, but no conditional-atlas or universal-collapse claim survives.
