# P4.1 condition-calibrated estimator repair v4

## Preserved result and repair boundary

P4.1 v3 remains **INCONCLUSIVE/F2** with held-out exact-label coverage `0.70`.
Its condition `(length=8, slip=0.2, horizon=8)` was not relabeled, moved into
training, or used to choose v4's representation, radius, split, or thresholds.
The unchanged v3 CLI path reproduces the same failure.

This is an F2 estimator repair, not a theory amendment or scientific recovery
cycle. The estimand remains exact finite-horizon `V_replan - V_fail` and the
policy-level P4 claim remains untested.

## Fresh preregistered method

Version `2.1-p4.1-estimator-v4` freezes a 112-condition grid before exact-label
generation. Counter-addressed condition keys assign 56 complete conditions to
training, 28 to calibration, and 28 to a fresh test split. Splitting whole
conditions prevents states from the same exact MDP from crossing split
boundaries. The failed v3 held-out tuple is explicitly excluded from all three
v4 splits.

The repair has two mechanism-based changes, both justified without v3 test
labels:

1. A fixed cubic basis represents normalized state progress, corridor length,
   slip, horizon, and all interactions through degree three. Known goal and
   failure terminal values are imposed exactly rather than estimated.
2. Calibration is condition-balanced. Each calibration condition contributes
   its 90th-percentile state error, then a finite-sample split-conformal upper
   quantile is taken across conditions. Longer corridors therefore do not get
   more calibration weight merely because they contain more states.

The preregistered success gate requires at least `0.90` aggregate exact-label
coverage, at least `0.80` of fresh conditions attaining `0.90` state coverage,
and full interval width no greater than `0.50` of the calibration-label range.

## One-shot fresh evaluation

The frozen v4 run returned **GO/E1** for the estimator gate:

- exact-label coverage: `259 / 260 = 0.996154`;
- test-condition success fraction: `28 / 28 = 1.0`;
- minimum test-condition coverage: `0.916667`;
- mean absolute error: `0.037551`;
- worst overestimate: `0.231664`;
- calibrated half-width: `0.223650`;
- full-width / calibration-label-range: `0.232`.

For context, the v3 pointwise ridge refit on the same fresh train/calibration
registry also reaches `0.919231` coverage, but with MAE `0.124363` and a wider
half-width `0.312856`. Therefore v4 repairs the failed estimator gate and is
more accurate on this registry, but this run does not establish unique
superiority of the structured estimator or explain all of v3's failure. The
broader, interpolation-focused registry itself contributes to recoverability.

## Verdict limits

GO authorizes locking this estimator for a separately preregistered P4.2
policy comparison only. It does not show that recovery-slack training improves
task success, beats shorter chunks or frequent verification at matched cost,
generalizes beyond corridor MDPs, resists proxy gaming, or provides a safety
certificate. Those remain BLOCKERs before any P4 mechanism or scale claim.

Source: frozen specification sections 7.1--7.6, proposal version 2.1;
portfolio amendment 2.1 changes scheduling only.
