# P6.2 frozen smoke result

Config: `configs/p6/p6_2/smoke.yaml`  
Config hash: `1d65b9e0adb823b91f2123746f518e61`  
Pilot/evaluation seeds: 611 / 621  
Repetitions per held-out cell: 160  
Verdict: **NARROW**

The pilot-fitted model's exact evaluation risks were `(0.0256, 0.0843, 0.1497)`. These values were recorded after fitting and were never certificate inputs.

| Held-out cell | Method | Model false certification | True false certification | Mean prefix | Nonvacuity |
|---|---:|---:|---:|---:|---:|
| stable | anytime max-certified | 0.0000 | 0.0000 | 1.6500 | 0.5500 |
| stable | shift-robust anytime | 0.0000 | 0.0000 | 1.3938 | 0.4646 |
| mild shift | anytime max-certified | 0.0000 | 0.0000 | 1.7438 | 0.5813 |
| mild shift | shift-robust anytime | 0.0000 | 0.0000 | 1.1000 | 0.3667 |
| adverse shift | anytime max-certified | 0.0000 | 0.0625 | 1.7125 | 0.5708 |
| adverse shift | shift-robust anytime | 0.0000 | 0.0000 | 1.0063 | 0.3354 |

For the primary shift-robust method, the worst held-out true false-certification count was 0/160, with one-sided 95% Clopper--Pearson upper bound 0.01855, below the preregistered 0.05 tolerance. Its minimum nonvacuity was 0.3354, so it was not an abstention-only solution.

The unadjusted anytime method's adverse-shift rate was 10/160 = 0.0625, with one-sided 95% upper bound 0.10371. This is the required empirical separation between model-relative and environment-relative validity. Always execute falsely certified the adverse prefix in every repetition; always abstain was valid but vacuous.

The threshold-only CSA-style e-process analogue observed 2/160 adverse-shift false certifications (one-sided 95% upper 0.03882) with nonvacuity 0.4625. This is useful bounded evidence, not a shift guarantee: unlike the paired-envelope method, its assumptions do not cover model/environment mismatch. The conservative nested-exit union observed 4/160 adverse false certifications (upper 0.05629), so it did not clear the locked 0.05 criterion. The shift-robust union and shift-robust anytime methods both observed zero adverse false certifications and selected mean prefix 1.0063 there; the anytime method therefore showed no material utility gain over the simpler robust union.

Conclusion: at least one method is empirically valid and nonvacuous throughout the preregistered shift family, but no GO claim is supported. P6 remains NARROW because the robust method depends on paired calibration/evaluation exchangeability, the study is a finite nested-Bernoulli analogue, and utility does not materially exceed the robust union baseline.
