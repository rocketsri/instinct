# P5.2 recovery cycle 2 — final archive result

- Verdict: **STOP/F3**, E1. Archive P5.2; do not start P5.3.
- Learned feedback-basis normalized AUC: 0.98326878.
- Strongest non-backprop comparator: sealed recovery-1 rule, AUC 0.96998705.
- Relative held-out gain: -0.01369268; the new mechanism is 1.3693% worse.
- New pilot-only meaningful-effect threshold: 0.03373213 (3.3732%).
- Primary gain-minus-threshold: -0.04742481.
- Strongest random feedback basis: 0.980451. Direct backprop: 0.943723.
- Final program: twelve coefficients plus step size, 104 bytes; no generated
  initializer, topology/task tokens, or persistent synapse-sized state.
- Permutation error: 5.20e-18. Locality error and cross-arm zero-shot spread:
  zero. Time-modulation effect: 0.104.

## Frontier and archival synthesis

1. V3's four-term rule produced a 0.2207% gain under a zero threshold.
2. Recovery 1's richer eight-term rule beat eight random controls by 0.4752%
   but missed its 1.6654% pilot-derived threshold.
3. Recovery 2's scientifically distinct coordinate/type/time feedback basis is
   worse than the frozen recovery-1 rule and misses its 3.3732% threshold.
4. Same-step direct backprop remains materially ahead in both recoveries.

All final scientific and instrumentation controls pass. The result is therefore
F3 mechanism absence/baseline dominance, not F0 implementation failure or F2
estimator failure. Both allowed recovery cycles are consumed. P5.2 is archived,
and the prerequisite for P5.3 joint initialization-plus-plasticity work is not
met.
