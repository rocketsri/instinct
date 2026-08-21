# P5.2 recovery cycle 1 — sealed result

- Verdict: **INCONCLUSIVE**, E1. P5.3 remains blocked.
- Fresh held-out learned normalized AUC: 0.98329452.
- Strongest of eight random rules: `random_rule_04`, AUC 0.98798946.
- Relative learned gain over that comparator: 0.00475202 (0.4752%).
- Pilot-calibration meaningful-effect threshold: 0.01665423 (1.6654%).
- Primary gain-minus-threshold: -0.01190222.
- Same-step direct-backprop AUC: 0.945353; it remains materially stronger.
- Program: eight coefficients plus step size, 72 bytes; conventional
  initialization is excluded and identical across arms.
- Permutation error: 3.47e-18. Remote-neuron locality error and zero-shot
  cross-arm spread: exactly zero.

Diagnosis: one lucky random comparator does not fully explain v3 because the
recovered rule also beats the strongest of eight fresh random rules. The richer
rule family improves pilot optimization but transfers only weakly. The original
zero threshold is the clearest defect: v3's 0.2207% gain and recovery's 0.4752%
gain both fall far below the 1.6654% pilot-derived threshold. This is valid
recovery-cycle evidence, not an F0/F2 failure and not a basis for P5.3.
