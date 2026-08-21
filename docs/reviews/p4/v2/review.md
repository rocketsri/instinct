# P4.0 implementation review v2

- **Resolved:** recovery labels now come from exact finite-horizon DP with an
  explicit failure origin and validated common disturbance kernels.
- **Resolved:** rare catastrophic lower quantiles, terminal masking, reward-unit
  rescaling, and deliberately biased-surrogate gaming are executable controls.
- **Resolved:** shorter chunks and periodic verification are compared as a
  two-dimensional matched-call dominance test.
- **Open before P4.1/P4.2:** 2D inertial environment, calibrated learned
  surrogate, frozen validation selection, and held-out disturbance training.
- **Verdict boundary:** constructed E0 mechanism gate, not a safety certificate.
