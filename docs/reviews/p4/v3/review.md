# P4.1 calibrated-surrogate pilot v3

- Exact finite-horizon replanning labels were generated across corridor lengths,
  slip probabilities, horizons, goals, and explicit failure states.
- A fixed polynomial ridge surrogate was fit on training conditions. Its
  symmetric residual radius was selected on separate calibration conditions.
- The held-out length/slip/horizon condition was evaluated once and was not used
  to refit coefficients or widen the interval.
- Held-out mean absolute error was approximately `0.186`, but interval coverage
  was only `0.70` against the preregistered `0.90` requirement.
- **Verdict:** INCONCLUSIVE/F2 estimator failure. Preserve P4.0 exact-label
  results; repair representation/calibration on pilot units before P4.2. This
  does not consume a scientific recovery cycle and cannot be converted into a
  positive result by tuning on the held-out condition.
