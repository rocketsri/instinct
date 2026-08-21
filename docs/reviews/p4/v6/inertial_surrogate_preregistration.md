# P4.1 inertial surrogate preregistration v6

The corridor v4 estimator is not transferred. A new quadratic state/dynamics
surrogate is fitted on two complete obstacle layouts, calibrated on a third,
and evaluated once on a fourth layout. Each layout crosses actuator-slip
levels `0.05/0.15/0.25` and replanning horizons `8/12/16`.

Features contain position, velocity, horizon, slip, normalized progress, the
minimum one-step collision probability, and all five action-specific collision
probabilities. Exact goal and collision values are imposed structurally.
Calibration first takes the 90th-percentile state residual within each complete
condition and then a finite-sample conformal upper quantile across conditions.

The frozen gate requires aggregate coverage at least `0.90`, at least `0.80`
of held-out conditions attaining `0.90` state coverage, and full interval width
no more than `0.75` of the held-out exact-value range. Passing only authorizes
locking this model for a fresh P4.2 policy comparison.
