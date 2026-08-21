# P4.2 bounded open-loop policy preregistration v8

The geometry-aware v2 inertial estimator is locked before policy optimization.
The P4.2 layout is absent from both inertial estimator versions. Policy search
sees mild cardinal velocity impulses; evaluation uses fresh diagonal impulses.

Task-only and slack-regularized chunks both replan every three steps. The beta
`0.1`, twelve-step horizon, progress weight `2.0`, and one-percentage-point
meaningful success threshold were frozen after a separate pilot layout. Shorter
chunks, per-step periodic replanning, a per-step surrogate verifier, and a
PACE-like boundary brake expose the extra-call frontier rather than pretending
their costs match.

This slice can return at most NARROW. A full P4.2 GO still requires a learned
policy, disturbance-type and magnitude replication, and adaptive baselines
matched on average replanning calls and measured wall-clock.
