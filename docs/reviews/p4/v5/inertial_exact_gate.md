# P4 exact 2-D inertial gate v5

## Why P4.2 did not start immediately

The successful v4 estimator is calibrated only for corridor length, slip, and
replanning horizon. The frozen P4 claim instead requires obstacles, inertia,
delayed replanning, and disturbances. Applying the corridor estimator to that
state space would be unmeasured extrapolation and would make a positive P4.2
result uninterpretable.

## Exact environment completion

The new finite MDP has 2-D position, velocity components in `[-2, 2]`, five
accelerations, obstacles, stochastic ignored commands, absorbing goal/collision
states, and declared post-prefix velocity impulses. Because acceleration is
bounded, some legal boundary states cannot avoid collision even under maximum
braking, while other legal states have zero immediate collision probability.

Exact finite-horizon replanning values retain the original `V_replan - V_fail`
origin and lower-tail quantile. Braking, myopic-progress, and exact-recovery
policies are evaluated under the same impulse kernel. This remains an E0
known-answer mechanism gate: the exact-recovery policy is an oracle control,
not a learned slack policy.

## Remaining blocker

Before P4.2, train and calibrate a fresh surrogate on disjoint inertial layouts,
impulse families, actuator-slip levels, and replanning horizons. Its test layouts
must be sealed before policy training. The corridor v4 result remains preserved
and cannot qualify the inertial model.

Source: frozen specification sections 7.1--7.6, proposal version 2.1;
portfolio amendment 2.1 changes scheduling only.
