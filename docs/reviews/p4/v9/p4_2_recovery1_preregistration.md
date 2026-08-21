# P4.2 recovery cycle 1 preregistration v9

The v8 same-call slack policy was 0.000324 worse than task-only. The estimator
interval radius is `1.059` over a value range of `2.95`; subtracting that radius
from every nonterminal prediction makes the hinge penalty nearly uniform. The
frozen objective targets estimated recovery slack itself, not a confidence
lower bound, so this recovery uses the calibrated point estimate for `R_j` and
retains interval coverage as an estimator audit.

Beta, chunk length, progress weight, meaningful gain, and clean-loss thresholds
remain unchanged. A fresh obstacle layout is used. Prior P4.2 held-out states
are sealed. Task/slack replanning calls remain exactly matched; shorter chunks
and per-step verification continue to expose their higher call counts.
