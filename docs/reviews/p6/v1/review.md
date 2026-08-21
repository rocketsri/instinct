# P6 pre-implementation review v1

- **Theory — BLOCKER resolved for instrumentation:** shared dynamic-rho confidence code was not time-uniform. Use fixed preregistered confidence spending and keep safety risk separate from confidence failure.
- **Heuristics — MAJOR:** attack with peeking, prefix selection, repeated decisions, nonmonotone risks, and poisoned exact truth.
- **Literature — BLOCKER for comparative GO:** add Anytime-Valid CRC, Conformal Selective Acting, and nested-exit AVCS baselines.
- **Synthesis:** conservative finite-family validity may run, but cannot beat itself as the union-bound baseline; verdict is at most NARROW.
- **Falsifier:** simultaneous undercoverage or vacuous bounds with no improvement over conservative baselines.
