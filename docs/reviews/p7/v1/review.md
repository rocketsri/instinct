# P7 pre-implementation review v1

- **Theory — BLOCKER:** no qualifying realistic P2.2 evidence exists. Simulator work is allowed; scientific P7.0 is forbidden.
- **Heuristics — MAJOR:** include burst/trickle, oldest-item deadline through completion, finite capacity, audit propensities, padding accounting, owner isolation, and serialized utilization.
- **Literature — BLOCKER for broad memory-selection claims:** TTCD already targets future-useful memory; LaCT/FSM/In-Place/RW-TTT set stronger systems precedents.
- **Synthesis:** fixed-shape residual claim unchanged; task/kernel execution remains dependency-blocked.
- **Falsifier:** scorer/synchronization erases throughput, queue is unstable, or rejected-item coverage fails.
