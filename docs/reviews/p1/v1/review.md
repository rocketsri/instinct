# P1 pre-implementation review v1

- **Theory — BLOCKER resolved:** the frozen identity requires separate `L_base_delay` and `epsilon_id`.
- **Heuristics — BLOCKER resolved for instrumentation:** a signed unequal-time handoff value is not irreversibility; use a matched-time failure event with explicit failure states.
- **Literature:** no new implementation blocker beyond the frozen M0–M5 and conditional-transfer boundary.
- **Synthesis:** original estimand unchanged; code repaired, no scientific amendment. All-pairs collapse/noncollapse controls and the M0–M5 whole-context registry now complete P1.0; sampled transfer remains P1.1 work.
- **Falsifier:** nonnumerical identity residual, inability to recover planted noncollapse, or flat/unidentifiable frontiers.
