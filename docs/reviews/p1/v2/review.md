# P1.1 pre-implementation review v2

## Classified findings

- **Theory — BLOCKER resolved in instrument, runner still pending:** recurring
  `fresh` arms do not remain at the same arrival state after divergence.
  Amendment P1 1.1 preserves recurring operational return and defines a
  separate matched one-handoff decomposition.
- **Theory/heuristics — BLOCKER resolved:** planning work now launches and is
  charged at the root even when it misses the horizon; per-lane calls and
  simulations exclude dead episodes.
- **Theory/heuristics — BLOCKER resolved in instrument:** sampled handoffs take
  observed per-episode latency and randomized event phase instead of imposing
  `floor(nu_e * nu_h * k)`.
- **Theory/heuristics — BLOCKER open for the P1.1 runner:** freeze separate
  oracle-identification and regret-evaluation units, hierarchical paired upper
  bounds, all held-out context combinations, causal descriptor extraction, and
  the three required environment/reflex families.
- **Literature — MAJOR/open baseline:** Finding the Time to Think directly
  covers adaptive variable-delay real-time RL with matched reflex MCTS on Snake
  and Tetris. P1's residual claim is only the offline decomposition,
  collapse-identifiability, and whole-context transfer atlas. P1.1 must add its
  lightweight gate plus planner/reflex match and mismatch.
- **Literature — later hardware control:** deployment claims require
  latest-completed/staggered inference at matched occupancy and cost.

## Synthesis

The original `Sigma` target, M0–M5 transfer comparison, and 20%-of-informative-
range criterion remain unchanged. Same-handoff semantics, observed event time,
and cumulative per-episode cost are clarified by P1 amendment 1.1. The shared
rollout accounting and matched handoff instrument are implemented, but no P1.1
scientific verdict is authorized until the open runner controls above pass.

## Falsifiers

- pre-arrival mismatch between `actual` and `fresh`;
- nonnumerical episodewise identity residual;
- free pending computation or work charged after termination;
- descriptor or split leakage;
- M4 point gains that disappear under paired context-level upper bounds;
- dominance by the mandatory learned gate or lookup on untouched contexts.

## Primary literature boundary

- Finding the Time to Think: <https://arxiv.org/html/2606.26463v2>
- Learning to Act While Thinking: <https://openreview.net/pdf?id=btbT4Lzzqj>
- Staggered asynchronous inference: <https://arxiv.org/abs/2412.14355>
- Handling Delay in Real-Time RL: <https://arxiv.org/abs/2503.23478>
