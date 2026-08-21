# Lightweight RL harness design

## Purpose

The portfolio needs repeatable small-game and control experiments, not a second
general-purpose RL framework. The local harness standardizes episode collection,
split ownership, evaluation, accounting, and checkpointing around the existing
`BatchedEnv`, `EnvState`, and counter-based `SeedScope` interfaces. Algorithm
implementations remain small enough to audit against a naive reference.

## Adopted conventions

1. **Reset and step contract.** Follow Gymnasium's explicit reset-before-step
   lifecycle and distinguish task termination from a finite-horizon truncation
   in collected records. Existing environments expose task termination; the
   collector owns and records horizon truncation separately.
2. **No silent autoreset.** A collector captures the terminal state and metrics
   before resetting a lane. Gymnasium vector environments may autoreset and move
   the terminal observation into `info`; that behavior is useful generally but
   too easy to misuse in exact counterfactual and recovery experiments.
3. **Lane identity is stable.** Randomness remains addressed by root seed,
   stream, episode, absolute tick, and lane ID. Batch position never becomes an
   RNG identity.
4. **Splits own whole episodes.** Training, model-selection validation, oracle
   identification, and final evaluation episode IDs are disjoint and persisted.
   A budget/action point cannot migrate between roles.
5. **Evaluation is separate and deterministic by default.** Following the
   Stable-Baselines3 pattern, periodic evaluation uses a separate environment
   instance/episode registry, fixed episode allocation, and a frozen policy
   snapshot. Per-episode returns and lengths are retained; only reporting their
   mean and standard deviation is insufficient for paired inference.
6. **Termination is not truncation.** Value-learning targets bootstrap across a
   time-limit truncation when the method calls for it and never bootstrap across
   a true terminal failure/goal. The distinction is stored in every transition.
7. **Readable reference learners.** Following CleanRL's research-friendly
   single-file philosophy, initial tabular/linear learners keep the complete
   update rule visible and test fast-versus-naive equivalence. Large framework
   abstractions are deferred until a proposal needs them.
8. **Checkpoint complete state.** Learner parameters, optimizer/statistics,
   global step, episode cursor, split registry, and seed scope identifiers must
   round-trip. Saving weights alone is not resume equivalence.
9. **Costs are first-class.** Environment steps, policy calls, learner updates,
   wall-clock, and proposal-specific planner/replanning work are separate
   counters. Matched-cost claims select a declared accounting basis rather than
   equating them implicitly.

## Compatibility boundary

Core experiments do not depend on Gymnasium, Stable-Baselines3, RLlib, or
Sample Factory. Optional adapters may expose a Gymnasium-compatible API later,
but internal experiments retain counterfactual lane IDs, explicit episode
registries, and controlled reset semantics. SB3 is a useful baseline provider
for standard tasks, not the source of truth for portfolio estimands.

RLlib/Sample Factory-style distributed sampling is intentionally deferred. It
would add process scheduling, serialization, and asynchronous policy-version
issues before the CPU mechanisms have passed. GPU work remains serialized by
the portfolio's available T4 capacity.

## Primary references

- Gymnasium custom environments and seeding:
  <https://gymnasium.farama.org/main/tutorials/environment_creation/>
- Gymnasium environment termination/truncation API:
  <https://gymnasium.farama.org/v0.28.1/api/env/>
- Gymnasium vector environment/autoreset behavior:
  <https://gymnasium.farama.org/v0.29.0/api/vector/>
- Stable-Baselines3 evaluation helper:
  <https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html>
- Stable-Baselines3 evaluation/checkpoint callbacks:
  <https://stable-baselines3.readthedocs.io/en/v2.3.2/guide/callbacks.html>
- CleanRL reference implementations:
  <https://github.com/vwxyzjn/cleanrl>
- Finding the Time to Think novelty/baseline boundary:
  <https://arxiv.org/abs/2606.26463>
