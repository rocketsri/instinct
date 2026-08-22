# instinct

Experiment infrastructure for a seven-proposal research portfolio on test-time inference (TTI),
test-time training (TTT), and the relationship between fast compiled behaviour and slow
deliberation.

The framing question across the portfolio:

> instinctive system = compiled fast behaviour + resource trigger + controlled plasticity

That is a *research lens*, not a claimed subfield. Each project has to stand on its own domain
contribution. The lens only becomes scientifically meaningful if these projects turn up measurable
exchange rates among freshness, progress, recoverability and retention — not merely because they all
happen to contain gates.

The frozen scientific specification is
`docs/AI_Instinct_Seven_Proposal_Execution_Plan.md`. All seven proposals now have
version-traced CLI smoke packages; only the bounded CPU gates described below
and several bounded learned pilots are implemented. Scale stages remain gated.

## What is here

| Program | Question | Status |
| --- | --- | --- |
| **P1** computation–freshness atlas | When does added computation lose value because the world advanced while you thought? | Exact Sigma instrument retained; conditional P1.1 atlas archived after two baseline-dominated recoveries |
| **P2** harmful-write TTT | Can a test-time memory tell useful plasticity from destructive plasticity? | P2.1 synthetic GO; the diagnosed F2 estimator repair ran on real T4/CIFAR-10 and stayed unidentifiable at every write fraction — archived as unidentifiable under the frozen benchmark; scientific recovery unconsumed; P7 stays locked |
| **P3** planner–reflex co-design | Can the fast policy improve planner arrival state? | Exact P3.0 retained; learned P3.1/P3.2 direction archived after two recoveries collapsed to simple baselines |
| **P4** prefix recovery slack | Does prefix-wise recovery training beat matched verification? | Exact environment and estimator retained; P4.2 archived after slack/verifier failures at matched calls |
| **P5** developmental program | Can fixed code generate initialization and local plasticity? | P5.2 archived after both local-plasticity recoveries missed pilot-derived thresholds; P5.3 locked |
| **P6** anytime-certified refinement | Risk certificates for action prefixes across compute exits. | P6.3/P6.4 bounded CPU GO; P6.5 realistic closed-form simulator confirmed validity/nonvacuity transfer unmodified, but the utility gain over robust union did not — closed NARROW |
| **P7** fixed-shape admission | Can semantic selection preserve large-update utilization? | queue/audit simulator only; locked on qualifying realistic P2.2 evidence |

## Why one codebase and not three

P1's measurements are the substrate for P2 and P3. P3's decisive test *is* re-running P1's atlas
under a retrained reflex — if the atlas describes the environment it should survive that; if it only
described one policy it will not. So the three programs share one statistics layer, one
compute-accounting layer, and one kill-condition harness.

## Methodological rules, enforced in code

These are checked by `tests/test_discipline.py`, not left to good intentions.

1. **No assumed functional forms.** The staleness fitter never fits `exp(-h*tau)` alone; it always
   evaluates the full family set including a nonparametric fit and a per-environment lookup table.
2. **No ratios with small denominators.** Effects are reported as differences in return units.
3. **Paired seeds everywhere,** via the counter-based CRN in `core/rng.py`.
4. **Matched total compute.** Comparisons log FLOPs *and* wall-clock and refuse to render when arms
   diverge beyond tolerance.
5. **Finite-use holdouts.** Retention probes carry a use ledger and are retired on exhaustion, so the
   reusable-holdout problem cannot migrate into the retention term.
6. **Residuals are reported, not hidden.** Decompositions telescope; the remainder is logged as a
   named interaction term.
7. **Every fast path has a naive reference,** with an equivalence test asserting they agree. Clever
   tricks in a research codebase are how results get retracted.

A tripped kill condition is a **result**. Stages report it and stop rather than quietly continuing.

## Performance

The sweeps are large outer products and the hardware is 4 CPUs plus one Colab T4, so throughput
decides how much science gets done. The wins that matter are structural, and all of the following
are *exact* — they return the same numbers as the naive path:

- The tabular arm needs **no Monte Carlo at all**. A fixed reflex and fixed delay make the
  plan-then-arrive process a finite Markov chain, so `J_mu(s, k)` is a linear solve that returns
  every start state at once.
- **MCTS budget sweeps are prefix-shared**: search once at `k_max`, snapshot the recommendation at
  every budget. `O(sum_k k)` becomes `O(k_max)`.
- **Counterfactual arms share prefixes** — `actual(k)` and `fresh(k)` differ only in which action
  lands at `s_d`.
- **Batched lanes, not per-episode Python loops**, which is only possible because the RNG is
  counter-based; sequential streams and batched forking do not compose.
- **Batched forking** for the P2 oracle, where kernel-launch overhead dominates FLOPs.

See the Performance section of the plan for the full list and `benchmarks/` for the measurements.

## Setup

```bash
uv sync --extra dev            # core + tooling, CPU only, no torch
uv sync --extra dev --extra nn # adds torch (CPU wheels) for the P2 work
uv run pytest
uv run instinct run --proposal p1_atlas --config configs/p1/p1_0/smoke.yaml
```

Torch resolves to CPU wheels locally by design: the only GPU available is a Colab T4, and Colab
images ship their own CUDA build.

### GPU (Colab T4)

```bash
bash scripts/colab_setup.sh     # installs google-colab-cli; auth is interactive
uv run instinct dispatch p2_llm --gpu T4
```

T4 is compute capability 7.5: fp16 tensor cores, **no bf16, no TF32**, and no FlashAttention-2
(Ampere+ only). The LM arm uses fp16 autocast with an fp32 master copy and the memory-efficient SDPA
backend. Reduced precision is confined to training inner loops — anything that lands in a reported
measurement is computed in fp32.

## Layout

```
src/instinct/
  core/      CRN rng, exact MDPs, batched envs, RL harness, planner, rollout, stats
  atlas/     P1: decomposition, sweep, curve fitting, transfer tests, kill harness
  ttt/       P2: fast weights, exact-fork oracle, frontier, gate, transactional
  p1_atlas/          P1 CLI/theory mapping
  p2_harmful_write/  P2 CLI/theory mapping
  p3_codesign/       P3 one-interval exact objective and gate
  p4_slack/          P4 quantile/slack instrument
  p5_development/    P5 coordinate generator and locality/symmetry checks
  p6_certification/  P6 finite-family Bernoulli process
  p7_chunking/       P7 dependency guard and queue simulator
  audit/     shared: exact counterfactuals, propensity estimators, blind-spot detection
  core/envs/ tabular, gridworld, arcade, and continuous delayed control
  core/compute/ FLOP and wall-clock accounting, Colab dispatch
configs/     smoke (seconds) / cpu (minutes) / t4 (hours)
reports/     generated, committed
specs/       P4, P5, P7 designs
```

## License

Apache-2.0.
