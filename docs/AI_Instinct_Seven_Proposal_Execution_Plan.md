# AI Instinct Research Portfolio

## Agent-executable experimental plan for seven proposals

**Version:** 2.0  
**Literature and novelty check:** 16 August 2026  
**Compute:** CPU experiments and interruptible free-tier Google Colab NVIDIA T4 sessions  
**Purpose:** turn the seven proposals into bounded, reproducible experiments that can produce a useful positive, negative, or inconclusive result without silently changing the hypothesis.

---

## 1. Executive decision

The seven proposals are not seven equally mature full projects.

| Class | Proposals | Decision |
| --- | --- | --- |
| **Primary programs** | P1 computation–freshness; P2 harmful-write TTT; P3 planner–reflex co-design | Build complete low-cost versions now. |
| **Bounded foundational study** | P6 anytime-certified action refinement | Run in exact MDPs now; continue only if certificates are nonvacuous. |
| **Dependency-gated feasibility studies** | P4 replanning slack; P5 developmental plasticity; P7 information-triggered full-chunk TTT | Give each a small decisive pilot; do not scale unless its prerequisite passes. |

A sound portfolio does not promise that every idea becomes a paper. It gives every idea a cheap, valid chance to demonstrate its mechanism before substantial compute is spent.

### 1.1 What “AI instinct” means here

“AI instinct” is a research lens, not a claimed new field:

1. **Compiled fast behavior:** a cheap pathway acts while deliberation is unavailable or unfinished.
2. **Resource triggering:** a rule decides when more inference, observation, or validation is worth its cost.
3. **Controlled plasticity:** a rule decides what information should change online memory.

No experiment must contain all three. Each paper must stand on its domain-specific contribution.

### 1.2 Evidence ladder

| Level | Meaning | Example |
| --- | --- | --- |
| **E0 — implementation validation** | Known-answer and equivalence tests pass. | Batched oracle equals a naive fork. |
| **E1 — constructed mechanism** | A deliberately planted effect is recovered. | Oracle detects sign-flipped writes. |
| **E2 — in-domain pilot** | Effect appears on a nontrivial frozen benchmark. | Selective TTT improves one corruption stream. |
| **E3 — held-out generalization** | Prediction transfers to unseen speeds, tasks, streams, or disturbances. | Conditional freshness model transfers to unseen speed–latency pairs. |
| **E4 — deployment transfer** | Result survives a changed measured system. | Simulated-delay policy transfers to measured CPU/T4 latency. |

The existing exact P1 run and synthetic P2 oracle are E0–E1 instruments, not E3 findings.

---

## 2. Portfolio-level scientific rules

### 2.1 Required pre-run statements

Every run manifest must state:

- **Estimand:** exact quantity measured.
- **Primary hypothesis:** one falsifiable comparison with one primary metric.
- **Non-claim:** nearby conclusion the experiment cannot support.

Every outcome is:

- **GO:** mechanism and practical-effect criteria pass.
- **STOP:** experiment could detect the target effect, but the effect is too small or a required baseline wins.
- **INCONCLUSIVE:** target was not identifiable, benchmark was flat, intervals were too wide, a baseline failed, or invariants failed.

Never turn INCONCLUSIVE into PASS or KILL.

### 2.2 Baseline discipline

For every comparison:

1. Match information available to each method.
2. Match deployable compute as well as nominal action/update counts.
3. Report wall-clock, peak memory, FLOPs or documented proxy, and task quality.
4. Include a simple baseline capable of showing whether the method merely plans, updates, or replans less.
5. Run headline baselines through the same optimized path. Slow references are for correctness only.

### 2.3 Statistical discipline

- Use common random numbers for paired simulated counterfactuals.
- Separate pilot seeds from locked evaluation seeds.
- Use one primary endpoint per proposal; other analyses are secondary.
- Report paired differences and confidence intervals in natural units, not unstable ratios.
- Stochastic T4 pilots need at least three training seeds; use five for a close decisive result. Use many paired evaluation episodes per seed.
- Cheap CPU experiments should use exact evaluation or 10–30 seeds.
- Do not give deterministic exact rows an invented near-zero Gaussian noise model. Use return-unit error, regret, or explicitly quantized description length.
- Freeze test split, seeds, and thresholds before decisive runs. Pilot results are not headline estimates.
- Use validation for model selection and one untouched final test set. An in-process use counter is not a persistent holdout ledger.

### 2.4 Failure taxonomy

| Code | Meaning | Required response |
| --- | --- | --- |
| **F0 code/invariant** | Shape error, leakage, divergence, reference mismatch. | Fix and rerun controls; no scientific interpretation. |
| **F1 unidentifiable design** | Flat target, structurally collapsed axes, contaminated probes, too few events. | Redesign experiment; do not tune method. |
| **F2 estimator failure** | Oracle/proxy misses exact or planted effect. | Repair or abandon estimator first. |
| **F3 mechanism absent** | Valid experiment finds no meaningful effect. | Stop or narrow claim. |
| **F4 baseline dominance** | Simpler matched method wins. | Stop method; preserve measurement result. |
| **F5 system failure** | Mechanism works but overhead/memory erases gain. | Reframe or optimize only a profiled removable bottleneck. |

### 2.5 Agent operating protocol

An assigned AI agent must:

1. Read this specification and `decision_log.md`.
2. Run smoke configuration before changing scientific code.
3. Check naive/reference paths when changing optimized code.
4. Add a known-answer regression test for every conclusion-changing bug.
5. Never change thresholds after viewing test results; create a new protocol version and test split.
6. Record command, git SHA, dirty status, config hash, dependencies, device, seeds, time, and memory.
7. Stop on NaN/Inf, unmatched compute, missing arms, exhausted holdouts, or failed invariants.
8. Produce GO, STOP, or INCONCLUSIVE with evidence level and reason.
9. Never describe unit tests or planted synthetic results as discoveries.
10. Finish the current stage gate before optimizing later proposals.

---

## 3. Compute and engineering contract

### 3.1 Resource envelope

Free Colab T4 sessions are interruptible. Jobs must be resumable rather than relying on session length.

| Run class | CPU limit | T4 limit | Purpose |
| --- | --- | --- | --- |
| Smoke | 15 min | 20 min | Imports, shapes, tiny artifact |
| Mechanism pilot | 2 CPU-hours | 2 GPU-hours | Establish effect and estimate variance |
| Decisive run | 12 CPU-hours, shardable | 6–8 GPU-hours per seed | Frozen evaluation after pilot |
| Pre-review cap | — | 24 T4-hours per proposal | Prevent open-ended tuning |

P4, P5, and P7 cannot use decisive-run compute before their gates pass.

### 3.2 T4 defaults

- Prefer FP16 autocast with FP32 master weights; do not assume BF16.
- Keep at least 2 GB VRAM headroom; target measured peak below 13.5 GB.
- Gradient accumulation must preserve intended update/chunk semantics.
- Benchmark eager before `torch.compile`; small mutable kernels may not benefit.
- Keep measured kernels fixed-shape. Content decisions should select into a fixed buffer or mask.
- Profile scoring, synchronization, transfers, queue delay, validation, rollback, and checkpointing.
- Checkpoint every 10–20 minutes and after each shard.
- Avoid training 1B+ models. Use small pretrained backbones or 1–100M-parameter models with small mutable modules.

### 3.3 Repository interface

Required commands:

```text
instinct validate --proposal pN --config CONFIG
instinct run      --proposal pN --config CONFIG --output RUN_DIR
instinct resume   --run RUN_DIR
instinct report   --run RUN_DIR
instinct compare  --runs RUN_A RUN_B ...
```

Required layout:

```text
configs/p1/.../smoke.yaml cpu.yaml t4.yaml decisive.yaml
configs/p2/...                    ...
src/instinct/core/
src/instinct/p1_atlas/
src/instinct/p2_harmful_write/
src/instinct/p3_codesign/
src/instinct/p4_slack/
src/instinct/p5_development/
src/instinct/p6_certification/
src/instinct/p7_chunking/
tests/
results/<proposal>/<run_id>/
  manifest.json
  config.lock.yaml
  metrics.parquet
  events.jsonl
  profile.json
  verdict.json
  report.md
```

The manifest includes git SHA, dirty status, config hash, dependency versions, device, seeds, times, and parent run. The verdict lists every prerequisite check.

### 3.4 Shared tests

- deterministic RNG and common-random-number pairing;
- fast/naive equivalence;
- rejection of unmatched compute;
- no train/validation/final-test overlap;
- finite-use probe ledger persistent across resumes;
- negative and planted positive controls;
- report counts and units agree with raw data;
- CLI smoke test per proposal;
- serialization/resume equivalence.

---

# 4. Proposal 1 — Conditional computation–freshness atlas

## 4.1 Question and hypothesis

Can a compact conditional model predict the useful planning budget at unseen combinations of environment speed, measured planning latency, state, and reflex?

This is TTI/real-time RL measurement, not a new gate.

**Primary hypothesis:** a model conditioned on measured delay plus cheap environment/reflex descriptors predicts held-out budget frontiers with lower budget regret than a fixed budget, a raw budget-only curve, product collapse, and nearest-neighbour lookup.

**Non-claim:** one universal exponential law. Failure of a raw curve to remain unchanged as speed changes is expected and does not kill the proposal.

## 4.2 Correct estimand

For state s, reflex mu, planning budget k, environment rate nu_e, and runtime condition h:

\[
\Sigma(s,k,\nu_e,h,\mu)
= [J_{actual}(s,k)-C_{hw}(k,h)]
- [J_{actual}(s,k_0)-C_{hw}(k_0,h)].
\]

Use the named identity

\[
\Sigma = G_{plan}+R_{intermediate}
-L_{arrival}-L_{wait}-C_{hw}
+L_{base-delay}+\varepsilon_{id}.
\]

- `G_plan`: better instantaneous decision from k versus base k0.
- `R_intermediate`: reward earned by the reflex during additional waiting.
- `L_arrival`: fresh-versus-stale action value at the same arrival state/time.
- `L_wait`: remaining opportunity/discount cost.
- `L_base-delay`: delay cost already paid by the base budget; never call this noise.
- `epsilon_id`: true reconstruction residual.

Measure irreversibility separately with a matched-time counterfactual or reachability-failure probability. A signed handoff-value difference is not automatically “damage.”

## 4.3 Competing models

- **M0:** global fixed budget.
- **M1:** budget-only curve f(k).
- **M2:** product-collapse model f(k, delta), delta = nu_e times measured planning time.
- **M3:** separate-axis model f(k, nu_e, h).
- **M4:** conditional model f(k, nu_e, h, state descriptors, reflex descriptors).
- **M5:** nearest-neighbour/environment lookup.

M2 versus M3 is identifiable only if hardware changes something beyond an algebraic product: fixed overhead, batching, planner quality at equal wall time, numerical precision, cost/energy, or latency variability. If speed and latency enter only through their product, report “collapse imposed by construction.”

## 4.4 P1.0 — CPU instrument validation

Use two exact tabular MDPs and one deliberately non-collapsing synthetic control.

1. Validate all counterfactual arms against direct simulation.
2. Confirm identity residual is numerical precision.
3. Construct one product-collapse surface and one surface with a separate hardware-quality term.
4. Confirm conditional transfer passes/fails in the correct direction.
5. Confirm flat frontiers return INCONCLUSIVE.

**Gate:** every known-answer test passes and transfer never treats raw-budget invariance as the law.  
**Compute:** CPU, under 30 minutes.

## 4.5 P1.1 — sampled real-time pilot

Use three small families:

- moving-target gridworld/chase;
- Snake or Tetris-lite;
- a continuous/discretized control task with delayed intervention and both recoverable and unrecoverable regions.

Per environment:

- at least three speeds;
- at least three latency regimes, including measured CPU/T4 planning where applicable;
- 8–12 budgets spanning useful, flat, and harmful regions;
- hold/safe, greedy/myopic, and planner-distilled or learned reflexes;
- paired rollouts with common random numbers;
- enough episodes to narrow budget-regret uncertainty below the frozen practical tolerance.

Share maximum-budget planner prefixes only when exact for that planner.

## 4.6 Transfer protocol

Split whole contexts, not individual budget points.

1. Train on some speed–latency pairs; test unseen combinations.
2. Hold out states/episodes used for descriptor evaluation.
3. Fit on two reflexes; predict a third using descriptors.
4. Optionally fit two environment families and test a third.
5. Calibrate simulated/CPU latency and test measured T4, or vice versa.

Primary metric: paired budget regret in return units. Secondary: Kendall tau, near-optimal budget rate, calibration, wall-clock, and decomposition.

A near-optimal budget lies within max(2 evaluation SE, 10% of target frontier range) of optimum. Freeze this before final evaluation.

## 4.7 Baselines

- smallest/largest fixed budgets;
- best fixed budget on training contexts;
- state-independent schedule by speed;
- raw budget curve copied across conditions;
- product-collapse model;
- nearest-neighbour lookup;
- when feasible, a lightweight Finding-the-Time-to-Think-style gate.

## 4.8 Verdict and failure handling

**GO:** M4 meaningfully lowers held-out regret versus M0/M2/M5 and its upper confidence bound is below 20% of the informative target range.

**NARROW:** transfer works within environments but not across families; publish only an in-domain diagnostic.

**STOP law claim:** lookup dominates on untouched contexts or descriptors do not improve fixed budgets.

**INCONCLUSIVE:** most frontiers are flat, axes are structurally collapsed, or planner quality does not change across budgets.

Recovery:

- Flat task: deepen/weaken planner so budget can matter; do not count flat task as evidence.
- Delay staircase: use continuous/event time or randomized phase offsets.
- Reflex reorders surface: condition on reflex; do not call it fatal.
- Family fails within-cell: use isotonic/spline/conditional model; do not let bad representation create a transfer verdict.
- Speed/hardware identical: add real timing or a separate cost/quality mechanism.

## 4.9 Deliverables

- `results/p1/atlas.parquet` with explicit `curve_id`;
- split and fitted-model registry;
- context-paired transfer report;
- collapse-identifiability report;
- return-unit plots with uncertainty;
- E0–E4 verdict.

---

# 5. Proposal 2 — Harmful-write TTT

## 5.1 Question and novelty boundary

Does knowledge of long-horizon retention effects buy a better retention–plasticity trade-off than updating less, periodic reset, elastic anchoring, or extra dense adaptation at the same compute?

Generic write gating is occupied by AURA; VANE already performs reversible future-validated TTT for VLA policies; FSM and other systems address forgetting. The residual is multi-horizon retention quality in a mutable fast-weight memory.

**Primary hypothesis:** at exactly matched admitted-write count, an offline counterfactual oracle improves the retention–plasticity frontier over surprise, periodic, and random selection. A deployable method is attempted only if this ceiling exists.

**Non-claim:** high AUC on planted sign flips proves only an E1 instrumentation check.

## 5.2 Utility

\[
U_t = \sum_{h\in H}w_h
[L(W_t;B_{t+h})-L(W_t+\Delta W_t;B_{t+h})]
-\lambda E_{z\sim A_t}
[L(W_t+\Delta W_t;z)-L(W_t;z)]-c_t.
\]

Report benefit, damage, and cost separately. Sweep lambda to trace a frontier. The future-looking oracle is offline analysis, not a trigger.

## 5.3 Corrections required first

1. Restrict candidates to replay window before top-m; assert exactly m writes.
2. Add naive references for frontier and decision logic.
3. Separate oracle probes, gate validation streams, and final downstream evaluation.
4. Persist probe use across resumes.
5. Report positive-event counts and AUC intervals.
6. Measure ranking staleness caused by moving away from the write-all trajectory.

## 5.4 P2.0 — E0/E1 synthetic validation

Use:

- sparse sign-flipped drifting regression;
- associative key collision;
- benign high-surprise novelty;
- dense mixed-task drift where “harmful” is intentionally ambiguous.

Controls:

- no planted harm;
- randomized harm labels;
- zero learning rate;
- retention-probe count sweep;
- batched fork equals clone/apply/evaluate/rollback;
- unstable learning rate raises on divergence.

This stage validates measurement only.

## 5.5 P2.1 — oracle frontier

At each write budget m replay:

- offline oracle;
- surprise/current loss;
- update norm;
- gradient alignment;
- periodic;
- random with many paired replicates;
- dense/all;
- no writes;
- elastic anchoring;
- periodic reset.

Use disjoint retention and fresh/plasticity evaluation and report Pareto dominance.

Compare:

- **frozen-reference oracle:** utilities from write-all trajectory;
- **iteratively rescored oracle:** recompute after selected blocks.

Their gap measures policy-induced oracle staleness.

## 5.6 P2.2 — one realistic T4-feasible benchmark

Preferred route: small pretrained ResNet/ViT on a long CIFAR-10-C or CIFAR-100-C changing-corruption stream, adapting only normalization, adapters, or a small fast-weight head.

Alternative: 5–50M sequence model on associative recall plus a small natural-language stream, with nonlinear fast weights and delayed queries.

Primary metric is downstream accuracy/recall. Aged reconstruction loss is acceptable only after showing correlation with downstream retention.

## 5.7 Phase-two decision

Use whole held-out streams, not random candidate splits:

- predictable pre-write utility -> train a small gate;
- not predictable before, but independent post-write audit predicts utility -> transactional validation outside already occupied VLA claims;
- damage low-dimensional and projected updates directly recover utility -> projection;
- none -> stop intervention and retain measurement result.

Spectral concentration alone never selects projection.

## 5.8 Mandatory compute comparisons

1. **Matched write count:** exactly m commits; tests selection quality.
2. **Matched deployable compute:** include features, shadow forwards, audits, rollback, and synchronization. Give dense adaptation extra steps, larger batches, or higher frequency using the same budget.

Report offline oracle cost separately.

## 5.9 Verdict

**GO to gate:** oracle improves untouched realistic frontier and pre-write features transfer across streams.

**GO to transaction:** disjoint post-write signal predicts future utility and overhead is smaller than recovered loss.

**GO to projection:** projected updates directly retain a frozen fraction of benefit while lowering damage.

**STOP:** reset or dense matched-compute adaptation matches oracle-guided selection, or aged loss fails to predict downstream retention.

**INCONCLUSIVE:** harmful events too rare, probe mixture changes target meaning, or memory diverges.

## 5.10 Compute and deliverables

- Synthetic oracle/frontier: CPU/T4 under 2 hours.
- Realistic pilot: under 2 T4-hours per seed.
- Decisive: 3–5 seeds and no more than 24 T4-hours before review.
- Deliver oracle table, matched frontier, oracle-staleness analysis, feature-transfer report, compute ledger, and branch verdict.

---

# 6. Proposal 3 — Planner–reflex co-design

## 6.1 Question and hypothesis

Can the fast policy acting during deliberation improve the state in which the slow planner’s result arrives, rather than merely imitate the planner or preserve options abstractly?

This is the portfolio’s most direct “instinctive behavior” proposal.

**Primary hypothesis:** a planner-aware reflex improves real-time return over hold, myopic greedy, planner-distilled, and reflex-only-trained policies across a band of delays without collapsing to inactivity.

**Non-claim:** monotonic improvement under arbitrary alternating optimization, or novelty for generic fast/slow architectures.

## 6.2 Joint objective

For delay d(k,h):

\[
J(\phi;k)=E\left[
\sum_{i=0}^{d-1}\gamma^i r(S_i,\mu_\phi(S_i))
+\gamma^d Q_\Pi(S_d,A_\Pi(S_0,k,\mu_\phi))
\right].
\]

The planner must simulate the same reflex that executes. Planner-model/deployed-reflex mismatch is a baseline and diagnostic.

## 6.3 P3.0 — exact tabular degeneracy test

Build small MDPs where:

- hold is optimal;
- greedy progress is optimal;
- a genuine interior reflex improves planner arrival;
- a bad reflex irreversibly destroys a trajectory.

Map the full reflex–budget objective by enumeration or policy iteration. Confirm the proposed optimizer recovers each case. If the objective cannot represent a useful interior policy in a constructed environment, stop and repair the objective before neural work.

## 6.4 P3.1 — counterfactual-label training

Preferred first optimizer:

1. Branch over reflex actions with common random numbers.
2. Execute the reflex prefix while holding pending planner computation fixed.
3. Estimate full option return per branch.
4. Imitate the best action or regress action advantages.
5. Refresh planner simulations after a fixed reflex-update block.

Run:

- reflex trained for immediate reward only;
- optionality/recovery-only reflex;
- planner-aware joint objective.

The optionality-only arm is a freeze-degeneracy control.

## 6.5 P3.2 — alternating co-design

Only after P3.1 wins:

- improve reflex against frozen planner model;
- refresh planner rollouts/model of reflex;
- update any budget gate only in an outer loop;
- accept blocks only if paired joint-objective validation does not regress beyond tolerance.

Log cross-play between planner version i and reflex version j. This separates coordination from fragile co-adaptation.

## 6.6 Environments, metrics, and baselines

Reuse P1 exact tabular/gridworld environments; add one small arcade environment only after the objective passes.

Primary metric: paired real-time return on held-out speed–latency combinations.

Secondary:

- intermediate reward;
- arrival-state planner value;
- reachability/failure rate;
- hold-action fraction;
- planner–reflex mismatch;
- shift in optimal planning budget;
- cross-play matrix.

Baselines:

- hold/no-op, myopic greedy, random;
- planner-distilled reflex;
- reflex trained without planner term;
- planner-aware reflex with shuffled planner values;
- planner adaptation to a fixed reflex;
- equal-compute policy gradient on the smallest task.

## 6.7 Verdict and recovery

**GO:** joint reflex beats progress-only and hold/distilled endpoints over multiple delays and unseen speed–latency pairs, without excessive inactivity.

**NARROW:** it works only with one planner; report co-adaptation, not general architecture.

**STOP:** all solutions lie on a simple hold–greedy interpolation or distillation matches the method.

Recovery:

- Freeze collapse -> constrain minimum progress; do not endlessly tune optionality weight.
- Planner nonstationarity -> slower alternating blocks and cross-play validation.
- High label variance -> more paired rollouts or exact small-state targets.
- One-delay-only improvement -> train on delay distribution or condition reflex on remaining planner time.

## 6.8 Compute and deliverables

P3.0 is CPU-only. P3.1 should fit within 2–6 T4-hours per seed. Deliver objective maps, policies, cross-play matrix, matched-return report, and P1 frontier before/after reflex learning.

---

# 7. Proposal 4 — Prefix recovery-slack training

## 7.1 Scope correction

This is not initially a VLA-scale robotics project or safety certificate. Test:

> Does prefix-wise recovery-value training improve recovery under held-out disturbances beyond shorter chunks or more frequent verification?

**Primary hypothesis:** a recovery-value regularizer improves the success–progress frontier under held-out disturbance types at matched replanning cost.

## 7.2 Experimental object

For chunk a(0:H-1) and disturbed prefix state:

\[
R_j(s)=Quantile_{q,\delta\sim D}
[V_{replan}(s_j^\delta)-V_{fail}].
\]

Train:

\[
L=L_{task}+\beta\sum_{j=1}^{H}[m-R_j(s)]_+.
\]

Use a lower quantile for the pilot. Worst-case optimization is likely to produce inactivity.

## 7.3 Stages

**P4.0 exact-label pilot:** a 2D navigation/manipulation surrogate with obstacles, inertia, delayed replanning, and disturbances. Obtain exact/high-quality replanning values. Verify an interior progress–recoverability frontier exists.

**P4.1 learned surrogate:** train and calibrate V_replan against exact labels. Freeze before policy training.

**P4.2 chunk training:** train task-only and slack-regularized policies on some disturbances; test both new magnitudes and qualitatively held-out disturbance types/dynamics.

Do not enter a large robotics stack unless P4.2 passes.

## 7.4 Baselines

- task-only chunks;
- shorter chunks at matched policy invocations;
- periodic replanning;
- PACE-like low-speed boundary heuristic where applicable;
- post-hoc verifier using the same V_replan;
- entropy/viable-action regularization;
- recovery-trajectory training if demonstrations exist.

## 7.5 Metrics and verdict

Primary: held-out-disturbance success at matched average replanning calls. Secondary: clean progress, failure rate, chunk length, interventions, recovery value, and wall-clock.

**GO:** slack training dominates task-only and short-chunk baselines on held-out disturbance types without large clean-task loss.

**STOP:** frontier is degenerate, verification gets the same gain more cheaply, or benefits disappear out of training distribution.

**Fallback:** if the learned surrogate fails, retain an exact-label analysis and stop the neural claim. If inactivity dominates, use a constrained minimum-progress formulation rather than indefinite beta tuning.

## 7.6 Compute cap

CPU for P4.0; at most 6 total T4-hours for P4.1–P4.2 before review. The output is a mechanism study or a decision not to proceed, not a full VLA comparison.

---

# 8. Proposal 5 — Developmental initialization-plus-plasticity

## 8.1 Scope correction

Width-agnostic weight generation is occupied by NNiT and adjacent weight-space/hypernetwork work. The remaining hypothesis is:

> Can one fixed-length coordinate-based program generate both initialization and a local deployment-time plasticity rule across unseen network sizes and unseen task combinations, without target trained weights?

This is the most literal instinct proposal and the highest risk.

**Primary hypothesis:** fixed-size developmental code plus local updates improves zero-shot-to-few-shot adaptation on jointly held-out task/architecture combinations over parameter-count-matched initialization and meta-learning baselines.

## 8.2 Model

Use normalized neuron/layer coordinates and type embeddings. Generator G_theta emits:

- initial weight/connectivity statistics;
- optional sparsity mask;
- local learning-rate coefficients;
- small local update rule.

A feasible rule family:

\[
\Delta w_{ij}=\eta_{ij}F_\theta(x_i,y_j,e_\ell,t,coordinates).
\]

Pre/post activity and layer-local/broadcast error may be used. In the strict local arm, deployment cannot access target trained weights or full backpropagated gradients.

Count generator parameters, coordinate embeddings, architecture tokens, task conditioning, and broadcast modules in code length.

## 8.3 Stages

**P5.0 symmetry/identifiability:** show consistent functions under width change; include neuron-permutation negative controls.

**P5.1 initialization only:** meta-train small regression/classification families and widths; evaluate unseen width/depth. This validates generator but is not novel alone.

**P5.2 plasticity only:** conventional shared initialization; learn local rule; test unseen task combinations and longer adaptation.

**P5.3 joint program:** emit initialization and plasticity. Hold out Cartesian task–architecture combinations, not merely one axis.

Low-cost tasks:

- sinusoid/polynomial regression compositions;
- parity, modular arithmetic, Boolean rules;
- MNIST/Fashion-MNIST as secondary modality;
- small control only after earlier stages.

## 8.4 Baselines

- standard/orthogonal initialization;
- width-aware and muP-style parameterization;
- first-order MAML/Reptile;
- Meta-SGD/per-parameter learning rates;
- size-conditioned hypernetwork/GHN;
- weight generator without plasticity;
- conventional backprop with same examples/steps;
- scale-matched random local rule.

NNiT is conceptually required, but a full diffusion reproduction is not mandatory on T4. If no lightweight official checkpoint is comparable, use a size-conditioned generator and explicitly avoid claiming superiority to NNiT.

## 8.5 Metrics and verdict

- zero-shot loss/accuracy;
- area under first 1–50 adaptation steps;
- jointly held-out task–architecture performance;
- code parameter count/compressed bytes;
- instantiated network size;
- local-rule versus backprop compute;
- permutation/coordinate sensitivity.

**GO:** joint program improves early adaptation on joint holdouts over both initialization-only and plasticity-only, not just random initialization.

**NARROW:** initialization transfers but local plasticity does not.

**STOP:** gains vanish after counting generator parameters, simple meta-initialization matches, or fixed neuron ordering is required.

Recovery: diagnose initialization and plasticity separately before increasing capacity. If meta-gradients are unstable, shorten horizons/use truncated or implicit gradients and bias-check against full unrolling on tiny tasks.

## 8.6 Compute cap

P5.0–P5.2: at most 8 total T4-hours. P5.3 scales only after clear cross-size evidence. A toy positive may support a workshop/foundational result, not automatically a high-impact architecture claim.

---

# 9. Proposal 6 — Anytime-valid action-prefix certification

## 9.1 Question and hypothesis

Can an agent adaptively stop computation and select an action prefix while maintaining valid sequential trajectory-risk bounds across exits, candidate prefixes, and repeated episode decisions?

The novelty is not generic conformal early exit. It is simultaneous action-prefix risk under dynamics.

**Primary hypothesis:** an anytime-valid simultaneous certificate achieves target episode violation rate under adaptive selection while executing more useful actions than a fixed worst-case/union-bound baseline.

## 9.2 Formal target

At compute exit ell and prefix p, estimate upper confidence process U(ell,p). Execute the longest deployable prefix satisfying U(ell,p) <= alpha_t, the remaining episode risk budget.

Handle:

1. optional stopping over exits;
2. selection among prefixes;
3. repeated episode decisions.

Use simultaneous confidence sequences/e-processes for the first two and preregistered alpha spending or an episode-level e-process for the third.

## 9.3 Stages

**P6.0 statistical unit problem:** bounded/Bernoulli trajectory costs with known truth. Confirm finite-sample coverage under adversarial stopping and prefix selection.

**P6.1 exact MDP:** the method sees Monte Carlo rollouts; exact dynamic programming supplies hidden truth only for evaluation.

**P6.2 model mismatch:** use learned/perturbed dynamics. Separately report validity relative to model risk and true environment risk.

## 9.4 Baselines and metrics

Baselines:

- raw uncertainty threshold;
- fixed-sample Hoeffding/Bernstein;
- Bonferroni/union bound;
- ordinary split conformal/CRC without anytime correction;
- nested prediction-set analogue;
- always abstain and always execute.

Primary: episode-level violation probability with interval over independent episodes. Secondary: return, abstention, prefix length, exits used, bound width, and shift failures.

Exact truth verifies coverage but is never input. Include stopping rules designed to break fixed-time intervals.

## 9.5 Verdict

**GO:** coverage holds and method executes materially more/longer than conservative union bound at similar return.

**NARROW to theory:** validity holds but bounds are vacuous; characterize why.

**STOP deployment claim:** model shift breaks true-environment validity.

**INCONCLUSIVE:** too few violations; construct risk levels near target instead of multiplying trivial episodes.

## 9.6 Compute and deliverables

CPU-only through P6.1. Deliver assumptions/theorem notes, calibration code, exact-risk benchmark, coverage tables, optional-stopping negative control, and nonvacuity curves.

---

# 10. Proposal 7 — Information-triggered fixed-shape TTT updates

## 10.1 Dependency and novelty

Start only if P2 shows:

1. update content matters at matched write/token count;
2. useful information is sparse enough for selection to improve a realistic stream.

LaCT, TNT, surprisal-routed residual caches, support-vector selection, and mutable-state serving occupy much of the area. The residual question:

> Can semantic admission choose content for a fixed-shape large update while preserving end-to-end T4 utilization and delayed recall of rejected information?

## 10.2 System

1. Score fixed-size incoming microchunks cheaply.
2. Admit selected items into token buffer.
3. Include low-rate random audit channel.
4. At exactly B tokens or maximum-age deadline, form one fixed-shape update.
5. If deadline fires early, apply documented padding/reservoir policy and charge its cost.

Semantic sparsity changes content, not measured kernel shape.

## 10.3 P7.0 — queue/kernel feasibility

Replay score traces through queue simulation. Sweep admission rate, burstiness, B, and deadline. Measure update frequency, token age, occupancy, padding, scorer overhead, and throughput.

On T4, compare fixed full chunks, masked fixed chunks, and variable microchunks. If scorer plus synchronization removes the large-chunk advantage, stop before task training.

## 10.4 P7.1/P7.2

**P7.1 constructed memory:** delayed associative recall with common items, rare critical items, distractors, and queries specifically targeting rejected microchunks. Match processed tokens and update compute.

**P7.2 one suitable modality:** prefer image-set, multi-view, or video-like data where selection does not violate autoregressive causality. Use a small T4-compatible model. For language, preserve token-order dependencies through a local path rather than dropping tokens blindly.

## 10.5 Baselines

- uniform contiguous B-token chunks;
- random admission;
- surprise admission;
- FIFO/reservoir;
- equal-token microchunk updates;
- dense fixed-rate updates;
- lightweight global/local hierarchical memory;
- external residual cache when feasible.

Full TNT/LaCT/FSM reproduction is not mandatory if incompatible with T4, but untested architectural claims must be explicit.

## 10.6 Metrics and verdict

Primary: task performance at matched measured wall-clock or matched update compute. Secondary: rejected-item recall, audit-estimated missed utility, update latency, throughput, utilization, VRAM, and selection bias.

**GO:** semantic admission beats uniform/random without losing fixed-shape throughput and rejected-item recall stays within frozen tolerance.

**STOP:** buffer delay erases gains, scoring removes throughput gain, uniform wins at equal tokens, or rejected-item queries fail systematically.

**Fallback:** quality without throughput becomes a memory-quality method; throughput with failed coverage requires a small exact/residual memory and becomes a hybrid architecture.

## 10.7 Compute cap

P7.0 under 2 T4-hours; P7.1 under 4. P7.2 starts only after both pass and must profile the actual T4 used for the claim.

---

# 11. Cross-proposal dependency graph and execution order

These proposals are a portfolio, not seven independent commitments. Results should flow between them:

- **P1 → P3:** P1 supplies measured latency/freshness regimes in which planner–reflex co-design is worth testing.
- **P1 → P2/P4:** P1 supplies realistic delay prices; P2 and P4 must count validation or replanning latency as part of the intervention.
- **P2 → P7:** P7 is forbidden unless P2 establishes that useful adaptation content is sparse at matched update count and compute.
- **P3 ↔ P4:** P3 studies the reflex while a decision is pending; P4 studies whether training makes useful new decisions arrive sooner. They may share environments and instrumentation but must retain separate hypotheses.
- **P4 → P6:** P4's time-to-useful-action estimates can become empirical inputs to P6, but P6's first validity result must remain exact and CPU-only.
- **P5:** independent high-risk branch. It must not consume the budget released by a failed primary proposal without a new decision.

## Stage A — Repair and freeze the experimental substrate

1. Put every run behind a single CLI/config interface.
2. Freeze seeds, environment versions, task splits, metrics, and compute accounting.
3. Add known-answer and negative-control tests before real sweeps.
4. Correct P1's transfer target and decomposition labels.
5. Correct P2's top-m/window ordering and add mandatory compute-matched baselines.
6. Require a manifest, resumable checkpoints, and an automatically generated run summary.

**Stage-A exit:** every proposal has a CPU smoke test; P1 and P2 reproduce their known-answer controls; no metric depends on post-hoc manual aggregation.

## Stage B — Run only the cheap falsification gates

Recommended order:

1. **P1.0:** exact atlas/decomposition validation.
2. **P2.0–P2.1:** synthetic isolation and replay-ordering correctness.
3. **P3.0:** exact tabular degeneracy test.
4. **P6.0–P6.1:** exact confidence-sequence validity and nonvacuity.
5. **P4.0:** trace/replay latency model.
6. **P5.0–P5.1:** task-family audit and developmental-program sanity check.
7. **P7.0:** only using P2 traces and only after the P2 dependency passes.

No learned T4 experiment starts before its associated cheap gate returns GO or a documented NARROW verdict.

## Stage C — T4 pilots

Priority order under a free-Colab budget:

1. P1 empirical latency perturbation.
2. P2 small-real-stream transaction policy.
3. P3 learned reflex on one stochastic environment.
4. P4 toy slack regularization, if P4.0 identifies a measurable mechanism.
5. P5 small topology holdout, if the oracle gap is nontrivial.
6. P7 constructed memory, if P2 and P7.0 pass.

P6 remains CPU-first. Larger learned refinements are optional, not required for its core result.

## Stage D — Select at most two paper-scale directions

Choose using preregistered evidence, not thematic preference. A direction should normally satisfy all of:

1. passes its mechanism gate on held-out seeds/tasks;
2. survives the strongest cheap baseline at matched resources;
3. has an effect larger than measurement noise and seed variance;
4. retains a novelty claim after the literature comparison;
5. can be completed within one additional free-Colab-scale experimental cycle.

If more than two qualify, prefer complementary risk: one robust empirical result and one higher-upside model/theory result.

---

# 12. Portfolio assessment, resource caps, and stopping budgets

| Proposal | Feasibility on CPU/T4 | Upside if positive | Principal risk | Maximum pre-decision budget | Current recommendation |
|---|---:|---:|---|---:|---|
| P1 computation–freshness atlas | High after correction | Medium–high empirical systems result | artificial exact-arm invariances mistaken for a law | 12 CPU-hours + 12 T4-hours | **Primary; continue after repair** |
| P2 harmful-write TTT | High for replay/small models | High model-level result | validation overhead or future-evidence leakage | 24 T4-hours total through first real stream | **Primary; strongest architectural candidate** |
| P3 planner–reflex co-design | High tabular, medium learned | Medium–high agent result | optimal reflex collapses to unconditional fallback | 18 T4-hours after P3.0 | **Primary if degeneracy test passes** |
| P4 replanning-slack training | High as toy study | Medium mechanism result | proxy gaming and trivial early exits | 6 T4-hours | **Bounded mechanism study** |
| P5 developmental plasticity programs | Medium toy, low scale | High conceptual upside | oracle gap vanishes; crowded weight-generation prior art | 8 T4-hours | **High risk; strict gate** |
| P6 anytime-certified action refinement | High exact/CPU | Medium theoretical result | valid but vacuous bounds | 24 CPU-hours before any learned extension | **Good independent theory track** |
| P7 information-triggered full-chunk TTT | Medium and dependency-gated | Medium hardware/model result | scorer/queue overhead destroys throughput | 2 T4-hours profiling + 4 T4-hours constructed task | **Do not start before P2 evidence** |

These are cumulative ceilings, not targets. Stop earlier when a frozen condition fires. Exceeding a ceiling requires a one-page continuation memo containing the failed assumption, new evidence, revised hypothesis, and exact extra budget.

## Portfolio-level stop rules

Stop or narrow the full program if any two of the following are observed across primary proposals:

- effects disappear under compute- or latency-matched baselines;
- success requires task-specific thresholds that do not transfer;
- claimed compute savings do not appear in wall-clock measurements on the T4;
- results depend on privileged future information unavailable at deployment;
- novelty reduces to a renamed router, early exit, cache, or ensemble without a new guarantee or trade-off;
- seed/task uncertainty is the same order as the reported benefit.

This is not a rule to abandon negative results. A clean impossibility, degeneracy, calibration failure, or hardware bottleneck can be reportable if it is demonstrated across representative conditions and changes how the field should evaluate the method.

---

# 13. Required run report and handoff format

Every agent must produce the following before declaring a milestone complete:

1. **Question:** one falsifiable sentence.
2. **Status:** GO, NARROW, STOP, or INCONCLUSIVE.
3. **Commit/config:** commit hash, config path, environment manifest, seed list, and data/task version.
4. **Resources:** CPU/GPU model, peak VRAM/RAM, wall-clock, training/update FLOPs when measurable, and number of interrupted/resumed sessions.
5. **Primary result:** preregistered metric with uncertainty and held-out result.
6. **Baselines:** which mandatory baselines ran and why any could not run.
7. **Controls:** known-answer, negative, leakage, and optional-stopping controls as applicable.
8. **Failure log:** OOMs, NaNs, timeouts, invalid cells, discarded runs, and deviations from config.
9. **Interpretation:** what the result establishes and what it does not establish.
10. **Decision:** exact next experiment or explicit termination.
11. **Artifacts:** machine-readable results, generated tables/plots, stdout/stderr, and a short README reproducing the verdict.

An agent may repair implementation errors without reopening a frozen scientific decision. Changing a primary metric, holdout, baseline, or stopping threshold after viewing results requires a new experiment version and must preserve the earlier report.

---

# 14. Literature and novelty anchors to check before claims

This is a minimum comparison set, not a substitute for a fresh search at writing time.

## P1: variable test-time computation and real-time decision delay

- [Finding the Time to Think: How Decision Latency Shapes Real-Time AI](https://arxiv.org/abs/2606.26463) directly studies variable decision time in real-time RL and includes cross-hardware transfer. P1 must differ through its controlled computation–freshness decomposition and transfer diagnostics, not merely by adding delay.
- [Adaptive Test-Time Compute Allocation](https://arxiv.org/abs/2604.14853) and [Conformal Thinking](https://arxiv.org/abs/2602.03814) occupy adaptive compute allocation and risk-controlled stopping. P1's contribution must be about delayed action value and conditional transfer.

## P2/P7: test-time learning, state transactions, and write selection

- [Test-Time Training Done Right (LaCT)](https://arxiv.org/abs/2505.23884) motivates large-chunk updates for hardware utilization; small frequent writes should not be assumed efficient.
- [VANE](https://arxiv.org/abs/2608.09448) is a direct novelty constraint: it isolates candidate VLA updates and commits them using subsequent evidence. P2 must establish a distinct transaction rule, unbiased deployable validator, guarantee, or better compute/evidence trade-off.
- [AURA](https://arxiv.org/abs/2606.02775), [FSM](https://arxiv.org/abs/2604.07350), and [In-Place TTT](https://arxiv.org/abs/2604.06169) cover action-gated, elastic, and efficient TTT variants.
- [RDumb](https://arxiv.org/abs/2306.05401) and [RDumb++](https://arxiv.org/abs/2601.15544) make reset/recovery baselines mandatory on nonstationary streams.
- [Forgetful Attention with Support-Vector Selection](https://arxiv.org/abs/2607.12204), [RW-TTT](https://arxiv.org/abs/2605.28053), and [Test-time Neuronal Training](https://arxiv.org/abs/2511.07343) constrain P7's claims about selection, mutable state, and chunked updates.
- [SR-TTT](https://arxiv.org/abs/2603.06642) is also a methodological warning: the paper's version history reports that earlier gains were affected by evaluation artifacts. All TTT proposals need state isolation and leakage tests.

## P3/P4: asynchronous action execution and replanning

- [Viability of Future Actions](https://arxiv.org/abs/2506.10871), [PACE](https://arxiv.org/abs/2606.00537), [CheckVLA](https://arxiv.org/abs/2607.26789), and [RePO-VLA](https://arxiv.org/abs/2605.09410) cover future-action viability, action-chunk execution, selective verification, and adaptive replanning. P3/P4 must compare against these neighboring intervention points and should not claim the generic fast/slow architecture as novel.

## P5: developmental priors and neural weight generation

- The [genomic bottleneck hypothesis](https://www.nature.com/articles/s41467-019-11786-6) and [Complex Computation from Developmental Priors](https://www.nature.com/articles/s41467-023-37980-1) motivate compressed developmental rules rather than learned endpoint weights.
- [NNiT](https://arxiv.org/abs/2603.00180) and [SANE](https://proceedings.mlr.press/v235/schurholt24a.html) make generic cross-architecture weight generation an occupied area. P5's defensible residual is joint generation of an initialization and a local plasticity program evaluated on unseen topology/task combinations.

## P6: anytime prediction and calibrated risk

- [Early-Exit Neural Networks with Nested Prediction Sets](https://proceedings.mlr.press/v244/jazbec24a.html), [SAFE-KD](https://arxiv.org/abs/2602.03043), and [CORA](https://arxiv.org/abs/2604.09155) cover calibrated early exit and risk control. P6 must contribute the action-specific conversion from interruptible refinement to sequential decision risk, with explicit assumptions and optional-stopping controls.

---

# 15. Final recommended portfolio

The most rational initial portfolio is:

1. **P1 as the measurement substrate**, but only after replacing unconditional transfer claims with conditional ones and adding a real empirical latency arm.
2. **P2 as the strongest model-level direction**, with VANE treated as close prior art and matched-compute/reset baselines treated as mandatory.
3. **P3 as the strongest agent-specific direction**, conditional on the exact tabular reflex being genuinely state- and latency-dependent.
4. **P6 as a cheap independent theory direction**, with nonvacuity given equal status to validity.
5. **P4 as a bounded mechanism test**, not a large VLA project.
6. **P5 as a capped high-risk bet**, only after an oracle-gap and topology-holdout gate.
7. **P7 as a dependent systems experiment**, not a standalone proposal until P2 demonstrates semantic sparsity.

This plan cannot guarantee a positive result. It is designed so that the most likely dead ends—invalid transfer laws, vacuous validators, reflex degeneracy, proxy gaming, missing oracle gaps, vacuous certificates, and nonexistent wall-clock savings—are tested cheaply before scarce T4 time is spent.
