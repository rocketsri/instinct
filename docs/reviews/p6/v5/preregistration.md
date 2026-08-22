# P6.5 realistic-simulator preregistration

`docs/reviews/p6/v5/continuation_memo.md` authorizes this stage (6 CPU-hours,
bringing the declared P6 total to 10 of 24) and specifies the discretization:
41 position bins over `[-1.0, 1.0]`, closed-form Gaussian transition
probabilities, no Monte Carlo in the ground truth. This document locks the
remaining choices before any rollout is drawn, including one correction to
the memo's own numbers, found by validating the discretization exactly the
way the memo itself prescribes ("what's lost" section): comparing closed-form
predictions against a high-sample Monte Carlo rollout of the real environment.

**Velocity-bin resolution, corrected from the memo's 25 to 61.** The memo
recommends 25 velocity bins (width 0.12) over `[-1.5, 1.5]`. At the nominal
parameters, one tick's velocity-noise standard deviation is `sqrt(dt) *
noise_std ~= 0.0139` -- about 8.6x smaller than that bin width. At that ratio,
the probability of a single tick landing outside the bin containing its mean
is negligible (about 8.66 standard deviations out), so the discretized chain
built at 25 bins behaves as a near-deterministic point process rather than a
diffusion: validating it against a 20,000-trajectory Monte Carlo rollout under
the schedule below showed the closed-form prefix risk diverging from the
empirical one by up to **13 percentage points** by tick 12 (`0.4326` exact vs.
`0.318` empirical). Widening to 61 bins (width ~= 0.049, ~3.5x the noise std)
reduces that gap to under 0.4 points at the same horizon and sample count,
while keeping the state count small (`41 * 61 + 1 = 2,502`). `velocity_bins:
61` in the config below reflects this correction, not the memo's original
number; see `src/instinct/p6_certification/realistic.py`'s module docstring
for the full validation numbers and `tests/test_p6_realistic.py` for the
regression test that would catch a future resolution regression.

## Frozen vs. new

Unchanged from P6.0-P6.4: `certificates.py::clopper_pearson_upper`/
`longest_certified`, `comparators.py::PrefixComparator.bounds`/`.snapshot`,
`mismatch.py::calibrate_shift_envelope`/`robust_snapshot`, and
`learned_dynamics.py::certify_prefixes`/`TransitionRiskModel` (reused
unmodified — the grid model is fit and sampled through the exact same class
P6.4 uses for its three-state chain). `confidence_delta: 0.05`,
`csa_bet_margin: 0.02` match P6.3/P6.4.

New, locked in `configs/p6/p6_5/cpu.yaml`:

- **One merged terminal state**, not the memo's two — explicitly permitted by
  the memo ("either is fine"); chosen so `TransitionRiskModel` (which
  hard-assumes one absorbing last state) is reusable unmodified.
- **`gamma: 0.97`** — a new free parameter this stage introduces (P6.0-P6.4's
  hazard chain is finite-horizon, undiscounted). Used only to run
  `TabularMDP.value_iteration()` once as a descriptive diagnostic
  (`optimal_value_at_reset`); it does not drive the certificate, which only
  ever sees Bernoulli rollout counts under one fixed `action_schedule`.
- **`action_schedule`**: 24 steps of `(drive, drive, hold)` repeated 8 times —
  a fixed, non-adaptive schedule under test, matching P6.0-P6.4's methodology
  of certifying a scripted prefix's cumulative risk rather than an adaptively
  controlled policy.
- **`sample_sizes: [512, 1024, 2048, 4096]`, `declared_looks: 4`** —
  `declared_looks` equals the number of sample sizes actually checked. This is
  a deliberate departure from P6.3/P6.4's `declared_looks: 8192`, which the
  v4 adversarial audit's B1 finding identified as a phantom exit schedule (the
  CSA specialization was credited a Ville guarantee against 8,192 exits no
  policy ever actually checked). Here the declared family is exactly the
  realized checks.
- **`repetitions: 64`, `pilot_trajectories: 4000`, `calibration_samples: 4096`**
  — sized to keep this stage comfortably CPU-bounded (realized wall-clock is
  well under the 6-hour cap; see the run's own timing metrics).
- **Independent-pair calibration coupling** — `model_calibration` and
  `env_calibration` (fed to `calibrate_shift_envelope`) are drawn with
  disjoint randomness (an independent NumPy generator for the model side, a
  fully separate `SeedScope` draw for the environment side), not the shared-
  uniform common-random-number coupling P6.2-P6.4 used. The v4 audit's B2
  finding is that CRN coupling minimizes discordance and is not available at
  deployment, where an agent cannot replay the environment's latent noise
  inside its own model. This stage uses the weaker, honest coupling.
- **`clustered_units` condition** (`clustered_unit_group_size: 8`) — an
  explicit exchangeability stress cell (v4 audit B4): environment-calibration
  rollouts are drawn with 8 trajectory indices sharing one underlying lane id,
  so they receive *identical* noise and the true number of independent noise
  roots is `calibration_samples / 8`, not `calibration_samples`. This
  condition is reported (`unique_noise_roots`, `effective_sample_fraction`)
  but is **not** part of the GO/STOP gate — it is descriptive evidence about
  whether validity survives a shrunk effective sample size, not a claim this
  stage is prepared to certify either way.

## What the certificate sees, and what stays evaluation-only

Exactly the P6.0-P6.4 discipline: `certify_prefixes` receives only Bernoulli
bad-event counts from `learned.sample_failures` (a model fit to real,
Gaussian-noise continuous rollouts of `InertialIntervention`, binned post-hoc
into grid states) and a sampled shift envelope. The closed-form discretized
`TabularMDP` — built once per condition via
`realistic.py::build_exact_mdp`, exact by construction, never sampled — is
used strictly for two post-selection evaluation quantities: `true_risk`
(never fed to the certificate) and `oracle_safe_fraction_recovered` (the
selected prefix as a fraction of the longest prefix the exact oracle itself
would certify at the same risk budget, per the v4 audit's B5 finding that
positive mean normalized prefix length alone does not establish useful
execution).

## Structural shift families

Four physically-motivated perturbations of `InertialIntervention`, all built
through the same closed-form discretization: `stable` (nominal), `increased_noise`
(`noise_std * 1.5`), `reduced_drag` (halves the gap between `drag` and 1,
i.e. less friction, harder to arrest momentum), `narrower_margin` (`boundary *
0.85`, less room for error). Plus the descriptive `clustered_units` cell
above.

## Verdict rule (stated before any rollout is drawn)

At the largest scale (4,096), on every structural-shift family *except*
`clustered_units`: shift-robust CSA's worst one-sided 95% Clopper-Pearson
upper bound on true false certification must be `<= 0.05`, and nonvacuity
must be positive in every family. **GO** additionally requires a mean
selected-prefix gain of at least `0.2` actions over shift-robust declared-exit
union — the same threshold P6.3 and P6.4 used, kept unchanged rather than
re-tuned for this smaller/coarser configuration. Valid, nonvacuous evidence
below that gain is **NARROW**. Invalid evidence (on any non-clustered family)
is **STOP**. The `clustered_units` cell's true-false-certification upper
bound is reported alongside the primary result but never enters this rule.

## Consequence

**GO**: preserves P6's certification result across the full P6.0-P6.5 track,
now against a genuinely continuous, closed-form-exact (not hand-specified
categorical) environment. No further P6 stage is required by the frozen
execution plan; anything past this would need its own fresh preregistration.
**NARROW/STOP**: the certificate core's decoupling from any particular
dynamics representation (the continuation memo's revised hypothesis) is
itself the finding worth reporting, independent of whether the utility gain
threshold is met — either way, P6.0-P6.4's own bounded results are unaffected
and stay banked.
