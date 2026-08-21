# P2 theory map

- Question and utility: frozen specification sections 5.1–5.2.
- Estimand: multi-horizon `U_t`; benefit, damage, and cost remain separate.
- Assumptions: full frozen horizon set; candidate restriction before top-m; oracle, audit, validation, and downstream units are disjoint.
- Controls: planted harm/no-harm, zero learning rate, fork/reference equivalence, matched write count and measured compute.
- Verdict: section 5.9. The oracle is an offline ceiling and never a deployable trigger.
- Source: proposal version 2.1; portfolio amendment 2.1 changes scheduling only.

## P2.2 bounded pilot mapping

- The natural-image pilot keeps the complete frozen horizon vector and computes
  `utility = benefit - retention_lambda * damage - cost` without combining the
  reported terms.
- Candidate, oracle-probe, validation-only, and downstream units occupy
  disjoint spatial bands. Candidate eligibility is frozen before any top-m
  selection, and matched-count arms commit exactly `m` writes.
- The validation-only arm may inspect only its disjoint same-timestep stream.
  The future-looking oracle is recorded as `oracle_ceiling` and its scoring
  work is excluded from deployable compute.
- This single-photograph affine-head experiment is an E2 instrumentation pilot,
  not the pretrained multi-image benchmark required by section 5.6. Its
  `realistic_scale_passed` field is therefore fail-closed and it cannot emit a
  P7 qualification artifact.

## P2.2 recovery-2 executable mapping

- `recovery2_benchmark.py` and `benchmarks/p2_recovery2_colab.py` implement the
  frozen v4 realistic contract without changing `U_t`. Each candidate row keeps
  weighted multi-horizon benefit, retention-audit damage, measured write cost,
  and `benefit - retention_lambda * damage - cost` as separate fields.
- The official CIFAR-10 archive and official TorchVision ResNet-18 weights are
  accepted only after their frozen byte counts and checksums pass. The CIFAR-10
  head is trained only on the official train split. Online adaptation can mutate
  only BN affine/running state and the CIFAR-10 linear head; the artifact records
  every parameter and buffer name.
- Full mode applies the seed-241 permutation to the official test split and
  freezes candidate, oracle-probe, validation, and downstream quarters before
  scoring. Candidate eligibility is restricted by age and the maximum horizon
  before any method calls top-m. Future oracle cost is evaluation-only.
- Never, fixed-frequency dense, random, validation, EATA-style,
  SAR-style, periodic-reset, and offline-oracle arms are replayed at write
  fractions 0.1, 0.25, and 0.5. Dense receives extra within-write optimization
  steps until its measured update time reaches the most expensive deployable
  selector budget; exact write counts and the remaining timing tolerance are
  explicit gates.
- Pilot mode draws all source and stream images from disjoint bands of the
  official train split, never indexes the reserved test split, and fails the
  realism gate by construction. P7 qualification is emitted only when every
  artifact, disjointness, complete-horizon, matched-count, matched-compute,
  baseline, source-accuracy, useful-sparsity, and oracle-frontier gate passes.
