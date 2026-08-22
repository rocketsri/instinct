# P2.2 recovery-2 F2 estimator repair preregistration

The v6 theory/literature synthesis (`docs/reviews/p2/v6/theory_literature_synthesis.md`)
classified the extended pilot as **F2 estimator/identifiability failure**, not
yet a mechanism failure. Never-write reached corrupted accuracy `0.585938`,
while the offline oracle reached only `0.550781`, `0.445312`, and `0.390625`
at write fractions `0.10`, `0.25`, and `0.50`; clean accuracy fell from
`0.832031` to `0.808594`, `0.699219`, and `0.632812` over the same fractions.
The oracle's selected candidates carried positive summed stored utility
(`0.503674`, `0.890522`, `0.665261`) despite that reversal. Within-fraction
Spearman correlations between stored utility and realized accuracy were
`-0.135`, `-0.170`, and `-0.383` — negative, not merely weak — so the frozen
utility does not yet identify the downstream target it is meant to predict.
The root cause the v6 doc isolated is reference-path staleness: candidate
scoring advances one model through every stream update
(`reference_write_all_update`) and records counterfactual terms at those
states, while top-m subsets are replayed from the source model with only
selected writes — a row scored at time `t` generally describes neither the
parameters nor BatchNorm buffers actually present at `t` in the sparse
replay.

Per the execution plan, an estimator/identifiability failure does not consume
a scientific recovery cycle. This preregistration locks the one permissible
pilot-only F2 repair the v6 doc specifies (its "One permissible pilot-only F2
repair", items 1–6) before any fresh train unit is opened. **It does not
consume recovery cycle 2**: it repairs the estimator strictly before reserved
evaluation, using only previously unused official train indices, and the
reserved seed-241 test partition stays closed regardless of outcome. This is
not a theory amendment — `U_t`, horizons, weights, lambda, write fractions,
the accuracy threshold, and the test partition are unchanged.

## Frozen vs. new

Unchanged, byte-for-byte, from `configs/p2/p2_2/recovery2.yaml`: the adapted
model (`resnet18_imagenet1k_v1`, BatchNorm-affine-and-CIFAR10-linear-head),
`horizons: [1, 2, 4]`, `horizon_weights: [0.5, 0.3, 0.2]`, `retention_lambda:
1.0`, `write_fractions: [0.1, 0.25, 0.5]`, `random_replicates: 8`, `min_age:
4`, `adaptation_lr: 0.001`, `cost_loss_per_second: 0.001`, `reset_every: 4`,
`compute_match_tolerance: 0.25`, `minimum_source_accuracy: 0.45`, and —
critically — `eval_partition_seed: 241`, which `BenchmarkConfig.__post_init__`
enforces in code and this repair never overrides. `n_steps: 32` and
`source_epochs: 2` match pilot2's already-run scale rather than the frozen
`full` scale.

New, locked in `configs/p2/p2_2/f2_repair.yaml`: `scoring_mode: f2_repair`;
`f2_repair_source_offset: 4096` (the first source-pool permutation position no
prior pilot artifact has materialized — `pilot.py` used `[0:2048]`, `pilot2.py`
used `[0:4096]`); `f2_repair_audit_pool_size: 4096` and
`f2_repair_downstream_label_pool_size: 1536`, two disjoint pools sliced from
`train_permutation[4096:40_000]` of the same seed-241 permutation
`build_partition_plan` already uses, never from the four fixed
candidate/oracle_probe/validation/downstream bands (which are byte-identical
across every pilot run because `partition_seed` is locked); `rescoring_block_size:
2`; `audit_probe_budget: 3`; `audit_ages: [4, 8, 12]`. `spearman_gate: 0.25`
and `minimum_accuracy_gain: 0.005` are restated from the frozen thresholds
below, not new values chosen here.

The repair itself, restated from the v6 doc's six numbered requirements: (1)
measure `U_t` at the actual per-replay `W_t`, never the write-all reference
path; (2) retain the frozen-reference oracle and add the spec-section-5.5
iteratively rescored oracle, reporting rank shift and selected-set overlap;
(3) estimate the retention-audit term over multiple ages and corruption
regimes from a fresh train-only audit mixture, rather than one fixed-age
batch; (4) obtain an isolated-write downstream damage label per candidate by
replaying from the same source state with and without that one write on fresh
train-only downstream units, supplying paired points for the identifiability
gate below; (5) evaluate selected arms once after locking, with dense's
realized utility accounting for all of its extra optimizer steps rather than
only its `m` admitted positions; (6) compute correlations within each write
fraction only — fractions are never pooled to manufacture predictiveness. No
beta/lambda/horizon/learning-rate search, corruption change, selector tuning,
test-split preview, or post-result threshold revision is permissible.

## Fresh seeds and fresh data

`seed: 251`, `pilot_seeds: [251]`, `eval_seeds: [252]` — disjoint from every
prior P2 config (smoke `221`/`222`, recovery1 `231`/`232`, recovery2
`241`/`242`). The audit pool and downstream-label pool
(`build_f2_repair_partition_plan`) are mutually disjoint from each other, from
the four fixed streams, and from the source-training band — verified by
`F2RepairPartitionPlan.disjoint` and recorded in `.manifest()` so a future
repair round can start past this one without re-deriving anything. Reusing
the same images for both scoring-time audit and isolated-write ground truth
would let the identifiability check measure self-consistency rather than
genuine identifiability, which is exactly what the disjointness check exists
to rule out.

## Exact archival condition (restated, not redefined, from v6)

After this one fresh train-only F2 repair, the reserved test partition does
**not** open unless both conditions hold:

1. **Identifiability**: paired candidate-level aged damage versus isolated
   downstream damage has Spearman `>= 0.25`, and candidate pathwise utility
   versus isolated downstream cross-entropy improvement also has Spearman
   `>= 0.25`. Both are computed on fresh disjoint train-only units, within
   each write fraction, without pooling fractions.
2. **Mechanism ceiling**: at least one frozen write fraction where the
   iteratively rescored offline oracle improves downstream accuracy by at
   least `0.005` over never-write and every matched deployable control
   (reset and dense included, with dense's corrected step accounting), while
   reporting the accompanying clean-retention result.

If condition 1 fails, archive P2 as **unidentifiable under the frozen utility
and benchmark**. If condition 1 passes but condition 2 fails — or reset/dense
matches the oracle at matched compute — archive P2 as a **valid mechanism
failure** under spec section 5.9. In either case, preserve P2.0/P2.1 as
measurement results, leave the full recovery unspent, do not attempt another
pilot repair, and keep P7 locked. Only if both conditions hold may a
separately preregistered recovery-2 test-partition run follow.

`evaluate_f2_repair_archival_condition` (`recovery2_benchmark.py`) implements
this disjunction exactly and is unit-tested against synthetic numbers in
`tests/test_p2_recovery2_f2repair.py`; the Torch-gated scoring/replay/rescoring
code that will feed it real numbers is implemented and reviewed separately,
after this document is locked.
