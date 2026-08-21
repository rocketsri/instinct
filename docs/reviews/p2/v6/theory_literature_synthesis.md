# P2.2 extended-pilot theory and literature synthesis v6

Date: 2026-08-17  
Evidence: sealed artifact
`/private/tmp/instinct-p2-r2-pilot2.json`, SHA-256
`6862af1886dbba734566c98d58fe79de68367ce75e96e859f6363309328f1feb`.
No CIFAR-10 test unit was opened for this review, no experiment was rerun, and
no threshold or model setting was changed.

## Decision

**Primary classification: F2 estimator/identifiability failure.** The pilot is
not a low-level instrumentation failure: checksums, train-only partition
disjointness, complete horizons, utility signs, candidate-before-top-m ordering,
write counts, downstream locking, baseline presence, and compute accounting all
passed. It is also not yet a valid mechanism failure. The stored candidate
utility is measured on a write-all reference trajectory, while each selected arm
is replayed on a different sparse trajectory and evaluated by final whole-stream
accuracy. The artifact never establishes that its local loss utility identifies
that downstream target.

There is nevertheless a strong **mechanism warning**: never-write has corrupted
accuracy `0.585938`, whereas the oracle obtains `0.550781`, `0.445312`, and
`0.390625` at fractions `0.10`, `0.25`, and `0.50`. Clean accuracy simultaneously
falls from `0.832031` to `0.808594`, `0.699219`, and `0.632812`. If a pathwise
estimator repair preserves this dominance, it becomes a mechanism failure and
P2 should be archived without spending the reserved test partition.

No theory amendment is justified. The frozen `U_t`, horizons, weights, lambda,
write fractions, accuracy threshold, and test partition remain unchanged.

## What the artifact establishes

- The source head is nondegenerate (`0.832031` clean accuracy), and the 24
  eligible candidates contain six positive-utility events. The earlier
  all-negative short-smoke degeneracy is repaired.
- The utility rows obey
  `utility = benefit - retention_lambda * damage - cost`; measured cost is tiny
  relative to the loss effects and cannot explain the reversal.
- The offline oracle chooses candidates with positive summed stored utility:
  `0.503674`, `0.890522`, and `0.665261`. Yet both corrupted cross-entropy and
  accuracy are worse than never-write at every fraction.
- The oracle's stored damage is favorable because several selected candidate
  rows have negative one-batch aged damage. That prediction conflicts with the
  large clean downstream degradation, so the current `A_t` sample does not
  identify retained whole-stream behavior.

As a sealed-artifact diagnostic, not a new experiment, Spearman correlations
were computed from existing rows. Across all one-step sparse arms, stored
utility versus accuracy is only `0.238`; versus negative cross-entropy it is
`0.276`. Pooling the 24 random arms gives `0.609`, but that is driven by write
fraction: within fractions the utility/accuracy correlations are `-0.135`,
`-0.170`, and `-0.383`. The within-budget comparisons are the relevant selection
test. These exploratory correlations are not the preregistered
aged-damage/downstream-damage statistic and cannot replace it; the artifact
therefore does not establish the previously frozen `0.25` identifiability gate.

## Why the estimand and downstream result diverge

### BLOCKER — reference-path staleness

Section 5.2 defines `U_t` at the actual pre-write state `W_t`. Candidate scoring
instead advances one model through every stream update (`reference_write_all_update`)
and records counterfactual terms at those states. Top-m subsets are then replayed
from the source model with only selected writes. Consequently, a row selected at
time `t` generally describes neither the parameters nor BatchNorm buffers present
at time `t` in the sparse replay.

This is the exact ranking-staleness problem that sections 5.3 and 5.5 required
the frozen-reference and iteratively rescored oracles to measure. Recovery 1
already diagnosed the same issue. The realistic runner preserved the
frozen-reference arm but did not add iterative rescoring or selected-path
counterfactual labels.

### BLOCKER — nonadditive arm accounting

Arm `benefit`, `damage`, and `utility` are sums of frozen one-step candidate
records. Sequential entropy updates and BatchNorm running-state changes interact,
so these sums are not the realized multiwrite utility of an arm. The mismatch is
largest for dense: it admits 2, 6, and 12 writes but executes 48, 51, and 57
optimizer steps to match compute, while its stored utility still sums one-step
candidate terms. Thus the decomposition is algebraically correct but is attached
to the wrong trajectory/update magnitude for arm-level inference.

### BLOCKER — the damage proxy is not identified

For each candidate, damage is evaluated on one oracle-audit batch exactly four
steps old. The frozen damage term is an expectation over `A_t`, and section 5.6
permits a proxy only after correlation with downstream retention is shown. The
runner does not report the preregistered aged-damage/downstream-damage Spearman
gate. The oracle's favorable summed damage alongside a `2.34`-, `13.28`-, and
`19.92`-point clean-accuracy loss is direct evidence that the current audit
sample is not an adequate estimator.

### MAJOR — local cross-entropy is not yet linked to final accuracy

Cross-entropy is a legitimate `L` in `U_t`, but P2.2's primary metric is
whole-stream downstream accuracy. A weighted loss change at horizons 1, 2, and 4
does not automatically identify accuracy after an entire nonlinear replay.
Accuracy is also discrete on this 256-image pilot. Both quantities should remain
reported; neither may be silently substituted for the other. The missing step is
predeclared correlation and frontier validation on fresh train-only units.

## Frontier comparison

The current literature supports the F2 diagnosis rather than a utility-sign
change.

- [VANE](https://arxiv.org/abs/2608.09448) isolates a candidate update on a
  shadow copy, keeps the old policy live, evaluates old and candidate states on
  matched subsequent inputs, and atomically commits model and optimizer state
  only after focused improvement without global regression. Its central causal
  safeguard is precisely the alignment missing when this pilot scores on a
  write-all path and deploys on a sparse path. VANE is an August 2026 preprint,
  so it is a close novelty boundary, not settled evidence of generality.
- [EATA](https://proceedings.mlr.press/v162/niu22a.html) filters unreliable
  high-entropy and redundant samples and adds a Fisher-weighted anti-forgetting
  regularizer. The pilot's `eata_style` arm only ranks
  `-entropy + nonredundancy` and applies the common entropy update. It is a useful
  heuristic selector, not a faithful EATA baseline, and cannot support a claim
  that EATA fails.
- [SAR](https://arxiv.org/abs/2302.12400) excludes noisy samples with large
  gradients and uses sharpness-aware reliable entropy minimization because
  small batches, mixed shifts, and BatchNorm can collapse. The pilot ranks
  `gradient_norm / entropy` in descending order and then performs ordinary
  entropy minimization. This favors large-gradient candidates—the opposite of
  SAR's reliability filter—and omits sharpness-aware optimization. Its result is
  not evidence against SAR.
- [CoTTA](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html)
  combines weight- and augmentation-averaged predictions with stochastic source
  restoration to limit error accumulation and forgetting. The pilot's reset arm
  periodically restores the entire adapted subset and has no teacher or
  augmentation averaging. At fraction `0.50`, the final twelfth write triggers a
  reset, structurally producing exactly the never-write endpoint. That equality
  is a reset-schedule control, not a CoTTA comparison.

These baseline-fidelity problems are MAJOR controls, but they do not cause the
oracle reversal: the offline oracle uses none of their scores. They do prevent
any claim that the completed baseline set represents the current TTA frontier.

## One permissible pilot-only F2 repair

A single versioned repair may use only previously unused official **train**
indices. It does not consume recovery cycle 2 because it repairs the estimator
before reserved evaluation. It must be preregistered before opening those fresh
train units and must preserve all current artifacts.

1. Keep the exact model, adaptable subset, entropy update, horizons `[1,2,4]`,
   weights `[0.5,0.3,0.2]`, lambda, cost conversion, write fractions, source
   fitting, corruption schedule, and thresholds.
2. Measure `U_t` at the actual `W_t` of each replay. Retain the frozen-reference
   oracle, add the section-5.5 iteratively rescored oracle, and report rank shift,
   selected-set overlap, and downstream differences. Never sum write-all labels
   as if they were realized sparse-arm utility.
3. On a fresh, predeclared train-only audit mixture, estimate `A_t` over multiple
   ages and corruption regimes rather than one batch. Keep benefit, damage, and
   cost separate. Exact labels may be used only for offline evaluation.
4. For each candidate, obtain an isolated-write downstream damage label by
   replaying from the same source state with and without that write on fresh
   train-only downstream units. This supplies enough paired points to test the
   already frozen Spearman `>= 0.25` identifiability gate without using the
   downstream stream for selection.
5. Evaluate selected arms once after locking. Report final cross-entropy and
   primary accuracy. Dense's realized utility must include all of its extra
   optimizer steps. Use within-fraction correlations; do not pool fractions to
   manufacture predictiveness.
6. Keep `eata_style`, `sar_style`, and periodic reset names for the existing
   heuristics. A claim against EATA, SAR, or CoTTA requires faithful official
   implementations and matched compute; adding those controls cannot alter the
   oracle or archival thresholds.

No beta/lambda/horizon/learning-rate search, corruption change, selector tuning,
test-split preview, or post-result threshold revision is permissible.

## Exact archival condition

After that one fresh train-only F2 repair, do **not** open the reserved test
partition unless both conditions hold:

1. **Identifiability:** paired candidate-level aged damage versus isolated
   downstream damage has Spearman `>= 0.25`, and candidate pathwise utility
   versus isolated downstream cross-entropy improvement also has Spearman
   `>= 0.25`. Both are computed on fresh disjoint train-only units without
   pooling write fractions.
2. **Mechanism ceiling:** at least one frozen write fraction, the iteratively
   rescored offline oracle improves downstream accuracy by at least the existing
   `0.005` threshold over never-write and every matched deployable control, while
   reporting the accompanying clean-retention result.

If condition 1 fails, archive P2 as **unidentifiable under the frozen utility and
benchmark**. If condition 1 passes but condition 2 fails—or reset/dense matches
the oracle at matched compute—archive P2 as a **valid mechanism failure** under
section 5.9. In either case, preserve P2.0/P2.1 as measurement results, leave the
full recovery unspent, do not attempt another pilot repair, and keep P7 locked.
