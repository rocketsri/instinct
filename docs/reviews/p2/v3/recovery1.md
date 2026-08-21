# P2.2 recovery cycle 1 review v3

## Preserved result and diagnosis

The original one-photo run remains preserved at its immutable result path. It
contained useful events—8 of 20 candidates had positive frozen utility—but its
identity-initialized affine head overfit short alternating color regimes.
Never-write had the lowest downstream MSE. The frozen write-all oracle ranking
also selected on a different trajectory from sparse replay, so positive frozen
candidate scores did not imply positive replay utility.

This is a benchmark/estimator degeneracy, not evidence that the original
utility sign was wrong: every stored row satisfied
`utility = benefit - retention_lambda * damage - cost`, and cost was only
`0.0001` per write versus damage up to `0.01697`.

## Recovery design

- **Theory unchanged:** the full section 5.2 horizon vector and signs remain.
  No amendment is required.
- **Fresh units:** three packaged natural images are divided into five spatial
  bands. Pretraining, candidate, oracle-probe, validation, and downstream pixels
  are disjoint. The old image-band evaluation is not reused.
- **Stronger boundary:** a ten-dimensional fixed nonlinear RGB feature map feeds
  a separately ridge-pretrained 3-channel head. Only that head adapts.
- **Persistent regimes:** each corruption lasts nine steps, longer than the
  maximum horizon four, so locally useful writes can exist without redefining
  future benefit.
- **Staleness repair:** after freezing the eligible window and `m=3`, the offline
  ceiling exhaustively evaluates all 3,276 subsets on oracle probes. Downstream
  units select nothing. Oracle compute is reported separately.
- **Baselines:** always, never, validation-only, eight random subsets, and exact
  oracle. Validation and random arms are charged identical candidate and
  validation work; all matched-count arms commit exactly three writes.

## Literature and code boundary (checked 2026-08-17)

- [VANE](https://arxiv.org/abs/2608.09448) already isolates candidate VLA updates,
  observes future consequences, and selectively commits supported updates. This
  recovery is only an offline multi-horizon ceiling; it is not a novel
  transactional policy and does not match VANE's robotics evaluation.
- [EATA](https://arxiv.org/abs/2204.02610),
  [SAR](https://github.com/mr-eggplant/SAR), and
  [CoTTA](https://arxiv.org/abs/2203.13591) make sample filtering,
  anti-forgetting regularization, reliable entropy updates, and stochastic
  source restoration mandatory realistic TTA baselines. The recovery's
  validation/random selectors do not replace them.
- [Adaptive and Selective Reset](https://arxiv.org/abs/2603.03796) strengthens
  the reset baseline beyond fixed periodic reset by choosing when and where to
  restore. It remains absent here.
- The official [TTT linear-attention analysis](https://github.com/nv-tlabs/tttla)
  shows that important KV-binding TTT architectures admit linear-attention
  interpretations and releases ViTTT/LaCT experiment code. This recovery's
  hand-built polynomial feature head does not match those architectures.
- The official [end-to-end TTT code](https://github.com/test-time-training/e2e)
  operates at 125M–3B parameters with meta-learned initialization and GPU-scale
  data. It bounds any scale or general-purpose memory claim from this pilot.
- The maintained [online TTA benchmark](https://github.com/mariodoebler/test-time-adaptation)
  exposes official/ported TENT, EATA, CoTTA, SAR, RDumb, and related baselines;
  recovery cycle 2 should use that baseline surface rather than bespoke proxies.

## Verdict contract

The recovery may establish provisional mechanism sparsity if the oracle beats
matched-count selectors, improves MSE over never-write, and aged damage
correlates with disjoint downstream damage. It still cannot pass the frozen
realism prerequisite: three sample images and a small nonlinear head are not a
pretrained ResNet/ViT on whole held-out CIFAR-C streams. No P7 qualification
artifact is emitted, irrespective of the pilot result.

Falsifiers remain baseline dominance, nonpositive oracle gain, correlation below
0.25, any stream overlap, incomplete horizons, unmatched count/compute, or a
realism claim based on the packaged-image proxy.
