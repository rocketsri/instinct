# P2.2 recovery cycle 2 final review v4

## Frozen realistic contract

Recovery 2 does not revise section 5.2. It requires a pretrained ImageNet
ResNet-18 feature extractor, a CIFAR-10 head trained only on the official train
split, changing held-out corruptions, whole-image downstream accuracy, complete
horizons, and separate benefit, damage, and measured cost.

The official 10,000-image CIFAR-10 test batch is permuted once with seed 241 and
reserved before execution:

- candidate indices `[0:2500]`;
- oracle-probe indices `[2500:5000]`;
- validation indices `[5000:7500]`;
- untouched downstream indices `[7500:10000]`.

No recovery-0 or recovery-1 unit is reused. Candidate restriction must occur
before top-m. Oracle futures remain offline-only. Dense adaptation receives
extra steps/batches until measured wall-clock matches selector feature,
validation, synchronization, and rollback cost. Never-write is a zero-update
anchor. Random, validation, EATA-style reliable/nonredundant filtering,
SAR-style entropy/gradient filtering, periodic reset, dense, and oracle arms are
required at matched write fractions 0.1, 0.25, and 0.5.

## Artifact/runtime audit

Local inspection found no CIFAR archive or pretrained weights. The only local
Conda Torch environment fails during import because NumPy/OpenBLAS cannot load
`libgfortran.5.dylib`; it is not an executable baseline environment.

A one-shot Colab T4 was then provisioned and automatically released. Torch and
TorchVision started the official CIFAR-10 download, but transfer stayed near
120–190 KB/s and the execution timed out after 2.46 MB of 170 MB. No partial
file, model weight, metric, or evaluation unit was retained.

Recovery 2 therefore requires these exact external artifacts:

1. `cifar-10-python.tar.gz`, 170,498,071 bytes, MD5
   `c58f30108f718f92721af3b95e74349a`, from
   <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>.
2. `resnet18-f37072fd.pth`, 46,830,571 bytes, SHA-256
   `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`, from
   <https://download.pytorch.org/models/resnet18-f37072fd.pth>.
3. A mutually compatible Torch/TorchVision runtime with CUDA on the target T4.

The preflight hashes files before opening the fresh stream registry. A missing,
wrong-size, or wrong-checksum artifact is a hard failure. There is deliberately
no sample-image, generated-array, or NumPy-head fallback.

## Preflight verdict and recovery accounting

The realistic experiment did not execute, so matched write count, matched
wall-clock, downstream accuracy, oracle advantage, and useful sparsity are all
unmeasured—not zero-effect results. Status is INCONCLUSIVE at E0 and no baseline
is marked as run. P7 remains locked and no qualification artifact is emitted.

This preflight does **not** consume recovery cycle 2. The portfolio execution
rules classify F0 implementation failures and failures before evaluation units
are opened as repairable without consuming a scientific recovery. P2 therefore
remains externally blocked at the frozen recovery-2 boundary rather than
scientifically archived. It may resume under this version after the exact
artifacts/runtime are supplied; it still may not overwrite the original or
recovery-1 results. No P7 work is unlocked while the realistic metrics remain
unmeasured.

### 2026-08-17 artifact addendum

Both external artifacts were subsequently retrieved and verified locally:
`170,498,071` bytes with MD5 `c58f30108f718f92721af3b95e74349a`
for CIFAR-10, and SHA-256
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`
for ResNet-18. The repeated preflight artifact is
`/tmp/instinct-p2-recovery2-preflight-verified/p2_harmful_write/p2_harmful_write-20260817T065804Z-09e650dc-134d93`.
Artifact checks now pass; the local environment still lacks a compatible
Torch/TorchVision CUDA runtime. No stream partition or evaluation baseline was
opened, so recovery-cycle accounting is unchanged.
