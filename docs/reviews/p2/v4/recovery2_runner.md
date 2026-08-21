# P2.2 recovery-2 T4 runner

Status: implementation only; no CIFAR-10 candidate, probe, validation, or
downstream unit was opened while adding this runner. The preserved v4 preflight
result remains unchanged.

## Scientific contract

The runner implements specification sections 5.1--5.10 and the frozen v4
recovery contract. It verifies the exact official CIFAR-10 archive and official
TorchVision ResNet-18 weights before dataset extraction. A CIFAR-10 linear head
is trained only on the official training split. Online writes mutate only the
linear head, BatchNorm affine parameters, and named BatchNorm running buffers.

For every eligible candidate, the artifact reports the complete horizon vector
`[1, 2, 4]`, weighted future benefit on oracle probes, aged retention damage,
measured update seconds and converted cost, and the unchanged utility identity.
Candidate age/horizon restriction is constructed before any top-m call. The
validation stream selects only validation scores, EATA-style and SAR-style arms
use pre-write reliable/nonredundant and entropy/gradient scores, and the future
oracle is marked as an offline ceiling with separate compute.

At each admitted-write fraction `0.1`, `0.25`, and `0.5`, the runner evaluates
never, fixed-frequency dense, eight random replicates, validation, EATA-style,
SAR-style, periodic reset, and oracle arms. All sparse arms other than never
commit exactly `m` writes. Dense uses the same fixed-frequency `m` admissions
and receives extra optimizer steps within admitted writes until its measured
update time reaches the largest deployable selector budget. Timing tolerance is
a qualification gate rather than an assumed equality.

The downstream indices are not materialized until all selected timestep sets
are locked and hashed. The JSON schema is `instinct.p2.recovery2.v1`; it includes
theory IDs, artifact checks, partition hashes, parameter scope, candidate terms,
arm metrics, compute ledgers, gate-by-gate status, and `qualifies_p7`. That last
field is true only if every realism and useful-sparsity gate passes.

## Safe pilot command

This downloads and verifies the official artifacts, but uses only disjoint
bands of the official **train** split for source fitting and smoke streams. It
cannot qualify P7 and does not consume the reserved seed-241 test partition.

```bash
python benchmarks/p2_recovery2_colab.py \
  --mode pilot \
  --artifact-dir /content/instinct-p2-artifacts \
  --data-root /content/instinct-p2-data \
  --output /content/p2-r2-pilot.json
```

## Frozen full T4 command

Run this once in a Colab T4 runtime only after the held-out evaluation is
authorized. The confirmation token is an intentional guard against accidental
test-partition consumption.

```bash
python benchmarks/p2_recovery2_colab.py \
  --mode full \
  --confirm OPEN_FROZEN_P2_R2_TEST_PARTITION \
  --artifact-dir /content/instinct-p2-artifacts \
  --data-root /content/instinct-p2-data \
  --output /content/p2-r2-full-seed241.json
```

Do not tune thresholds or hyperparameters after inspecting this full artifact.
The standalone JSON is intended for later CLI ingestion; it does not overwrite
the original, recovery-1, or v4 preflight result paths.
