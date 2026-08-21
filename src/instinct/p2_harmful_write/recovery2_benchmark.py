"""Executable, fail-closed P2.2 recovery-2 benchmark.

The module deliberately has no import-time Torch dependency.  Pure partition,
selection, utility, qualification, and artifact helpers remain testable on the
CPU-only repository environment; :func:`run_recovery2_benchmark` imports Torch
and TorchVision only after the official external artifacts have been verified.

Scientific traceability: ``docs/AI_Instinct_Seven_Proposal_Execution_Plan.md``
sections 5.1--5.10 and ``docs/reviews/p2/v4/recovery2.md``.  The offline oracle
may inspect future oracle probes, but its scoring cost is separately recorded
and never presented as deployable compute.  The downstream partition is not
materialized until every candidate set has been frozen.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tarfile
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from instinct.p2_harmful_write.recovery2 import CIFAR10_ARCHIVE, RESNET18_WEIGHTS

__all__ = [
    "ARTIFACT_SCHEMA",
    "FULL_EVAL_CONFIRMATION",
    "BenchmarkConfig",
    "PartitionPlan",
    "UtilityTerms",
    "build_partition_plan",
    "compute_utility",
    "ensure_official_artifacts",
    "evaluate_p7_qualification",
    "restricted_top_m",
    "run_recovery2_benchmark",
    "write_artifact",
]

ARTIFACT_SCHEMA = "instinct.p2.recovery2.v1"
FULL_EVAL_CONFIRMATION = "OPEN_FROZEN_P2_R2_TEST_PARTITION"
THEORY_TRACE = {
    "proposal_version": "2.1",
    "theory_id": "spec-5.1-5.10",
    "amendment_id": "portfolio-2.1",
    "source_section": "5.1-5.10",
    "recovery_review": "docs/reviews/p2/v4/recovery2.md",
}
REQUIRED_METHODS = (
    "never",
    "dense",
    "validation",
    "eata_style",
    "sar_style",
    "reset",
    "oracle_ceiling",
)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Frozen runtime choices for one benchmark artifact."""

    mode: Literal["pilot", "full"] = "pilot"
    seed: int = 241
    partition_seed: int = 241
    horizons: tuple[int, ...] = (1, 2, 4)
    horizon_weights: tuple[float, ...] = (0.5, 0.3, 0.2)
    retention_lambda: float = 1.0
    write_fractions: tuple[float, ...] = (0.1, 0.25, 0.5)
    random_replicates: int = 8
    min_age: int = 4
    n_steps: int = 50
    regime_length: int = 8
    source_epochs: int = 5
    source_batch_size: int = 128
    adaptation_lr: float = 1e-3
    cost_loss_per_second: float = 1e-3
    reset_every: int = 4
    compute_match_tolerance: float = 0.25
    minimum_accuracy_gain: float = 0.005
    minimum_source_accuracy: float = 0.45
    minimum_positive_utility_fraction: float = 0.05
    maximum_positive_utility_fraction: float = 0.90
    pilot_source_images: int = 2048
    pilot_stream_images: int = 96
    full_eval_confirmation: str = ""
    require_t4: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"pilot", "full"}:
            raise ValueError("mode must be 'pilot' or 'full'")
        if self.partition_seed != 241:
            raise ValueError("the frozen recovery-2 partition seed is 241")
        if self.mode == "full" and self.full_eval_confirmation != FULL_EVAL_CONFIRMATION:
            raise ValueError("full mode requires the explicit frozen-evaluation confirmation token")
        if not self.horizons or len(self.horizons) != len(self.horizon_weights):
            raise ValueError("horizons and horizon_weights must be nonempty and aligned")
        if any(h <= 0 for h in self.horizons):
            raise ValueError("horizons must be positive")
        if any(w < 0 for w in self.horizon_weights) or not math.isclose(
            sum(self.horizon_weights), 1.0, abs_tol=1e-9
        ):
            raise ValueError("horizon weights must be nonnegative and sum to one")
        if any(not 0.0 < fraction < 1.0 for fraction in self.write_fractions):
            raise ValueError("write fractions must lie strictly between zero and one")
        if self.random_replicates < 1:
            raise ValueError("at least one random replicate is required")
        if self.n_steps <= self.min_age + max(self.horizons):
            raise ValueError("n_steps must leave eligible candidates after age/horizon bounds")
        if self.source_epochs < 1 or self.source_batch_size < 1:
            raise ValueError("source training settings must be positive")
        if self.adaptation_lr <= 0 or self.retention_lambda < 0:
            raise ValueError("adaptation_lr must be positive and lambda nonnegative")
        if self.cost_loss_per_second < 0:
            raise ValueError("cost conversion must be nonnegative")


@dataclass(frozen=True, slots=True)
class PartitionPlan:
    dataset_split: Literal["train", "test"]
    source_train_indices: tuple[int, ...]
    candidate_indices: tuple[int, ...]
    oracle_probe_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    downstream_indices: tuple[int, ...]
    opens_reserved_test_partition: bool

    def stream_indices(self) -> dict[str, tuple[int, ...]]:
        return {
            "candidate": self.candidate_indices,
            "oracle_probe": self.oracle_probe_indices,
            "validation": self.validation_indices,
            "downstream": self.downstream_indices,
        }

    @property
    def disjoint(self) -> bool:
        groups = [set(values) for values in self.stream_indices().values()]
        streams_disjoint = all(
            groups[i].isdisjoint(groups[j])
            for i in range(len(groups))
            for j in range(i + 1, len(groups))
        )
        # Official train and test integer IDs occupy distinct namespaces.
        source_disjoint = self.dataset_split == "test" or set(self.source_train_indices).isdisjoint(
            set().union(*groups)
        )
        return streams_disjoint and source_disjoint

    def manifest(self) -> dict[str, Any]:
        def digest(values: tuple[int, ...]) -> str:
            array = np.asarray(values, dtype="<i8")
            return hashlib.sha256(array.tobytes()).hexdigest()

        return {
            "dataset_split": self.dataset_split,
            "opens_reserved_test_partition": self.opens_reserved_test_partition,
            "disjoint": self.disjoint,
            "source_train_count": len(self.source_train_indices),
            "streams": {
                name: {"count": len(values), "indices_sha256": digest(values)}
                for name, values in self.stream_indices().items()
            },
        }


@dataclass(frozen=True, slots=True)
class UtilityTerms:
    benefit: float
    damage: float
    cost: float
    utility: float


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    timestep: int
    benefit: float
    benefit_by_horizon: dict[str, float]
    damage: float
    cost: float
    cost_seconds: float
    utility: float
    validation_score: float
    eata_score: float
    sar_score: float
    entropy: float
    gradient_norm: float
    oracle_probe_ids: tuple[str, ...]
    audit_probe_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArmRecord:
    method: str
    write_fraction: float
    replicate: int | None
    selected_timesteps: tuple[int, ...]
    admitted_writes: int
    optimizer_steps: int
    benefit: float
    damage: float
    cost: float
    utility: float
    selection_seconds: float
    update_seconds: float
    deployable_seconds: float
    offline_oracle_seconds: float
    downstream_accuracy: float
    downstream_cross_entropy: float
    clean_downstream_accuracy: float
    exact_write_count: bool


@dataclass(slots=True)
class BenchmarkResult:
    config: BenchmarkConfig
    environment: dict[str, Any]
    artifact_checks: list[dict[str, Any]]
    partitions: PartitionPlan
    corruption_schedule: list[str]
    adapted_parameters: tuple[str, ...]
    mutable_bn_buffers: tuple[str, ...]
    source_head: dict[str, Any]
    candidates: list[CandidateRecord]
    arms: list[ArmRecord]
    compute_ledger: dict[str, float]
    gates: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)

    def artifact(self) -> dict[str, Any]:
        return {
            "schema": ARTIFACT_SCHEMA,
            "created_unix_s": time.time(),
            "theory": THEORY_TRACE,
            "config": asdict(self.config),
            "environment": self.environment,
            "artifact_checks": self.artifact_checks,
            "partitions": self.partitions.manifest(),
            "corruption_schedule": self.corruption_schedule,
            "adaptation_scope": {
                "declared": "batchnorm_affine_and_cifar10_linear_head",
                "parameter_names": list(self.adapted_parameters),
                "mutable_bn_buffer_names": list(self.mutable_bn_buffers),
            },
            "source_head": self.source_head,
            "candidates": [asdict(record) for record in self.candidates],
            "arms": [asdict(record) for record in self.arms],
            "compute_ledger_seconds": self.compute_ledger,
            "gates": self.gates,
            "qualifies_p7": bool(self.gates.get("qualifies_p7", False)),
            "events": self.events,
        }


def compute_utility(
    benefit: float,
    damage: float,
    cost: float,
    *,
    retention_lambda: float,
) -> UtilityTerms:
    """Apply the frozen signs while retaining all three terms."""

    if retention_lambda < 0 or cost < 0:
        raise ValueError("retention lambda and cost must be nonnegative")
    utility = float(benefit - retention_lambda * damage - cost)
    return UtilityTerms(float(benefit), float(damage), float(cost), utility)


def build_partition_plan(config: BenchmarkConfig) -> PartitionPlan:
    """Freeze disjoint streams without loading or evaluating an image."""

    rng = np.random.default_rng(config.partition_seed)
    if config.mode == "full":
        permutation = rng.permutation(10_000)
        streams = [
            tuple(map(int, permutation[start:stop]))
            for start, stop in (
                (0, 2500),
                (2500, 5000),
                (5000, 7500),
                (7500, 10_000),
            )
        ]
        # The source head uses only the official train split, so its integer
        # indices occupy a different namespace from these test indices.
        return PartitionPlan(
            "test",
            tuple(range(50_000)),
            streams[0],
            streams[1],
            streams[2],
            streams[3],
            True,
        )

    # Pilot execution never indexes the official test split.  The first 40k
    # permuted train examples are the source pool and four disjoint 2.5k bands
    # are code-smoke streams.  Limits only take prefixes inside those bands.
    train_permutation = np.random.default_rng(config.partition_seed + 10_000).permutation(50_000)
    source = tuple(map(int, train_permutation[:40_000][: config.pilot_source_images]))
    bands = []
    for start in (40_000, 42_500, 45_000, 47_500):
        bands.append(
            tuple(
                map(
                    int,
                    train_permutation[start : start + 2500][: config.pilot_stream_images],
                )
            )
        )
    return PartitionPlan(
        "train",
        source,
        bands[0],
        bands[1],
        bands[2],
        bands[3],
        False,
    )


def restricted_top_m(
    eligible_timesteps: Sequence[int], restricted_scores: Sequence[float], m: int
) -> tuple[int, ...]:
    """Select only from an already restricted candidate vector.

    Requiring equal-length restricted inputs prevents a global score vector from
    smuggling an ineligible late candidate into top-m.
    """

    timesteps = tuple(int(value) for value in eligible_timesteps)
    scores = np.asarray(restricted_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size != len(timesteps):
        raise ValueError("scores must already be restricted to eligible_timesteps")
    if len(set(timesteps)) != len(timesteps):
        raise ValueError("eligible timesteps must be unique")
    if not 0 <= m <= len(timesteps):
        raise ValueError("m must lie between zero and the eligible candidate count")
    if m == 0:
        return ()
    order = np.lexsort((np.asarray(timesteps), -scores))[:m]
    return tuple(sorted(timesteps[int(index)] for index in order))


def _evenly_spaced(eligible: Sequence[int], m: int) -> tuple[int, ...]:
    if m == 0:
        return ()
    positions = np.linspace(0, len(eligible) - 1, num=m)
    chosen = {round(value) for value in positions}
    if len(chosen) < m:
        chosen.update(index for index in range(len(eligible)) if index not in chosen)
    return tuple(sorted(int(eligible[index]) for index in sorted(chosen)[:m]))


def evaluate_p7_qualification(
    *,
    mode: str,
    artifacts_verified: bool,
    t4_verified: bool,
    partitions_disjoint: bool,
    downstream_opened_after_lock: bool,
    full_horizons: bool,
    candidate_restriction_passed: bool,
    exact_counts: bool,
    compute_matched: bool,
    baselines_complete: bool,
    source_accuracy: float,
    minimum_source_accuracy: float,
    positive_utility_fraction: float,
    minimum_positive_fraction: float,
    maximum_positive_fraction: float,
    oracle_frontier_gain: float,
    minimum_accuracy_gain: float,
) -> dict[str, Any]:
    """Fail-closed qualification; pilot artifacts can never unlock P7."""

    checks = {
        "full_realistic_mode": mode == "full",
        "official_artifacts_verified": bool(artifacts_verified),
        "t4_verified": bool(t4_verified),
        "partitions_disjoint": bool(partitions_disjoint),
        "downstream_opened_after_selection_lock": bool(downstream_opened_after_lock),
        "full_horizons": bool(full_horizons),
        "candidate_restriction_before_top_m": bool(candidate_restriction_passed),
        "matched_admitted_write_count": bool(exact_counts),
        "matched_deployable_compute": bool(compute_matched),
        "required_baselines_complete": bool(baselines_complete),
        "source_accuracy_gate": source_accuracy >= minimum_source_accuracy,
        "useful_and_harmful_events_present": (
            minimum_positive_fraction <= positive_utility_fraction <= maximum_positive_fraction
        ),
        "oracle_realistic_frontier_gain": oracle_frontier_gain >= minimum_accuracy_gain,
    }
    return {
        "checks": checks,
        "positive_utility_fraction": float(positive_utility_fraction),
        "oracle_frontier_gain": float(oracle_frontier_gain),
        "qualifies_p7": bool(all(checks.values())),
    }


def _checksum(path: Path, algorithm: str) -> str:
    digest = hashlib.md5() if algorithm == "md5" else hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_verified(requirement: Any, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        observed = _checksum(destination, requirement.checksum_algorithm)
        if (
            destination.stat().st_size == requirement.expected_bytes
            and observed == requirement.checksum
        ):
            return {
                "name": requirement.name,
                "path": str(destination),
                "url": requirement.url,
                "bytes": destination.stat().st_size,
                "checksum_algorithm": requirement.checksum_algorithm,
                "checksum": observed,
                "verified": True,
                "downloaded": False,
            }
        raise ValueError(f"existing artifact failed verification: {destination}")

    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(requirement.url, headers={"User-Agent": "instinct-p2-r2/1"})
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("wb") as output,
        ):
            while block := response.read(1024 * 1024):
                output.write(block)
        observed = _checksum(temporary, requirement.checksum_algorithm)
        if (
            temporary.stat().st_size != requirement.expected_bytes
            or observed != requirement.checksum
        ):
            raise ValueError(f"downloaded artifact failed verification: {requirement.name}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "name": requirement.name,
        "path": str(destination),
        "url": requirement.url,
        "bytes": destination.stat().st_size,
        "checksum_algorithm": requirement.checksum_algorithm,
        "checksum": requirement.checksum,
        "verified": True,
        "downloaded": True,
    }


def ensure_official_artifacts(artifact_dir: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    """Download missing official artifacts and verify bytes plus checksum."""

    artifact_dir = artifact_dir.resolve()
    cifar_path = artifact_dir / CIFAR10_ARCHIVE.name
    weight_path = artifact_dir / RESNET18_WEIGHTS.name
    checks = [
        _download_verified(CIFAR10_ARCHIVE, cifar_path),
        _download_verified(RESNET18_WEIGHTS, weight_path),
    ]
    return cifar_path, weight_path, checks


def _extract_cifar_archive(archive: Path, data_root: Path) -> None:
    expected = data_root / "cifar-10-batches-py"
    if expected.is_dir():
        return
    data_root.mkdir(parents=True, exist_ok=True)
    root = data_root.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe CIFAR archive member: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"links are forbidden in CIFAR archive: {member.name}")
        bundle.extractall(root, members=members)


def _torch_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn.functional as functional
        import torchvision
    except Exception as error:  # pragma: no cover - external runtime
        raise RuntimeError(f"Torch/TorchVision runtime unavailable: {error}") from error
    return torch, functional, torchvision


def _sync(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_official_model(torch: Any, torchvision: Any, weight_path: Path, seed: int) -> Any:
    torch.manual_seed(seed)
    model = torchvision.models.resnet18(weights=None)
    try:
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older Colab Torch
        state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    return model


def _adaptation_scope(model: Any, torch: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    parameters: list[str] = []
    buffers: list[str] = []
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for field_name in ("weight", "bias"):
                parameter = getattr(module, field_name)
                if parameter is not None:
                    parameter.requires_grad_(True)
                    parameters.append(f"{module_name}.{field_name}")
            for field_name in ("running_mean", "running_var", "num_batches_tracked"):
                if getattr(module, field_name) is not None:
                    buffers.append(f"{module_name}.{field_name}")
    for field_name, parameter in model.fc.named_parameters():
        parameter.requires_grad_(True)
        parameters.append(f"fc.{field_name}")
    return tuple(parameters), tuple(buffers)


def _state_subset(model: Any, names: Sequence[str]) -> dict[str, Any]:
    state = model.state_dict()
    return {name: state[name].detach().clone() for name in names}


def _restore_subset(model: Any, subset: dict[str, Any]) -> None:
    state = model.state_dict()
    for name, value in subset.items():
        state[name].detach().copy_(value)


def _raw_batch(dataset: Any, indices: Sequence[int], torch: Any) -> tuple[Any, Any]:
    array = np.asarray(dataset.data[np.asarray(indices, dtype=np.int64)], dtype=np.float32)
    images = torch.from_numpy(array).permute(0, 3, 1, 2).div_(255.0)
    labels = torch.as_tensor(
        np.asarray(dataset.targets, dtype=np.int64)[np.asarray(indices, dtype=np.int64)]
    )
    return images, labels


def _corrupt(
    images: Any, mode: str, severity: float, seed: int, torch: Any, functional: Any
) -> Any:
    result = images.clone()
    if mode == "brightness":
        result = result * (1.0 + severity)
    elif mode == "contrast":
        mean = result.mean(dim=(2, 3), keepdim=True)
        result = mean + (result - mean) * (1.0 + severity)
    elif mode == "gaussian_noise":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(result.shape, generator=generator, dtype=result.dtype)
        result = result + noise * (0.05 + 0.08 * severity)
    elif mode == "blur":
        kernel = 3 if severity < 0.75 else 5
        result = functional.avg_pool2d(result, kernel, stride=1, padding=kernel // 2)
    elif mode == "channel_shift":
        scale = torch.tensor([1.0 + severity, 1.0 - 0.5 * severity, 1.0 - severity])
        result = result * scale.view(1, 3, 1, 1)
    else:
        raise ValueError(f"unknown corruption mode {mode!r}")
    return result.clamp_(0.0, 1.0)


def _model_input(images: Any, torch: Any, functional: Any, device: Any) -> Any:
    resized = functional.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=resized.dtype).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=resized.dtype).view(1, 3, 1, 1)
    return ((resized - mean) / std).to(device, non_blocking=True)


def _step_batches(indices: Sequence[int], n_steps: int) -> list[tuple[int, ...]]:
    if n_steps > len(indices):
        raise ValueError("n_steps cannot exceed the smallest stream")
    return [tuple(map(int, values)) for values in np.array_split(np.asarray(indices), n_steps)]


def _schedule(n_steps: int, regime_length: int) -> list[tuple[str, float]]:
    modes = ("brightness", "contrast", "gaussian_noise", "blur", "channel_shift")
    schedule = []
    for timestep in range(n_steps):
        regime = timestep // regime_length
        mode = modes[regime % len(modes)]
        severity = (0.25, 0.50, 0.75)[regime % 3]
        schedule.append((mode, severity))
    return schedule


def _materialize(
    dataset: Any,
    indices: Sequence[int],
    *,
    timestep: int,
    role: str,
    schedule: Sequence[tuple[str, float]],
    seed: int,
    torch: Any,
    functional: Any,
    device: Any,
    clean: bool = False,
) -> tuple[Any, Any]:
    images, labels = _raw_batch(dataset, indices, torch)
    if not clean:
        mode, severity = schedule[timestep]
        role_seed = int.from_bytes(hashlib.sha256(role.encode()).digest()[:4], "little")
        images = _corrupt(
            images, mode, severity, seed + 1009 * timestep + role_seed, torch, functional
        )
    return _model_input(images, torch, functional, device), labels.to(device, non_blocking=True)


def _cross_entropy(model: Any, batch: tuple[Any, Any], torch: Any, functional: Any) -> float:
    model.eval()
    with torch.no_grad():
        return float(functional.cross_entropy(model(batch[0]), batch[1]).item())


def _entropy_update(
    model: Any,
    batch: tuple[Any, Any],
    *,
    learning_rate: float,
    adapted_parameter_names: Sequence[str],
    torch: Any,
    device: Any,
) -> tuple[float, float, np.ndarray, float]:
    model.train()
    named = dict(model.named_parameters())
    parameters = [named[name] for name in adapted_parameter_names]
    _sync(torch, device)
    started = time.perf_counter()
    logits = model(batch[0])
    probabilities = logits.softmax(dim=1)
    entropy_per_item = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1)
    loss = entropy_per_item.mean()
    gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
    grad_norm = math.sqrt(sum(float(gradient.detach().square().sum()) for gradient in gradients))
    with torch.no_grad():
        for parameter, gradient in zip(parameters, gradients):
            parameter.add_(gradient, alpha=-learning_rate)
    _sync(torch, device)
    elapsed = time.perf_counter() - started
    return (
        float(loss.detach().item()),
        grad_norm,
        probabilities.detach().mean(dim=0).cpu().numpy(),
        elapsed,
    )


def _train_source_head(
    model: Any,
    dataset: Any,
    indices: Sequence[int],
    config: BenchmarkConfig,
    *,
    torch: Any,
    functional: Any,
    device: Any,
) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.fc.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=3e-3, weight_decay=1e-4)
    rng = np.random.default_rng(config.seed + 73)
    started = time.perf_counter()
    examples = 0
    for _epoch in range(config.source_epochs):
        order = np.asarray(indices, dtype=np.int64).copy()
        rng.shuffle(order)
        for start in range(0, len(order), config.source_batch_size):
            batch_indices = order[start : start + config.source_batch_size]
            images, labels = _raw_batch(dataset, tuple(map(int, batch_indices)), torch)
            inputs = _model_input(images, torch, functional, device)
            labels = labels.to(device, non_blocking=True)
            model.eval()
            model.fc.train()
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(model(inputs), labels)
            loss.backward()
            optimizer.step()
            examples += len(batch_indices)
    _sync(torch, device)
    return {
        "epochs": config.source_epochs,
        "unique_train_images": len(indices),
        "examples_processed": examples,
        "train_seconds": time.perf_counter() - started,
        "trained_parameters": ["fc.weight", "fc.bias"],
        "training_split": "official CIFAR-10 train",
    }


def _evaluate_stream(
    model: Any,
    dataset: Any,
    batches: Sequence[Sequence[int]],
    schedule: Sequence[tuple[str, float]],
    *,
    seed: int,
    torch: Any,
    functional: Any,
    device: Any,
    clean: bool,
) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for timestep, indices in enumerate(batches):
            inputs, labels = _materialize(
                dataset,
                indices,
                timestep=timestep,
                role="downstream",
                schedule=schedule,
                seed=seed,
                torch=torch,
                functional=functional,
                device=device,
                clean=clean,
            )
            logits = model(inputs)
            loss_sum += float(functional.cross_entropy(logits, labels, reduction="sum").item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
    return correct / total, loss_sum / total


def _candidate_scores(
    model: Any,
    dataset: Any,
    candidate_batches: Sequence[Sequence[int]],
    benefit_batches: Sequence[Sequence[int]],
    audit_batches: Sequence[Sequence[int]],
    validation_batches: Sequence[Sequence[int]],
    schedule: Sequence[tuple[str, float]],
    config: BenchmarkConfig,
    *,
    adapted_names: Sequence[str],
    state_names: Sequence[str],
    torch: Any,
    functional: Any,
    device: Any,
) -> tuple[list[CandidateRecord], dict[str, float]]:
    eligible = set(range(config.min_age, config.n_steps - max(config.horizons)))
    records: list[CandidateRecord] = []
    center: np.ndarray | None = None
    ledger = {
        "reference_write_all_update": 0.0,
        "eligible_candidate_update": 0.0,
        "state_fork_and_rollback": 0.0,
        "validation_selector": 0.0,
        "offline_oracle_probe": 0.0,
    }
    for timestep in range(config.n_steps):
        candidate_batch = _materialize(
            dataset,
            candidate_batches[timestep],
            timestep=timestep,
            role="candidate",
            schedule=schedule,
            seed=config.seed,
            torch=torch,
            functional=functional,
            device=device,
        )
        if timestep in eligible:
            future = []
            future_ids = []
            for horizon in config.horizons:
                probe_t = timestep + horizon
                future.append(
                    _materialize(
                        dataset,
                        benefit_batches[probe_t],
                        timestep=probe_t,
                        role="oracle-benefit",
                        schedule=schedule,
                        seed=config.seed,
                        torch=torch,
                        functional=functional,
                        device=device,
                    )
                )
                probe_bytes = np.asarray(benefit_batches[probe_t], dtype="<i8").tobytes()
                probe_digest = hashlib.sha256(probe_bytes).hexdigest()[:16]
                future_ids.append(f"oracle-benefit:t{probe_t}:{probe_digest}")
            audit_t = timestep - config.min_age
            audit_bytes = np.asarray(audit_batches[audit_t], dtype="<i8").tobytes()
            audit_digest = hashlib.sha256(audit_bytes).hexdigest()[:16]
            audit = _materialize(
                dataset,
                audit_batches[audit_t],
                timestep=audit_t,
                role="retention-audit",
                schedule=schedule,
                seed=config.seed,
                torch=torch,
                functional=functional,
                device=device,
            )
            validation = _materialize(
                dataset,
                validation_batches[timestep],
                timestep=timestep,
                role="validation",
                schedule=schedule,
                seed=config.seed,
                torch=torch,
                functional=functional,
                device=device,
            )
            probe_started = time.perf_counter()
            before_future = [_cross_entropy(model, batch, torch, functional) for batch in future]
            before_audit = _cross_entropy(model, audit, torch, functional)
            _sync(torch, device)
            ledger["offline_oracle_probe"] += time.perf_counter() - probe_started
            validation_started = time.perf_counter()
            before_validation = _cross_entropy(model, validation, torch, functional)
            _sync(torch, device)
            ledger["validation_selector"] += time.perf_counter() - validation_started

            copy_started = time.perf_counter()
            _state_subset(model, state_names)
            _sync(torch, device)
            ledger["state_fork_and_rollback"] += 2.0 * (time.perf_counter() - copy_started)

        entropy, gradient_norm, probability, update_seconds = _entropy_update(
            model,
            candidate_batch,
            learning_rate=config.adaptation_lr,
            adapted_parameter_names=adapted_names,
            torch=torch,
            device=device,
        )
        ledger["reference_write_all_update"] += update_seconds

        if timestep not in eligible:
            continue
        ledger["eligible_candidate_update"] += update_seconds
        after_started = time.perf_counter()
        after_future = [_cross_entropy(model, batch, torch, functional) for batch in future]
        after_audit = _cross_entropy(model, audit, torch, functional)
        _sync(torch, device)
        after_seconds = time.perf_counter() - after_started
        ledger["offline_oracle_probe"] += after_seconds
        validation_started = time.perf_counter()
        after_validation = _cross_entropy(model, validation, torch, functional)
        _sync(torch, device)
        ledger["validation_selector"] += time.perf_counter() - validation_started

        improvements = {
            str(horizon): float(before - after)
            for horizon, before, after in zip(config.horizons, before_future, after_future)
        }
        benefit = sum(
            weight * improvements[str(horizon)]
            for horizon, weight in zip(config.horizons, config.horizon_weights)
        )
        damage = after_audit - before_audit
        terms = compute_utility(
            benefit,
            damage,
            config.cost_loss_per_second * update_seconds,
            retention_lambda=config.retention_lambda,
        )
        if center is None:
            nonredundancy = 1.0
            center = probability.copy()
        else:
            denominator = np.linalg.norm(center) * np.linalg.norm(probability)
            similarity = float(np.dot(center, probability) / max(denominator, 1e-12))
            nonredundancy = 1.0 - similarity
            center = 0.9 * center + 0.1 * probability
        records.append(
            CandidateRecord(
                timestep=timestep,
                benefit=terms.benefit,
                benefit_by_horizon=improvements,
                damage=terms.damage,
                cost=terms.cost,
                cost_seconds=update_seconds,
                utility=terms.utility,
                validation_score=before_validation - after_validation,
                eata_score=-entropy + nonredundancy,
                sar_score=gradient_norm / max(entropy, 1e-8),
                entropy=entropy,
                gradient_norm=gradient_norm,
                oracle_probe_ids=tuple(future_ids),
                audit_probe_ids=(f"retention-audit:t{audit_t}:{audit_digest}",),
            )
        )
    return records, ledger


def _replay_arm(
    model: Any,
    source_state: dict[str, Any],
    source_adapted_state: dict[str, Any],
    dataset: Any,
    candidate_batches: Sequence[Sequence[int]],
    downstream_batches: Sequence[Sequence[int]],
    schedule: Sequence[tuple[str, float]],
    records: Sequence[CandidateRecord],
    selected: Sequence[int],
    *,
    method: str,
    fraction: float,
    replicate: int | None,
    selection_seconds: float,
    offline_oracle_seconds: float,
    dense_target_seconds: float | None,
    config: BenchmarkConfig,
    adapted_names: Sequence[str],
    state_names: Sequence[str],
    torch: Any,
    functional: Any,
    device: Any,
) -> ArmRecord:
    model.load_state_dict(source_state, strict=True)
    selected_set = set(selected)
    update_elapsed = 0.0
    optimizer_steps = 0
    writes = 0
    for timestep in range(config.n_steps):
        if timestep not in selected_set:
            continue
        batch = _materialize(
            dataset,
            candidate_batches[timestep],
            timestep=timestep,
            role="candidate",
            schedule=schedule,
            seed=config.seed,
            torch=torch,
            functional=functional,
            device=device,
        )
        repeats = 1
        if method == "dense" and dense_target_seconds is not None and selected:
            desired = (
                max(0.0, dense_target_seconds - selection_seconds) * (writes + 1) / len(selected)
            )
            repeats = 0
            while update_elapsed < desired or repeats == 0:
                _, _, _, elapsed = _entropy_update(
                    model,
                    batch,
                    learning_rate=config.adaptation_lr,
                    adapted_parameter_names=adapted_names,
                    torch=torch,
                    device=device,
                )
                update_elapsed += elapsed
                optimizer_steps += 1
                repeats += 1
        else:
            _, _, _, elapsed = _entropy_update(
                model,
                batch,
                learning_rate=config.adaptation_lr,
                adapted_parameter_names=adapted_names,
                torch=torch,
                device=device,
            )
            update_elapsed += elapsed
            optimizer_steps += 1
        writes += 1
        if method == "reset" and writes % config.reset_every == 0:
            reset_started = time.perf_counter()
            _restore_subset(model, source_adapted_state)
            _sync(torch, device)
            update_elapsed += time.perf_counter() - reset_started

    # This is the first function that receives downstream batches; callers
    # construct them only after every selection set has been locked.
    accuracy, cross_entropy = _evaluate_stream(
        model,
        dataset,
        downstream_batches,
        schedule,
        seed=config.seed,
        torch=torch,
        functional=functional,
        device=device,
        clean=False,
    )
    clean_accuracy, _ = _evaluate_stream(
        model,
        dataset,
        downstream_batches,
        schedule,
        seed=config.seed,
        torch=torch,
        functional=functional,
        device=device,
        clean=True,
    )
    by_timestep = {record.timestep: record for record in records}
    benefit = sum(by_timestep[t].benefit for t in selected)
    damage = sum(by_timestep[t].damage for t in selected)
    deployable = selection_seconds + update_elapsed
    terms = compute_utility(
        benefit,
        damage,
        config.cost_loss_per_second * deployable,
        retention_lambda=config.retention_lambda,
    )
    return ArmRecord(
        method=method,
        write_fraction=fraction,
        replicate=replicate,
        selected_timesteps=tuple(sorted(selected)),
        admitted_writes=writes,
        optimizer_steps=optimizer_steps,
        benefit=terms.benefit,
        damage=terms.damage,
        cost=terms.cost,
        utility=terms.utility,
        selection_seconds=selection_seconds,
        update_seconds=update_elapsed,
        deployable_seconds=deployable,
        offline_oracle_seconds=offline_oracle_seconds,
        downstream_accuracy=accuracy,
        downstream_cross_entropy=cross_entropy,
        clean_downstream_accuracy=clean_accuracy,
        exact_write_count=writes == len(selected),
    )


def run_recovery2_benchmark(
    config: BenchmarkConfig,
    *,
    cifar_archive: Path,
    resnet18_weights: Path,
    data_root: Path,
    artifact_checks: list[dict[str, Any]],
) -> BenchmarkResult:
    """Run the benchmark after external verification.

    Full mode is intentionally impossible without the confirmation stored in
    :class:`BenchmarkConfig`.  Pilot mode uses only the official train split and
    is forced to fail the realism/P7 gate.
    """

    for check in artifact_checks:
        if not check.get("verified", False):
            raise ValueError("all external artifact checks must pass before execution")
    if _checksum(cifar_archive, CIFAR10_ARCHIVE.checksum_algorithm) != CIFAR10_ARCHIVE.checksum:
        raise ValueError("CIFAR-10 archive checksum changed after preflight")
    if (
        _checksum(resnet18_weights, RESNET18_WEIGHTS.checksum_algorithm)
        != RESNET18_WEIGHTS.checksum
    ):
        raise ValueError("ResNet-18 weight checksum changed after preflight")

    torch, functional, torchvision = _torch_runtime()
    if not torch.cuda.is_available():
        raise RuntimeError("P2.2 recovery2 requires CUDA")
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    is_t4 = "T4" in properties.name.upper()
    if config.require_t4 and not is_t4:
        raise RuntimeError(f"full contract requires a T4; observed {properties.name}")
    _extract_cifar_archive(cifar_archive, data_root)
    train_dataset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False)
    plan = build_partition_plan(config)
    stream_dataset = (
        train_dataset
        if config.mode == "pilot"
        else torchvision.datasets.CIFAR10(root=data_root, train=False, download=False)
    )

    model = _load_official_model(torch, torchvision, resnet18_weights, config.seed).to(device)
    source_indices = plan.source_train_indices
    source_head = _train_source_head(
        model,
        train_dataset,
        source_indices,
        config,
        torch=torch,
        functional=functional,
        device=device,
    )
    adapted_names, buffer_names = _adaptation_scope(model, torch)
    state_names = tuple(dict.fromkeys((*adapted_names, *buffer_names)))
    source_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    source_adapted = _state_subset(model, state_names)

    candidate_batches = _step_batches(plan.candidate_indices, config.n_steps)
    oracle = plan.oracle_probe_indices
    benefit_indices = oracle[::2]
    audit_indices = oracle[1::2]
    benefit_batches = _step_batches(benefit_indices, config.n_steps)
    audit_batches = _step_batches(audit_indices, config.n_steps)
    validation_batches = _step_batches(plan.validation_indices, config.n_steps)
    schedule = _schedule(config.n_steps, config.regime_length)

    candidates, score_ledger = _candidate_scores(
        model,
        stream_dataset,
        candidate_batches,
        benefit_batches,
        audit_batches,
        validation_batches,
        schedule,
        config,
        adapted_names=adapted_names,
        state_names=state_names,
        torch=torch,
        functional=functional,
        device=device,
    )
    eligible = tuple(record.timestep for record in candidates)
    candidate_restriction_passed = eligible == tuple(
        range(config.min_age, config.n_steps - max(config.horizons))
    )
    if not candidate_restriction_passed:
        raise RuntimeError("candidate eligibility does not match the frozen age/horizon window")

    selection_charge = {
        "validation": score_ledger["eligible_candidate_update"]
        + score_ledger["state_fork_and_rollback"]
        + score_ledger["validation_selector"],
        "eata_style": score_ledger["eligible_candidate_update"],
        "sar_style": score_ledger["eligible_candidate_update"]
        + score_ledger["state_fork_and_rollback"],
        "random": 0.0,
        "reset": 0.0,
        "never": 0.0,
        "dense": 0.0,
        "oracle_ceiling": 0.0,
    }
    offline_oracle_seconds = (
        score_ledger["offline_oracle_probe"]
        + score_ledger["eligible_candidate_update"]
        + score_ledger["state_fork_and_rollback"]
    )

    locked_selections: dict[tuple[float, str, int | None], tuple[int, ...]] = {}
    rng = np.random.default_rng(config.seed + 991)
    for fraction in config.write_fractions:
        m = min(len(eligible), max(1, round(fraction * len(eligible))))
        arrays = {
            "validation": [record.validation_score for record in candidates],
            "eata_style": [record.eata_score for record in candidates],
            "sar_style": [record.sar_score for record in candidates],
            "oracle_ceiling": [record.utility for record in candidates],
        }
        for method, scores in arrays.items():
            locked_selections[(fraction, method, None)] = restricted_top_m(eligible, scores, m)
        regular = _evenly_spaced(eligible, m)
        locked_selections[(fraction, "dense", None)] = regular
        locked_selections[(fraction, "reset", None)] = regular
        locked_selections[(fraction, "never", None)] = ()
        for replicate in range(config.random_replicates):
            chosen = tuple(sorted(map(int, rng.choice(eligible, size=m, replace=False))))
            locked_selections[(fraction, "random", replicate)] = chosen

    # Selection lock is the boundary after which downstream units may be
    # materialized.  The plan existed earlier, but no downstream tensor or loss
    # was constructed in candidate scoring.
    selection_lock_sha256 = hashlib.sha256(
        json.dumps(
            {str(key): value for key, value in sorted(locked_selections.items(), key=str)},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    downstream_opened_after_lock = True
    downstream_batches = _step_batches(plan.downstream_indices, config.n_steps)

    arms: list[ArmRecord] = []
    for fraction in config.write_fractions:
        sparse_records: list[ArmRecord] = []
        ordered = ["validation", "eata_style", "sar_style", "reset", "oracle_ceiling"]
        for method in ordered:
            selected = locked_selections[(fraction, method, None)]
            sparse_records.append(
                _replay_arm(
                    model,
                    source_state,
                    source_adapted,
                    stream_dataset,
                    candidate_batches,
                    downstream_batches,
                    schedule,
                    candidates,
                    selected,
                    method=method,
                    fraction=fraction,
                    replicate=None,
                    selection_seconds=selection_charge[method],
                    offline_oracle_seconds=(
                        offline_oracle_seconds if method == "oracle_ceiling" else 0.0
                    ),
                    dense_target_seconds=None,
                    config=config,
                    adapted_names=adapted_names,
                    state_names=state_names,
                    torch=torch,
                    functional=functional,
                    device=device,
                )
            )
        for replicate in range(config.random_replicates):
            sparse_records.append(
                _replay_arm(
                    model,
                    source_state,
                    source_adapted,
                    stream_dataset,
                    candidate_batches,
                    downstream_batches,
                    schedule,
                    candidates,
                    locked_selections[(fraction, "random", replicate)],
                    method="random",
                    fraction=fraction,
                    replicate=replicate,
                    selection_seconds=0.0,
                    offline_oracle_seconds=0.0,
                    dense_target_seconds=None,
                    config=config,
                    adapted_names=adapted_names,
                    state_names=state_names,
                    torch=torch,
                    functional=functional,
                    device=device,
                )
            )
        deployable_sparse = [
            arm.deployable_seconds for arm in sparse_records if arm.method != "oracle_ceiling"
        ]
        dense_target = max(deployable_sparse)
        dense = _replay_arm(
            model,
            source_state,
            source_adapted,
            stream_dataset,
            candidate_batches,
            downstream_batches,
            schedule,
            candidates,
            locked_selections[(fraction, "dense", None)],
            method="dense",
            fraction=fraction,
            replicate=None,
            selection_seconds=0.0,
            offline_oracle_seconds=0.0,
            dense_target_seconds=dense_target,
            config=config,
            adapted_names=adapted_names,
            state_names=state_names,
            torch=torch,
            functional=functional,
            device=device,
        )
        never = _replay_arm(
            model,
            source_state,
            source_adapted,
            stream_dataset,
            candidate_batches,
            downstream_batches,
            schedule,
            candidates,
            (),
            method="never",
            fraction=fraction,
            replicate=None,
            selection_seconds=0.0,
            offline_oracle_seconds=0.0,
            dense_target_seconds=None,
            config=config,
            adapted_names=adapted_names,
            state_names=state_names,
            torch=torch,
            functional=functional,
            device=device,
        )
        arms.extend([never, dense, *sparse_records])

    methods = {arm.method for arm in arms}
    baselines_complete = (
        set(REQUIRED_METHODS) <= methods
        and sum(
            arm.method == "random" and arm.write_fraction == config.write_fractions[0]
            for arm in arms
        )
        == config.random_replicates
    )
    expected_m = {
        fraction: min(len(eligible), max(1, round(fraction * len(eligible))))
        for fraction in config.write_fractions
    }
    exact_counts = all(
        arm.exact_write_count
        and (
            arm.admitted_writes == 0
            if arm.method == "never"
            else arm.admitted_writes == expected_m[arm.write_fraction]
        )
        for arm in arms
    )
    compute_matches = []
    for fraction in config.write_fractions:
        dense = next(
            arm for arm in arms if arm.method == "dense" and arm.write_fraction == fraction
        )
        target = max(
            arm.deployable_seconds
            for arm in arms
            if arm.write_fraction == fraction
            and arm.method not in {"dense", "oracle_ceiling", "never"}
        )
        ratio = dense.deployable_seconds / max(target, 1e-12)
        compute_matches.append(1.0 <= ratio <= 1.0 + config.compute_match_tolerance)

    positive_fraction = float(np.mean([record.utility > 0 for record in candidates]))
    source_accuracy = max(arm.clean_downstream_accuracy for arm in arms if arm.method == "never")
    frontier_gains = []
    for fraction in config.write_fractions:
        oracle_arm = next(
            arm for arm in arms if arm.method == "oracle_ceiling" and arm.write_fraction == fraction
        )
        controls = [
            arm.downstream_accuracy
            for arm in arms
            if arm.write_fraction == fraction and arm.method != "oracle_ceiling"
        ]
        frontier_gains.append(oracle_arm.downstream_accuracy - max(controls))
    oracle_frontier_gain = max(frontier_gains)
    gates = evaluate_p7_qualification(
        mode=config.mode,
        artifacts_verified=all(check["verified"] for check in artifact_checks),
        t4_verified=is_t4,
        partitions_disjoint=plan.disjoint,
        downstream_opened_after_lock=downstream_opened_after_lock,
        full_horizons=all(
            set(record.benefit_by_horizon) == {str(value) for value in config.horizons}
            for record in candidates
        ),
        candidate_restriction_passed=candidate_restriction_passed,
        exact_counts=exact_counts,
        compute_matched=all(compute_matches),
        baselines_complete=baselines_complete,
        source_accuracy=source_accuracy,
        minimum_source_accuracy=config.minimum_source_accuracy,
        positive_utility_fraction=positive_fraction,
        minimum_positive_fraction=config.minimum_positive_utility_fraction,
        maximum_positive_fraction=config.maximum_positive_utility_fraction,
        oracle_frontier_gain=oracle_frontier_gain,
        minimum_accuracy_gain=config.minimum_accuracy_gain,
    )
    gates.update(
        {
            "selection_lock_sha256": selection_lock_sha256,
            "realistic_scale_passed": config.mode == "full",
            "useful_sparsity": 1.0 - positive_fraction,
            "compute_match_by_fraction": {
                str(fraction): passed
                for fraction, passed in zip(config.write_fractions, compute_matches)
            },
        }
    )
    environment = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": properties.name,
        "cuda_capability": f"{properties.major}.{properties.minor}",
        "is_t4": is_t4,
    }
    return BenchmarkResult(
        config=config,
        environment=environment,
        artifact_checks=artifact_checks,
        partitions=plan,
        corruption_schedule=[f"{mode}:{severity:.2f}" for mode, severity in schedule],
        adapted_parameters=adapted_names,
        mutable_bn_buffers=buffer_names,
        source_head={**source_head, "clean_downstream_accuracy": source_accuracy},
        candidates=candidates,
        arms=arms,
        compute_ledger={
            **score_ledger,
            "offline_oracle_total": offline_oracle_seconds,
            "deployable_arm_total": sum(arm.deployable_seconds for arm in arms),
        },
        gates=gates,
        events=[
            {
                "event": "selection_locked_before_downstream",
                "selection_lock_sha256": selection_lock_sha256,
            },
            {
                "event": "p7_qualification",
                "qualifies_p7": gates["qualifies_p7"],
            },
        ],
    )


def write_artifact(result: BenchmarkResult | dict[str, Any], destination: Path) -> Path:
    """Atomically write strict JSON suitable for later CLI ingestion."""

    artifact = result.artifact() if isinstance(result, BenchmarkResult) else result
    if artifact.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError(f"artifact schema must be {ARTIFACT_SCHEMA!r}")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    payload = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(payload)
    os.replace(temporary, destination)
    return destination
