"""Fail-closed artifact contract for final P2 realistic recovery.

Recovery 2 must not silently fall back to another generated image task.  This
module therefore checks the exact official CIFAR-10 archive, official
TorchVision ResNet-18 weights, a working Torch/TorchVision runtime, and CUDA.
Only after every condition passes may a realistic runner reserve the fixed,
fresh whole-stream partitions documented below.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = [
    "CIFAR10_ARCHIVE",
    "RESNET18_WEIGHTS",
    "ArtifactRequirement",
    "Recovery2Preflight",
    "inspect_recovery2",
]


@dataclass(frozen=True, slots=True)
class ArtifactRequirement:
    name: str
    url: str
    checksum_algorithm: str
    checksum: str
    expected_bytes: int


CIFAR10_ARCHIVE = ArtifactRequirement(
    name="cifar-10-python.tar.gz",
    url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    checksum_algorithm="md5",
    checksum="c58f30108f718f92721af3b95e74349a",
    expected_bytes=170_498_071,
)

RESNET18_WEIGHTS = ArtifactRequirement(
    name="resnet18-f37072fd.pth",
    url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
    checksum_algorithm="sha256",
    checksum="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    expected_bytes=46_830_571,
)


@dataclass(frozen=True, slots=True)
class ArtifactCheck:
    requirement: ArtifactRequirement
    path: str
    exists: bool
    size_matches: bool
    checksum_matches: bool
    observed_bytes: int
    observed_checksum: str

    @property
    def passed(self) -> bool:
        return self.exists and self.size_matches and self.checksum_matches


@dataclass(frozen=True, slots=True)
class Recovery2Preflight:
    torch_importable: bool
    torchvision_importable: bool
    cuda_available: bool
    runtime_detail: str
    artifacts: tuple[ArtifactCheck, ...]
    fixed_partitions: dict[str, tuple[int, int]]
    realistic_contract_passed: bool
    qualifies_p7: bool = False

    def metric_row(self) -> dict[str, object]:
        return {
            "record_type": "p2_2_recovery2_preflight",
            "torch_importable": self.torch_importable,
            "torchvision_importable": self.torchvision_importable,
            "cuda_available": self.cuda_available,
            "runtime_detail": self.runtime_detail,
            "artifact_checks": [
                {
                    **asdict(check.requirement),
                    "path": check.path,
                    "exists": check.exists,
                    "size_matches": check.size_matches,
                    "checksum_matches": check.checksum_matches,
                    "observed_bytes": check.observed_bytes,
                    "observed_checksum": check.observed_checksum,
                }
                for check in self.artifacts
            ],
            "fixed_partitions": {
                name: f"permuted_test_indices[{start}:{stop}]"
                for name, (start, stop) in self.fixed_partitions.items()
            },
            "realistic_contract_passed": self.realistic_contract_passed,
            "matched_count_passed": False,
            "matched_compute_passed": False,
            "useful_sparsity": 0.0,
            "qualifies_p7": self.qualifies_p7,
        }


def _checksum(path: Path, algorithm: str) -> str:
    digest = hashlib.md5() if algorithm == "md5" else hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path_value: object, requirement: ArtifactRequirement) -> ArtifactCheck:
    path = Path(str(path_value)) if path_value else Path("__missing__")
    exists = bool(path_value) and path.is_file()
    observed_bytes = path.stat().st_size if exists else 0
    observed = _checksum(path, requirement.checksum_algorithm) if exists else ""
    checksum_matches = observed == requirement.checksum
    return ArtifactCheck(
        requirement,
        str(path) if path_value else "",
        exists,
        observed_bytes == requirement.expected_bytes,
        checksum_matches,
        observed_bytes,
        observed,
    )


def _runtime() -> tuple[bool, bool, bool, str]:
    has_torch = importlib.util.find_spec("torch") is not None
    has_vision = importlib.util.find_spec("torchvision") is not None
    if not (has_torch and has_vision):
        return has_torch, has_vision, False, "torch and torchvision are both required"
    try:
        torch = importlib.import_module("torch")
        vision = importlib.import_module("torchvision")
        cuda = bool(torch.cuda.is_available())
        return True, True, cuda, f"torch={torch.__version__}, torchvision={vision.__version__}"
    except Exception as error:  # pragma: no cover - depends on broken external runtimes
        return False, False, False, f"runtime import failed: {type(error).__name__}: {error}"


def inspect_recovery2(params: dict[str, object]) -> Recovery2Preflight:
    """Inspect prerequisites without downloading, mutating, or reserving eval units."""

    torch_ok, vision_ok, cuda, detail = _runtime()
    artifacts = (
        _artifact(params.get("cifar10_archive"), CIFAR10_ARCHIVE),
        _artifact(params.get("resnet18_weights"), RESNET18_WEIGHTS),
    )
    # Apply a seed-241 permutation only after artifacts pass. These ranges are
    # whole, disjoint CIFAR-10 test-image streams and are fresh relative to R0/R1.
    partitions = {
        "candidate": (0, 2500),
        "oracle_probe": (2500, 5000),
        "validation": (5000, 7500),
        "downstream": (7500, 10000),
    }
    passed = bool(torch_ok and vision_ok and cuda and all(item.passed for item in artifacts))
    return Recovery2Preflight(
        torch_ok,
        vision_ok,
        cuda,
        detail,
        artifacts,
        partitions,
        passed,
        False,
    )
