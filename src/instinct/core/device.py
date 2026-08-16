"""What machine a run happened on, for the manifest's ``device`` field.

CPU identity is read unconditionally — it costs nothing and it is the only
device every run has. GPU identity goes through :func:`instinct.core.optional.
try_import_torch`, because the same manifest schema has to hold whether a run
executed on a bare CPU smoke box or a Colab T4, and the field is simply absent
(not an error) in the first case.

:func:`assert_t4_defaults` is spec section 3.2's T4 checklist, restricted to
what is actually observable from inside the process: BF16/TF32 are opt-in torch
flags, so their state is a fact this module can read; "keep VRAM headroom below
13.5 GB peak" is a target for :mod:`instinct.core.profile` to measure after the
fact, not something checkable at any single instant, so this function checks
headroom *now* rather than promising the peak.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from typing import Any

from instinct.core.optional import try_import_torch

__all__ = ["T4ComplianceError", "assert_t4_defaults", "capture_device_info"]

#: spec section 3.2: "keep at least 2 GB VRAM headroom".
MIN_VRAM_HEADROOM_GB = 2.0


class T4ComplianceError(RuntimeError):
    """A T4 default from spec section 3.2 was violated."""


def _cpu_model() -> str:
    """A human-readable CPU identifier, best-effort across platforms."""
    proc = platform.processor()
    if proc:
        return proc
    try:
        # macOS's platform.processor() is often empty; sysctl has the real name.
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.machine() or "unknown"


@dataclass(frozen=True, slots=True)
class _GpuInfo:
    name: str
    total_vram_gb: float
    bf16_enabled: bool
    tf32_enabled: bool


def _gpu_info() -> _GpuInfo | None:
    torch = try_import_torch()
    if torch is None or not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return _GpuInfo(
        name=props.name,
        total_vram_gb=props.total_memory / (1024**3),
        bf16_enabled=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction")
        else False,
        tf32_enabled=bool(torch.backends.cuda.matmul.allow_tf32),
    )


def capture_device_info() -> dict[str, Any]:
    """CPU always; GPU fields only when torch and CUDA are both available."""
    info: dict[str, Any] = {"cpu": _cpu_model(), "cpu_count": _cpu_count()}
    gpu = _gpu_info()
    if gpu is not None:
        info["gpu"] = gpu.name
        info["gpu_total_vram_gb"] = round(gpu.total_vram_gb, 3)
        info["gpu_bf16_enabled"] = gpu.bf16_enabled
        info["gpu_tf32_enabled"] = gpu.tf32_enabled
    return info


def _cpu_count() -> int:
    import os

    return os.cpu_count() or 0


def assert_t4_defaults(*, min_headroom_gb: float = MIN_VRAM_HEADROOM_GB) -> None:
    """Raise if the running process violates spec section 3.2's T4 defaults.

    A no-op when torch or CUDA is unavailable: this check exists to catch a
    misconfigured T4 job, not to force CUDA onto a CPU-only run.
    """
    torch = try_import_torch()
    if torch is None or not torch.cuda.is_available():
        return

    if torch.backends.cuda.matmul.allow_tf32:
        raise T4ComplianceError(
            "TF32 is enabled (torch.backends.cuda.matmul.allow_tf32=True). "
            "T4 is compute capability 7.5 and does not have TF32 tensor cores; "
            "this flag is a no-op there at best and a correctness footgun on "
            "any other device this code might run on. Set it False for T4 runs."
        )
    if getattr(torch.backends.cudnn, "allow_tf32", False):
        raise T4ComplianceError(
            "TF32 is enabled for cuDNN (torch.backends.cudnn.allow_tf32=True). "
            "Not supported on T4 (compute capability 7.5); set it False."
        )

    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024**3)
    if free_gb < min_headroom_gb:
        raise T4ComplianceError(
            f"only {free_gb:.2f} GB VRAM free, below the {min_headroom_gb:.1f} GB headroom "
            "spec section 3.2 requires. This check runs before the workload allocates its "
            "own tensors, so headroom this low now means the run is already too close to "
            "the T4's 16 GB ceiling to trust the 13.5 GB peak target."
        )
