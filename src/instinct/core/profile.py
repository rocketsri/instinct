"""``profile.json``: what a run actually cost, beyond the compute ledger.

:class:`~instinct.core.compute.budget.ComputeLedger` already tracks wall-clock,
simulations, FLOPs, env steps, and planner calls *per arm*, which is what
matched-compute comparisons need. ``profile.json`` is the run-wide complement
spec section 3.3 lists separately: peak RAM and VRAM, which are not per-arm
quantities (the OS reports a process's peak, not one attributable to whichever
arm happened to be running when the peak occurred) and which spec section 3.2
asks to be profiled explicitly ("scoring, synchronization, transfers, queue
delay, validation, rollback, and checkpointing").

RAM is read via ``psutil`` unconditionally — every run has RAM to measure.
VRAM goes through :func:`instinct.core.optional.try_import_torch` for the same
reason :mod:`instinct.core.device` does: ``None`` on a CPU-only run is correct,
not missing data.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil

from instinct.core.optional import try_import_torch

if TYPE_CHECKING:
    from instinct.core.compute.budget import ComputeLedger

__all__ = ["PROFILE_NAME", "current_ram_mb", "peak_vram_mb", "write_profile_json"]

PROFILE_NAME = "profile.json"


def current_ram_mb() -> float:
    """This process's current resident set size, in MB.

    Not the peak — ``psutil`` has no portable peak-RSS query, only the OS's
    "maximum resident set size" counter on POSIX (``ru_maxrss``, units differ by
    platform) or nothing at all on some containers. A caller wanting a peak
    samples this repeatedly (e.g. from the periodic checkpoint hook) and keeps
    the running max, which is what ``write_profile_json``'s ``ram_peak_mb``
    parameter expects to receive.
    """
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)


def peak_vram_mb() -> float | None:
    """Peak CUDA memory allocated by this process, in MB, or ``None`` off-GPU."""
    torch = try_import_torch()
    if torch is None or not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024**2)


def write_profile_json(
    path: str | Path,
    ledger: ComputeLedger,
    *,
    ram_peak_mb: float,
    vram_peak_mb: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write ``profile.json``: the ledger's per-arm costs plus run-wide peaks.

    ``extra`` is for the specific bottlenecks spec section 3.2 names —
    scoring/synchronization/transfer/queue-delay/validation/rollback/checkpoint
    time — which are proposal-specific measurements a caller accumulates and
    hands in rather than something this generic writer could know how to time.
    """
    payload: dict[str, Any] = {
        "ledger_label": ledger.label,
        "arms": ledger.as_records(),
        "ram_peak_mb": ram_peak_mb,
        "vram_peak_mb": vram_peak_mb,
    }
    if extra:
        payload["extra"] = dict(extra)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))
