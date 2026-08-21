"""Finding-the-Time-to-Think-compatible gate boundary and preflight.

The official method uses a learned gate over a frozen planner and selects one
of a finite set of planning budgets.  This adapter freezes that narrow API for
P1 without pretending that P1's heuristic FTT-lite schedule is the released
PPO gate or that its deterministic lookahead is AlphaZero/MCTS.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["FTTAdapter", "FTTPreflight", "inspect_external_ftt"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
GateFn = Callable[[FloatArray], IntArray]


@dataclass(frozen=True, slots=True)
class FTTAdapter:
    """Select a budget from state, planner-policy, and planner-value features."""

    budgets: tuple[int, ...]
    gate: GateFn

    def __post_init__(self) -> None:
        if not self.budgets or any(value <= 0 for value in self.budgets):
            raise ValueError("FTT budget options must be nonempty and positive")
        if len(set(self.budgets)) != len(self.budgets):
            raise ValueError("FTT budget options must be unique")

    def choose(
        self,
        state_features: FloatArray,
        planner_policy: FloatArray,
        planner_value: FloatArray,
    ) -> IntArray:
        state = np.asarray(state_features, dtype=np.float64)
        policy = np.asarray(planner_policy, dtype=np.float64)
        value = np.asarray(planner_value, dtype=np.float64)
        if state.ndim != 2 or policy.ndim != 2 or value.shape != (state.shape[0], 1):
            raise ValueError("FTT inputs must be batched matrices with scalar planner value")
        if policy.shape[0] != state.shape[0]:
            raise ValueError("FTT input batch dimensions must align")
        observation = np.concatenate((state, policy, value), axis=1)
        option = np.asarray(self.gate(observation), dtype=np.int64)
        if option.shape != (state.shape[0],) or np.any(
            (option < 0) | (option >= len(self.budgets))
        ):
            raise ValueError("FTT gate returned an invalid budget-option index")
        choices = np.asarray(self.budgets, dtype=np.int64)
        return choices[option]


@dataclass(frozen=True, slots=True)
class FTTPreflight:
    source_present: bool
    checkpoint_present: bool
    manifest_present: bool
    official_equivalence_available: bool
    source_path: str
    checkpoint_path: str
    detail: str


def inspect_external_ftt(params: Mapping[str, Any]) -> FTTPreflight:
    """Inspect the released repo/checkpoint without importing its JAX stack."""

    source = Path(str(params.get("ftt_source", ""))) if params.get("ftt_source") else None
    checkpoint = (
        Path(str(params.get("ftt_checkpoint", "")))
        if params.get("ftt_checkpoint")
        else None
    )
    source_present = bool(
        source
        and (source / "committed_action").is_dir()
        and (source / "clock").is_dir()
        and (source / "README.md").is_file()
    )
    checkpoint_present = bool(checkpoint and checkpoint.is_file())
    manifest = source / "checkpoints/MANIFEST.md" if source else None
    manifest_present = bool(manifest and manifest.is_file())
    available = source_present and checkpoint_present and manifest_present
    detail = json.dumps(
        {
            "source": str(source) if source else "",
            "source_present": source_present,
            "checkpoint": str(checkpoint) if checkpoint else "",
            "checkpoint_present": checkpoint_present,
            "manifest_present": manifest_present,
            "required_repo": "https://github.com/Aneeshers/realtime-rl-code",
        },
        sort_keys=True,
    )
    return FTTPreflight(
        source_present,
        checkpoint_present,
        manifest_present,
        available,
        str(source) if source else "",
        str(checkpoint) if checkpoint else "",
        detail,
    )


def load_t4_cells(path_value: object) -> tuple[bool, str]:
    """Validate an external measured-T4 timing cell artifact."""

    if not path_value:
        return False, "no T4 timing artifact configured"
    path = Path(str(path_value))
    if not path.is_file():
        return False, f"missing T4 timing artifact {path}"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return False, f"invalid T4 timing JSON: {error}"
    cells: Sequence[object] = data.get("cells", []) if isinstance(data, dict) else []
    required = {"budget", "event_rate_hz", "latency_s", "occupancy", "device"}
    passed = len(cells) >= 9 and all(
        isinstance(cell, dict)
        and required <= set(cell)
        and cell.get("device") == "T4"
        and float(cell.get("latency_s", 0.0)) > 0
        for cell in cells
    )
    return passed, f"path={path}, cells={len(cells)}"
