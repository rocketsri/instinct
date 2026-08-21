"""Canonical composite checkpoint for an in-progress RL training split."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from instinct.core.rl.accounting import AccountingHook
from instinct.core.rl.registry import EpisodeRegistry
from instinct.core.rl.types import Learner
from instinct.core.rng import SeedScope

__all__ = ["TrainingCheckpoint"]


@dataclass(frozen=True, slots=True)
class TrainingCheckpoint:
    next_episode_index: int
    learner_state: dict[str, Any]
    registry_state: dict[str, list[int]]
    accounting_state: dict[str, int | float]
    seed_root: int
    seed_path: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.next_episode_index < 0:
            raise ValueError("next_episode_index must be nonnegative")
        if self.seed_root < 0:
            raise ValueError("seed_root must be nonnegative")

    @classmethod
    def capture(
        cls,
        next_episode_index: int,
        learner: Learner,
        registry: EpisodeRegistry,
        accounting: AccountingHook,
        scope: SeedScope,
    ) -> TrainingCheckpoint:
        return cls(
            next_episode_index=next_episode_index,
            learner_state=learner.state_dict(),
            registry_state=registry.state_dict(),
            accounting_state=accounting.state_dict(),
            seed_root=scope.root_seed,
            seed_path=scope.path,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "next_episode_index": self.next_episode_index,
            "learner_state": self.learner_state,
            "registry_state": self.registry_state,
            "accounting_state": self.accounting_state,
            "seed_root": self.seed_root,
            "seed_path": list(self.seed_path),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> TrainingCheckpoint:
        learner = state.get("learner_state")
        registry = state.get("registry_state")
        accounting = state.get("accounting_state")
        if not isinstance(learner, Mapping):
            raise ValueError("learner_state must be a mapping")
        if not isinstance(registry, Mapping):
            raise ValueError("registry_state must be a mapping")
        if not isinstance(accounting, Mapping):
            raise ValueError("accounting_state must be a mapping")
        seed_path = state.get("seed_path")
        if not isinstance(seed_path, list) or not all(
            isinstance(value, str) for value in seed_path
        ):
            raise ValueError("seed_path must be a list of strings")
        return cls(
            next_episode_index=int(state.get("next_episode_index", -1)),
            learner_state=dict(learner),
            registry_state={str(key): list(value) for key, value in registry.items()},
            accounting_state={str(key): value for key, value in accounting.items()},
            seed_root=int(state.get("seed_root", -1)),
            seed_path=tuple(seed_path),
        )

    def restored_scope(self) -> SeedScope:
        """Reconstruct the exact counter-based RNG namespace for resumption."""

        return SeedScope(self.seed_root, self.seed_path)

    def restore(
        self,
        learner: Learner,
        registry: EpisodeRegistry,
        accounting: AccountingHook,
    ) -> int:
        learner.load_state_dict(self.learner_state)
        registry.load_state_dict(self.registry_state)
        accounting.load_state_dict(self.accounting_state)
        return self.next_episode_index
