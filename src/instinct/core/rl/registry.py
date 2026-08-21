"""Explicit train/validation/evaluation episode ownership."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from instinct.core.holdout import SplitRegistry

__all__ = ["EpisodeRegistry", "EpisodeRole"]

EpisodeRole = Literal["train", "validation", "eval"]
_ROLES: tuple[EpisodeRole, ...] = ("train", "validation", "eval")


@dataclass
class EpisodeRegistry:
    """Fail-closed registry for whole episode IDs.

    Registration is explicit; collection refuses unknown IDs as well as IDs
    assigned to a different role. This prevents a convenient evaluation call
    from quietly turning final episodes into additional training data.
    """

    _split: SplitRegistry = field(default_factory=SplitRegistry)

    def register(self, role: EpisodeRole, episode_ids: Iterable[int]) -> None:
        if role not in _ROLES:
            raise ValueError(f"unknown episode role {role!r}")
        ids = [int(episode_id) for episode_id in episode_ids]
        if any(episode_id < 0 or episode_id >= 2**32 for episode_id in ids):
            raise ValueError("episode IDs must lie in [0, 2**32)")
        self._split.assign_all(ids, role)

    def require(self, role: EpisodeRole, episode_ids: Sequence[int]) -> None:
        wrong = [
            (int(episode_id), self._split.role_of(int(episode_id)))
            for episode_id in episode_ids
            if self._split.role_of(int(episode_id)) != role
        ]
        if wrong:
            raise ValueError(f"episodes are not registered for {role!r}: {wrong[:5]}")

    def ids(self, role: EpisodeRole) -> tuple[int, ...]:
        values = self._split.role(role)
        if not all(isinstance(value, int) for value in values):
            raise TypeError("episode registry contains a non-integer ID")
        return tuple(sorted(value for value in values if isinstance(value, int)))

    def state_dict(self) -> dict[str, list[int]]:
        return {role: list(self.ids(role)) for role in _ROLES}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        restored = EpisodeRegistry()
        for role in _ROLES:
            raw = state.get(role, [])
            if not isinstance(raw, list):
                raise ValueError(f"registry role {role!r} must be a list")
            restored.register(role, [int(value) for value in raw])
        self._split = restored._split

    @classmethod
    def from_splits(
        cls,
        *,
        train: Iterable[int],
        validation: Iterable[int],
        eval: Iterable[int],
    ) -> EpisodeRegistry:
        out = cls()
        out.register("train", train)
        out.register("validation", validation)
        out.register("eval", eval)
        return out
