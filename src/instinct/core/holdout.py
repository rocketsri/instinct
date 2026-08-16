"""The generic form of "no train/validation/final-test overlap" (spec section 3.4).

Every proposal that holds out contexts, states, streams, or probes needs this
same check, and every proposal's holdout unit is different — P1 holds out
speed/latency/environment/reflex *contexts* (spec section 4.6: "split whole
contexts, not individual budget points"), P2 holds out *probes* (already
served by :class:`instinct.ttt.probes.ProbeLedger`'s finite-use budget, a
different and complementary guarantee — a probe not being reused indefinitely
is not the same claim as a probe never crossing from train into test). This
module is the one proposal-agnostic primitive underneath all of them:
register which role (``"train"``, ``"val"``, ``"test"``, or any other role
name a proposal wants) each identifier belongs to, and get a hard error the
moment an identifier would belong to two roles at once, rather than a silent
metric that quietly mixes them.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field

__all__ = ["OverlapError", "SplitRegistry"]


class OverlapError(ValueError):
    """An identifier was assigned to more than one role."""


@dataclass
class SplitRegistry:
    """Tracks which role each identifier belongs to; raises on any overlap.

    Deliberately minimal: no notion of what an "identifier" is (a context key,
    a probe id, a state index — whatever a proposal's split actually happens
    to be over), because the invariant that matters is role-exclusivity, not
    any particular splitting strategy.
    """

    _role_of: dict[Hashable, str] = field(default_factory=dict)

    def assign(self, identifier: Hashable, role: str) -> None:
        """Assign ``identifier`` to ``role``. Raises if it already has a different role.

        Assigning the same identifier to the same role twice is a no-op, not an
        error — a proposal re-registering a context it already placed is not a
        leak.
        """
        existing = self._role_of.get(identifier)
        if existing is not None and existing != role:
            raise OverlapError(
                f"identifier {identifier!r} was assigned to role {existing!r} and is now "
                f"being assigned to role {role!r}. An identifier used for both is exactly "
                "the train/test overlap this registry exists to catch."
            )
        self._role_of[identifier] = role

    def assign_all(self, identifiers: Iterable[Hashable], role: str) -> None:
        for identifier in identifiers:
            self.assign(identifier, role)

    def role_of(self, identifier: Hashable) -> str | None:
        return self._role_of.get(identifier)

    def role(self, name: str) -> frozenset[Hashable]:
        return frozenset(k for k, v in self._role_of.items() if v == name)

    def __len__(self) -> int:
        return len(self._role_of)
