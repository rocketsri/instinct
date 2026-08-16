"""Spec section 3.4: no train/validation/final-test overlap, checked generically."""

from __future__ import annotations

import pytest

from instinct.core.holdout import OverlapError, SplitRegistry


def test_disjoint_assignments_are_accepted() -> None:
    reg = SplitRegistry()
    reg.assign_all(["a", "b"], "train")
    reg.assign_all(["c"], "val")
    reg.assign_all(["d", "e"], "test")
    assert reg.role("train") == frozenset({"a", "b"})
    assert reg.role("test") == frozenset({"d", "e"})
    assert len(reg) == 5


def test_reassigning_the_same_identifier_to_a_different_role_raises() -> None:
    reg = SplitRegistry()
    reg.assign("ctx-1", "train")
    with pytest.raises(OverlapError):
        reg.assign("ctx-1", "test")


def test_reassigning_to_the_same_role_is_a_harmless_no_op() -> None:
    reg = SplitRegistry()
    reg.assign("ctx-1", "train")
    reg.assign("ctx-1", "train")  # must not raise
    assert reg.role_of("ctx-1") == "train"


def test_unregistered_identifier_has_no_role() -> None:
    reg = SplitRegistry()
    assert reg.role_of("never-seen") is None


def test_context_level_identifiers_survive_being_tuples() -> None:
    """P1's holdout is over whole (speed, latency) contexts, not budget points
    (spec section 4.6) — identifiers are naturally tuples, not scalars."""
    reg = SplitRegistry()
    reg.assign((1.0, 0.1), "train")
    reg.assign((2.0, 0.2), "test")
    with pytest.raises(OverlapError):
        reg.assign((1.0, 0.1), "test")
