"""Exact fork and rollback for cheap-to-copy state.

**This is evaluation infrastructure, not a contribution.** Forking a simulator to
ask "what would have happened if the trigger had said yes" is standard practice
and nothing here is novel. It lives in the repo because the audited-trigger
experiments need a *correct* version of it, and the incorrect versions are
subtle enough to be worth writing down.

The incorrect version, specifically
-----------------------------------
The obvious fork is ``state.copy()``. For a container of NumPy arrays, the
obvious ``copy`` is shallow: the fork's dict is new, the arrays inside it are
the parent's. The fork then advances, writes in place, and the parent's state
has silently changed. Every subsequent measurement in the parent branch is
contaminated, and it is contaminated *plausibly* — no exception, no NaN, just a
trajectory that came from somewhere else.

So the default fork here is :func:`copy.deepcopy`, which is always right and
sometimes slow, and a caller who wants the fast shallow-ish path
(:meth:`~instinct.core.env.EnvState.copy`, which does copy its arrays) passes it
explicitly. Rule 7: the fast copier is the fast path, ``deepcopy`` is the naive
reference, and ``tests/test_infra.py`` asserts they agree.

The second incorrect version is forgetting that a fork must not *consume*
randomness the parent would have drawn. That one is already solved:
:mod:`instinct.core.rng` is counter-based, so a lane's noise is a pure function
of its identity and the tick. A fork that keeps its lane ids sees exactly the
noise its parent would have seen, which is what makes the comparison
counterfactual rather than merely similar. :func:`branch_actions` relies on
that, and it is the reason this module can be twenty lines of copying rather
than a replay engine.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar

import numpy as np

from instinct.core.env import BatchedEnv, EnvState, StepResult
from instinct.core.rng import SeedScope

__all__ = [
    "CounterfactualError",
    "assert_unchanged",
    "branch_actions",
    "fork",
    "fork_scope",
    "rollback_on_exit",
    "states_equal",
]

S = TypeVar("S")

Copier = Callable[[Any], Any]


class CounterfactualError(AssertionError):
    """A fork leaked into its parent, or a rollback did not restore."""


def fork(state: S, *, copier: Copier | None = None) -> S:
    """An independent copy of ``state``.

    Defaults to :func:`copy.deepcopy`. Pass ``copier=lambda s: s.copy()`` when
    the type's own copy is known to be deep enough — :class:`EnvState` qualifies,
    because its ``copy`` copies every field array — and expect the equivalence
    test to cover the claim.
    """
    if copier is None:
        return copy.deepcopy(state)
    out: S = copier(state)
    return out


def states_equal(a: Any, b: Any) -> bool:
    """Structural equality that understands NumPy arrays.

    ``==`` on arrays returns an array, so a plain ``a == b`` on anything holding
    arrays is either a ``ValueError`` or, worse, a truthiness check on a
    one-element array that happens to work until the batch grows. This walks the
    structure instead.

    NaN compares equal to NaN here. A state field that legitimately holds NaN
    (a masked-out lane, say) must not make "unchanged" impossible to assert.
    """
    if isinstance(a, EnvState) or isinstance(b, EnvState):
        if not (isinstance(a, EnvState) and isinstance(b, EnvState)):
            return False
        return states_equal(
            {"lane_ids": a.lane_ids, **a.fields}, {"lane_ids": b.lane_ids, **b.fields}
        )
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        arr_a, arr_b = np.asarray(a), np.asarray(b)
        if arr_a.shape != arr_b.shape or arr_a.dtype.kind != arr_b.dtype.kind:
            return False
        if arr_a.dtype.kind == "f":
            return bool(np.array_equal(arr_a, arr_b, equal_nan=True))
        return bool(np.array_equal(arr_a, arr_b))
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        if not (isinstance(a, Mapping) and isinstance(b, Mapping)):
            return False
        return set(a) == set(b) and all(states_equal(a[k], b[k]) for k in a)
    if isinstance(a, str | bytes) or isinstance(b, str | bytes):
        return bool(a == b)
    if isinstance(a, Sequence) or isinstance(b, Sequence):
        if not (isinstance(a, Sequence) and isinstance(b, Sequence)):
            return False
        return len(a) == len(b) and all(states_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return bool(a == b)


def assert_unchanged(before: Any, after: Any, *, what: str = "state") -> None:
    """Fail loudly when a fork modified what it was supposed to leave alone.

    The whole hazard of fork-based counterfactuals is that a violation produces
    a wrong number rather than an error, so the check has to be explicit and it
    has to raise.
    """
    if not states_equal(before, after):
        raise CounterfactualError(
            f"{what} changed across a counterfactual fork. The fork aliased its parent: "
            "some array was shared rather than copied, and the parent branch is now "
            "contaminated. Use fork(..., copier=None) (deepcopy) until the fast copier "
            "is fixed."
        )


@contextmanager
def fork_scope(state: S, *, copier: Copier | None = None, verify: bool = True) -> Iterator[S]:
    """Yield a fork of ``state``, and check the original survived untouched.

    >>> import numpy as np
    >>> from instinct.core.env import EnvState
    >>> s = EnvState(lane_ids=np.arange(2), fields={"x": np.zeros(2)})
    >>> with fork_scope(s) as branch:
    ...     branch.fields["x"] += 1.0
    >>> float(s.fields["x"].sum())
    0.0

    ``verify`` snapshots the parent as well, which doubles the copying. It is on
    by default because these forks are cheap by construction — this module is
    only for cheap-to-copy state — and because an aliasing bug found at the end
    of a sweep costs the whole sweep.
    """
    reference = copy.deepcopy(state) if verify else None
    branch = fork(state, copier=copier)
    yield branch
    if verify:
        assert_unchanged(reference, state, what="parent state")


@contextmanager
def rollback_on_exit(state: S) -> Iterator[S]:
    """Hand out the *original* object and restore its contents afterwards.

    For state that callers insist on mutating in place. Restoration is done by
    copying a snapshot's contents back into the live object's arrays, so any
    reference the caller kept still points at correct data — replacing the object
    would leave stale references behind, which is its own quiet bug.

    Only supports :class:`EnvState` and mappings of arrays, because in-place
    restoration cannot be written generically without guessing.
    """
    snapshot = copy.deepcopy(state)
    try:
        yield state
    finally:
        _restore_into(state, snapshot)


def _restore_into(target: Any, snapshot: Any) -> None:
    if isinstance(target, EnvState) and isinstance(snapshot, EnvState):
        target.lane_ids[...] = snapshot.lane_ids
        for name, arr in target.fields.items():
            arr[...] = snapshot.fields[name]
        return
    if isinstance(target, dict) and isinstance(snapshot, Mapping):
        for name, value in target.items():
            if isinstance(value, np.ndarray):
                value[...] = snapshot[name]
            else:
                target[name] = snapshot[name]
        return
    raise TypeError(
        f"cannot restore {type(target).__name__} in place; "
        "use fork_scope and rebind instead of mutating."
    )


def branch_actions(
    env: BatchedEnv,
    state: EnvState,
    actions: Sequence[int] | np.ndarray,
    *,
    scope: SeedScope,
    episode: int = 0,
    tick: int = 0,
    verify: bool = True,
) -> dict[int, StepResult]:
    """Step the same state under several candidate actions, one branch each.

    Returns ``{action: StepResult}``. Every branch sees **identical** environment
    noise, because lane ids are preserved and the RNG is counter-based, so the
    difference between two branches is the action and nothing else. This is the
    audit primitive: it is what tells you what the declined action would have
    returned.

    Note the contrast with :meth:`~instinct.core.env.EnvState.relabel`, which
    exists for the opposite case — branches that must *not* share noise. Sharing
    is right here and wrong there, and getting it backwards silently changes
    what is being estimated, so neither is the default.
    """
    reference = copy.deepcopy(state) if verify else None
    out: dict[int, StepResult] = {}
    for a in np.asarray(actions, dtype=np.int64).ravel().tolist():
        branch = state.copy()
        act = np.full(branch.n_lanes, int(a), dtype=np.int64)
        out[int(a)] = env.step(branch, act, scope=scope, episode=episode, tick=tick)
    if verify:
        assert_unchanged(reference, state, what="branched state")
    return out
