"""The use ledger: finite-budget holdouts that retire when spent.

This module is here to stop one specific way the whole result could be wrong.

The retention term of ``U_t`` is an expectation over aged information ``A_t``.
The cheap implementation is to carve off a retention set once and evaluate every
candidate write against it forever. That is the reusable-holdout problem, moved
somewhere it is harder to see: with hundreds of candidates scored against the
same probes, "damage to ``A_t``" degrades into "damage to these particular
rows", any gate learned downstream overfits them, and the frontier that gets
plotted is a frontier against a memorized set. Nothing raises. The numbers just
stop meaning what the axis label says.

So every probe carries a use budget and is **retired** when it is exhausted, and
asking for a retired probe raises :class:`ProbeExhaustedError` rather than quietly
handing it back. The failure mode becomes a stack trace instead of an
optimistic curve.

Two consequences the callers have to live with, and should:

* The aged pool is consumable. A long stream needs a supply of aged
  information, which means ``A_t`` has to *rotate* — which is what the
  specification asked for anyway ("aged information, not a permanently reused
  holdout").
* Budgets are part of the experiment's compute accounting. ``usage_report``
  reports what was actually spent, so a comparison that claims matched compute
  can be checked rather than asserted.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "AGED",
    "AUDIT",
    "FRESH",
    "Probe",
    "ProbeExhaustedError",
    "ProbeLedger",
    "UsageReport",
]

AGED = "aged"
FRESH = "fresh"
AUDIT = "audit"


class ProbeExhaustedError(RuntimeError):
    """Raised when a retired probe is requested again.

    Deliberately not a warning. A warning in a sweep is a line in a log nobody
    reads, and the resulting curve is indistinguishable from a valid one.
    """


@dataclass(frozen=True, slots=True)
class Probe:
    """One indivisible unit of held-out evidence.

    ``born`` is the stream timestep the information dates from; it is what
    ``age`` is measured against, and the only reason a probe knows anything
    about the stream that produced it.
    """

    probe_id: str
    inputs: Tensor
    targets: Tensor
    born: int
    kind: str = AGED
    source_chunk: int = -1

    @property
    def n_rows(self) -> int:
        return int(self.inputs.shape[0])


@dataclass(frozen=True, slots=True)
class UsageReport:
    """What the ledger actually spent, for the compute-matching audit."""

    registered: int
    charges: int
    rows_evaluated: int
    retired: int
    remaining_budget: int


@dataclass(slots=True)
class _Entry:
    probe: Probe
    budget: int
    used: int = 0


@dataclass(slots=True)
class ProbeLedger:
    """A pool of finite-use probes.

    Not thread-safe and not meant to be: a single ledger per experiment run is
    the point, because the budget is a property of the *run*, not of a worker.
    """

    default_budget: int = 2
    _entries: dict[str, _Entry] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)
    _charges: int = 0
    _rows: int = 0

    # -- registration ------------------------------------------------------

    def register(self, probe: Probe, budget: int | None = None) -> None:
        if probe.probe_id in self._entries:
            raise ValueError(f"probe {probe.probe_id!r} is already registered")
        b = self.default_budget if budget is None else budget
        if b < 1:
            raise ValueError(f"budget={b} must be at least 1")
        self._entries[probe.probe_id] = _Entry(probe=probe, budget=b)
        self._order.append(probe.probe_id)

    def register_all(self, probes: Iterable[Probe], budget: int | None = None) -> None:
        for p in probes:
            self.register(p, budget)

    # -- inspection --------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def remaining(self, probe_id: str) -> int:
        return self._entry(probe_id).budget - self._entry(probe_id).used

    def is_retired(self, probe_id: str) -> bool:
        return self.remaining(probe_id) <= 0

    def peek(self, probe_id: str) -> Probe:
        """Read a probe's identity without charging it.

        Only for bookkeeping — shapes, ``born``, ``source_chunk``. Anything that
        touches ``inputs``/``targets`` for evaluation must go through
        :meth:`charge`, or the budget means nothing.
        """
        return self._entry(probe_id).probe

    def available(
        self,
        *,
        kind: str | None = None,
        born_at_most: int | None = None,
        born_at_least: int | None = None,
        source_chunks: Sequence[int] | None = None,
    ) -> list[str]:
        """Ids of probes that still have budget, in registration order."""
        allowed = None if source_chunks is None else set(source_chunks)
        out: list[str] = []
        for pid in self._order:
            e = self._entries[pid]
            if e.used >= e.budget:
                continue
            if kind is not None and e.probe.kind != kind:
                continue
            if born_at_most is not None and e.probe.born > born_at_most:
                continue
            if born_at_least is not None and e.probe.born < born_at_least:
                continue
            if allowed is not None and e.probe.source_chunk not in allowed:
                continue
            out.append(pid)
        return out

    def usage_report(self) -> UsageReport:
        retired = sum(1 for e in self._entries.values() if e.used >= e.budget)
        left = sum(max(0, e.budget - e.used) for e in self._entries.values())
        return UsageReport(
            registered=len(self._entries),
            charges=self._charges,
            rows_evaluated=self._rows,
            retired=retired,
            remaining_budget=left,
        )

    # -- spending ----------------------------------------------------------

    def charge(self, probe_id: str, uses: int = 1) -> Probe:
        """Spend ``uses`` from a probe's budget and hand back its data.

        Raises if the probe is unknown, or if it is retired, or if this single
        request would take it past its budget. The last case is worth being
        strict about: a partial charge would let a caller evaluate rows it has
        not paid for and still see no error.
        """
        e = self._entry(probe_id)
        if uses < 1:
            raise ValueError(f"uses={uses} must be at least 1")
        left = e.budget - e.used
        if left <= 0:
            raise ProbeExhaustedError(
                f"probe {probe_id!r} was retired after {e.used}/{e.budget} uses; "
                "reusing it would reintroduce the reusable-holdout problem into the retention term"
            )
        if uses > left:
            raise ProbeExhaustedError(
                f"probe {probe_id!r} has {left} use(s) left but {uses} were requested"
            )
        e.used += uses
        self._charges += uses
        self._rows += uses * e.probe.n_rows
        return e.probe

    def draw(self, probe_ids: Sequence[str], uses: int = 1) -> list[Probe]:
        """Charge several probes at once.

        All-or-nothing: the availability of every id is checked before any
        budget moves, so a failed draw leaves the ledger untouched and the
        caller can be re-run.
        """
        for pid in probe_ids:
            e = self._entry(pid)
            if e.budget - e.used < uses:
                raise ProbeExhaustedError(
                    f"probe {pid!r} has {e.budget - e.used} use(s) left but {uses} were requested; "
                    "the draw was rejected whole and no budget was spent"
                )
        return [self.charge(pid, uses) for pid in probe_ids]

    def _entry(self, probe_id: str) -> _Entry:
        try:
            return self._entries[probe_id]
        except KeyError:
            raise KeyError(f"no probe registered under {probe_id!r}") from None


def stack_probes(probes: Sequence[Probe]) -> tuple[Tensor, Tensor]:
    """Concatenate probe rows into one ``(N, d_in)`` / ``(N, d_out)`` pair.

    The oracle evaluates a timestep's probes in a single forward, so they are
    concatenated rather than looped over. Returns empty tensors for an empty
    draw, which is a legitimate state late in a stream once the aged pool has
    been spent.
    """
    if not probes:
        raise ValueError("cannot stack an empty probe list; callers must handle the empty draw")
    return (
        torch.cat([p.inputs for p in probes], dim=0),
        torch.cat([p.targets for p in probes], dim=0),
    )


def probe_row_weights(probes: Sequence[Probe]) -> np.ndarray:
    """Uniform weight per *probe*, spread across that probe's rows.

    Weighting by row instead would silently give a probe with more rows more
    say in the retention term, which is a sampling choice, not an accident to
    inherit from how the chunks happened to be split.
    """
    total = len(probes)
    return np.concatenate([np.full(p.n_rows, 1.0 / (total * p.n_rows)) for p in probes])
