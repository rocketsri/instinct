"""The seam between the CLI and the seven proposals.

This file is the reason Stage B's six independent branches (P1, P2, P3, P4,
P5, P6) can build in parallel without ever touching each other's directories
or this one: :data:`PROPOSAL_MODULES` names all seven proposal packages once,
here, and nothing else in ``core/`` imports from any of them. A proposal
package satisfies :class:`ProposalPlugin` *structurally* — by exposing
``validate``/``run``/``resume`` functions at module level — so registering a
new proposal never requires editing a shared registry class or subclassing
anything; it only requires the module to exist at the name this file already
reserved for it.

``load_plugin`` will raise :class:`ModuleNotFoundError` for every proposal
until its Stage B branch lands — that is expected during Stage A and remains
correct behavior afterward for any proposal not yet built: a config that
names an unbuilt proposal should fail loudly, not silently no-op.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from instinct.core.configschema import ProposalRunConfig
    from instinct.core.results import ResultsWriter
    from instinct.core.verdict import PrerequisiteCheck, VerdictReport

__all__ = ["PROPOSAL_MODULES", "ProposalPlugin", "load_plugin"]

#: The seven proposals, fixed at Stage A. Frozen here — see the module
#: docstring for why no Stage B branch ever needs to edit this tuple.
PROPOSAL_MODULES: tuple[str, ...] = (
    "p1_atlas",
    "p2_harmful_write",
    "p3_codesign",
    "p4_slack",
    "p5_development",
    "p6_certification",
    "p7_chunking",
)


@runtime_checkable
class ProposalPlugin(Protocol):
    """What ``instinct.<proposal>.run`` must expose.

    A plain module satisfies this Protocol structurally (no base class, no
    registration call) as long as it defines these three functions at module
    scope with these signatures. That is deliberate: a proposal package is
    free to organize its internals however its own science demands, as long
    as this one entry point is stable.
    """

    def validate(self, config: ProposalRunConfig) -> list[PrerequisiteCheck]:
        """Cheap checks only: config sanity, upstream-dependency status,
        known-answer/control setup. Must never open a run directory or spend
        compute — see ``cli.py``'s ``validate`` command."""
        ...

    def run(self, config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
        """Execute the proposal's gate/pilot/decisive run and return its verdict."""
        ...

    def resume(
        self, config: ProposalRunConfig, writer: ResultsWriter, checkpoint_state: Mapping[str, Any]
    ) -> VerdictReport:
        """Pick up from ``checkpoint_state`` (as saved by this proposal's own
        ``run`` via ``core.checkpoint.CheckpointStore``) and finish the run."""
        ...


def load_plugin(proposal_id: str) -> ModuleType:
    """Import ``instinct.<proposal_id>.run`` and return it as a :class:`ProposalPlugin`.

    Raises :class:`ValueError` for a name outside :data:`PROPOSAL_MODULES`
    (a typo should fail before an import is even attempted) and lets
    :class:`ModuleNotFoundError` propagate for a real, registered proposal
    whose package does not exist yet — that is a "this proposal isn't built"
    error, and disguising it as anything else would hide real progress.
    """
    if proposal_id not in PROPOSAL_MODULES:
        raise ValueError(
            f"unknown proposal {proposal_id!r}; expected one of {PROPOSAL_MODULES}"
        )
    module = import_module(f"instinct.{proposal_id}.run")
    missing = [name for name in ("validate", "run", "resume") if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            f"instinct.{proposal_id}.run is missing required plugin function(s) {missing}"
        )
    return module
