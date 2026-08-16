"""The config fields spec sections 2.1, 2.3, and 3.1 require of every proposal run.

:mod:`instinct.core.config` already gives every run a strict, typed,
content-hashed config (:class:`~instinct.core.config.RunConfig`). What it does
not yet have is the portfolio's own required content — the three pre-run
statements spec section 2.1 says every manifest must carry, the compute
envelope spec section 3.1 puts a ceiling on per run class, and the pilot/eval
seed separation spec section 2.3 requires. :class:`ProposalRunConfig` adds
exactly those on top of :class:`~instinct.core.config.RunConfig`, through the
same strict :func:`~instinct.core.config.from_mapping` loader — an unknown key
in a proposal's ``smoke.yaml`` fails the same way an unknown key in the base
config already does.

``ComputeLimits`` defaults come from spec section 3.1's table so a config only
has to *override* a limit it wants tighter than the class default, not restate
the whole table on every proposal's every config file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from instinct.core.config import RunConfig, from_mapping, load_config

__all__ = [
    "COMPUTE_LIMIT_DEFAULTS",
    "ComputeLimits",
    "PreregistrationConfig",
    "ProposalRunConfig",
    "ResumeConfig",
    "RunClass",
    "load_proposal_config",
    "proposal_config_from_mapping",
]

RunClass = Literal["smoke", "cpu", "t4", "decisive"]


@dataclass(frozen=True, slots=True)
class PreregistrationConfig:
    """Spec section 2.1's three required pre-run statements.

    Required, not optional, because the discipline that matters ("never turn
    INCONCLUSIVE into PASS or KILL") only works if the falsifiable claim and its
    non-claim were written down *before* the run, not reconstructed afterward to
    fit whatever the numbers showed.
    """

    estimand: str
    primary_hypothesis: str
    non_claim: str


#: Spec section 3.1's resource envelope, keyed by run class. A config's
#: ``compute.run_class`` selects its row; explicit ``*_max`` fields in the YAML
#: override the row's value rather than being required to restate it.
COMPUTE_LIMIT_DEFAULTS: dict[RunClass, dict[str, float | None]] = {
    "smoke": {"wall_clock_minutes_max": 15.0, "cpu_hours_max": None, "gpu_hours_max": None},
    "cpu": {"wall_clock_minutes_max": None, "cpu_hours_max": 2.0, "gpu_hours_max": 2.0},
    "t4": {"wall_clock_minutes_max": None, "cpu_hours_max": 2.0, "gpu_hours_max": 2.0},
    "decisive": {"wall_clock_minutes_max": None, "cpu_hours_max": 12.0, "gpu_hours_max": 8.0},
}


@dataclass(frozen=True, slots=True)
class ComputeLimits:
    """Spec section 3.1's per-run-class ceiling, resolved for one config.

    ``wall_clock_minutes_max`` is the smoke tier's limit (15 CPU-min / 20 T4-min
    — the two are not distinguished here because a smoke run is defined by being
    trivially fast on *either* device, not by which one it happens to run on).
    ``cpu_hours_max``/``gpu_hours_max`` govern the mechanism-pilot and decisive
    tiers.

    A YAML config only has to name its ``run_class``; any ``*_max`` field left
    unset is filled from :data:`COMPUTE_LIMIT_DEFAULTS` in ``__post_init__``, so
    every constructed ``ComputeLimits`` is already resolved — there is no
    separate "raw" and "resolved" form for a caller to confuse. An explicit
    ``*_max`` in the YAML overrides its row's default rather than being
    required to restate the whole table. A field still ``None`` after that
    means "no ceiling of that kind applies to this run class" (e.g. a smoke run
    has no CPU/GPU-hour ceiling because 15 minutes already bounds it tighter
    than any hour figure would).
    """

    run_class: RunClass = "smoke"
    wall_clock_minutes_max: float | None = None
    cpu_hours_max: float | None = None
    gpu_hours_max: float | None = None

    def __post_init__(self) -> None:
        defaults = COMPUTE_LIMIT_DEFAULTS[self.run_class]
        if self.wall_clock_minutes_max is None:
            object.__setattr__(self, "wall_clock_minutes_max", defaults["wall_clock_minutes_max"])
        if self.cpu_hours_max is None:
            object.__setattr__(self, "cpu_hours_max", defaults["cpu_hours_max"])
        if self.gpu_hours_max is None:
            object.__setattr__(self, "gpu_hours_max", defaults["gpu_hours_max"])


@dataclass(frozen=True, slots=True)
class ResumeConfig:
    """How a run checkpoints itself for spec section 3.2's interruption model."""

    checkpoint_every_minutes: float = 15.0
    allow_resume: bool = True


@dataclass(frozen=True, slots=True)
class ProposalRunConfig(RunConfig):
    """Everything :class:`~instinct.core.config.RunConfig` has, plus the portfolio's rules.

    ``seeds`` is the full set a run is allowed to touch; ``pilot_seeds`` and
    ``eval_seeds`` partition it per spec section 2.3 ("separate pilot seeds from
    locked evaluation seeds"). They are not required to partition ``seeds``
    exactly — a proposal may keep extra seeds in reserve — but a shared test in
    ``tests/test_core_holdout.py`` asserts ``pilot_seeds`` and ``eval_seeds``
    never overlap each other, which is the actual invariant spec 2.3 cares
    about.
    """

    preregistration: PreregistrationConfig = field(
        default_factory=lambda: PreregistrationConfig("", "", "")
    )
    seeds: list[int] = field(default_factory=list)
    pilot_seeds: list[int] = field(default_factory=list)
    eval_seeds: list[int] = field(default_factory=list)
    compute: ComputeLimits = field(default_factory=ComputeLimits)
    resume: ResumeConfig = field(default_factory=ResumeConfig)


def load_proposal_config(path: str | Path) -> ProposalRunConfig:
    """Read a ``configs/pN/.../*.yaml`` file into a validated :class:`ProposalRunConfig`.

    Thin wrapper over :func:`instinct.core.config.load_config` so every
    proposal loads its config the same way and gets the same unknown-key
    rejection and nearest-match suggestion on a typo.
    """
    return load_config(path, ProposalRunConfig)


# Re-exported so a proposal config module needs only this import for the
# common case of building a ``ProposalRunConfig`` from a plain mapping in a
# test fixture, without reaching into ``instinct.core.config`` separately.
def proposal_config_from_mapping(data: dict) -> ProposalRunConfig:  # pragma: no cover - thin
    return from_mapping(ProposalRunConfig, data)
