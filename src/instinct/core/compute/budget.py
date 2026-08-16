"""Matched-compute accounting, and the tripwire that enforces it.

Rule 4 says two arms may only be compared at matched total compute. The reason is
mundane and it is the reason most "our method is better" plots are wrong: if arm
A ran 3x longer than arm B, the difference between them is not attributable to
the intervention, and no amount of statistics downstream repairs that. So the
comparison has to refuse to render rather than render something misleading —
hence :func:`assert_matched`, which raises.

Which currency?
---------------
The obvious answer is FLOPs, because FLOPs are hardware-independent and easy to
count analytically. On our hardware that answer is **wrong**, and we measured it
rather than assumed it. From ``reports/t4_hardware.md`` (section 3), a SwiGLU
fast-weight update at batch 32 on a T4:

    width  64 -> 512  is an 8x width increase, a 64x FLOP increase,
    and it costs 80.7 us versus 84.8 us. Constant wall-clock.

The kernels in our regime are launch-bound, not arithmetic-bound. A FLOP count
in that regime is a count of work the GPU was never waiting on, so matching two
arms on FLOPs can match them on a quantity that has almost no relationship to
the resource actually being spent. Two arms matched to within 1% on FLOPs can
differ by 60x in wall-clock, and vice versa.

Therefore: **wall-clock is the default basis**, FLOPs are recorded and reported
but are advisory. The ledger stores both because the regime is a property of the
hardware and the kernel size, not a law — the LM arm at width 1024+ starts to be
arithmetic-bound (the same table shows real GFLOP/s appearing there), and if a
future run lands in that regime the FLOP column is the one that will matter.
Recording both costs nothing; recording only one loses the ability to notice the
regime changed.

Wall-clock is noisy, and this is why the tolerance has an absolute floor as well
as a relative one. A 4-CPU shared container will happily add 5 ms of scheduler
jitter to a 20 ms arm; failing CI on that would train everyone to raise the
tolerance until it means nothing. Set ``atol`` to the jitter you can measure and
``rtol`` to the mismatch you actually care about.

There are no ratios in the failure report (rule 2). A mismatch is reported as a
*gap in basis units* — seconds, simulations, FLOPs — alongside both endpoints.
The tolerance is computed as ``atol + rtol * max(cost)``, the ``np.isclose``
form, whose denominator is the largest cost present and therefore cannot be
small when the gap is not.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Literal, get_args

__all__ = [
    "DEFAULT_BASIS",
    "Basis",
    "ComputeCost",
    "ComputeLedger",
    "ComputeMismatchError",
    "MatchReport",
    "assert_matched",
    "check_matched",
]

# The three things worth counting. `simulations` is the planner-side unit the
# rollout engine already tracks (`ArmTrace.simulations`); it is the cleanest
# basis for the tabular arms, where there is no GPU and wall-clock is dominated
# by NumPy call overhead.
Basis = Literal["wall_clock", "simulations", "flops"]

DEFAULT_BASIS: Basis = "wall_clock"

#: Why the default is not "flops". Kept as a constant so an error message can
#: point at the measurement rather than at an opinion.
FLOPS_ADVISORY = (
    "FLOPs are advisory on this hardware: reports/t4_hardware.md section 3 measures "
    "a 64x FLOP increase (width 64->512) at constant wall-clock, because the kernels "
    "are launch-bound. Match on wall_clock unless you have measured that your regime "
    "is arithmetic-bound."
)


class ComputeMismatchError(AssertionError):
    """Raised when two arms did not get the same amount of compute.

    Deliberately an :class:`AssertionError` rather than a :class:`ValueError`:
    this is a violated invariant of the experiment, not a bad argument, and it
    should read as a failed check in a test report.
    """


@dataclass(frozen=True, slots=True)
class ComputeCost:
    """What one arm spent.

    All fields are cumulative totals for the arm, never rates. A rate divides by
    a wall-clock that may itself be the thing under comparison, which is exactly
    the small-denominator trap rule 2 exists to avoid.
    """

    wall_clock_s: float = 0.0
    simulations: int = 0
    flops: int = 0
    env_steps: int = 0
    planner_calls: int = 0

    def __add__(self, other: ComputeCost) -> ComputeCost:
        return ComputeCost(
            wall_clock_s=self.wall_clock_s + other.wall_clock_s,
            simulations=self.simulations + other.simulations,
            flops=self.flops + other.flops,
            env_steps=self.env_steps + other.env_steps,
            planner_calls=self.planner_calls + other.planner_calls,
        )

    def amount(self, basis: Basis) -> float:
        """The scalar this cost contributes on a given matching basis."""
        if basis == "wall_clock":
            return float(self.wall_clock_s)
        if basis == "simulations":
            return float(self.simulations)
        if basis == "flops":
            return float(self.flops)
        raise ValueError(f"unknown basis {basis!r}; expected one of {get_args(Basis)}")

    def as_record(self) -> dict[str, float | int]:
        return {
            "wall_clock_s": self.wall_clock_s,
            "simulations": self.simulations,
            "flops": self.flops,
            "env_steps": self.env_steps,
            "planner_calls": self.planner_calls,
        }


@dataclass
class ComputeLedger:
    """Per-arm compute totals, accumulated as the experiment runs.

    Mutable on purpose: a run adds to it from many places (the rollout loop, the
    planner, the oracle) and the whole point is that the total is assembled from
    every contributor rather than estimated once at the end. An estimate made at
    the end is the thing that lets an arm quietly acquire extra compute.

    >>> led = ComputeLedger("p1-smoke")
    >>> led.record("actual", wall_clock_s=1.00, simulations=512)
    >>> led.record("fresh", wall_clock_s=1.02, simulations=512)
    >>> check_matched(led, rtol=0.05).matched
    True
    """

    label: str = ""
    costs: dict[str, ComputeCost] = field(default_factory=dict)

    # -- recording --------------------------------------------------------

    def record(
        self,
        arm: str,
        *,
        wall_clock_s: float = 0.0,
        simulations: int = 0,
        flops: int = 0,
        env_steps: int = 0,
        planner_calls: int = 0,
    ) -> None:
        """Add to an arm's running total. Creates the arm if it is new."""
        delta = ComputeCost(
            wall_clock_s=wall_clock_s,
            simulations=simulations,
            flops=flops,
            env_steps=env_steps,
            planner_calls=planner_calls,
        )
        self.costs[arm] = self.costs.get(arm, ComputeCost()) + delta

    @contextmanager
    def measure(
        self,
        arm: str,
        *,
        simulations: int = 0,
        flops: int = 0,
        env_steps: int = 0,
        planner_calls: int = 0,
    ) -> Iterator[None]:
        """Time a block and charge it to ``arm``, along with any counted work.

        Uses :func:`time.perf_counter`, not :func:`time.process_time`. Wall-clock
        is the currency precisely because the T4 measurement says the process is
        often *waiting* — on kernel launches, on the host-device round trip — and
        process time does not see waiting. An arm that waits twice as long has
        spent twice as much of the resource we actually have.

        The block is charged even when it raises, so a crashed arm still shows
        the compute it burned before crashing.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(
                arm,
                wall_clock_s=time.perf_counter() - start,
                simulations=simulations,
                flops=flops,
                env_steps=env_steps,
                planner_calls=planner_calls,
            )

    # -- reading ----------------------------------------------------------

    @property
    def arms(self) -> tuple[str, ...]:
        """Arm names in insertion order, so reports are stable across runs."""
        return tuple(self.costs)

    def cost(self, arm: str) -> ComputeCost:
        if arm not in self.costs:
            raise KeyError(f"no compute recorded for arm {arm!r}; have {self.arms}")
        return self.costs[arm]

    def total(self) -> ComputeCost:
        out = ComputeCost()
        for c in self.costs.values():
            out = out + c
        return out

    def merged(self, other: ComputeLedger) -> ComputeLedger:
        """Combine two ledgers, summing per arm.

        Needed because sharded and parallel runs each build their own ledger;
        matching has to be checked on the sum, not on one shard that happened to
        be balanced.
        """
        out = replace(self, costs=dict(self.costs))
        for arm, cost in other.costs.items():
            out.costs[arm] = out.costs.get(arm, ComputeCost()) + cost
        return out

    def as_records(self) -> list[dict[str, float | int | str]]:
        """Flat rows for the run manifest / a parquet table."""
        rows: list[dict[str, float | int | str]] = []
        for arm in self.arms:
            row: dict[str, float | int | str] = {"label": self.label, "arm": arm}
            row.update(self.costs[arm].as_record())
            rows.append(row)
        return rows


@dataclass(frozen=True, slots=True)
class MatchReport:
    """The outcome of a matched-compute check, in basis units throughout."""

    basis: Basis
    matched: bool
    lo_arm: str
    hi_arm: str
    lo: float
    hi: float
    tolerance: float
    n_arms: int

    @property
    def gap(self) -> float:
        """Largest pairwise difference, which is simply ``hi - lo``."""
        return self.hi - self.lo

    def __str__(self) -> str:
        unit = {"wall_clock": "s", "simulations": "sims", "flops": "FLOP"}[self.basis]
        verdict = "matched" if self.matched else "MISMATCHED"
        return (
            f"{verdict} on {self.basis}: gap {self.gap:.6g} {unit} "
            f"({self.lo_arm}={self.lo:.6g}, {self.hi_arm}={self.hi:.6g}), "
            f"tolerance {self.tolerance:.6g} {unit}, {self.n_arms} arms"
        )


def _extremes(values: list[float]) -> tuple[int, int]:
    """Indices of the cheapest and most expensive arm.

    The fast path for "largest pairwise gap": the maximum of ``|x_i - x_j|`` over
    all pairs is ``max(x) - min(x)``, so one linear scan replaces the O(n^2)
    double loop. Rule 7 applies even to something this small —
    :func:`_extremes_naive` is the reference and ``tests/test_infra.py`` asserts
    they agree.
    """
    lo = min(range(len(values)), key=lambda i: values[i])
    hi = max(range(len(values)), key=lambda i: values[i])
    return lo, hi


def _extremes_naive(values: list[float]) -> tuple[int, int]:
    """Reference: enumerate every pair and keep the widest. Never used in anger."""
    best = (0, 0)
    best_gap = -1.0
    for i in range(len(values)):
        for j in range(len(values)):
            gap = values[j] - values[i]
            if gap > best_gap:
                best_gap, best = gap, (i, j)
    return best


def _as_costs(
    ledger: ComputeLedger | Mapping[str, ComputeCost],
) -> dict[str, ComputeCost]:
    return dict(ledger.costs) if isinstance(ledger, ComputeLedger) else dict(ledger)


def check_matched(
    ledger: ComputeLedger | Mapping[str, ComputeCost],
    *,
    basis: Basis = DEFAULT_BASIS,
    rtol: float = 0.05,
    atol: float = 0.0,
    arms: Iterable[str] | None = None,
) -> MatchReport:
    """Non-raising form: report whether the arms got the same compute.

    ``arms`` restricts the check to a subset, which is the usual case — the
    ``base`` arm is *supposed* to be cheaper, and including it would make every
    check fail for the wrong reason. Matching is a claim about the arms being
    compared, not about every arm that exists.

    Tolerance is ``atol + rtol * max(cost)``. The relative part is scaled by the
    largest cost rather than by a difference or a mean, so the denominator can
    never be small while the gap is large.
    """
    costs = _as_costs(ledger)
    names = list(arms) if arms is not None else list(costs)
    missing = [a for a in names if a not in costs]
    if missing:
        raise KeyError(f"no compute recorded for arms {missing}; have {sorted(costs)}")
    if len(names) < 2:
        # One arm is trivially matched with itself. Say so rather than raising:
        # a smoke run with a single arm is legitimate.
        only = names[0] if names else ""
        amount = costs[only].amount(basis) if names else 0.0
        return MatchReport(basis, True, only, only, amount, amount, atol, len(names))

    values = [costs[a].amount(basis) for a in names]
    i_lo, i_hi = _extremes(values)
    tolerance = atol + rtol * max(abs(v) for v in values)
    return MatchReport(
        basis=basis,
        matched=(values[i_hi] - values[i_lo]) <= tolerance,
        lo_arm=names[i_lo],
        hi_arm=names[i_hi],
        lo=values[i_lo],
        hi=values[i_hi],
        tolerance=tolerance,
        n_arms=len(names),
    )


def assert_matched(
    ledger: ComputeLedger | Mapping[str, ComputeCost],
    *,
    basis: Basis = DEFAULT_BASIS,
    rtol: float = 0.05,
    atol: float = 0.0,
    arms: Iterable[str] | None = None,
    what: str = "arms",
) -> MatchReport:
    """Matched-compute tripwire. Raises :class:`ComputeMismatchError` when it trips.

    Call this *before* rendering any comparison. The failure it prevents is not
    a crash — an unmatched comparison produces a perfectly plausible number that
    attributes a compute difference to the intervention. Nothing downstream can
    detect that, which is why the check has to be here and has to raise.

    Returns the report on success so a caller can log the observed gap; a
    comparison that passed with a 4.9% gap is worth having in the manifest.
    """
    report = check_matched(ledger, basis=basis, rtol=rtol, atol=atol, arms=arms)
    if not report.matched:
        detail = FLOPS_ADVISORY if basis == "flops" else ""
        raise ComputeMismatchError(
            f"{what} are not matched on compute. {report}. "
            "Refusing to compare: a compute difference of this size is "
            "indistinguishable from an effect of the intervention."
            + (f"\n{detail}" if detail else "")
        )
    return report
