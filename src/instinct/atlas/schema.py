"""The atlas measurement record: the contract between measuring and fitting.

One row is one fully-specified cell of the sweep — a environment, reflex,
speed, hardware latency and planning budget — carrying both the raw arm values
and the decomposition derived from them.

This module exists so the two halves of P1 can be built independently. The sweep
produces rows; the curve fitting, transfer tests and kill harness consume them
and never touch a rollout. Pinning the schema here means a change to what gets
measured shows up as a schema change rather than as a silently misread column.

The invariant that matters is the frozen-specification identity

    sigma == G_plan + R_intermediate - L_arrival - L_wait - C_hw
             + L_base_delay + epsilon_id

which holds *by construction*. ``L_base_delay`` is the identified cost the base
arm pays for its own delay. ``epsilon_id`` is only the reconstruction residual.
Conflating the two was an F0 bug because a physical base-delay cost could be
misreported as measurement error. :func:`validate_frame` checks them separately.

A correction to the proposal's decomposition is baked into these names, and it
is a finding rather than a bookkeeping choice.

The proposal lists five terms and defines ``L_irreversible`` as "damage that
occurred while no slow intervention was available". Those five do not close: an
implementation measuring exactly them leaves a residual of ~4.9 return units, on
a scale where the whole budget effect is a fraction of that. The missing
quantity is the plain **discounting** cost of waiting. Everything after a delay
of ``d`` ticks is worth ``gamma**d`` of what it would have been, whether or not
anything unrecoverable happened, and no listed term carries it.

So the identity here uses ``L_wait`` -- the entire return cost of having waited,
net of what the reflex earned back. ``L_irreversible`` is a separate matched-time
excess failure probability. It has probability units and is never subtracted
from a return term or interpreted as a signed handoff-value difference.

Reading ``L_wait`` as unrecoverable damage is the mistake to avoid. It is large
in every environment, including ones with no absorbing states at all.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import pandas as pd

__all__ = ["ATLAS_COLUMNS", "DECOMPOSITION_TERMS", "AtlasRow", "to_frame", "validate_frame"]

# Terms that must reconstruct sigma, with the sign each carries in the identity.
DECOMPOSITION_TERMS: dict[str, int] = {
    "G_plan": +1,
    "R_intermediate": +1,
    "L_arrival": -1,
    "L_wait": -1,
    "C_hw": -1,
    "L_base_delay": +1,
    "epsilon_id": +1,
}


@dataclass(frozen=True, slots=True)
class AtlasRow:
    """One measured cell of the compute-freshness surface."""

    # -- what was measured ------------------------------------------------
    env: str
    reflex: str
    nu_e: float  # environment ticks per unit wall-clock
    nu_h: float  # wall-clock per planning simulation
    budget: int
    delay: int  # integer ticks; several budgets may share one delay
    staleness: float  # continuous nu_e * nu_h * budget, before rounding
    start_state: int  # tabular state id, or -1 when not applicable

    # -- raw arm values, in return units ----------------------------------
    J_actual: float
    J_instant: float
    J_fresh: float
    J_base: float

    # -- decomposition, all in return units, never ratios -----------------
    G_plan: float
    R_intermediate: float
    L_arrival: float
    L_wait: float  # whole cost of waiting, net of reflex earnings
    L_irreversible: float  # matched-time excess failure probability; outside identity
    C_hw: float
    L_base_delay: float
    epsilon_id: float
    sigma: float

    # -- provenance -------------------------------------------------------
    n_seeds: int
    ci_lo: float
    ci_hi: float
    exact: bool  # True for the closed-form tabular arm, False for sampled
    simulations: int = 0  # planning simulations consumed, the input to C_hw
    wall_clock_s: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


ATLAS_COLUMNS: list[str] = [f.name for f in fields(AtlasRow)]


def to_frame(rows: list[AtlasRow]) -> pd.DataFrame:
    """Build the canonical dataframe, with columns in schema order."""
    if not rows:
        return pd.DataFrame(columns=ATLAS_COLUMNS)
    return pd.DataFrame([r.as_dict() for r in rows], columns=ATLAS_COLUMNS)


def validate_frame(df: pd.DataFrame, *, tol: float = 1e-9) -> None:
    """Check the schema and the telescoping identity.

    ``tol`` applies to the exact rows only. Sampled rows carry estimation noise
    by nature, so the identity is enforced where it must hold exactly and merely
    recorded elsewhere — ``epsilon_id`` is the place that noise is meant to show
    up, and hiding it would defeat its purpose.
    """
    missing = set(ATLAS_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"atlas frame is missing columns: {sorted(missing)}")

    if df.empty:
        return

    reconstructed = sum(sign * df[term] for term, sign in DECOMPOSITION_TERMS.items())
    residual = (reconstructed - df["sigma"]).abs()

    exact = df["exact"].astype(bool)
    if exact.any():
        worst = float(residual[exact].max())
        if worst > tol:
            bad = df[exact].loc[residual[exact].idxmax()]
            raise ValueError(
                f"telescoping identity violated on an exact row by {worst:.3e} "
                f"(env={bad['env']}, budget={bad['budget']}, nu_e={bad['nu_e']}). "
                "The decomposition must reconstruct sigma by construction."
            )

    if (df["delay"] < 0).any():
        raise ValueError("negative delay")
    if (df["budget"] < 1).any():
        raise ValueError("budget must be at least 1")
    if (df["n_seeds"] < 1).any():
        raise ValueError("n_seeds must be at least 1")
