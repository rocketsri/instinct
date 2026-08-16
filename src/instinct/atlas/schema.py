"""The atlas measurement record: the contract between measuring and fitting.

One row is one fully-specified cell of the sweep — a environment, reflex,
speed, hardware latency and planning budget — carrying both the raw arm values
and the decomposition derived from them.

This module exists so the two halves of P1 can be built independently. The sweep
produces rows; the curve fitting, transfer tests and kill harness consume them
and never touch a rollout. Pinning the schema here means a change to what gets
measured shows up as a schema change rather than as a silently misread column.

The invariant that matters is the telescoping identity

    sigma == G_plan + R_intermediate - L_arrival - L_irreversible - C_hw + eps_cross

which holds *by construction* because the terms are a telescoping chain over
adjacent counterfactual arms, not five independent measurements that happen to
add up. ``eps_cross`` is therefore not a bucket: in the exact arm it collapses to
exactly one identified quantity, the cost the *base* arm pays for its own delay,
and it is exactly zero whenever the base budget lands immediately. In a sampled
cell it additionally carries estimation noise, and a cell where it grows
comparable to the budget effect is a cell whose decomposition cannot be trusted —
one of P1's stated kill conditions. :func:`validate_frame` is not optional.

One column deserves a warning, because its name is easy to misread.
``L_irreversible`` is *the whole cost of having waited, net of what the reflex
earned back*, which bundles plain discounting together with genuinely
unrecoverable damage. Waiting always costs something under a discount factor,
absorbing states or not. The unrecoverable part is measured separately by
``Decomposition.L_unrecoverable`` in ``atlas/decomposition.py`` and deliberately
sits outside this identity. Do not read ``L_irreversible`` as "damage that could
not be undone" — it is not, and an earlier version of the estimator conflated the
two and reported ~4.9 units of irreversible damage in an environment with no
absorbing states at all.
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
    "L_irreversible": -1,
    "C_hw": -1,
    "eps_cross": +1,
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
    L_irreversible: float
    C_hw: float
    eps_cross: float
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
    recorded elsewhere — ``eps_cross`` is the place that noise is meant to show
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
