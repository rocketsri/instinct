"""Known-answer conditional surfaces for the P1.0 transfer gate."""

from __future__ import annotations

import pandas as pd


def conditional_surface(*, collapse: bool = False, flat: bool = False) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    budgets = (1, 2, 4, 8, 16, 24)
    context = 0
    for nu_e in (0.5, 1.0, 2.0):
        for nu_h in (0.25, 0.5, 1.0):
            for state in (0.0, 0.5, 1.0):
                for reflex in (0.0, 1.0):
                    for environment in (0.0, 1.0):
                        hardware_quality = 0.0 if collapse else nu_h**2
                        # Hold out whole contexts. The rule mixes speed/latency,
                        # state, reflex, and environment so every axis appears in
                        # both partitions while exact context IDs remain disjoint.
                        split = (
                            "test"
                            if (context + int(2 * state) + int(reflex) + int(environment)) % 5 == 0
                            else "train"
                        )
                        context_id = f"ctx-{context:04d}"
                        for budget in budgets:
                            if flat:
                                sigma = 0.25
                            else:
                                progress = (
                                    0.3
                                    if collapse
                                    else 0.26 + 0.06 * state + 0.07 * reflex + 0.04 * environment
                                )
                                curvature = 0.004 + 0.003 * nu_e * nu_h
                                if not collapse:
                                    curvature += (
                                        0.003 * nu_h
                                        + 0.0015 * state
                                        + 0.001 * environment
                                        + 0.001 * hardware_quality
                                    )
                                sigma = progress * budget - curvature * budget**2
                            rows.append(
                                {
                                    "context_id": context_id,
                                    "split": split,
                                    "budget": budget,
                                    "sigma": sigma,
                                    "nu_e": nu_e,
                                    "nu_h": nu_h,
                                    "state_descriptor": state,
                                    "reflex_descriptor": reflex,
                                    "environment_descriptor": environment,
                                    "hardware_quality": hardware_quality,
                                }
                            )
                        context += 1
    return pd.DataFrame(rows)
