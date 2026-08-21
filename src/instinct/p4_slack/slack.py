from __future__ import annotations

import numpy as np
import numpy.typing as npt

from instinct.core.mdp import TabularMDP

FloatArray = npt.NDArray[np.float64]


def lower_quantile(values: FloatArray, q: float) -> float:
    """inf{r: empirical CDF(r) >= q}, including ties."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("quantile values must be a nonempty vector")
    if not 0.0 < q <= 1.0:
        raise ValueError("q must lie in (0, 1]")
    return float(np.quantile(x, q, method="inverted_cdf"))


def recovery_slack(replan_values: FloatArray, *, v_fail: float, q: float) -> FloatArray:
    values = np.asarray(replan_values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("replan_values must have shape (prefix, disturbance)")
    return np.array([lower_quantile(row - v_fail, q) for row in values])


def slack_penalty(
    slack: FloatArray, *, margin: float, active: npt.NDArray[np.bool_] | None = None
) -> float:
    r = np.asarray(slack, dtype=np.float64)
    mask = np.ones(r.shape, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if mask.shape != r.shape:
        raise ValueError("active mask must align with prefix slack")
    return float(np.maximum(margin - r[mask], 0.0).sum())


def finite_horizon_replan_value(
    mdp: TabularMDP,
    *,
    horizon: int,
    v_fail: float,
) -> FloatArray:
    """Exact optimal replanning value with explicit terminal failure value."""
    if horizon < 0:
        raise ValueError("replanning horizon must be nonnegative")
    failure = (
        np.zeros(mdp.n_states, dtype=bool)
        if mdp.failure is None
        else np.asarray(mdp.failure, dtype=bool)
    )
    value = np.zeros(mdp.n_states, dtype=np.float64)
    value[failure] = v_fail
    for _ in range(horizon):
        continuation = np.einsum("sat,t->sa", mdp.P, value)
        updated = np.max(mdp.R + mdp.gamma * continuation, axis=1)
        value = np.where(mdp.terminal, value, updated)
    return value


def disturbed_replan_values(
    value: FloatArray,
    disturbance_kernel: npt.NDArray[np.float64],
) -> FloatArray:
    """Apply common prefix-by-disturbance state distributions to exact values."""
    v = np.asarray(value, dtype=np.float64)
    kernel = np.asarray(disturbance_kernel, dtype=np.float64)
    if v.ndim != 1 or kernel.ndim != 3 or kernel.shape[2] != v.size:
        raise ValueError("kernel must have shape (prefix, disturbance, state)")
    if np.any(kernel < 0) or not np.allclose(kernel.sum(axis=2), 1.0, atol=1e-10):
        raise ValueError("each disturbance row must be a probability distribution")
    return np.einsum("pds,s->pd", kernel, v)


def pareto_dominated(
    candidate: tuple[float, float],
    baselines: list[tuple[float, float]],
    *,
    tolerance: float = 1e-12,
) -> bool:
    """Whether a progress/success point is weakly dominated by a baseline."""
    progress, success = candidate
    return any(
        base_progress >= progress - tolerance
        and base_success >= success - tolerance
        and (base_progress > progress + tolerance or base_success > success + tolerance)
        for base_progress, base_success in baselines
    )


def reward_rescaling_consistent(
    values: FloatArray,
    *,
    v_fail: float,
    margin: float,
    q: float,
    scale: float,
    offset: float,
) -> bool:
    """Check that value origin/units and the slack margin transform together."""
    if scale <= 0:
        raise ValueError("reward scale must be positive")
    original = recovery_slack(values, v_fail=v_fail, q=q)
    transformed = recovery_slack(
        scale * values + offset,
        v_fail=scale * v_fail + offset,
        q=q,
    )
    left = slack_penalty(original, margin=margin)
    right = slack_penalty(transformed, margin=scale * margin)
    return bool(np.isclose(right, scale * left, atol=1e-10))
