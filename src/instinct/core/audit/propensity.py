"""Off-policy value estimators for audited triggers: IPS, SNIPS, doubly robust.

**This is evaluation infrastructure, not a contribution.** Horvitz-Thompson
weighting, its self-normalized variant and the doubly robust estimator are
textbook (Horvitz & Thompson 1952; Hesterberg 1995; Dudík, Langford & Li 2011).
They are here because the audited-trigger experiments log decisions under one
trigger and need to score another, and because the standard implementations
quietly do two things this repo does not allow.

The first is dividing by tiny propensities. IPS is a ratio, and rule 2 exists
because ratios with small denominators produce numbers that look like effects.
The honest version of that rule for an off-policy estimator is not "don't use
one" — it is: refuse to run when the denominators are too small to support the
estimate, report the effective sample size *always*, and make clipping an
explicit choice with a visible bias rather than a silent floor. So
:func:`ips` raises below ``min_propensity`` unless the caller has asked for
clipping, and every estimate carries its ESS and its largest weight. An IPS
estimate over 10,000 logged episodes with an ESS of 4 is not an estimate, and
the only way anyone finds that out is if the number is printed next to it.

The second is reporting a point estimate alone. Every function here returns an
interval. The intervals come from a bootstrap over logged records rather than
from a normal approximation, because importance weights are heavy-tailed almost
by construction and the sample mean of a heavy-tailed variable converges to
normality slowly enough that the approximation is worse than useless at the
sample sizes these audits produce.

A note on pairing. :mod:`instinct.core.stats` pairs by default because arms
share seeds. That does not apply here: logged records are one draw each from the
behaviour policy, and there is no counterfactual twin to pair with — that is the
entire reason weighting is needed. Where a twin *does* exist, because the audit
forked the simulator (:mod:`instinct.core.audit.counterfactual`), use the paired
estimators instead. They are strictly better and it is not close.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "OffPolicyEstimate",
    "PropensityError",
    "doubly_robust",
    "effective_sample_size",
    "importance_weights",
    "ips",
    "self_normalized_ips",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: Below this behaviour probability, a single record can dominate the estimate.
#: Not a universal constant — it is the point at which one record in a thousand
#: carries weight 1000 — but it is a defensible default and it is visible.
DEFAULT_MIN_PROPENSITY = 1e-3


class PropensityError(ValueError):
    """The logged propensities cannot support the requested estimate."""


@dataclass(frozen=True, slots=True)
class OffPolicyEstimate:
    """A point estimate, an interval, and the diagnostics that qualify them.

    ``ess`` is Kish's effective sample size: the number of equally-weighted
    records that would carry the same information. Read it before the estimate.
    A large ``n`` with a small ``ess`` means the answer came from a handful of
    records and the interval, however it was computed, is optimistic.
    """

    value: float
    lo: float
    hi: float
    n: int
    method: str
    ess: float
    max_weight: float
    clipped_fraction: float = 0.0

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0

    def __str__(self) -> str:
        return (
            f"{self.value:+.5f} [{self.lo:+.5f}, {self.hi:+.5f}] "
            f"({self.method}, n={self.n}, ess={self.ess:.1f}, "
            f"max_w={self.max_weight:.3g}, clipped={self.clipped_fraction:.1%})"
        )


def effective_sample_size(weights: FloatArray) -> float:
    """Kish's ESS, ``(sum w)^2 / sum w^2``.

    Zero when all weights are zero. Equal to ``n`` when they are all equal, and
    it falls fast: one weight ten times the others already costs most of the
    sample.
    """
    w = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(w**2))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(w) ** 2 / denom)


def importance_weights(
    behavior_probs: FloatArray,
    target_probs: FloatArray,
    *,
    clip: float | None = None,
    min_propensity: float = DEFAULT_MIN_PROPENSITY,
) -> tuple[FloatArray, float]:
    """``pi_target / pi_behavior``, with the small-denominator check up front.

    Returns ``(weights, clipped_fraction)``.

    Raises when any behaviour probability is below ``min_propensity`` and no
    ``clip`` was requested. This is the rule-2 tripwire: a propensity of 1e-9 in
    the log produces a weight of a billion, that record becomes the estimate, and
    nothing downstream reveals it. Asking for a ``clip`` is how a caller says "I
    know, and I accept the bias".
    """
    mu = np.asarray(behavior_probs, dtype=np.float64)
    pi = np.asarray(target_probs, dtype=np.float64)
    if mu.shape != pi.shape:
        raise PropensityError(f"probability shapes differ: {mu.shape} vs {pi.shape}")
    if mu.ndim != 1:
        raise PropensityError(f"expected 1-D per-record probabilities, got {mu.ndim}-D")
    if np.any(mu <= 0.0):
        raise PropensityError(
            "behaviour probability of zero for a logged action: the action was taken, "
            "so its logged propensity cannot be zero. The log and the policy disagree."
        )
    if np.any(pi < 0.0):
        raise PropensityError("target probabilities must be non-negative")

    worst = float(mu.min())
    if worst < min_propensity and clip is None:
        raise PropensityError(
            f"smallest logged propensity is {worst:.3g}, below min_propensity="
            f"{min_propensity:.3g}. One record would carry weight {1.0 / worst:.3g}. "
            "Pass clip=<max weight> to accept the bias explicitly, lower "
            "min_propensity if you have a reason, or collect a log with support."
        )

    w = pi / mu
    if clip is None:
        return w, 0.0
    if clip <= 0.0:
        raise PropensityError(f"clip must be positive, got {clip}")
    clipped = w > clip
    return np.minimum(w, clip), float(np.mean(clipped))


def _bootstrap_interval(
    values: FloatArray,
    statistic: str,
    *,
    weights: FloatArray | None = None,
    n_boot: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    """Percentile bootstrap over records.

    Percentile, not BCa, unlike :mod:`instinct.core.stats`. The acceleration
    term is a jackknife third moment, and with heavy-tailed importance weights
    the jackknife is dominated by whichever single record has the largest weight
    — the correction becomes noise. Percentile is cruder and does not pretend.
    """
    n = values.size
    if n < 2:
        v = float(values.mean()) if n else float("nan")
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    if statistic == "mean":
        boots = values[idx].mean(axis=1)
    else:  # self-normalized ratio: resample numerator and denominator together
        if weights is None:
            raise PropensityError("the ratio bootstrap needs the weights it normalizes by")
        num = values[idx].sum(axis=1)
        den = weights[idx].sum(axis=1)
        boots = np.divide(num, den, out=np.full(n_boot, np.nan), where=den > 0)
    lo, hi = np.nanquantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def ips(
    rewards: FloatArray,
    behavior_probs: FloatArray,
    target_probs: FloatArray,
    *,
    clip: float | None = None,
    min_propensity: float = DEFAULT_MIN_PROPENSITY,
    alpha: float = 0.05,
    n_boot: int = 5_000,
    seed: int = 0,
) -> OffPolicyEstimate:
    """Inverse propensity scoring: ``mean(w * r)``.

    Unbiased when the log has support for the target policy, and only then.
    Unbiasedness is the reason it is here and the variance is the reason it is
    never the only thing reported — see the ESS on the returned estimate.
    """
    r = np.asarray(rewards, dtype=np.float64)
    w, clipped = importance_weights(
        behavior_probs, target_probs, clip=clip, min_propensity=min_propensity
    )
    if r.shape != w.shape:
        raise PropensityError(f"rewards shape {r.shape} does not match weights {w.shape}")
    scores = w * r
    lo, hi = _bootstrap_interval(scores, "mean", n_boot=n_boot, alpha=alpha, seed=seed)
    return OffPolicyEstimate(
        value=float(scores.mean()) if scores.size else float("nan"),
        lo=lo,
        hi=hi,
        n=int(r.size),
        method="ips",
        ess=effective_sample_size(w),
        max_weight=float(w.max()) if w.size else 0.0,
        clipped_fraction=clipped,
    )


def _ips_naive(
    rewards: FloatArray, behavior_probs: FloatArray, target_probs: FloatArray
) -> float:
    """Reference: the definition, one record at a time. Never used in anger."""
    r_list = [float(x) for x in rewards]
    total = 0.0
    behavior = [float(x) for x in behavior_probs]
    target = [float(x) for x in target_probs]
    for r, mu, pi in zip(r_list, behavior, target):
        total += (pi / mu) * r
    return total / len(r_list) if r_list else float("nan")


def self_normalized_ips(
    rewards: FloatArray,
    behavior_probs: FloatArray,
    target_probs: FloatArray,
    *,
    clip: float | None = None,
    min_propensity: float = DEFAULT_MIN_PROPENSITY,
    alpha: float = 0.05,
    n_boot: int = 5_000,
    seed: int = 0,
) -> OffPolicyEstimate:
    """SNIPS: ``sum(w r) / sum(w)``.

    Biased, consistent, and usually much lower variance than :func:`ips`,
    because dividing by the realized weight total cancels the run-to-run
    variation in how much total weight the log happened to carry. It is also
    range-preserving: the estimate cannot exceed the largest observed reward,
    which plain IPS routinely does when weights are heavy.

    The denominator here is ``sum(w)``, which concentrates at ``n`` and is not a
    small-denominator hazard — but it *can* be zero if the target policy assigns
    zero probability to every logged action, and that case raises rather than
    returning a NaN somebody has to trace.
    """
    r = np.asarray(rewards, dtype=np.float64)
    w, clipped = importance_weights(
        behavior_probs, target_probs, clip=clip, min_propensity=min_propensity
    )
    if r.shape != w.shape:
        raise PropensityError(f"rewards shape {r.shape} does not match weights {w.shape}")
    total_w = float(w.sum())
    if total_w <= 0.0:
        raise PropensityError(
            "total importance weight is zero: the target policy gives zero probability "
            "to every logged action. There is nothing to estimate from this log."
        )
    scores = w * r
    lo, hi = _bootstrap_interval(
        scores, "ratio", weights=w, n_boot=n_boot, alpha=alpha, seed=seed
    )
    return OffPolicyEstimate(
        value=float(scores.sum() / total_w),
        lo=lo,
        hi=hi,
        n=int(r.size),
        method="snips",
        ess=effective_sample_size(w),
        max_weight=float(w.max()) if w.size else 0.0,
        clipped_fraction=clipped,
    )


def doubly_robust(
    rewards: FloatArray,
    actions: IntArray,
    behavior_probs: FloatArray,
    target_action_probs: FloatArray,
    q_hat: FloatArray,
    *,
    clip: float | None = None,
    min_propensity: float = DEFAULT_MIN_PROPENSITY,
    alpha: float = 0.05,
    n_boot: int = 5_000,
    seed: int = 0,
) -> OffPolicyEstimate:
    """Doubly robust: model-based baseline plus weighted residual correction.

    ``target_action_probs`` and ``q_hat`` are ``(n_records, n_actions)``:
    the target policy's action distribution per record, and a reward model's
    prediction for every action. ``actions`` and ``behavior_probs`` describe what
    the behaviour policy actually did.

    Per record::

        v_hat_i = sum_a pi(a | x_i) q_hat(x_i, a)
        dr_i    = v_hat_i + w_i * (r_i - q_hat(x_i, a_i))

    "Doubly robust" means consistent if *either* the propensities or the reward
    model are right. The practical value here is narrower and more useful: when
    ``q_hat`` is decent, the residual ``r - q_hat`` is small, so the heavy
    importance weights multiply a small number and the variance collapses. That
    is what makes an audit with a few hundred logged records readable at all.

    ``q_hat`` must not have been fitted on these same records, or the residuals
    are optimistically small and the interval is too narrow. Fit it on a
    disjoint split; rule 5's finite-use holdout machinery is the right home for
    that split.
    """
    r = np.asarray(rewards, dtype=np.float64)
    a = np.asarray(actions, dtype=np.int64)
    pi = np.asarray(target_action_probs, dtype=np.float64)
    q = np.asarray(q_hat, dtype=np.float64)
    if pi.shape != q.shape:
        raise PropensityError(f"target probs {pi.shape} and q_hat {q.shape} must match")
    if pi.ndim != 2:
        raise PropensityError(f"expected (n_records, n_actions) arrays, got {pi.ndim}-D")
    if r.shape != a.shape or r.shape[0] != pi.shape[0]:
        raise PropensityError("rewards, actions and per-action arrays disagree on n_records")
    if a.size and (a.min() < 0 or a.max() >= pi.shape[1]):
        raise PropensityError("logged action index outside the action set")

    rows = np.arange(r.size)
    pi_taken = pi[rows, a]
    w, clipped = importance_weights(
        behavior_probs, pi_taken, clip=clip, min_propensity=min_propensity
    )
    v_hat = np.sum(pi * q, axis=1)
    scores = v_hat + w * (r - q[rows, a])
    lo, hi = _bootstrap_interval(scores, "mean", n_boot=n_boot, alpha=alpha, seed=seed)
    return OffPolicyEstimate(
        value=float(scores.mean()) if scores.size else float("nan"),
        lo=lo,
        hi=hi,
        n=int(r.size),
        method="doubly_robust",
        ess=effective_sample_size(w),
        max_weight=float(w.max()) if w.size else 0.0,
        clipped_fraction=clipped,
    )


def _doubly_robust_naive(
    rewards: FloatArray,
    actions: IntArray,
    behavior_probs: FloatArray,
    target_action_probs: FloatArray,
    q_hat: FloatArray,
) -> float:
    """Reference: the estimator written as its own definition. Tests only."""
    total = 0.0
    for i in range(len(rewards)):
        a = int(actions[i])
        v_hat = 0.0
        for j in range(target_action_probs.shape[1]):
            v_hat += float(target_action_probs[i, j]) * float(q_hat[i, j])
        w = float(target_action_probs[i, a]) / float(behavior_probs[i])
        total += v_hat + w * (float(rewards[i]) - float(q_hat[i, a]))
    return total / len(rewards) if len(rewards) else float("nan")
