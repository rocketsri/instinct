"""Classifying the oracle result into the four outcomes that select phase two.

Phase zero produces a number per candidate write. What it has to produce for the
*program* is a decision, and there are exactly four of them. Writing the rule
down here, before the oracle is run, is the point: a threshold chosen after
seeing the curve is not a decision rule, it is a description.

============================================  ==================================
predictable beforehand AND clustered          learn a gate (2A)
predictable beforehand, NOT clustered         project the update, do not gate it
not predictable beforehand, detectable after  transactional validation (2B)
neither                                       the probes are wrong, or harmful
                                              writes are not identifiable
============================================  ==================================

**Predictability** is held-out rank agreement between cheap features and ``U_t``.
Rank agreement, not fit quality: a gate only has to *order* candidates, and a
model can fit the magnitudes well while ordering them wrongly. Held out, because
in-sample agreement is a foregone conclusion with five features and a hundred
candidates.

The combination of features is deliberately crude — sign on the training half,
then average of z-scored ranks — because rule 1 forbids assuming a functional
form, and any fitted weighting is one. A more capable predictor would move the
threshold in the *permissive* direction, so a "not predictable" verdict from
this rule is the conservative one and a "predictable" verdict is not an artifact
of the model class. Per-feature scores are reported alongside so a reader can
see whether one feature is carrying everything.

**Clusteredness** asks whether the damage lives in a low-dimensional subspace.
If it does, you do not need to reject the write: you can keep the component that
helps and project out the component that hurts. The measure is the fraction of
the retention-damage Gram matrix's spectral energy in its top ``r``
eigendirections, where the Gram is built from damage-weighted update directions
so that a large harmless write does not count as damage energy.

Spectral concentration is a *necessary* condition for projection to work, not a
sufficient one — the harmful subspace could coincide with the useful one, in
which case projecting it out removes the benefit too. When the caller can supply
``recovered_utility``, that stronger check is used and reported; when it cannot,
the report says which of the two it measured. Do not read the weaker one as the
stronger one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from scipy import stats as scipy_stats

__all__ = [
    "ClusterReport",
    "DecisionReport",
    "DecisionThresholds",
    "Outcome",
    "PredictReport",
    "classify",
    "damage_clustering",
    "held_out_rank_score",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


class Outcome(StrEnum):
    LEARN_GATE = "learn_gate"
    PROJECT_UPDATE = "project_update"
    TRANSACTIONAL_VALIDATION = "transactional_validation"
    PROBES_OR_TARGET_WRONG = "probes_or_target_wrong"


@dataclass(frozen=True, slots=True)
class DecisionThresholds:
    """Where each branch is taken. Stated up front, swept afterwards.

    ``spearman`` at 0.35 and ``auc`` at 0.70 are not derived from anything —
    they are the level at which a cheap gate is worth building rather than a
    level at which the correlation is "significant". They should be reported
    with the verdict and the verdict re-checked at neighbouring values.
    """

    spearman: float = 0.35
    auc: float = 0.70
    cluster_energy: float = 0.80
    cluster_rank: int = 4
    recovered_utility: float = 0.60
    train_fraction: float = 0.5


@dataclass(frozen=True, slots=True)
class PredictReport:
    spearman: float
    auc: float
    per_feature_spearman: dict[str, float]
    residuals: FloatArray
    residual_rms: float
    n_train: int
    n_test: int
    features: tuple[str, ...]

    def predictable(self, th: DecisionThresholds) -> bool:
        return abs(self.spearman) >= th.spearman or self.auc >= th.auc


@dataclass(frozen=True, slots=True)
class ClusterReport:
    rank: int
    energy_top_r: float
    spectrum: FloatArray
    recovered_utility_fraction: float | None
    basis_source: str

    def clustered(self, th: DecisionThresholds) -> bool:
        if self.recovered_utility_fraction is not None:
            return self.recovered_utility_fraction >= th.recovered_utility
        return self.energy_top_r >= th.cluster_energy

    @property
    def evidence(self) -> str:
        if self.recovered_utility_fraction is not None:
            return "recovered_utility"
        return "spectral_energy"


@dataclass(frozen=True, slots=True)
class DecisionReport:
    outcome: Outcome
    pre: PredictReport
    post: PredictReport
    cluster: ClusterReport
    thresholds: DecisionThresholds
    caveats: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        return (
            f"{self.outcome.value}: pre rho={self.pre.spearman:+.3f} auc={self.pre.auc:.3f}, "
            f"post rho={self.post.spearman:+.3f} auc={self.post.auc:.3f}, "
            f"cluster({self.cluster.evidence}) r={self.cluster.rank} "
            f"energy={self.cluster.energy_top_r:.3f}"
        )


# ----------------------------------------------------------------------------


def _rank_z(x: FloatArray) -> FloatArray:
    """Ranks, centred and scaled. Ties averaged.

    Ranks rather than values so a single outlying update norm cannot dominate
    the combination — which is exactly what happens on these streams, where the
    conflicting chunks produce gradients an order of magnitude larger than the
    rest.
    """
    r = scipy_stats.rankdata(x).astype(np.float64)
    sd = r.std()
    return (r - r.mean()) / sd if sd > 1e-12 else np.zeros_like(r)


def _auc(score: FloatArray, positive: BoolArray) -> float:
    """Rank AUC for "this candidate is one of the harmful ones".

    Returns 0.5 — the uninformative value — when one class is empty, rather than
    NaN. A NaN would propagate into the verdict as a silent False.
    """
    pos, neg = int(positive.sum()), int((~positive).sum())
    if pos == 0 or neg == 0:
        return 0.5
    r = scipy_stats.rankdata(score)
    return float((r[positive].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _split(n: int, train_fraction: float) -> tuple[IntArray, IntArray]:
    """Split by stream position, not at random.

    Candidates from adjacent timesteps share their aged probes and much of their
    memory state, so a random split leaks. A prefix/suffix split is the honest
    version and also the one a deployed gate would face: fit on early stream,
    apply to later stream.
    """
    n_train = round(train_fraction * n)
    n_train = min(max(n_train, 1), n - 1)
    return np.arange(n_train, dtype=np.int64), np.arange(n_train, n, dtype=np.int64)


def held_out_rank_score(
    features: Mapping[str, FloatArray],
    utility: FloatArray,
    *,
    harmful: BoolArray | None = None,
    names: Sequence[str] | None = None,
    train_fraction: float = 0.5,
) -> PredictReport:
    """Can these features order ``U_t`` on data they were not tuned on?

    The only thing fitted is a sign per feature, taken on the training half. The
    test half then sees a fixed, unweighted average of signed rank-z scores.
    Residuals are reported in rank space (predicted rank minus actual rank,
    normalized), because that is the space the score lives in; residuals of a
    regression that was never run would be meaningless.
    """
    keys = tuple(names) if names is not None else tuple(features)
    if not keys:
        raise ValueError("no features supplied")
    u = np.asarray(utility, dtype=np.float64)
    n = u.shape[0]
    if n < 4:
        raise ValueError(f"need at least 4 candidates to hold out, got {n}")
    tr, te = _split(n, train_fraction)
    pos = np.zeros(n, dtype=bool) if harmful is None else np.asarray(harmful, dtype=bool)

    per_feature: dict[str, float] = {}
    signs: dict[str, float] = {}
    for k in keys:
        f = np.asarray(features[k], dtype=np.float64)
        if f.shape[0] != n:
            raise ValueError(f"feature {k!r} has {f.shape[0]} entries, expected {n}")
        rho_tr = scipy_stats.spearmanr(f[tr], u[tr]).statistic
        rho_tr = 0.0 if not np.isfinite(rho_tr) else float(rho_tr)
        signs[k] = 1.0 if rho_tr >= 0 else -1.0
        rho_te = scipy_stats.spearmanr(f[te], u[te]).statistic
        per_feature[k] = 0.0 if not np.isfinite(rho_te) else float(rho_te)

    combined_te = np.mean([signs[k] * _rank_z(np.asarray(features[k])[te]) for k in keys], axis=0)
    rho = scipy_stats.spearmanr(combined_te, u[te]).statistic
    rho = 0.0 if not np.isfinite(rho) else float(rho)
    # AUC is stated for "predicts a *harmful* write", so the score is negated:
    # low utility is the positive class.
    auc = _auc(-combined_te, pos[te])

    pred_rank = _rank_z(combined_te)
    true_rank = _rank_z(u[te])
    residuals = true_rank - pred_rank
    return PredictReport(
        spearman=rho,
        auc=auc,
        per_feature_spearman=per_feature,
        residuals=residuals,
        residual_rms=float(np.sqrt(np.mean(residuals**2))),
        n_train=int(tr.size),
        n_test=int(te.size),
        features=keys,
    )


def damage_clustering(
    deltas_flat: FloatArray,
    damage: FloatArray,
    *,
    rank: int = 4,
    recovered_utility: Callable[[FloatArray], float] | None = None,
) -> ClusterReport:
    """Does the retention damage live in a few directions?

    The Gram matrix is built from update directions weighted by ``sqrt(damage)``,
    clipped at zero: a write that *helps* retention contributes no damage energy,
    and a write with a huge norm but no damage does not masquerade as a
    dominant direction. Unit-normalizing the directions first is what makes this
    a statement about *where* damage points rather than about how big the
    updates were.

    ``recovered_utility`` — if the caller can afford it — takes the top-``r``
    basis (rows are basis vectors in parameter space) and returns the fraction
    of oracle utility recovered by writing the projected updates instead. That
    is the real question; spectral energy is the affordable proxy.
    """
    D = np.asarray(deltas_flat, dtype=np.float64)
    if D.ndim != 2:
        raise ValueError(f"deltas_flat must be (n_candidates, n_params), got {D.shape}")
    w = np.sqrt(np.clip(np.asarray(damage, dtype=np.float64), 0.0, None))
    norms = np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-30)
    W = (D / norms) * w[:, None]

    # Gram over candidates: n x n is tiny next to n_params x n_params, and its
    # nonzero spectrum is identical.
    G = W @ W.T
    eig = np.linalg.eigvalsh(G)[::-1]
    eig = np.clip(eig, 0.0, None)
    total = float(eig.sum())
    r = int(min(rank, eig.shape[0]))
    energy = float(eig[:r].sum() / total) if total > 1e-30 else 0.0

    frac: float | None = None
    if recovered_utility is not None:
        # Left singular vectors of W are the damage directions in parameter space.
        u_, s_, vt = np.linalg.svd(W, full_matrices=False)
        frac = float(recovered_utility(vt[:r]))
        del u_, s_

    return ClusterReport(
        rank=r,
        energy_top_r=energy,
        spectrum=eig,
        recovered_utility_fraction=frac,
        basis_source="damage_weighted_unit_updates",
    )


def classify(
    features: Mapping[str, FloatArray],
    utility: FloatArray,
    deltas_flat: FloatArray,
    damage: FloatArray,
    *,
    pre_features: Sequence[str],
    post_features: Sequence[str],
    harmful: BoolArray | None = None,
    thresholds: DecisionThresholds | None = None,
    recovered_utility: Callable[[FloatArray], float] | None = None,
) -> DecisionReport:
    """Run the four-way rule and say which phase two the evidence selects.

    ``post_features`` must be things a system could only know *after* committing
    — a cheap audit of the memory, not a re-reading of the oracle's own aged set.
    If a post feature is secretly derived from the target, this function will
    report a confident ``transactional_validation`` for a system that cannot
    actually validate anything, and nothing here can detect that. Keeping the
    audit pool disjoint from the retention pool is the caller's job; see
    :func:`instinct.ttt.chunker.build_ledger`.
    """
    th = thresholds or DecisionThresholds()
    pre = held_out_rank_score(
        features, utility, harmful=harmful, names=pre_features, train_fraction=th.train_fraction
    )
    post = held_out_rank_score(
        features, utility, harmful=harmful, names=post_features, train_fraction=th.train_fraction
    )
    cluster = damage_clustering(
        deltas_flat, damage, rank=th.cluster_rank, recovered_utility=recovered_utility
    )

    caveats: list[str] = []
    if cluster.recovered_utility_fraction is None:
        caveats.append(
            "clusteredness rests on spectral energy alone; low-dimensional damage does not "
            "guarantee the harmful subspace is separable from the useful one"
        )
    if pre.n_test < 20:
        caveats.append(f"held-out set is only {pre.n_test} candidates; the verdict is provisional")
    if harmful is not None and (int(np.sum(harmful)) == 0 or int(np.sum(~harmful)) == 0):
        caveats.append("no ground-truth contrast available; AUC is uninformative by construction")

    if pre.predictable(th):
        outcome = Outcome.LEARN_GATE if cluster.clustered(th) else Outcome.PROJECT_UPDATE
    elif post.predictable(th):
        outcome = Outcome.TRANSACTIONAL_VALIDATION
    else:
        outcome = Outcome.PROBES_OR_TARGET_WRONG

    return DecisionReport(
        outcome=outcome, pre=pre, post=post, cluster=cluster, thresholds=th, caveats=tuple(caveats)
    )
