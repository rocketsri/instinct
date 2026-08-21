"""Auditable P2.2 pilot on a natural-image corruption stream.

This is deliberately a bounded pilot, not a CIFAR-C substitute.  It uses the
real Grace Hopper photograph shipped as Matplotlib sample data, partitions the
image into disjoint spatial bands, and adapts only a 3x4 affine RGB head.  The
small model makes every counterfactual inspectable while the explicit realism
check prevents this pilot from being promoted to the frozen specification's
pretrained ResNet/ViT claim.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.metadata import distribution
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy import stats as scipy_stats

__all__ = ["P2PilotConfig", "P2PilotResult", "run_natural_image_pilot"]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class P2PilotConfig:
    n_steps: int = 28
    patch_size: int = 10
    horizons: tuple[int, ...] = (1, 2, 4)
    horizon_weights: tuple[float, ...] = (0.5, 0.3, 0.2)
    min_age: int = 4
    write_fraction: float = 0.35
    learning_rate: float = 0.18
    retention_lambda: float = 1.0
    write_cost: float = 1e-4
    random_replicates: int = 8
    correlation_threshold: float = 0.25
    utility_margin: float = 1e-5

    def __post_init__(self) -> None:
        if self.n_steps < 12 or self.patch_size < 2:
            raise ValueError("P2.2 pilot needs at least 12 steps and 2x2 patches")
        if not self.horizons or len(self.horizons) != len(self.horizon_weights):
            raise ValueError("horizons and horizon_weights must be nonempty and aligned")
        if any(h <= 0 for h in self.horizons):
            raise ValueError("horizons must be positive")
        if any(w < 0 for w in self.horizon_weights) or not np.isclose(
            sum(self.horizon_weights), 1.0
        ):
            raise ValueError("horizon weights must be nonnegative and sum to one")
        if not 0.0 < self.write_fraction < 1.0:
            raise ValueError("write_fraction must lie in (0, 1)")
        if self.learning_rate <= 0 or self.retention_lambda < 0 or self.write_cost < 0:
            raise ValueError("learning rate must be positive and penalties nonnegative")
        if self.random_replicates < 2:
            raise ValueError("at least two random replicates are required")


@dataclass(frozen=True, slots=True)
class _Unit:
    unit_id: str
    x: FloatArray
    y: FloatArray


@dataclass(frozen=True, slots=True)
class _CandidateScore:
    t: int
    utility: float
    benefit: float
    damage: float
    cost: float
    validation_utility: float
    aged_downstream_damage: float
    update_norm: float
    oracle_probe_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Arm:
    method: str
    n_writes: int
    selected_timesteps: tuple[int, ...]
    benefit: float
    damage: float
    cost: float
    utility: float
    downstream_mse: float
    downstream_psnr: float
    deployable_compute_units: int
    offline_oracle_units: int


@dataclass(frozen=True, slots=True)
class P2PilotResult:
    benchmark_id: str
    source_sha256: str
    candidates: tuple[_CandidateScore, ...]
    arms: tuple[_Arm, ...]
    eligible_timesteps: tuple[int, ...]
    m: int
    streams_disjoint: bool
    full_horizons_only: bool
    matched_count_passed: bool
    matched_compute_passed: bool
    aged_downstream_spearman: float
    identifiability_passed: bool
    realistic_scale_passed: bool
    oracle_advantage: float
    useful_sparsity: float
    qualifies_p7: bool

    def metric_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for candidate in self.candidates:
            row = asdict(candidate)
            row["record_type"] = "p2_2_candidate"
            row["oracle_probe_ids"] = ",".join(candidate.oracle_probe_ids)
            rows.append(row)
        for arm in self.arms:
            row = asdict(arm)
            row["record_type"] = "p2_2_frontier"
            row["selected_timesteps"] = ",".join(map(str, arm.selected_timesteps))
            row.update(
                {
                    "matched_count_passed": self.matched_count_passed,
                    "matched_compute_passed": self.matched_compute_passed,
                    "aged_downstream_spearman": self.aged_downstream_spearman,
                    "identifiability_passed": self.identifiability_passed,
                    "realistic_scale_passed": self.realistic_scale_passed,
                    "oracle_advantage": self.oracle_advantage,
                    "useful_sparsity": self.useful_sparsity,
                    "qualifies_p7": self.qualifies_p7,
                    "benchmark_id": self.benchmark_id,
                }
            )
            rows.append(row)
        return rows


@lru_cache(maxsize=1)
def _load_image() -> tuple[FloatArray, str]:
    path = Path(
        str(
            distribution("matplotlib").locate_file(
                "matplotlib/mpl-data/sample_data/grace_hopper.jpg"
            )
        )
    )
    raw = path.read_bytes()
    with Image.open(path) as source:
        image = np.asarray(source.convert("RGB"), dtype=np.float64)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("natural-image benchmark source must be RGB")
    image = image[:, :, :3]
    if image.max() > 1.0:
        image /= 255.0
    return image, hashlib.sha256(raw).hexdigest()


def _corrupt(clean: FloatArray, t: int, rng: np.random.Generator) -> FloatArray:
    # Blockwise camera/color shifts create a changing-corruption stream.  The
    # target remains the untouched natural-image patch.
    modes = (
        (np.array([0.72, 0.92, 1.12]), 0.06),
        (np.array([1.15, 0.78, 0.86]), -0.03),
        (np.array([0.86, 1.12, 0.75]), 0.02),
        (np.array([1.08, 0.90, 1.16]), -0.05),
    )
    scale, bias = modes[(t // 4) % len(modes)]
    noise = rng.normal(0.0, 0.018, size=clean.shape)
    return np.clip(clean * scale + bias + noise, 0.0, 1.0)


def _streams(cfg: P2PilotConfig, seed: int) -> tuple[dict[str, tuple[_Unit, ...]], str]:
    image, digest = _load_image()
    h, w, _ = image.shape
    band_width = w // 4
    streams: dict[str, tuple[_Unit, ...]] = {}
    names = ("candidate", "oracle_probe", "validation", "downstream")
    for band, name in enumerate(names):
        units: list[_Unit] = []
        left = band * band_width
        right = (band + 1) * band_width if band < 3 else w
        for t in range(cfg.n_steps):
            rng = np.random.default_rng(np.random.SeedSequence([seed, band, t]))
            top = int(rng.integers(0, h - cfg.patch_size + 1))
            x0 = int(rng.integers(left, right - cfg.patch_size + 1))
            clean = image[top : top + cfg.patch_size, x0 : x0 + cfg.patch_size].reshape(-1, 3)
            corrupted = _corrupt(clean, t, rng)
            features = np.concatenate(
                [corrupted, np.ones((corrupted.shape[0], 1), dtype=np.float64)], axis=1
            )
            units.append(_Unit(f"{name}:{t}:{top}:{x0}", features, clean.copy()))
        streams[name] = tuple(units)
    return streams, digest


def _initial_weights() -> FloatArray:
    weights = np.zeros((3, 4), dtype=np.float64)
    weights[:, :3] = np.eye(3)
    return weights


def _loss(weights: FloatArray, unit: _Unit) -> float:
    residual = unit.x @ weights.T - unit.y
    return float(np.mean(residual * residual))


def _delta(weights: FloatArray, unit: _Unit, learning_rate: float) -> FloatArray:
    residual = unit.x @ weights.T - unit.y
    gradient = (2.0 / unit.x.shape[0]) * residual.T @ unit.x
    delta = -learning_rate * gradient
    norm = float(np.linalg.vector_norm(delta))
    if norm > 0.25:
        delta *= 0.25 / norm
    return delta


def _improvement(weights: FloatArray, delta: FloatArray, unit: _Unit) -> float:
    return _loss(weights, unit) - _loss(weights + delta, unit)


def _score_candidates(
    streams: dict[str, tuple[_Unit, ...]], cfg: P2PilotConfig
) -> tuple[tuple[_CandidateScore, ...], tuple[int, ...], int]:
    weights = _initial_weights()
    eligible = tuple(range(cfg.min_age, cfg.n_steps - max(cfg.horizons)))
    scores: list[_CandidateScore] = []
    probe_uses: dict[str, int] = {}
    for t, candidate in enumerate(streams["candidate"]):
        delta = _delta(weights, candidate, cfg.learning_rate)
        if t in eligible:
            future = [streams["oracle_probe"][t + h] for h in cfg.horizons]
            benefit = sum(
                weight * _improvement(weights, delta, unit)
                for weight, unit in zip(cfg.horizon_weights, future)
            )
            aged = streams["oracle_probe"][t - cfg.min_age]
            damage = -min(0.0, _improvement(weights, delta, aged))
            validation_utility = (
                _improvement(weights, delta, streams["validation"][t]) - cfg.write_cost
            )
            downstream_damage = -min(
                0.0,
                _improvement(weights, delta, streams["downstream"][t - cfg.min_age]),
            )
            probe_ids = tuple(unit.unit_id for unit in (*future, aged))
            for probe_id in probe_ids:
                probe_uses[probe_id] = probe_uses.get(probe_id, 0) + 1
            scores.append(
                _CandidateScore(
                    t=t,
                    utility=benefit - cfg.retention_lambda * damage - cfg.write_cost,
                    benefit=benefit,
                    damage=damage,
                    cost=cfg.write_cost,
                    validation_utility=validation_utility,
                    aged_downstream_damage=downstream_damage,
                    update_norm=float(np.linalg.vector_norm(delta)),
                    oracle_probe_ids=probe_ids,
                )
            )
        # Frozen-reference oracle: score on, then advance, the write-all path.
        weights += delta
    return tuple(scores), eligible, max(probe_uses.values(), default=0)


def _select_top(scores: FloatArray, eligible: tuple[int, ...], m: int) -> tuple[int, ...]:
    if scores.shape != (len(eligible),):
        raise ValueError("selection scores must already be restricted to eligible candidates")
    order = np.argsort(-scores, kind="stable")[:m]
    return tuple(sorted(eligible[int(index)] for index in order))


def _replay(
    method: str,
    selected: tuple[int, ...],
    streams: dict[str, tuple[_Unit, ...]],
    cfg: P2PilotConfig,
    *,
    deployable_compute_units: int,
    offline_oracle_units: int = 0,
) -> _Arm:
    selected_set = set(selected)
    weights = _initial_weights()
    benefit = damage = cost = 0.0
    for t, candidate in enumerate(streams["candidate"]):
        delta = _delta(weights, candidate, cfg.learning_rate)
        if t not in selected_set:
            continue
        future = [streams["downstream"][t + h] for h in cfg.horizons if t + h < cfg.n_steps]
        if len(future) == len(cfg.horizons) and t >= cfg.min_age:
            benefit += sum(
                weight * _improvement(weights, delta, unit)
                for weight, unit in zip(cfg.horizon_weights, future)
            )
            aged = streams["downstream"][t - cfg.min_age]
            damage += -min(0.0, _improvement(weights, delta, aged))
        cost += cfg.write_cost
        weights += delta
    downstream_mse = float(np.mean([_loss(weights, unit) for unit in streams["downstream"]]))
    return _Arm(
        method=method,
        n_writes=len(selected),
        selected_timesteps=tuple(sorted(selected)),
        benefit=benefit,
        damage=damage,
        cost=cost,
        utility=benefit - cfg.retention_lambda * damage - cost,
        downstream_mse=downstream_mse,
        downstream_psnr=float(-10.0 * np.log10(max(downstream_mse, 1e-12))),
        deployable_compute_units=deployable_compute_units,
        offline_oracle_units=offline_oracle_units,
    )


def run_natural_image_pilot(seed: int, cfg: P2PilotConfig) -> P2PilotResult:
    """Run the P2.2 pilot without using exact truth in any deployable arm."""

    streams, digest = _streams(cfg, seed)
    ids = {name: {unit.unit_id for unit in units} for name, units in streams.items()}
    streams_disjoint = all(
        ids[left].isdisjoint(ids[right])
        for i, left in enumerate(ids)
        for right in tuple(ids)[i + 1 :]
    )
    candidates, eligible, max_probe_uses = _score_candidates(streams, cfg)
    m = max(1, round(cfg.write_fraction * len(eligible)))
    oracle_scores = np.array([score.utility for score in candidates])
    validation_scores = np.array([score.validation_utility for score in candidates])
    oracle_selected = _select_top(oracle_scores, eligible, m)
    validation_selected = _select_top(validation_scores, eligible, m)

    # Candidate gradients and two validation losses are charged to both
    # deployable matched-count arms. Random deliberately discards the validation
    # score, giving it the same budget rather than a cheaper code path.
    matched_compute = cfg.n_steps + 2 * len(eligible)
    arms: list[_Arm] = [
        _replay(
            "always",
            tuple(range(cfg.n_steps)),
            streams,
            cfg,
            deployable_compute_units=cfg.n_steps,
        ),
        _replay("never", (), streams, cfg, deployable_compute_units=0),
        _replay(
            "validation_only",
            validation_selected,
            streams,
            cfg,
            deployable_compute_units=matched_compute,
        ),
        _replay(
            "oracle_ceiling",
            oracle_selected,
            streams,
            cfg,
            deployable_compute_units=matched_compute,
            offline_oracle_units=len(eligible) * (2 * (len(cfg.horizons) + 1)),
        ),
    ]
    for replicate in range(cfg.random_replicates):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 991, replicate]))
        selected = tuple(sorted(int(t) for t in rng.choice(eligible, size=m, replace=False)))
        arms.append(
            _replay(
                f"random-{replicate}",
                selected,
                streams,
                cfg,
                deployable_compute_units=matched_compute,
            )
        )

    matched_arms = [
        arm
        for arm in arms
        if arm.method == "validation_only" or arm.method.startswith("random-")
    ]
    matched_count_passed = all(arm.n_writes == m for arm in matched_arms) and len(
        oracle_selected
    ) == m
    matched_compute_passed = len({arm.deployable_compute_units for arm in matched_arms}) == 1
    aged = np.array([score.damage for score in candidates])
    downstream = np.array([score.aged_downstream_damage for score in candidates])
    correlation = scipy_stats.spearmanr(aged, downstream).statistic
    correlation = 0.0 if not np.isfinite(correlation) else float(correlation)
    identifiability_passed = correlation >= cfg.correlation_threshold and max_probe_uses <= 4

    oracle_arm = next(arm for arm in arms if arm.method == "oracle_ceiling")
    simple_arms = [
        arm
        for arm in arms
        if arm.method == "validation_only" or arm.method.startswith("random-")
    ]
    best_simple = max(arm.utility for arm in simple_arms)
    oracle_advantage = oracle_arm.utility - best_simple
    useful_sparsity = (
        1.0 - m / len(eligible)
        if oracle_advantage > cfg.utility_margin and matched_count_passed
        else 0.0
    )
    # This benchmark is a real natural image but not the frozen spec's
    # pretrained multi-image ResNet/ViT benchmark.  Fail closed by construction.
    realistic_scale_passed = False
    qualifies_p7 = bool(
        realistic_scale_passed
        and identifiability_passed
        and matched_count_passed
        and matched_compute_passed
        and useful_sparsity > 0
    )
    return P2PilotResult(
        benchmark_id="matplotlib-grace-hopper-corruption-v1",
        source_sha256=digest,
        candidates=candidates,
        arms=tuple(arms),
        eligible_timesteps=eligible,
        m=m,
        streams_disjoint=streams_disjoint,
        full_horizons_only=all(
            candidate.t + max(cfg.horizons) < cfg.n_steps for candidate in candidates
        ),
        matched_count_passed=matched_count_passed,
        matched_compute_passed=matched_compute_passed,
        aged_downstream_spearman=correlation,
        identifiability_passed=identifiability_passed,
        realistic_scale_passed=realistic_scale_passed,
        oracle_advantage=oracle_advantage,
        useful_sparsity=useful_sparsity,
        qualifies_p7=qualifies_p7,
    )
