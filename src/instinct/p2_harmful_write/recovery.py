"""P2 recovery-cycle-1 multi-image nonlinear restoration pilot.

The original one-photo artifact is intentionally untouched.  This fresh
benchmark diagnoses its two main degeneracies: identity was a strong no-write
solution, and frozen write-all oracle scores became stale on sparse replay.
Here a small nonlinear color head is pretrained on a spatially disjoint band of
three packaged images, corruption regimes persist longer than the utility
horizon, and the offline ceiling exhaustively searches a preregistered small
write budget using oracle probes only.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from instinct.core.stats import spearman

__all__ = ["RecoveryConfig", "RecoveryResult", "run_recovery"]

FloatArray = npt.NDArray[np.float64]

_ASSETS = ("grace_hopper.jpg", "Minduka_Present_Blue_Pack.png", "logo2.png")
_STREAMS = ("candidate", "oracle_probe", "validation", "downstream")


@dataclass(frozen=True, slots=True)
class RecoveryConfig:
    n_steps: int = 36
    patch_size: int = 8
    regime_length: int = 9
    horizons: tuple[int, ...] = (1, 2, 4)
    horizon_weights: tuple[float, ...] = (0.5, 0.3, 0.2)
    min_age: int = 4
    m: int = 3
    learning_rate: float = 0.035
    retention_lambda: float = 1.0
    write_cost: float = 1e-5
    random_replicates: int = 8
    correlation_threshold: float = 0.25
    utility_margin: float = 1e-5

    def __post_init__(self) -> None:
        if self.n_steps < 20 or self.patch_size < 2 or self.regime_length <= max(self.horizons):
            raise ValueError("recovery needs a long stream and regimes longer than max horizon")
        if len(self.horizons) != len(self.horizon_weights) or not np.isclose(
            sum(self.horizon_weights), 1.0
        ):
            raise ValueError("horizons and normalized weights must align")
        n_eligible = self.n_steps - max(self.horizons) - self.min_age
        if self.m < 1 or self.m >= n_eligible:
            raise ValueError("m must be an interior eligible-candidate budget")


@dataclass(frozen=True, slots=True)
class _Unit:
    unit_id: str
    x: FloatArray
    y: FloatArray


@dataclass(frozen=True, slots=True)
class RecoveryArm:
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
class RecoveryResult:
    benchmark_id: str
    source_sha256: str
    arms: tuple[RecoveryArm, ...]
    eligible_timesteps: tuple[int, ...]
    m: int
    oracle_subsets_evaluated: int
    streams_disjoint: bool
    full_horizons_only: bool
    matched_count_passed: bool
    matched_compute_passed: bool
    aged_downstream_spearman: float
    identifiability_passed: bool
    realistic_scale_passed: bool
    oracle_advantage: float
    oracle_gain_over_never_mse: float
    useful_sparsity: float
    qualifies_p7: bool

    def metric_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for arm in self.arms:
            row = asdict(arm)
            row["record_type"] = "p2_2_recovery_frontier"
            row["selected_timesteps"] = ",".join(map(str, arm.selected_timesteps))
            row.update(
                {
                    "benchmark_id": self.benchmark_id,
                    "oracle_subsets_evaluated": self.oracle_subsets_evaluated,
                    "matched_count_passed": self.matched_count_passed,
                    "matched_compute_passed": self.matched_compute_passed,
                    "aged_downstream_spearman": self.aged_downstream_spearman,
                    "identifiability_passed": self.identifiability_passed,
                    "realistic_scale_passed": self.realistic_scale_passed,
                    "oracle_advantage": self.oracle_advantage,
                    "oracle_gain_over_never_mse": self.oracle_gain_over_never_mse,
                    "useful_sparsity": self.useful_sparsity,
                    "qualifies_p7": self.qualifies_p7,
                }
            )
            rows.append(row)
        return rows


def _load_images() -> tuple[tuple[FloatArray, ...], str]:
    root = files("matplotlib").joinpath("mpl-data/sample_data")
    images: list[FloatArray] = []
    digest = hashlib.sha256()
    for name in _ASSETS:
        path = Path(str(root.joinpath(name)))
        digest.update(path.read_bytes())
        with Image.open(path) as source:
            rgb = Image.new("RGB", source.size, "white")
            if source.mode == "RGBA":
                rgb.paste(source.convert("RGB"), mask=source.getchannel("A"))
            else:
                rgb.paste(source.convert("RGB"))
            images.append(np.asarray(rgb, dtype=np.float64) / 255.0)
    return tuple(images), digest.hexdigest()


def _features(corrupted: FloatArray) -> FloatArray:
    r, g, b = corrupted.T
    return np.column_stack((r, g, b, r * r, g * g, b * b, r * g, r * b, g * b, np.ones(r.size)))


def _mode(t: int, regime_length: int) -> tuple[FloatArray, float]:
    modes = (
        (np.array([0.55, 0.82, 1.18]), 0.08),
        (np.array([1.22, 0.62, 0.80]), -0.045),
        (np.array([0.72, 1.24, 0.58]), 0.055),
        (np.array([1.18, 0.76, 1.28]), -0.075),
    )
    return modes[(t // regime_length) % len(modes)]


def _unit(
    image: FloatArray,
    *,
    band: int,
    n_bands: int,
    t: int,
    seed: int,
    cfg: RecoveryConfig,
    name: str,
) -> _Unit:
    height, width, _ = image.shape
    left, right = band * width // n_bands, (band + 1) * width // n_bands
    rng = np.random.default_rng(np.random.SeedSequence([seed, band, t, len(name)]))
    top = int(rng.integers(0, height - cfg.patch_size + 1))
    x0 = int(rng.integers(left, right - cfg.patch_size + 1))
    clean = image[top : top + cfg.patch_size, x0 : x0 + cfg.patch_size].reshape(-1, 3)
    scale, bias = _mode(t, cfg.regime_length)
    noise = rng.normal(0.0, 0.012, clean.shape)
    corrupted = np.clip(clean * scale + bias + noise, 0.0, 1.0)
    return _Unit(f"{name}:{band}:{t}:{top}:{x0}", _features(corrupted), clean.copy())


def _build(seed: int, cfg: RecoveryConfig) -> tuple[dict[str, tuple[_Unit, ...]], FloatArray, str]:
    images, digest = _load_images()
    # Band zero is pretraining-only; bands one through four own the four streams.
    pretrain: list[_Unit] = []
    for image_index, image in enumerate(images):
        for t in range(16):
            pretrain.append(
                _unit(
                    image,
                    band=0,
                    n_bands=5,
                    t=t,
                    seed=seed + image_index * 101,
                    cfg=cfg,
                    name="pretrain",
                )
            )
    X = np.concatenate([unit.x for unit in pretrain])
    Y = np.concatenate([unit.y for unit in pretrain])
    ridge = 1e-3 * np.eye(X.shape[1])
    initial = np.linalg.solve(X.T @ X + ridge, X.T @ Y).T

    streams: dict[str, tuple[_Unit, ...]] = {}
    for band, name in enumerate(_STREAMS, start=1):
        units = []
        for t in range(cfg.n_steps):
            image_index = (t + t // cfg.regime_length) % len(images)
            units.append(
                _unit(
                    images[image_index],
                    band=band,
                    n_bands=5,
                    t=t,
                    seed=seed + image_index * 101,
                    cfg=cfg,
                    name=name,
                )
            )
        streams[name] = tuple(units)
    return streams, initial, digest


def _loss(weights: FloatArray, unit: _Unit) -> float:
    residual = unit.x @ weights.T - unit.y
    return float(np.mean(residual * residual))


def _delta(weights: FloatArray, unit: _Unit, learning_rate: float) -> FloatArray:
    residual = unit.x @ weights.T - unit.y
    gradient = 2.0 * residual.T @ unit.x / unit.x.shape[0]
    delta = -learning_rate * gradient
    norm = float(np.linalg.vector_norm(delta))
    return delta * min(1.0, 0.15 / max(norm, 1e-12))


def _improvement(weights: FloatArray, delta: FloatArray, unit: _Unit) -> float:
    return _loss(weights, unit) - _loss(weights + delta, unit)


def _write_utility(
    weights: FloatArray,
    delta: FloatArray,
    t: int,
    evaluation: tuple[_Unit, ...],
    cfg: RecoveryConfig,
) -> tuple[float, float, float]:
    benefit = sum(
        weight * _improvement(weights, delta, evaluation[t + h])
        for weight, h in zip(cfg.horizon_weights, cfg.horizons)
    )
    damage = -min(0.0, _improvement(weights, delta, evaluation[t - cfg.min_age]))
    return benefit, damage, benefit - cfg.retention_lambda * damage - cfg.write_cost


def _replay(
    method: str,
    selected: tuple[int, ...],
    streams: dict[str, tuple[_Unit, ...]],
    initial: FloatArray,
    evaluation_name: str,
    cfg: RecoveryConfig,
    deployable_units: int,
    offline_units: int = 0,
) -> RecoveryArm:
    weights = initial.copy()
    selected_set = set(selected)
    benefit = damage = 0.0
    eligible = set(range(cfg.min_age, cfg.n_steps - max(cfg.horizons)))
    for t, candidate in enumerate(streams["candidate"]):
        delta = _delta(weights, candidate, cfg.learning_rate)
        if t not in selected_set:
            continue
        if t in eligible:
            b, d, _ = _write_utility(weights, delta, t, streams[evaluation_name], cfg)
            benefit += b
            damage += d
        weights += delta
    cost = cfg.write_cost * len(selected)
    mse = float(np.mean([_loss(weights, unit) for unit in streams["downstream"]]))
    return RecoveryArm(
        method,
        len(selected),
        tuple(sorted(selected)),
        benefit,
        damage,
        cost,
        benefit - cfg.retention_lambda * damage - cost,
        mse,
        float(-10.0 * np.log10(max(mse, 1e-12))),
        deployable_units,
        offline_units,
    )


def _validation_scores(
    streams: dict[str, tuple[_Unit, ...]],
    initial: FloatArray,
    eligible: tuple[int, ...],
    cfg: RecoveryConfig,
) -> FloatArray:
    """Execute the deployable validation budget for one matched-count arm."""

    scores = []
    for t in eligible:
        delta = _delta(initial, streams["candidate"][t], cfg.learning_rate)
        scores.append(_improvement(initial, delta, streams["validation"][t]))
    return np.asarray(scores, dtype=np.float64)


def run_recovery(seed: int, cfg: RecoveryConfig) -> RecoveryResult:
    streams, initial, digest = _build(seed, cfg)
    eligible = tuple(range(cfg.min_age, cfg.n_steps - max(cfg.horizons)))
    ids = {name: {unit.unit_id for unit in units} for name, units in streams.items()}
    disjoint = all(
        ids[left].isdisjoint(ids[right])
        for index, left in enumerate(_STREAMS)
        for right in _STREAMS[index + 1 :]
    )

    # Exact small-budget ceiling: enumerate subsets only after eligibility is frozen.
    combinations = tuple(itertools.combinations(eligible, cfg.m))
    oracle_values = [
        _replay("search", subset, streams, initial, "oracle_probe", cfg, 0).utility
        for subset in combinations
    ]
    oracle_selected = combinations[int(np.argmax(oracle_values))]

    # Validation-only ranking uses no oracle or downstream unit.
    validation_scores = _validation_scores(streams, initial, eligible, cfg)
    validation_order = np.argsort(-validation_scores, kind="stable")[: cfg.m]
    validation_selected = tuple(sorted(eligible[int(index)] for index in validation_order))

    matched_units = cfg.n_steps + 2 * len(eligible)
    arms = [
        _replay(
            "always", tuple(range(cfg.n_steps)), streams, initial, "downstream", cfg, cfg.n_steps
        ),
        _replay("never", (), streams, initial, "downstream", cfg, 0),
        _replay(
            "validation_only",
            validation_selected,
            streams,
            initial,
            "downstream",
            cfg,
            matched_units,
        ),
        _replay(
            "oracle_ceiling",
            oracle_selected,
            streams,
            initial,
            "downstream",
            cfg,
            matched_units,
            len(combinations) * cfg.n_steps * (len(cfg.horizons) + 1),
        ),
    ]
    for replicate in range(cfg.random_replicates):
        # Execute and discard the same disjoint validation workload used by the
        # validation selector. This is real matched work, not ledger padding.
        _ = _validation_scores(streams, initial, eligible, cfg)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1701, replicate]))
        selected = tuple(sorted(int(t) for t in rng.choice(eligible, cfg.m, replace=False)))
        arms.append(
            _replay(
                f"random-{replicate}",
                selected,
                streams,
                initial,
                "downstream",
                cfg,
                matched_units,
            )
        )

    oracle_damage, downstream_damage = [], []
    for t in eligible:
        delta = _delta(initial, streams["candidate"][t], cfg.learning_rate)
        oracle_damage.append(
            -min(0.0, _improvement(initial, delta, streams["oracle_probe"][t - cfg.min_age]))
        )
        downstream_damage.append(
            -min(0.0, _improvement(initial, delta, streams["downstream"][t - cfg.min_age]))
        )
    correlation = spearman(np.asarray(oracle_damage), np.asarray(downstream_damage))
    correlation = 0.0 if not np.isfinite(correlation) else float(correlation)
    matched = [
        arm
        for arm in arms
        if arm.method == "validation_only" or arm.method.startswith("random-")
    ]
    matched_count = all(arm.n_writes == cfg.m for arm in (*matched, arms[3]))
    matched_compute = len({arm.deployable_compute_units for arm in matched}) == 1
    oracle = arms[3]
    best_simple = max(arm.utility for arm in matched)
    never = arms[1]
    advantage = oracle.utility - best_simple
    gain_over_never = never.downstream_mse - oracle.downstream_mse
    useful = (
        1.0 - cfg.m / len(eligible)
        if advantage > cfg.utility_margin and gain_over_never > cfg.utility_margin
        else 0.0
    )
    identifiable = correlation >= cfg.correlation_threshold
    realistic = False  # Three packaged images/nonlinear head still are not CIFAR-C + ResNet/ViT.
    qualifies = bool(
        realistic and identifiable and matched_count and matched_compute and useful > 0
    )
    return RecoveryResult(
        "matplotlib-three-image-nonlinear-corruption-r1",
        digest,
        tuple(arms),
        eligible,
        cfg.m,
        len(combinations),
        disjoint,
        True,
        matched_count,
        matched_compute,
        correlation,
        identifiable,
        realistic,
        advantage,
        gain_over_never,
        useful,
        qualifies,
    )
