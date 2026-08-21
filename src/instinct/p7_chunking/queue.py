from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Item:
    item_id: str
    arrival: float
    score: float
    stream_id: str = "default"


@dataclass(frozen=True, slots=True)
class QueueConfig:
    buffer_size: int = 8
    max_age: float = 10.0
    threshold: float = 0.5
    audit_probability: float = 0.05
    scorer_time: float = 0.01
    update_time: float = 0.1

    def __post_init__(self) -> None:
        if self.buffer_size < 1 or self.max_age <= 0:
            raise ValueError("buffer_size and max_age must be positive")
        if not 0 <= self.audit_probability <= 1:
            raise ValueError("audit_probability must lie in [0,1]")
        if self.scorer_time < 0 or self.update_time < 0:
            raise ValueError("service times must be nonnegative")


@dataclass(frozen=True, slots=True)
class UpdateEvent:
    stream_id: str
    trigger: str
    started: float
    completed: float
    real_tokens: int
    padding_tokens: int
    kernel_shape: tuple[int]
    maximum_age_at_completion: float
    item_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QueueTrace:
    updates: tuple[UpdateEvent, ...]
    admitted: int
    rejected: int
    audited_rejections: int
    scorer_time: float
    update_time: float
    elapsed: float
    audit_item_ids: tuple[str, ...]
    audit_propensities: tuple[float, ...]
    admitted_item_ids: tuple[str, ...]
    maximum_score_queue_delay: float
    arrival_span: float

    @property
    def utilization(self) -> float:
        return (self.scorer_time + self.update_time) / max(self.elapsed, 1e-12)

    @property
    def offered_load(self) -> float:
        return (self.scorer_time + self.update_time) / max(self.arrival_span, 1e-12)

    @property
    def audit_effective_sample_size(self) -> float:
        if not self.audit_propensities:
            return 0.0
        weights = 1.0 / np.asarray(self.audit_propensities, dtype=np.float64)
        return float(weights.sum() ** 2 / np.sum(weights**2))


def simulate_queue(items: list[Item], cfg: QueueConfig, *, seed: int = 0) -> QueueTrace:
    rng = np.random.default_rng(seed)
    buffers: dict[str, list[Item]] = {}
    updates: list[UpdateEvent] = []
    resource_free = 0.0
    admitted = rejected = audited = 0
    audit_ids: list[str] = []
    audit_propensities: list[float] = []
    admitted_ids: list[str] = []
    maximum_score_queue_delay = 0.0

    def flush(stream_id: str, trigger: str, ready_at: float) -> None:
        nonlocal resource_free
        buf = buffers.get(stream_id, [])
        if not buf:
            return
        start = max(resource_free, ready_at)
        completed = start + cfg.update_time
        oldest = min(item.arrival for item in buf)
        real = len(buf)
        updates.append(
            UpdateEvent(
                stream_id,
                trigger,
                start,
                completed,
                real,
                cfg.buffer_size - real,
                (cfg.buffer_size,),
                completed - oldest,
                tuple(item.item_id for item in buf),
            )
        )
        resource_free = completed
        buffers[stream_id] = []

    ordered = sorted(items, key=lambda item: (item.arrival, item.item_id))
    for item in ordered:
        # Score and update share the serialized resource in this conservative model.
        score_done = max(resource_free, item.arrival) + cfg.scorer_time
        maximum_score_queue_delay = max(maximum_score_queue_delay, score_done - item.arrival)
        resource_free = score_done
        for stream_id, buf in list(buffers.items()):
            if buf and score_done >= min(x.arrival for x in buf) + cfg.max_age:
                flush(stream_id, "deadline", score_done)
        rejected_by_scorer = item.score < cfg.threshold
        audit = rejected_by_scorer and rng.random() < cfg.audit_probability
        if audit:
            audited += 1
            audit_ids.append(item.item_id)
            audit_propensities.append(cfg.audit_probability)
        if not rejected_by_scorer:
            admitted += 1
            admitted_ids.append(item.item_id)
            buffers.setdefault(item.stream_id, []).append(item)
            if len(buffers[item.stream_id]) == cfg.buffer_size:
                flush(item.stream_id, "full", resource_free)
        else:
            rejected += 1
    for stream_id, buf in list(buffers.items()):
        if buf:
            flush(stream_id, "end", resource_free)
    elapsed = max(resource_free, ordered[-1].arrival if ordered else 0.0)
    arrival_span = (
        ordered[-1].arrival - ordered[0].arrival if len(ordered) > 1 else max(elapsed, 1e-12)
    )
    return QueueTrace(
        tuple(updates),
        admitted,
        rejected,
        audited,
        len(ordered) * cfg.scorer_time,
        len(updates) * cfg.update_time,
        elapsed,
        tuple(audit_ids),
        tuple(audit_propensities),
        tuple(admitted_ids),
        maximum_score_queue_delay,
        arrival_span,
    )
