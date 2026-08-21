from __future__ import annotations

from instinct.p7_chunking.queue import Item, QueueConfig, simulate_queue


def test_audit_channel_never_changes_update_content() -> None:
    items = [Item(str(i), i * 0.1, 0.0 if i % 2 else 1.0) for i in range(30)]
    trace = simulate_queue(
        items,
        QueueConfig(buffer_size=4, audit_probability=1.0),
        seed=4,
    )
    updated = {item_id for event in trace.updates for item_id in event.item_ids}
    assert updated == set(trace.admitted_item_ids)
    assert updated.isdisjoint(trace.audit_item_ids)
    assert trace.audited_rejections == trace.rejected


def test_real_tokens_are_unique_across_padding_flushes() -> None:
    items = [Item(str(i), i * 3.0, 1.0) for i in range(7)]
    trace = simulate_queue(items, QueueConfig(buffer_size=4, max_age=1.0))
    updated = [item_id for event in trace.updates for item_id in event.item_ids]
    assert len(updated) == len(set(updated)) == trace.admitted
    assert all(event.real_tokens + event.padding_tokens == 4 for event in trace.updates)


def test_offered_load_detects_backpressure_even_when_realized_utilization_is_bounded() -> None:
    items = [Item(str(i), i * 0.001, 1.0) for i in range(100)]
    trace = simulate_queue(
        items,
        QueueConfig(buffer_size=2, scorer_time=0.01, update_time=0.1),
    )
    assert trace.offered_load > 1.0
    assert trace.maximum_score_queue_delay > 1.0
    assert trace.utilization <= 1.0 + 1e-12
