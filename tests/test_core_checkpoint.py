"""CheckpointStore: atomic writes, save/load round-trip, and cadence gating."""

from __future__ import annotations

import json
import os

from instinct.core.checkpoint import CheckpointStore


def test_save_then_load_round_trips_a_nested_mapping(tmp_path) -> None:
    store = CheckpointStore(run_dir=tmp_path)
    state = {"next_step": 3, "rows": [{"step": 0, "value": 1.5}, {"step": 1, "value": 2.5}]}
    store.save(state)
    assert store.load() == state


def test_load_returns_none_before_any_save(tmp_path) -> None:
    store = CheckpointStore(run_dir=tmp_path)
    assert store.load() is None


def test_second_save_overwrites_rather_than_appending(tmp_path) -> None:
    store = CheckpointStore(run_dir=tmp_path)
    store.save({"next_step": 1})
    store.save({"next_step": 2})
    assert store.load() == {"next_step": 2}


def test_a_truncated_write_never_replaces_the_prior_valid_checkpoint(tmp_path) -> None:
    """Simulates a crash mid-write: a stray, unfinished temp file must not be
    mistaken for the checkpoint, and the last complete save must still read back
    cleanly. This is what atomic rename via os.replace buys."""
    store = CheckpointStore(run_dir=tmp_path)
    store.save({"next_step": 1, "note": "first complete checkpoint"})

    # Simulate a process dying mid-write: a leftover temp file sits beside the
    # real checkpoint, truncated and invalid.
    stray_tmp = store.path.with_name(f"{store.path.name}.999999.tmp")
    stray_tmp.write_text("{not valid json")

    assert store.load() == {"next_step": 1, "note": "first complete checkpoint"}
    # The stray temp file is inert: it is never the file CheckpointStore reads.
    assert json.loads(store.path.read_text())["state"]["next_step"] == 1
    stray_tmp.unlink()


def test_checkpoint_written_via_atomic_rename_not_left_as_tmp(tmp_path) -> None:
    store = CheckpointStore(run_dir=tmp_path)
    store.save({"a": 1})
    names = os.listdir(tmp_path)
    assert store.filename in names
    assert not any(n.endswith(".tmp") for n in names)


def test_should_checkpoint_fires_immediately_before_any_save(tmp_path) -> None:
    store = CheckpointStore(run_dir=tmp_path)
    assert store.should_checkpoint(elapsed_s=0.0, every_s=600.0) is True


def test_should_checkpoint_respects_the_interval_after_a_save(tmp_path) -> None:
    store = CheckpointStore(run_dir=tmp_path)
    store.save({"a": 1})
    assert store.should_checkpoint(elapsed_s=0.0, every_s=600.0) is False


def test_should_checkpoint_is_false_for_a_nonpositive_interval(tmp_path) -> None:
    store = CheckpointStore(run_dir=tmp_path)
    assert store.should_checkpoint(elapsed_s=0.0, every_s=0.0) is False
