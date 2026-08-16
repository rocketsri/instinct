"""In-run checkpoints: the state a resumed run needs that the manifest does not carry.

Nothing before this module saved *in-progress* state. ``core/compute/colab.py``'s
:class:`~instinct.core.compute.colab.ShardManifest` tracks which *shards* of a
sharded sweep finished — coarse, cross-process state, useful for resharding a
whole job. This module is the finer-grained thing spec section 3.1 requires for
every run class, not only sharded ones: a free Colab T4 session is interruptible
mid-run, and the work between "the process died" and "the next checkpoint" is
gone. So a proposal's ``run()`` calls :meth:`CheckpointStore.save` periodically
with whatever it needs to pick back up — a stream position, a ``ProbeLedger``'s
use counts, a training step — and ``instinct resume`` hands that state back to
``plugin.resume``.

The two are not interchangeable: a job can use both (shard-level resume *and*
in-shard checkpointing) or either alone, and conflating them is how a resume
path ends up trusting stale shard state for what actually changed mid-shard.

Atomic writes reuse :mod:`instinct.core.cache`'s pattern (write to a temp file
in the same directory, ``os.replace``) for the same reason: a process killed
mid-write must not leave a checkpoint that reads as valid JSON but is actually
truncated, which is worse than no checkpoint because it fails silently instead
of falling back to "no checkpoint" and resuming from scratch.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from instinct.core.cache import to_canonical

__all__ = ["CHECKPOINT_NAME", "CheckpointStore"]

CHECKPOINT_NAME = "checkpoint.json"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


@dataclass
class CheckpointStore:
    """One checkpoint slot per run directory.

    Deliberately a single slot, not a history: a resumed run wants the *latest*
    state, and keeping every intermediate checkpoint around trades disk for a
    debugging convenience nothing here needs. A proposal that wants its own
    checkpoint history can layer that on top of ``state`` itself.
    """

    run_dir: Path
    filename: str = CHECKPOINT_NAME
    _last_saved_at: float = 0.0

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)

    @property
    def path(self) -> Path:
        return self.run_dir / self.filename

    def save(self, state: Mapping[str, Any]) -> None:
        """Overwrite the checkpoint atomically.

        ``state`` must be canonicalizable (see :func:`instinct.core.cache.
        to_canonical`) — the same constraint every other persisted artifact in
        this repo has, and for the same reason: an object that stringifies with
        an embedded ``id()`` would make a checkpoint that cannot be told apart
        from a different run's.
        """
        payload = json.dumps(
            {"saved_at": time.time(), "state": to_canonical(dict(state))},
            sort_keys=True,
        )
        _atomic_write_text(self.path, payload)
        self._last_saved_at = time.time()

    def load(self) -> dict[str, Any] | None:
        """The most recent checkpoint, or ``None`` if there isn't one yet.

        ``None`` rather than raising: a run's first checkpoint interval has not
        elapsed yet is a normal state, not a resume failure, and ``instinct
        resume`` on a run that died before its first checkpoint has nothing to
        hand back but a config and a manifest — which is still a valid, if
        expensive, resume (the proposal starts over but keeps its run id and
        provenance chain).
        """
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text())
        state = data.get("state", {})
        return dict(state) if isinstance(state, Mapping) else {}

    def should_checkpoint(self, elapsed_s: float, every_s: float) -> bool:
        """Whether ``every_s`` has elapsed since the last save this process made.

        Measured against *this process's* last save (``_last_saved_at``, which
        starts at ``0.0``), not wall-clock since run start: a resumed run's
        checkpoint cadence should not be thrown off by however long the previous
        process ran before it died, and the never-saved-yet state trivially
        clears the threshold so the first checkpoint fires promptly rather than
        waiting a full interval. ``elapsed_s`` is accepted but not consulted for
        the decision; it is there for a caller to log alongside the check.
        """
        del elapsed_s
        if every_s <= 0:
            return False
        return (time.time() - self._last_saved_at) >= every_s
