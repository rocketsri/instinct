"""Run manifests: what ran, from what code, on what machine, for how long.

Every run writes a ``manifest.json`` next to its results. Not because manifests
are good hygiene in the abstract, but because of a specific recurring failure: a
results file exists, the number in it disagrees with the number in the report,
and there is no way to tell whether the code changed, the config changed, the
seed changed, or the machine changed. Six months of a project can be spent
answering that question badly.

The manifest is written **twice** — once at the start with ``status="running"``,
once at the end with the outcome. A run killed by the OOM killer, by Colab
preemption, or by a timeout therefore leaves a manifest that says it started and
never finished, instead of leaving a directory of partial results indexed by
nothing. A missing manifest and a manifest that says ``running`` mean different
things, and both are more informative than silence.

Results are jsonl and parquet, and both are append-friendly for a reason. A
sweep that dies at 80% should leave 80% of its rows readable, which rules out
formats that need a clean close to be parseable.

The compute ledger travels with the manifest. Rule 4's tripwire
(:func:`instinct.compute.budget.assert_matched`) can only be checked against
what was actually spent, so a run that does not record its spend cannot be
audited for matched compute later.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import time
import traceback
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

from instinct.compute.budget import ComputeLedger
from instinct.core.cache import to_canonical
from instinct.core.config import config_hash

__all__ = [
    "JsonlWriter",
    "Provenance",
    "RunManifest",
    "RunRecorder",
    "list_runs",
    "load_manifest",
    "resolve_run",
    "run_context",
]

#: Packages whose versions actually change results. Kept explicit rather than
#: dumping ``pip freeze``: a 300-line environment blob is not read by anyone,
#: and the point of provenance is that somebody reads it.
TRACKED_PACKAGES = (
    "instinct",
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "torch",
    "transformers",
)

MANIFEST_NAME = "manifest.json"


def _git(*args: str) -> str:
    """Run a git command, returning "" when git or the repo is unavailable.

    Provenance capture must never be the thing that kills a run. A missing SHA
    is a degraded manifest; an exception here would be a lost experiment.
    """
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _dep_versions(packages: Sequence[str] = TRACKED_PACKAGES) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in packages:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue  # optional extras are legitimately absent
    return out


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where this run came from. Captured once, at start."""

    git_sha: str
    git_branch: str
    git_dirty: bool
    host: str
    platform: str
    python: str
    dep_versions: dict[str, str]

    @staticmethod
    def capture() -> Provenance:
        status = _git("status", "--porcelain")
        return Provenance(
            git_sha=_git("rev-parse", "HEAD"),
            git_branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
            # A dirty tree means the SHA does not describe the code that ran.
            # Recording the flag is the difference between a reproducible result
            # and one that merely looks reproducible.
            git_dirty=bool(status),
            host=socket.gethostname(),
            platform=platform.platform(),
            python=platform.python_version(),
            dep_versions=_dep_versions(),
        )


@dataclass
class RunManifest:
    """The record written alongside every run's results."""

    run_id: str
    experiment: str
    config_hash: str
    config: Any
    provenance: Provenance
    started_at: str
    finished_at: str | None = None
    wall_clock_s: float | None = None
    status: str = "running"  # running | ok | failed
    error: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    compute: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "config_hash": self.config_hash,
            "config": to_canonical(self.config),
            "provenance": to_canonical(self.provenance),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wall_clock_s": self.wall_clock_s,
            "status": self.status,
            "error": self.error,
            "summary": to_canonical(self.summary),
            "compute": self.compute,
            "artifacts": self.artifacts,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True))


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Read a manifest from a run directory or a direct file path."""
    p = Path(path)
    if p.is_dir():
        p = p / MANIFEST_NAME
    if not p.exists():
        raise FileNotFoundError(f"no manifest at {p}")
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{p}: manifest is not an object")
    return data


def make_run_id(experiment: str, cfg_hash: str, when: datetime | None = None) -> str:
    """A sortable, self-describing run id.

    ``<experiment>-<utc timestamp>-<config hash prefix>``. Timestamp first after
    the name so lexical order is chronological order; the hash prefix so two runs
    of the same experiment in the same second are distinguishable and so the
    directory name already tells you whether two runs used the same config.
    """
    stamp = (when or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{experiment or 'run'}-{stamp}-{cfg_hash[:8]}"


class JsonlWriter:
    """Append-only jsonl sink, flushed on every record.

    Flushing every line is not free, and it is the right trade here: these files
    exist to be readable after a crash, and a buffered writer loses exactly the
    records that describe what was happening when things went wrong.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh: TextIO = path.open("a", encoding="utf-8")

    def write(self, record: Mapping[str, Any]) -> None:
        self._fh.write(json.dumps(to_canonical(record), sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


@dataclass
class RunRecorder:
    """Handle handed to an experiment: where to put things, and what to record.

    The experiment does not choose paths. It asks for a named stream or table and
    the recorder decides where it lands, which is what keeps every run directory
    the same shape and makes ``instinct report`` possible at all.
    """

    run_dir: Path
    manifest: RunManifest
    ledger: ComputeLedger = field(default_factory=ComputeLedger)
    _streams: dict[str, JsonlWriter] = field(default_factory=dict, repr=False)

    def stream(self, name: str) -> JsonlWriter:
        """A named jsonl stream, created on first use."""
        if name not in self._streams:
            self._streams[name] = JsonlWriter(self.run_dir / f"{name}.jsonl")
        return self._streams[name]

    def event(self, name: str, **fields: Any) -> None:
        """Append one record to a named stream, stamped with elapsed time."""
        self.stream(name).write({"t": time.time(), **fields})

    def table(self, name: str, rows: Sequence[Mapping[str, Any]] | pd.DataFrame) -> Path:
        """Write a parquet table into the run directory and register it."""
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame.from_records(list(rows))
        path = self.run_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        if name not in self.manifest.artifacts:
            self.manifest.artifacts.append(name)
        return path

    def note(self, **summary: Any) -> None:
        """Add to the manifest's summary block — the numbers worth reading first."""
        self.manifest.summary.update(summary)

    def close(self) -> None:
        for writer in self._streams.values():
            writer.close()
        self._streams.clear()


@contextmanager
def run_context(
    experiment: str,
    config: Any,
    *,
    output_root: str | Path = "runs",
    run_id: str | None = None,
) -> Iterator[RunRecorder]:
    """Open a run directory, write its manifest, and close it out honestly.

    On the way out the manifest records ``ok`` or ``failed`` with the traceback,
    then the exception is re-raised. Swallowing it would leave a manifest that
    claims success, which is worse than no manifest at all.
    """
    cfg_hash = config_hash(config)
    rid = run_id or make_run_id(experiment, cfg_hash)
    run_dir = Path(output_root) / experiment / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(UTC)
    manifest = RunManifest(
        run_id=rid,
        experiment=experiment,
        config_hash=cfg_hash,
        config=config,
        provenance=Provenance.capture(),
        started_at=started.isoformat(timespec="seconds"),
    )
    manifest.write(run_dir / MANIFEST_NAME)

    recorder = RunRecorder(run_dir=run_dir, manifest=manifest)
    clock = time.perf_counter()
    try:
        yield recorder
    except BaseException as exc:
        manifest.status = "failed"
        manifest.error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        raise
    else:
        manifest.status = "ok"
    finally:
        manifest.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
        manifest.wall_clock_s = time.perf_counter() - clock
        manifest.compute = [dict(r) for r in recorder.ledger.as_records()]
        manifest.write(run_dir / MANIFEST_NAME)
        recorder.close()


def list_runs(output_root: str | Path = "runs") -> list[dict[str, Any]]:
    """Every run under a root, newest first.

    Sorted by ``started_at`` rather than by mtime: mtime changes when a manifest
    is rewritten at the end, so mtime order is finish order, and a long run that
    started first would sort last.
    """
    root = Path(output_root)
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    for manifest_path in root.glob(f"*/*/{MANIFEST_NAME}"):
        try:
            out.append(load_manifest(manifest_path))
        except (OSError, ValueError, json.JSONDecodeError):
            # A manifest being written right now is not a reason to fail a listing.
            continue
    out.sort(key=lambda m: str(m.get("started_at", "")), reverse=True)
    return out


def resolve_run(run_id: str, output_root: str | Path = "runs") -> Path:
    """Find a run directory from a full or partial run id.

    Prefix matching because run ids carry a timestamp and a hash and nobody
    retypes those correctly. An ambiguous prefix raises rather than picking one:
    reporting on the wrong run is the failure to avoid.
    """
    root = Path(output_root)
    candidates = [
        p.parent for p in root.glob(f"*/*/{MANIFEST_NAME}") if p.parent.name.startswith(run_id)
    ]
    if not candidates:
        raise FileNotFoundError(f"no run under {root} whose id starts with {run_id!r}")
    if len(sorted({c.name for c in candidates})) > 1:
        names = sorted(c.name for c in candidates)
        raise ValueError(f"run id {run_id!r} is ambiguous; matches {names}")
    return candidates[0]
