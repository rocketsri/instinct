"""Content-addressed result cache: one grid point, one key, one file.

The sweeps here are outer products, and outer products get edited. Somebody adds
a budget, or a third value of ``nu_h``, or fixes one environment. Recomputing the
whole grid for that is how a 40-minute iteration loop becomes a 6-hour one, and a
6-hour loop is how experiments stop being run.

So results are keyed by *content*: a hash of the configuration that produced them
plus a code version. Adding a grid point changes exactly one key, so exactly one
point re-runs. Nothing is keyed by position in a list, by filename, or by run
order, all of which shift when the grid is edited and silently invalidate work
that was still good.

Why blake2b over canonical JSON, and not :func:`hash`
-----------------------------------------------------
Python's built-in ``hash`` is salted per process (PYTHONHASHSEED), so a key
computed today would not match the same config tomorrow. Every cache lookup
would miss, the cache would grow without bound, and — worse — it would look like
it was working. :mod:`instinct.core.rng` makes the same choice for the same
reason.

Canonicalization has to be strict, because the failure mode is asymmetric. A
*miss* costs one recomputation. A false *hit* silently serves the results of a
different experiment, and there is no downstream check that would catch it. So:
keys are sorted, floats are canonicalized, and any type the canonicalizer does
not recognize raises instead of falling back to ``str(obj)`` — because
``str`` of most objects embeds an ``id()``, which is a memory address, which
makes the key process-dependent in a way that only shows up as a mysterious
cache miss rate.

``1`` and ``1.0`` hash differently, and that is deliberate. Collapsing them would
require deciding that an int-valued float means the same thing as the int in
every config field, which is exactly the sort of assumption this repo does not
make. The cost of being wrong here is one extra run.

Why ``code_version`` is a string you bump by hand
-------------------------------------------------
The tempting alternatives are worse. Hashing the git SHA invalidates every cached
result on every commit, including commits that only touch a README. Hashing the
source files invalidates on a comment change and, more insidiously, does not
invalidate when the change is in a dependency. A hand-bumped version is honest
about what it is: a claim by the author that the computation changed. It is
recorded in every entry's sidecar so a stale claim is at least auditable.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from hashlib import blake2b
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "CacheEntry",
    "CacheError",
    "ResultCache",
    "canonical_json",
    "content_hash",
    "key_for",
    "to_canonical",
]

#: Bump when the *meaning* of a cached result changes and the config does not
#: capture it. This is the cache-wide escape hatch; per-experiment versions live
#: in the experiment's own config.
CACHE_FORMAT_VERSION = "1"

_KEY_BYTES = 16  # 128-bit keys; collision probability is not a real risk here


class CacheError(RuntimeError):
    """Raised when a cache entry is unusable or inconsistent with its sidecar."""


def to_canonical(obj: Any) -> Any:
    """Reduce an object to JSON-safe primitives, deterministically.

    Recognized: ``None``, ``bool``, ``int``, ``float``, ``str``, ``bytes``,
    mappings, sequences, sets, :class:`~pathlib.Path`, dataclasses, numpy scalars
    and arrays, and :class:`~enum.Enum`-like objects exposing ``.value``.

    Everything else raises. That is the point — see the module docstring. An
    unrecognized object silently stringified into a key is the one failure this
    module exists to prevent.
    """
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, bool):  # before int: bool is an int subclass
        return obj
    if isinstance(obj, Enum):
        return to_canonical(obj.value)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return _canonical_float(float(obj))
    if isinstance(obj, bytes):
        return {"__bytes__": obj.hex()}
    if isinstance(obj, Path):
        # posix form so a key does not depend on the host's path separator
        return {"__path__": obj.as_posix()}
    if isinstance(obj, np.generic):
        return to_canonical(obj.item())
    if isinstance(obj, np.ndarray):
        return {"__array__": to_canonical(obj.tolist()), "dtype": str(obj.dtype)}
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_canonical(asdict(obj))
    if isinstance(obj, Mapping):
        # Sorted so two configs that differ only in insertion order agree.
        return {str(k): to_canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, set | frozenset):
        return {"__set__": sorted(repr(to_canonical(v)) for v in obj)}
    if isinstance(obj, Sequence):
        return [to_canonical(v) for v in obj]
    raise TypeError(
        f"cannot canonicalize {type(obj).__name__} for a cache key. "
        "Add it to instinct.core.cache.to_canonical rather than stringifying it: "
        "str() of most objects embeds an id(), which would make the key differ "
        "between processes and turn every lookup into a miss."
    )


def _canonical_float(x: float) -> Any:
    """Normalize the float values whose textual forms are not unique.

    ``-0.0`` and ``0.0`` compare equal and mean the same thing everywhere in a
    config, but ``json.dumps`` writes them differently, so they would key
    differently. NaN and the infinities are written as tagged strings because
    ``json.dumps`` emits bare ``NaN``/``Infinity``, which is not valid JSON and
    would not survive a round trip through a strict reader.
    """
    if x == 0.0:
        return 0.0
    if math.isnan(x):
        return {"__float__": "nan"}
    if math.isinf(x):
        return {"__float__": "inf" if x > 0 else "-inf"}
    return x


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text for any canonicalizable object."""
    return json.dumps(
        to_canonical(obj),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_hash(obj: Any, *, digest_size: int = _KEY_BYTES) -> str:
    """Stable hex digest of an object's canonical form.

    blake2b, not :func:`hash`: the built-in is salted per process, so a key made
    in one run would not match the same content in the next.
    """
    return blake2b(canonical_json(obj).encode("utf-8"), digest_size=digest_size).hexdigest()


def key_for(config: Mapping[str, Any] | Any, *, code_version: str = "") -> str:
    """The cache key for one grid point.

    ``code_version`` is folded in as a sibling field rather than concatenated to
    the digest, so a config that happens to contain a ``code_version`` key cannot
    collide with the version supplied here.

    >>> key_for({"b": 2, "a": 1}) == key_for({"a": 1, "b": 2})
    True
    >>> key_for({"a": 1}, code_version="v1") == key_for({"a": 1}, code_version="v2")
    False
    """
    return content_hash(
        {
            "config": config,
            "code_version": code_version,
            "cache_format": CACHE_FORMAT_VERSION,
        }
    )


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """One stored result, plus enough provenance to distrust it later.

    The sidecar carries the *full canonical config*, not only its hash. A hash
    tells you two things differ; the config tells you how, which is what you
    actually need at 2am when a cell you expected to hit is missing.
    """

    key: str
    config: Any
    code_version: str
    payload_kind: str  # "parquet" | "json"
    created_at: str
    n_rows: int
    path: Path

    def as_record(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "code_version": self.code_version,
            "payload_kind": self.payload_kind,
            "created_at": self.created_at,
            "n_rows": self.n_rows,
            "path": str(self.path),
            "config": self.config,
        }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write via a temp file in the same directory, then rename.

    A run that dies mid-write must not leave a half-written parquet that the
    next run happily reads as a cache hit. ``os.replace`` is atomic within a
    filesystem, so an entry either exists complete or does not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


@dataclass
class ResultCache:
    """A directory of content-addressed results.

    Layout::

        <root>/<key[:2]>/<key>.parquet   # or .json
        <root>/<key[:2]>/<key>.meta.json

    Sharded by the first two hex characters because a full P1 sweep produces
    thousands of entries and a directory with thousands of siblings is slow to
    list on every filesystem worth naming.

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as d:
    ...     cache = ResultCache(root=Path(d), code_version="demo.v1")
    ...     calls = []
    ...     def run(cfg):
    ...         calls.append(cfg)
    ...         return [{"k": cfg["k"], "value": cfg["k"] * 2}]
    ...     _ = cache.map_grid([{"k": 1}, {"k": 2}], run)
    ...     _ = cache.map_grid([{"k": 1}, {"k": 2}, {"k": 3}], run)  # only k=3 re-runs
    ...     [c["k"] for c in calls]
    [1, 2, 3]
    """

    root: Path
    code_version: str = ""
    enabled: bool = True
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    # -- addressing -------------------------------------------------------

    def key_for(self, config: Mapping[str, Any] | Any) -> str:
        return key_for(config, code_version=self.code_version)

    def _dir_for(self, key: str) -> Path:
        return self.root / key[:2]

    def _meta_path(self, key: str) -> Path:
        return self._dir_for(key) / f"{key}.meta.json"

    def _payload_path(self, key: str, kind: str) -> Path:
        suffix = "parquet" if kind == "parquet" else "json"
        return self._dir_for(key) / f"{key}.{suffix}"

    # -- reading ----------------------------------------------------------

    def entry(self, config: Mapping[str, Any] | Any) -> CacheEntry | None:
        """The stored entry for a config, or ``None``.

        Returns ``None`` rather than raising when the payload is missing but the
        sidecar is present: that is what a crash between the two writes looks
        like, and the right response is to recompute, not to abort the sweep.
        """
        key = self.key_for(config)
        meta_path = self._meta_path(key)
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        payload = self._payload_path(key, meta["payload_kind"])
        if not payload.exists():
            return None
        return CacheEntry(
            key=key,
            config=meta["config"],
            code_version=meta["code_version"],
            payload_kind=meta["payload_kind"],
            created_at=meta["created_at"],
            n_rows=int(meta["n_rows"]),
            path=payload,
        )

    def has(self, config: Mapping[str, Any] | Any) -> bool:
        return self.enabled and self.entry(config) is not None

    def get(self, config: Mapping[str, Any] | Any) -> Any:
        """Load a cached result. Raises :class:`CacheError` if absent.

        Verifies that the sidecar's canonical config matches the one asked for.
        A 128-bit blake2b collision is not the concern; a hand-edited cache
        directory, a copied entry, or a key scheme changed underneath an existing
        cache all are, and they present as exactly this mismatch.
        """
        entry = self.entry(config)
        if entry is None:
            raise CacheError(f"no cache entry for key {self.key_for(config)}")
        if canonical_json(entry.config) != canonical_json(to_canonical(config)):
            raise CacheError(
                f"cache entry {entry.key} holds a different config than the one requested. "
                "The cache directory is inconsistent; delete the entry rather than trusting it."
            )
        if entry.payload_kind == "parquet":
            return pd.read_parquet(entry.path)
        return json.loads(entry.path.read_text())

    # -- writing ----------------------------------------------------------

    def put(self, config: Mapping[str, Any] | Any, value: Any) -> CacheEntry:
        """Store a result.

        Tabular values (a :class:`pandas.DataFrame`, or a sequence of mappings)
        go to parquet; anything else goes to JSON. Parquet because the sweep
        outputs are long and narrow and the whole grid gets concatenated for
        analysis, and JSON because not every result is a table and a cache that
        only holds tables gets worked around.

        The payload is written before the sidecar. If the process dies between
        them, :meth:`entry` sees no sidecar and the point recomputes — the safe
        direction. The reverse order would leave a sidecar advertising a payload
        that is not there.
        """
        key = self.key_for(config)
        frame = _as_frame(value)
        if frame is not None:
            kind, n_rows = "parquet", int(frame.shape[0])
            path = self._payload_path(key, kind)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
        else:
            kind = "json"
            n_rows = len(value) if isinstance(value, Sequence) and not isinstance(value, str) else 1
            path = self._payload_path(key, kind)
            _atomic_write_bytes(path, canonical_json(value).encode("utf-8"))

        entry = CacheEntry(
            key=key,
            config=to_canonical(config),
            code_version=self.code_version,
            payload_kind=kind,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            n_rows=n_rows,
            path=path,
        )
        meta = entry.as_record()
        meta.pop("path")
        _atomic_write_bytes(self._meta_path(key), json.dumps(meta, sort_keys=True).encode("utf-8"))
        return entry

    # -- the thing callers actually use -----------------------------------

    def get_or_compute(
        self, config: Mapping[str, Any] | Any, compute: Callable[[Any], Any]
    ) -> tuple[Any, bool]:
        """Return ``(value, was_hit)``, computing and storing on a miss.

        ``compute`` receives the config, so a caller can pass the same function
        for every grid point.
        """
        if self.enabled and self.has(config):
            self.hits += 1
            return self.get(config), True
        self.misses += 1
        value = compute(config)
        if self.enabled:
            self.put(config, value)
        return value, False

    def map_grid(
        self,
        configs: Sequence[Mapping[str, Any] | Any],
        compute: Callable[[Any], Any],
    ) -> list[Any]:
        """Evaluate a grid, running only the points that are not already stored.

        This is the method the whole module exists for: adding one point to
        ``configs`` costs one evaluation, not ``len(configs)``.
        """
        return [self.get_or_compute(cfg, compute)[0] for cfg in configs]

    # -- housekeeping -----------------------------------------------------

    def iter_entries(self) -> Iterator[CacheEntry]:
        """Every entry in the cache, for auditing what a directory contains."""
        for meta_path in sorted(self.root.glob("*/*.meta.json")):
            meta = json.loads(meta_path.read_text())
            key = meta_path.name.removesuffix(".meta.json")
            yield CacheEntry(
                key=key,
                config=meta["config"],
                code_version=meta["code_version"],
                payload_kind=meta["payload_kind"],
                created_at=meta["created_at"],
                n_rows=int(meta["n_rows"]),
                path=self._payload_path(key, meta["payload_kind"]),
            )

    def evict(self, config: Mapping[str, Any] | Any) -> bool:
        """Drop one entry. Returns whether anything was there."""
        key = self.key_for(config)
        removed = False
        for path in (
            self._meta_path(key),
            self._payload_path(key, "parquet"),
            self._payload_path(key, "json"),
        ):
            if path.exists():
                path.unlink()
                removed = True
        return removed

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


def _as_frame(value: Any) -> pd.DataFrame | None:
    """Coerce tabular values to a DataFrame; return ``None`` for non-tabular.

    A list of mappings is the shape experiment code naturally produces, and
    turning it into a frame here means callers do not have to import pandas to
    get a parquet-backed cache entry.
    """
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        rows = list(value)
        if rows and all(isinstance(r, Mapping) for r in rows):
            return pd.DataFrame.from_records([dict(r) for r in rows])
    return None
