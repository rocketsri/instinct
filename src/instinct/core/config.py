"""Typed configs, loaded from YAML, with unknown keys treated as errors.

The failure this module exists to prevent is quiet and expensive. You write

.. code-block:: yaml

    gama: 0.99

in a sweep config, the loader ignores the key it does not recognize, the run
uses the default ``gamma=0.95``, and the results are filed under a name that
says 0.99. Nothing crashes. Nothing in the analysis can detect it. The
experiment is simply wrong, and it stays wrong until somebody re-derives a
number by hand.

So :func:`from_mapping` rejects unknown keys, rejects missing required keys, and
coerces each value to the field's declared type instead of trusting whatever
YAML's scalar resolver produced. It also suggests the nearest known key on a
typo, because the error is only useful if it names the fix.

Configs are frozen dataclasses. Immutability matters more than usual here: the
config's hash is the cache key and part of the run manifest, so a config mutated
mid-run would produce results filed under a description of something that never
ran.

The grid helpers (:func:`expand_grid`) live here rather than in the sweep code
because the cache keys off individual grid points, and both sides have to agree
on exactly what a point is.
"""

from __future__ import annotations

import difflib
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal, TypeVar, Union, cast, get_args, get_origin

import yaml

from instinct.core.cache import content_hash, to_canonical

__all__ = [
    "CacheConfig",
    "ConfigError",
    "RunConfig",
    "apply_overrides",
    "config_hash",
    "expand_grid",
    "from_mapping",
    "load_config",
    "to_dict",
]

T = TypeVar("T")


class ConfigError(ValueError):
    """A config could not be built. Always names the offending key."""


# ---------------------------------------------------------------------------
# The configs themselves
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """Where cached grid points live and what invalidates them."""

    enabled: bool = True
    root: Path = Path(".cache/instinct")
    #: Hand-bumped when the computation changes in a way the config does not
    #: express. See the ``ResultCache`` docstring for why this is not derived
    #: from the git SHA.
    code_version: str = ""


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Everything a run needs that is not experiment-specific.

    ``params`` is deliberately an untyped mapping: each experiment validates its
    own parameters against its own dataclass. Forcing every experiment's schema
    into this file would make the shared config a merge conflict magnet, and the
    hashing path does not care — :func:`to_canonical` handles nested primitives.
    """

    experiment: str = ""
    seed: int = 0
    output_root: Path = Path("runs")
    tag: str = ""
    notes: str = ""
    cache: CacheConfig = field(default_factory=CacheConfig)
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Construction from plain mappings
# ---------------------------------------------------------------------------


def from_mapping(cls: type[T], data: Mapping[str, Any], *, path: str = "") -> T:
    """Build a dataclass from a mapping, strictly.

    Strictly means: unknown keys raise, missing keys without defaults raise, and
    every value is coerced to its declared type. YAML resolves ``1e-3`` to a
    float but ``1`` to an int and ``yes`` to a bool, so untyped acceptance means
    the type of a config field depends on how somebody happened to write it.
    """
    if not is_dataclass(cls):
        raise ConfigError(f"{cls.__name__} is not a dataclass")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}

    unknown = [k for k in data if k not in known]
    if unknown:
        raise ConfigError(_unknown_key_message(unknown, known, path))

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        where = f"{path}.{f.name}" if path else f.name
        if f.name not in data:
            continue  # dataclass defaults apply; a missing required field errors below
        kwargs[f.name] = _coerce(data[f.name], hints[f.name], where)

    try:
        return cast(T, cls(**kwargs))
    except TypeError as exc:  # missing required field
        raise ConfigError(f"cannot build {cls.__name__} at {path or '<root>'}: {exc}") from exc


def _unknown_key_message(unknown: Sequence[str], known: set[str], path: str) -> str:
    lines = []
    for key in unknown:
        close = difflib.get_close_matches(key, sorted(known), n=1)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        lines.append(f"  {path + '.' if path else ''}{key}{hint}")
    return (
        "unknown config key(s):\n"
        + "\n".join(lines)
        + f"\nknown keys here: {sorted(known)}\n"
        "Unknown keys are an error, not a warning: silently ignoring one means "
        "the run used a default while the config claims otherwise."
    )


def _coerce(value: Any, hint: Any, where: str) -> Any:
    """Convert one YAML-derived value to its declared type, or explain why not."""
    origin = get_origin(hint)

    if hint is Any or hint is object:
        return value

    # Optional[X] / X | None / unions generally.
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(hint) if a is not type(None)]
        if value is None:
            if len(args) != len(get_args(hint)):
                return None
            raise ConfigError(f"{where}: None is not allowed for {hint}")
        # Try each member; the first that accepts wins. Unions in configs are
        # rare and shallow, so first-match is adequate and predictable.
        for arg in args:
            try:
                return _coerce(value, arg, where)
            except ConfigError:
                continue
        raise ConfigError(f"{where}: {value!r} matches no member of {hint}")

    if origin is Literal:
        allowed = get_args(hint)
        if value not in allowed:
            raise ConfigError(f"{where}: {value!r} is not one of {list(allowed)}")
        return value

    if is_dataclass(hint) and isinstance(hint, type):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a mapping for {hint.__name__}, got {type(value)}")
        return from_mapping(hint, value, path=where)

    if origin in (list, Sequence):
        (inner,) = get_args(hint) or (Any,)
        seq = _as_sequence(value, where)
        return [_coerce(v, inner, f"{where}[{i}]") for i, v in enumerate(seq)]

    if origin is tuple:
        # Distinct name from the list binding above: that one is a list, this a
        # tuple, and reusing the name makes the two branches unsound to a checker.
        targs = get_args(hint)
        elems = list(_as_sequence(value, where))
        if len(targs) == 2 and targs[1] is Ellipsis:
            return tuple(_coerce(v, targs[0], f"{where}[{i}]") for i, v in enumerate(elems))
        if len(targs) != len(elems):
            raise ConfigError(f"{where}: expected {len(targs)} items, got {len(elems)}")
        return tuple(_coerce(v, a, f"{where}[{i}]") for i, (v, a) in enumerate(zip(elems, targs)))

    if origin in (dict, Mapping):
        kt, vt = get_args(hint) or (Any, Any)
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a mapping, got {type(value)}")
        return {
            _coerce(k, kt, where): _coerce(v, vt, f"{where}.{k}") for k, v in value.items()
        }

    if hint is Path:
        if not isinstance(value, str | Path):
            raise ConfigError(f"{where}: expected a path string, got {type(value)}")
        return Path(value)

    if hint is bool:
        # Not `bool(value)`: that turns the string "false" into True, which is
        # the single most common config bug in any YAML-driven system.
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected a bool, got {value!r}")
        return value

    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an int, got {value!r}")
        return int(value)

    if hint is float:
        # int -> float is the one widening we accept: YAML writes `1` for a
        # field that means 1.0 constantly, and the meaning is unambiguous.
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{where}: expected a float, got {value!r}")
        return float(value)

    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a str, got {value!r}")
        return value

    if isinstance(hint, type) and isinstance(value, hint):
        return value
    raise ConfigError(f"{where}: cannot coerce {value!r} to {hint}")


def _as_sequence(value: Any, where: str) -> Sequence[Any]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ConfigError(f"{where}: expected a list, got {value!r}")
    return value


# Hoisted out of the signature: a call in a default argument is evaluated once at
# import time anyway, so naming it makes that explicit rather than surprising.
_DEFAULT_CONFIG_CLS: type = cast(type, RunConfig)


def load_config(path: str | Path, cls: type[T] = _DEFAULT_CONFIG_CLS) -> T:
    """Read a YAML file into a config dataclass.

    ``yaml.safe_load``, never ``yaml.load``: a config file should not be able to
    construct arbitrary Python objects, and the difference has bitten enough
    projects to be worth stating.
    """
    text = Path(path).read_text()
    data = yaml.safe_load(text)
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data)}")
    return from_mapping(cls, data)


def apply_overrides(data: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``a.b=value`` overrides to a nested mapping, returning a new one.

    Values are parsed with the YAML scalar resolver so ``--set seed=3`` gives an
    int and ``--set cache.enabled=false`` gives a bool. Overriding a key that
    does not exist is allowed here; :func:`from_mapping` is what rejects it,
    with a better message and a nearest-match suggestion.
    """
    out = {k: dict(v) if isinstance(v, Mapping) else v for k, v in data.items()}
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override {item!r} is not of the form key=value")
        dotted, _, raw = item.partition("=")
        parts = dotted.split(".")
        cursor: dict[str, Any] = out
        for part in parts[:-1]:
            nxt = cursor.get(part)
            cursor[part] = dict(nxt) if isinstance(nxt, Mapping) else {}
            cursor = cursor[part]
        cursor[parts[-1]] = yaml.safe_load(raw)
    return out


# ---------------------------------------------------------------------------
# Hashing and grids
# ---------------------------------------------------------------------------


def to_dict(config: Any) -> Any:
    """A config as plain JSON-safe data, for manifests and cache sidecars."""
    return to_canonical(config)


def config_hash(config: Any) -> str:
    """Stable hash of a config, identical across processes and machines."""
    return content_hash(config)


def expand_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of per-key value lists, in a deterministic order.

    Keys are sorted before the product is taken, so the order of points depends
    on the *content* of the grid and not on how the YAML happened to be written.
    That is what lets a resumed or re-sharded sweep line up with an earlier one.

    >>> expand_grid({"b": [1, 2], "a": ["x"]})
    [{'a': 'x', 'b': 1}, {'a': 'x', 'b': 2}]
    """
    keys = sorted(grid)
    for k in keys:
        if isinstance(grid[k], str | bytes) or not isinstance(grid[k], Sequence):
            raise ConfigError(f"grid axis {k!r} must be a list of values, got {grid[k]!r}")
    return [dict(zip(keys, combo)) for combo in product(*(grid[k] for k in keys))]
