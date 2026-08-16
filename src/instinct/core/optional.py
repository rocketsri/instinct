"""The lazy import that keeps core CPU-only.

Core has to run on a bare 4-CPU container with no ``nn`` extra installed, and it
also has to report device/VRAM state when it happens to be running under torch
(Colab, or a local machine with the extra installed). Those are in tension only
if importing torch is unconditional; it is not, here.

:func:`try_import_torch` is the one place that import lives. Everything under
``core/`` that wants to know about CUDA — :mod:`instinct.core.device`,
:mod:`instinct.core.profile` — calls this instead of ``import torch`` at module
scope, so importing ``instinct.core`` on a CPU-only smoke box never pays for or
fails on a missing wheel.
"""

from __future__ import annotations

from types import ModuleType

__all__ = ["try_import_torch"]


def try_import_torch() -> ModuleType | None:
    """``torch`` if the ``nn`` extra is installed, else ``None``.

    Never raises. A missing torch is an expected, common state for the core test
    suite and for CPU-only proposals (P1, P3, P4, P6), not an error condition.
    """
    try:
        import torch
    except ImportError:
        return None
    return torch
