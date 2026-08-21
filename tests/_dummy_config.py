"""Shared helper for building a minimal, valid proposal config YAML in tests.

Not a test module itself (no ``test_`` prefix, so pytest does not collect it);
``tests/test_cli_smoke.py``, ``tests/test_core_resume_equivalence.py``, and
``tests/test_core_plugin.py`` all need the same "write a config the dummy
fixture proposal can run against" step, and this is the one place that shape
is defined.
"""

from __future__ import annotations

from pathlib import Path

import yaml

DUMMY_CONFIG: dict = {
    "experiment": "dummy",
    "seed": 7,
    "seeds": [7],
    "pilot_seeds": [],
    "eval_seeds": [],
    "preregistration": {
        "estimand": "the fixture's final counter value",
        "primary_hypothesis": "the fixture always reaches its configured total_steps",
        "non_claim": "nothing about any real proposal",
    },
    "theory": {
        "proposal_version": "fixture-v1",
        "theory_id": "tests/fixtures/dummy_proposal",
        "amendment_id": "none",
    },
    "compute": {"run_class": "smoke"},
    "resume": {"checkpoint_every_minutes": 15.0, "allow_resume": True},
    "params": {"total_steps": 5},
}


def write_dummy_config(path: Path, **param_overrides: object) -> Path:
    """Write a valid dummy-proposal config YAML to ``path`` and return it.

    ``param_overrides`` are merged into ``params`` (e.g. ``abort_after=2``).
    """
    data = dict(DUMMY_CONFIG)
    data["params"] = {**DUMMY_CONFIG["params"], **param_overrides}
    path.write_text(yaml.safe_dump(data, sort_keys=True))
    return path
