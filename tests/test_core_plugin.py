"""ProposalPlugin/load_plugin: the seam Stage B builds against.

Exercised against ``tests/fixtures/dummy_proposal/run.py`` rather than any
real proposal — none of ``p1_atlas`` through ``p7_chunking`` exist yet at
Stage A, and ``load_plugin`` correctly raising ``ModuleNotFoundError`` for
each of them is itself covered below, not worked around.
"""

from __future__ import annotations

import pytest

import instinct.core.plugin as plugin_mod
from instinct.core.plugin import PROPOSAL_MODULES, ProposalPlugin, load_plugin
from tests.fixtures.dummy_proposal import run as dummy_run


def test_all_seven_proposals_are_registered() -> None:
    assert PROPOSAL_MODULES == (
        "p1_atlas",
        "p2_harmful_write",
        "p3_codesign",
        "p4_slack",
        "p5_development",
        "p6_certification",
        "p7_chunking",
    )


def test_unknown_proposal_id_is_rejected_before_any_import() -> None:
    with pytest.raises(ValueError, match="unknown proposal"):
        load_plugin("p8_does_not_exist")


@pytest.mark.parametrize("proposal_id", PROPOSAL_MODULES)
def test_unbuilt_proposal_fails_loudly_not_silently(proposal_id: str) -> None:
    """None of the seven exist yet at Stage A. That must be a loud import
    error, not a silent no-op plugin — a config naming an unbuilt proposal
    should fail exactly like a typo would."""
    with pytest.raises(ModuleNotFoundError):
        load_plugin(proposal_id)


def test_dummy_fixture_satisfies_the_protocol_structurally() -> None:
    assert isinstance(dummy_run, ProposalPlugin)


def test_load_plugin_returns_the_resolved_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plugin_mod, "import_module", lambda name: dummy_run)
    resolved = load_plugin("p1_atlas")
    assert resolved is dummy_run


def test_load_plugin_rejects_a_module_missing_required_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Incomplete:
        def validate(self, config):
            return []

    monkeypatch.setattr(plugin_mod, "import_module", lambda name: _Incomplete())
    with pytest.raises(AttributeError, match=r"run.*resume"):
        load_plugin("p1_atlas")
