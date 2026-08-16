"""A minimal fake proposal package, for shared-substrate tests only.

Not wired into :data:`instinct.core.plugin.PROPOSAL_MODULES` — the shared
tests that need a plugin monkeypatch ``instinct.cli.load_plugin`` (for CLI
smoke tests) or ``instinct.core.plugin.import_module`` (for plugin-loader
unit tests) to hand back :mod:`tests.fixtures.dummy_proposal.run` instead of
resolving a real proposal. See ``tests/test_cli_smoke.py`` and
``tests/test_core_plugin.py``.
"""
