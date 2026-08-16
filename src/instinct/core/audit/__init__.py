"""Shared plumbing for audited triggers.

**Everything in this package is evaluation infrastructure, not a contribution.**
That framing is load-bearing rather than modest. Several of the portfolio's
programs contain a gate — spend more compute or don't, accept a fast-weight
write or don't — and every one of them needs the same three things to be
evaluated honestly:

* :mod:`instinct.core.audit.counterfactual` — fork the state, take the declined
  action anyway under identical randomness, roll back. What *would* have
  happened.
* :mod:`instinct.core.audit.propensity` — score a trigger against a log collected
  under a different one, with the effective sample size printed next to every
  estimate.
* :mod:`instinct.core.audit.blindspot` — find the regions the trigger keeps declining
  where no audit ever looked, and which therefore cannot have falsified it.

None of these is new (Horvitz-Thompson weighting is from 1952; the
selective-labels framing is from 2017). If a paper from this repo describes a
trigger as audited, the audit is only as good as the coverage this package
measures, and the claim to make about that coverage is a measurement, never a
methodological claim.

The `README <../../../README.md>`_ rule these serve is rule 5's cousin: a gate
evaluated only on the decisions it chose to make is evaluated on a sample it
selected. Reporting that is a result; not noticing it is a retraction.
"""

from __future__ import annotations

from instinct.core.audit.blindspot import (
    BlindspotReport,
    CellCoverage,
    detect_blindspots,
    nearest_audit_distance,
    quantile_cells,
)
from instinct.core.audit.counterfactual import (
    CounterfactualError,
    assert_unchanged,
    branch_actions,
    fork,
    fork_scope,
    rollback_on_exit,
    states_equal,
)
from instinct.core.audit.propensity import (
    OffPolicyEstimate,
    PropensityError,
    doubly_robust,
    effective_sample_size,
    importance_weights,
    ips,
    self_normalized_ips,
)

__all__ = [
    "BlindspotReport",
    "CellCoverage",
    "CounterfactualError",
    "OffPolicyEstimate",
    "PropensityError",
    "assert_unchanged",
    "branch_actions",
    "detect_blindspots",
    "doubly_robust",
    "effective_sample_size",
    "fork",
    "fork_scope",
    "importance_weights",
    "ips",
    "nearest_audit_distance",
    "quantile_cells",
    "rollback_on_exit",
    "self_normalized_ips",
    "states_equal",
]
