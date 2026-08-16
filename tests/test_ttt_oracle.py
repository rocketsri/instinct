"""The exact-fork oracle: does it find harmful writes that were planted?

Phase zero of P2 rests entirely on one claim — that forking a fast-weight state,
applying a candidate update, and scoring it against future chunks and aged probes
identifies writes that damage retention. Everything downstream (the frontier, the
predictability analysis, the 2A-versus-2B decision) inherits that claim.

So the load-bearing test is not that the oracle runs. It is
:func:`test_oracle_recovers_planted_harmful_writes`: on a stream where harmful
writes are *constructed*, and therefore known, the oracle must rank them at the
bottom. If it cannot, the oracle is broken and no conclusion drawn from it means
anything — which is exactly the failure mode that would otherwise be reported as
"harmful writes are not identifiable".

Second in importance is :func:`test_batched_forking_matches_the_naive_reference`,
the rule-7 guard. The batched path packs many candidate forks into one tensor and
evaluates them in a single pass; the reference clones, applies and rolls back one
at a time. They must produce identical numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from instinct.core.rng import SeedScope  # noqa: E402
from instinct.ttt.chunker import build_ledger  # noqa: E402
from instinct.ttt.fastweight import (  # noqa: E402
    FastWeightConfig,
    FastWeights,
    chunk_delta,
    row_losses,
)
from instinct.ttt.oracle import OracleConfig, run_oracle  # noqa: E402
from instinct.ttt.probes import ProbeExhaustedError  # noqa: E402
from instinct.ttt.streams import (  # noqa: E402
    associative_recall_stream,
    drifting_regression_stream,
)

pytestmark = pytest.mark.needs_torch


def _small_stream(**kw):
    # conflict_every=13 keeps the conflicting task sparse. That matters: see
    # test_recall_degrades_as_the_conflicting_task_gets_denser for why a dense
    # conflict makes the "harmful" label itself ambiguous.
    params = dict(
        d_in=5, d_out=3, n_chunks=40, chunk_size=6, drift=0.05, seed=0,
        conflict_every=13, conflict_offset=3,
    )
    params.update(kw)
    return drifting_regression_stream(**params)


def _fw_cfg(stream) -> FastWeightConfig:
    # lr=0.1 with the normalized update. lr=0.4 unnormalized diverges to NaN by
    # step 12 and the oracle happily scores the wreckage.
    return FastWeightConfig(d_in=stream.d_in, d_hidden=16, d_out=stream.d_out, lr=0.1)


def _cfg(**kw) -> OracleConfig:
    params = dict(horizons=(1, 2, 4), min_age=4, n_aged_probes=4, min_forks=8, max_forks=32)
    params.update(kw)
    return OracleConfig(**params)


# -- the known-answer check ----------------------------------------------


def test_oracle_recovers_planted_harmful_writes() -> None:
    """Constructed harmful writes must land at the bottom of the ranking.

    The conflict chunks come from ``-A_t``: absorbing one actively undoes what
    the memory has learned, so its retention damage is real rather than nominal.
    Scored by AUC, since what the downstream frontier consumes is the *ordering*
    of candidates, not a threshold.
    """
    stream = _small_stream()
    ledger = build_ledger(stream, rows_per_probe=2, aged_budget=3)
    result = run_oracle(
        stream, _fw_cfg(stream), _cfg(), SeedScope(11).child("oracle"), ledger=ledger
    )

    truth = result.harmful_truth
    assert truth.any(), "the stream must actually contain planted harmful writes"
    assert not truth.all(), "and must also contain benign ones, or AUC is undefined"

    # AUC of (-utility) against the planted labels: harmful writes should have
    # the lowest utility, so ranking by -utility should surface them first.
    order = np.argsort(-result.utility, kind="stable")
    ranked = truth[order]
    n_pos, n_neg = int(truth.sum()), int((~truth).sum())
    # Mann-Whitney U, computed from the rank positions of the positives.
    pos_ranks = np.flatnonzero(ranked) + 1
    auc = (pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    assert auc > 0.85, (
        f"oracle AUC {auc:.3f} on planted harmful writes. Below chance-plus-margin "
        "means the oracle is not measuring retention damage."
    )
    assert result.utility[truth].mean() < result.utility[~truth].mean(), (
        "planted harmful writes must score worse on average than benign ones"
    )


def test_oracle_finds_collisions_in_associative_recall() -> None:
    """The same claim on a second, structurally different stream.

    A colliding key overwrites an earlier association, so the damage is a
    discrete overwrite rather than a gradual rotation. One stream agreeing could
    be a coincidence of that stream's construction.
    """
    stream = associative_recall_stream(n_chunks=40, collide_every=13, seed=3)
    if not stream.harmful_mask.any():
        pytest.skip("this stream configuration planted no collisions")
    ledger = build_ledger(stream, rows_per_probe=2, aged_budget=99)
    result = run_oracle(
        stream, _fw_cfg(stream), _cfg(n_aged_probes=8), SeedScope(12).child("oracle"),
        ledger=ledger,
    )
    truth = result.harmful_truth
    if truth.all() or not truth.any():
        pytest.skip("no usable positive/negative split after the min_age filter")
    assert result.utility[truth].mean() < result.utility[~truth].mean()


def test_a_stream_with_no_planted_harm_shows_no_separation() -> None:
    """The negative control, and the reason the positive result means anything.

    With ``conflict_every`` disabled there is nothing harmful to find, so any
    apparent separation would be the oracle responding to something other than
    retention damage.
    """
    # Both knobs pushed past the stream length: the planting rule fires at
    # t % conflict_every == conflict_offset, so a large period alone still
    # plants one at t == offset.
    stream = _small_stream(conflict_every=10_000, conflict_offset=10_000)
    assert not stream.harmful_mask.any()
    ledger = build_ledger(stream, rows_per_probe=2, aged_budget=3)
    result = run_oracle(
        stream, _fw_cfg(stream), _cfg(), SeedScope(13).child("oracle"), ledger=ledger
    )
    assert np.isfinite(result.utility).all()
    assert not result.harmful_truth.any()


# -- rule 7: the fast path equals the reference ---------------------------


def test_batched_forking_matches_the_naive_reference() -> None:
    """Many forks in one tensor must equal one-at-a-time clone/apply/rollback.

    float64 throughout, so a disagreement here is a bug and not rounding.
    """
    stream = _small_stream(n_chunks=18)
    fw, cfg = _fw_cfg(stream), _cfg()

    batched = run_oracle(
        stream, fw, cfg, SeedScope(21).child("o"),
        ledger=build_ledger(stream, rows_per_probe=2, aged_budget=99),
    )
    naive = run_oracle(
        stream, fw, cfg, SeedScope(21).child("o"),
        ledger=build_ledger(stream, rows_per_probe=2, aged_budget=99),
        naive=True,
    )

    assert np.array_equal(batched.t, naive.t)
    for field_name in ("utility", "benefit", "damage", "cost"):
        assert np.allclose(
            getattr(batched, field_name), getattr(naive, field_name), atol=1e-9
        ), f"{field_name} differs between the batched and naive paths"


def test_batched_path_uses_far_fewer_forward_calls() -> None:
    """The measured 14x fork speedup only exists if batching actually batches."""
    stream = _small_stream(n_chunks=18)
    fw, cfg = _fw_cfg(stream), _cfg()
    batched = run_oracle(
        stream, fw, cfg, SeedScope(21).child("o"),
        ledger=build_ledger(stream, rows_per_probe=2, aged_budget=99),
    )
    naive = run_oracle(
        stream, fw, cfg, SeedScope(21).child("o"),
        ledger=build_ledger(stream, rows_per_probe=2, aged_budget=99),
        naive=True,
    )
    assert batched.n_forward_calls < naive.n_forward_calls
    assert batched.max_forks_used >= 8, (
        "8 forks cost the same as 1 on the target hardware, so never fork fewer"
    )


def test_config_refuses_to_fork_fewer_than_eight() -> None:
    """A measured hardware fact, enforced rather than left as a comment."""
    with pytest.raises(ValueError, match="8 forks cost the same"):
        OracleConfig(min_forks=4)


# -- exact rollback -------------------------------------------------------


def test_fork_and_rollback_leaves_the_state_bitwise_identical() -> None:
    """If a fork leaks into the reference state, every later U_t is wrong."""
    stream = _small_stream(n_chunks=6)
    state = FastWeights.init(_fw_cfg(stream), 1, SeedScope(31).child("fw"))
    before = state.snapshot()

    chunk = stream.chunks[0]
    delta = chunk_delta(state, chunk.keys, chunk.values)
    forked = state.clone()
    forked.apply_(delta)
    _ = row_losses(forked, chunk.keys, chunk.values)

    after = state.snapshot()
    for lhs, rhs in zip(
        (before.w1, before.w2, before.w3), (after.w1, after.w2, after.w3)
    ):
        assert torch.equal(lhs, rhs), "the reference state was mutated by a fork"


def test_applying_a_delta_actually_changes_the_state() -> None:
    """Guards the opposite failure: a no-op update would make every U_t zero."""
    stream = _small_stream(n_chunks=6)
    state = FastWeights.init(_fw_cfg(stream), 1, SeedScope(32).child("fw"))
    chunk = stream.chunks[0]
    before = float(row_losses(state, chunk.keys, chunk.values).mean())
    state.apply_(chunk_delta(state, chunk.keys, chunk.values))
    after = float(row_losses(state, chunk.keys, chunk.values).mean())
    assert after < before, "a write should reduce loss on the chunk it absorbed"


# -- the probe use ledger -------------------------------------------------


def test_probe_ledger_raises_once_a_probe_is_exhausted() -> None:
    """Finite-use holdouts, enforced.

    Without this the reusable-holdout problem simply migrates into the retention
    term: probes get queried repeatedly, the oracle starts fitting them, and the
    retention estimate stops being an out-of-sample quantity.
    """
    stream = _small_stream(n_chunks=8)
    ledger = build_ledger(stream, rows_per_probe=2, aged_budget=2, with_audit=False)
    probe_id = next(iter(ledger._entries))

    ledger.charge(probe_id)
    ledger.charge(probe_id)
    assert ledger.is_retired(probe_id)
    with pytest.raises(ProbeExhaustedError):
        ledger.charge(probe_id)


def test_retired_probes_are_not_offered_again() -> None:
    stream = _small_stream(n_chunks=8)
    ledger = build_ledger(stream, rows_per_probe=2, aged_budget=1, with_audit=False)
    first = ledger.available()
    assert first
    ledger.charge(first[0])
    assert first[0] not in ledger.available()


def test_oracle_stays_within_its_probe_budget() -> None:
    """The oracle must not silently exceed the ledger it was handed."""
    stream = _small_stream(n_chunks=14)
    ledger = build_ledger(stream, rows_per_probe=2, aged_budget=2)
    result = run_oracle(
        stream, _fw_cfg(stream), _cfg(), SeedScope(41).child("o"), ledger=ledger
    )
    report = ledger.usage_report()
    assert result.probe_rows_spent > 0
    # Every charge is checked against the budget at the point of use, so the
    # ledger cannot record more charges than it registered budget for.
    assert report.charges <= report.registered * 2
    assert report.remaining_budget >= 0


# -- shape and provenance -------------------------------------------------


def test_utility_decomposes_into_its_reported_parts() -> None:
    """benefit - lambda*damage - cost must reconstruct the utility."""
    stream = _small_stream(n_chunks=16)
    cfg = _cfg(retention_lambda=0.7, cost=0.01)
    result = run_oracle(
        stream, _fw_cfg(stream), cfg, SeedScope(51).child("o"),
        ledger=build_ledger(stream, rows_per_probe=2, aged_budget=3),
    )
    expected = result.benefit - cfg.retention_lambda * result.damage - result.cost
    assert np.allclose(result.utility, expected, atol=1e-12)


def test_features_align_with_candidates() -> None:
    """Features feed the predictability analysis; a misalignment would fake a signal."""
    stream = _small_stream(n_chunks=16)
    result = run_oracle(
        stream, _fw_cfg(stream), _cfg(), SeedScope(52).child("o"),
        ledger=build_ledger(stream, rows_per_probe=2, aged_budget=3),
    )
    n = len(result)
    assert n > 0
    for name, values in result.features.items():
        assert values.shape == (n,), f"feature {name!r} has shape {values.shape}, expected {(n,)}"
    assert result.harmful_truth.shape == (n,)
    assert result.deltas_flat.shape[0] == n


def test_oracle_is_reproducible() -> None:
    stream = _small_stream(n_chunks=14)
    runs = [
        run_oracle(
            stream, _fw_cfg(stream), _cfg(), SeedScope(61).child("o"),
            ledger=build_ledger(stream, rows_per_probe=2, aged_budget=3),
        ).utility
        for _ in range(2)
    ]
    assert np.array_equal(runs[0], runs[1])


# -- what the investigation turned up -------------------------------------


def test_an_unnormalized_update_diverges_and_is_caught() -> None:
    """The bug that made every earlier oracle number meaningless.

    With ``normalize_update=False`` at ``lr=0.4`` the write_all reference
    trajectory blows up: primary-task loss reached 6.5e2 by step 8, 1e300 by
    step 11 and NaN by step 12. Nothing raised. The oracle scored the wreckage
    and returned a full table of finite-looking utilities whose AUC against the
    planted labels was 0.059 — strongly *anti*-correlated — which reads exactly
    like the "harmful writes are not identifiable" kill condition rather than
    like arithmetic that had already died.

    So divergence is now a loud failure, and the guard is tested rather than
    trusted.
    """
    stream = _small_stream(n_chunks=40)
    unstable = FastWeightConfig(
        d_in=stream.d_in, d_hidden=16, d_out=stream.d_out, lr=0.4, normalize_update=False
    )
    state = FastWeights.init(unstable, 1, SeedScope(71).child("fw"))
    with pytest.raises(FloatingPointError, match="diverged"):
        for chunk in stream.chunks:
            state.apply_(chunk_delta(state, chunk.keys, chunk.values))


def test_the_default_config_survives_a_long_stream() -> None:
    """The normalized default must not need per-stream tuning to stay finite."""
    stream = _small_stream(n_chunks=60)
    state = FastWeights.init(_fw_cfg(stream), 1, SeedScope(72).child("fw"))
    for chunk in stream.chunks:
        state.apply_(chunk_delta(state, chunk.keys, chunk.values))
    snap = state.snapshot()
    assert all(bool(torch.isfinite(w).all()) for w in (snap.w1, snap.w2, snap.w3))


def _auc(utility: np.ndarray, truth: np.ndarray) -> float:
    order = np.argsort(-utility, kind="stable")
    ranked = truth[order]
    n_pos, n_neg = int(truth.sum()), int((~truth).sum())
    if not n_pos or not n_neg:
        return float("nan")
    pos_ranks = np.flatnonzero(ranked) + 1
    return float((pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def test_recall_degrades_as_the_conflicting_task_gets_denser() -> None:
    """A real limitation of the setup, pinned down rather than tuned around.

    Measured AUC against conflict period, at the stable config:

        period 20 (5% conflict)  -> 1.000
        period 13 (7%)           -> 0.955
        period  9 (12%)          -> 0.729
        period  6 (17%)          -> 0.727

    The mechanism is contamination of the retention term. Aged probes are drawn
    from earlier chunks, and when the conflicting task is common some of those
    probes came from *other conflict chunks*. A new conflict write helps on
    those, so its measured damage falls. In the limit the label stops being
    well-defined: "harmful" only means anything relative to a task that
    dominates the stream.

    This is a caveat P2 has to carry into the real-data arm, where the mixture
    is not controllable, so it is asserted rather than left as a footnote.
    """
    scores = {}
    for period in (20, 6):
        stream = _small_stream(n_chunks=40, conflict_every=period)
        if int(stream.harmful_mask.sum()) < 2:
            pytest.skip(f"period {period} planted too few conflicts")
        result = run_oracle(
            stream, _fw_cfg(stream), _cfg(n_aged_probes=8), SeedScope(11).child("oracle"),
            ledger=build_ledger(stream, rows_per_probe=2, aged_budget=99),
        )
        scores[period] = _auc(result.utility, result.harmful_truth)

    assert scores[20] > scores[6], (
        f"sparse conflicts should be easier to detect: {scores}"
    )
    assert scores[20] > 0.9, f"a sparse conflicting task must be clearly detectable: {scores}"


def test_more_retention_probes_help_rather_than_hurt() -> None:
    """The direction that was inverted while the memory was diverging.

    Under the broken config AUC fell as probes were added (0.889 -> 0.796 ->
    0.704), which is backwards and was the first clue that the numbers were not
    measuring what they claimed. A better retention estimate must not make the
    ranking worse.
    """
    stream = _small_stream(n_chunks=40)
    scores = []
    for n_probes in (4, 12):
        result = run_oracle(
            stream, _fw_cfg(stream), _cfg(n_aged_probes=n_probes),
            SeedScope(11).child("oracle"),
            ledger=build_ledger(stream, rows_per_probe=2, aged_budget=99),
        )
        scores.append(_auc(result.utility, result.harmful_truth))
    assert scores[1] >= scores[0] - 0.05, f"more probes should not degrade recall: {scores}"
