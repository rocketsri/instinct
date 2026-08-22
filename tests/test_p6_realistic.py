from __future__ import annotations

from dataclasses import asdict

import numpy as np

from instinct.core.configschema import load_proposal_config, proposal_config_from_mapping
from instinct.core.envs.control import InertialIntervention
from instinct.core.rng import SeedScope
from instinct.p6_certification.realistic import (
    build_exact_mdp,
    default_grid,
    default_shifts,
    exact_prefix_risks_from_distribution,
    reset_distribution,
    rollout_grid_states,
    run_realistic_benchmark,
    validate_p6_5,
)

CONFIG_PATH = "configs/p6/p6_5/cpu.yaml"


def _small_grid_env() -> tuple[InertialIntervention, "default_grid"]:  # type: ignore[valid-type]
    env = InertialIntervention()
    # Validated resolution (see realistic.py's module docstring): velocity bins
    # must be fine enough relative to the per-tick noise std (~0.0139) that the
    # discretized chain isn't near-deterministic. 21x71 keeps a fast unit test
    # while staying well inside the regime where closed-form matches Monte Carlo.
    grid = default_grid(env, position_bins=21, velocity_bins=71, velocity_range=(-1.5, 1.5))
    return env, grid


def test_exact_mdp_validates_cleanly() -> None:
    env, grid = _small_grid_env()
    mdp = build_exact_mdp(env, grid, gamma=0.97)
    mdp.validate()
    assert mdp.P.shape == (grid.n_states, env.n_actions, grid.n_states)
    row_sums = mdp.P.sum(axis=2)
    assert np.allclose(row_sums, 1.0, atol=1e-9)
    assert np.all(mdp.P[grid.terminal_state, :, grid.terminal_state] == 1.0)
    assert np.all(mdp.R[grid.terminal_state] == 0.0)


def test_reset_distribution_sums_to_one_and_is_nonnegative() -> None:
    _, grid = _small_grid_env()
    dist = reset_distribution(grid)
    assert dist.shape == (grid.n_states,)
    assert np.all(dist >= 0.0)
    assert np.isclose(dist.sum(), 1.0)
    # The reset rectangle lies entirely inside the legal region, so no reset
    # mass should land in the merged terminal state.
    assert dist[grid.terminal_state] == 0.0


def test_exact_prefix_risks_from_distribution_are_nondecreasing() -> None:
    env, grid = _small_grid_env()
    mdp = build_exact_mdp(env, grid, gamma=0.97)
    matrices = np.transpose(mdp.P, (1, 0, 2))
    dist = reset_distribution(grid)
    schedule = (2, 2, 1) * 4
    risks = exact_prefix_risks_from_distribution(matrices, schedule, dist)
    assert risks.shape == (len(schedule),)
    assert np.all(risks >= 0.0) and np.all(risks <= 1.0)
    assert np.all(np.diff(risks) >= -1e-12)


def test_discretized_exact_prediction_matches_monte_carlo_rollout() -> None:
    """The memo's own validation standard: compare the closed-form oracle
    against a high-sample-count rollout of the real continuous environment."""
    env, grid = _small_grid_env()
    mdp = build_exact_mdp(env, grid, gamma=0.97)
    matrices = np.transpose(mdp.P, (1, 0, 2))
    dist = reset_distribution(grid)
    schedule = (2, 2, 1) * 4
    exact_risks = exact_prefix_risks_from_distribution(matrices, schedule, dist)

    n = 20_000
    states = rollout_grid_states(env, grid, schedule, n, seed=4021)
    empirical_failed = states[:, 1:] == grid.terminal_state
    empirical_risks = empirical_failed.mean(axis=0)

    # Discretization coarsening is an acknowledged, bounded source of error
    # (see the module docstring and the preregistration's "what's lost"
    # framing); this tolerance is generous relative to Monte Carlo noise at
    # n=20,000 (~0.5% at p=0.5) plus that coarsening.
    assert np.max(np.abs(exact_risks - empirical_risks)) < 0.05


def test_rollout_grid_states_group_size_shares_noise() -> None:
    env, grid = _small_grid_env()
    schedule = (2, 1, 0, 2)
    grouped = rollout_grid_states(env, grid, schedule, 16, seed=7, group_size=4)
    # Every block of 4 trajectories shares one lane id, so their discretized
    # trajectories must be identical.
    for block in range(4):
        rows = grouped[block * 4 : (block + 1) * 4]
        assert np.all(rows == rows[0])


def test_rollout_grid_states_is_reproducible_and_uses_real_noise() -> None:
    env, grid = _small_grid_env()
    schedule = (2, 2, 1)
    a = rollout_grid_states(env, grid, schedule, 32, seed=99)
    b = rollout_grid_states(env, grid, schedule, 32, seed=99)
    np.testing.assert_array_equal(a, b)
    c = rollout_grid_states(env, grid, schedule, 32, seed=100)
    assert not np.array_equal(a, c)


def test_default_shifts_are_valid_environments_and_include_clustered_condition() -> None:
    env = InertialIntervention()
    shifts = default_shifts(env, clustered_unit_group_size=8)
    names = {shift.name for shift in shifts}
    assert names == {"stable", "increased_noise", "reduced_drag", "narrower_margin", "clustered_units"}
    clustered = next(s for s in shifts if s.name == "clustered_units")
    assert clustered.group_size == 8
    for shift in shifts:
        shift.env.__post_init__()  # re-validates dataclass invariants


def test_run_realistic_benchmark_small_smoke() -> None:
    env, grid = _small_grid_env()
    shifts = default_shifts(env, clustered_unit_group_size=2)[:2]  # stable + increased_noise only
    schedule = (2, 2, 1) * 4  # long enough that the pilot reliably reaches terminal
    results, learned, profile, metadata = run_realistic_benchmark(
        env=env,
        grid=grid,
        gamma=0.97,
        action_schedule=schedule,
        shifts=shifts,
        pilot_seed=11,
        eval_seed=12,
        pilot_trajectories=500,
        calibration_samples=256,
        repetitions=3,
        sample_sizes=(64, 128),
        episode_risk=0.2,
        confidence_delta=0.05,
        csa_bet_margin=0.02,
    )
    assert profile.pilot_trajectories == 500
    assert learned.horizon == len(schedule)
    assert {r.samples for r in results} == {64, 128}
    assert {r.condition for r in results} == {"stable", "increased_noise"}
    assert all(r.mean_end_to_end_latency_us >= r.mean_certificate_latency_us for r in results)
    assert all(0.0 <= r.nonvacuity_rate <= 1.0 for r in results)
    assert metadata["declared_looks"] == 2
    assert "calibration_scope" in metadata


def test_validate_p6_5_passes_on_locked_config() -> None:
    config = load_proposal_config(CONFIG_PATH)
    checks = validate_p6_5(config)
    failed = [check for check in checks if not check.passed]
    assert not failed, failed


def test_validate_p6_5_flags_phantom_declared_looks() -> None:
    config = load_proposal_config(CONFIG_PATH)
    bad_params = dict(config.params)
    bad_params["declared_looks"] = 8192  # far more than len(sample_sizes)
    mapping = {
        "experiment": config.experiment,
        "seed": config.seed,
        "seeds": list(config.seeds),
        "pilot_seeds": list(config.pilot_seeds),
        "eval_seeds": list(config.eval_seeds),
        "preregistration": asdict(config.preregistration),
        "theory": asdict(config.theory),
        "compute": asdict(config.compute),
        "resume": asdict(config.resume),
        "params": bad_params,
    }
    bad_config = proposal_config_from_mapping(mapping)
    checks = validate_p6_5(bad_config)
    named = {check.name: check for check in checks}
    assert not named["declared looks equal actual checked exits (v4 audit B1)"].passed
