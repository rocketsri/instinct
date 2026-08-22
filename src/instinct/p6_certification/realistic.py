"""P6.5 realistic-simulator integration: a discretized, closed-form-exact
:class:`InertialIntervention`, certified with the unchanged P6 certificate core.

Per ``docs/reviews/p6/v5/continuation_memo.md``, this stage tests the revised
hypothesis that ``certificates.py``/``comparators.py``/``mismatch.py`` generalize
to a genuinely stochastic, non-toy environment without modification. Nothing in
this module imports or edits those three files' internals; it only produces the
same shapes (Bernoulli bad counts, paired binary discordance arrays) that
:mod:`instinct.p6_certification.learned_dynamics` already produces for the
three-state toy chain.

Two departures from the memo's literal numbers, both explicitly permitted by
the memo itself:

* **One merged terminal state**, not two. The memo says either is fine ("both
  are absorbing with identical zero-reward semantics"); one state means the
  fitted grid model can reuse :class:`~instinct.p6_certification.learned_dynamics.TransitionRiskModel`
  unchanged (its ``fit``/``sample_failures``/``exact_prefix_risks`` machinery
  hard-assumes a single last-index absorbing failure state).
* **Velocity-bin-center routing.** For a fixed pre-step cell and action, the
  next continuous state is Gaussian in velocity with position an exact affine
  function of velocity. Splitting a velocity bin's mass across the (rare) cases
  where its implied position range straddles two position bins would need a
  second, nested integral; instead each velocity bin's mass is routed to the
  single position bin implied by that bin's *center*. Since one tick's position
  displacement from noise (``dt * velocity_bin_width`` ~= 0.014) is much smaller
  than a position bin (0.05 wide), this under-counts straddling by a bounded,
  small amount -- reported as part of the discretization error in the
  preregistration, not asserted away.

**A third, load-bearing departure from the memo's literal numbers:** the memo
recommends 25 velocity bins (width 0.12) over ``[-1.5, 1.5]``. Validating that
against real rollouts (see ``tests/test_p6_realistic.py``) showed this
resolution is badly under-resolved relative to the true per-tick velocity
noise std (``sqrt(dt) * noise_std`` ~= 0.0139 at the nominal parameters, ~8.6x
smaller than a 0.12-wide bin): at 8.66 standard deviations, the probability of
landing in any bin but the one containing the mean is astronomically small, so
the discretized chain collapses to a near-deterministic point process instead
of a diffusion, and multi-step boundary-crossing risk was found to disagree
with Monte Carlo by up to 13 percentage points over a 12-tick horizon (0.4326
exact vs. 0.318 empirical at ``position_bins=41, velocity_bins=25``). Widening
to **61 velocity bins** (width ~= 0.049, ~3.5x the noise std) reduces that gap
to under 0.4 percentage points at the same horizon and 20,000 Monte Carlo
samples, while keeping the state count (41*61+1=2,502) cheap for
``TabularMDP.value_iteration()``. ``default_grid``'s ``velocity_bins`` default
is 61, not the memo's 25, for this reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from time import perf_counter_ns
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import beta, norm

from instinct.core.configschema import ProposalRunConfig
from instinct.core.envs.control import ACCELERATIONS, InertialIntervention
from instinct.core.mdp import TabularMDP
from instinct.core.proposal import common_checks
from instinct.core.results import ResultsWriter
from instinct.core.rng import SeedScope
from instinct.core.verdict import PrerequisiteCheck, Status, VerdictReport
from instinct.p6_certification.certificates import longest_certified
from instinct.p6_certification.learned_dynamics import TransitionRiskModel, certify_prefixes
from instinct.p6_certification.mismatch import calibrate_shift_envelope

__all__ = [
    "InertialGrid",
    "RealisticFitProfile",
    "RealisticResult",
    "RealisticShift",
    "build_exact_mdp",
    "default_grid",
    "default_shifts",
    "exact_prefix_risks_from_distribution",
    "reset_distribution",
    "rollout_grid_states",
    "run_p6_5",
    "run_realistic_benchmark",
    "validate_p6_5",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


# -- discretization ---------------------------------------------------------


@dataclass(frozen=True)
class InertialGrid:
    """A uniform grid over ``(position, velocity)`` plus one merged terminal state."""

    position_edges: FloatArray
    velocity_edges: FloatArray

    def __post_init__(self) -> None:
        if self.position_edges.ndim != 1 or self.position_edges.size < 2:
            raise ValueError("position_edges must be a 1-D array with at least 2 entries")
        if self.velocity_edges.ndim != 1 or self.velocity_edges.size < 2:
            raise ValueError("velocity_edges must be a 1-D array with at least 2 entries")
        if np.any(np.diff(self.position_edges) <= 0) or np.any(np.diff(self.velocity_edges) <= 0):
            raise ValueError("grid edges must be strictly increasing")

    @property
    def n_position_bins(self) -> int:
        return int(self.position_edges.size - 1)

    @property
    def n_velocity_bins(self) -> int:
        return int(self.velocity_edges.size - 1)

    @property
    def n_grid_states(self) -> int:
        return self.n_position_bins * self.n_velocity_bins

    @property
    def terminal_state(self) -> int:
        return self.n_grid_states

    @property
    def n_states(self) -> int:
        return self.n_grid_states + 1

    @property
    def boundary(self) -> float:
        return float(self.position_edges[-1])

    def index(self, position_bin: IntArray | int, velocity_bin: IntArray | int) -> Any:
        return np.asarray(position_bin) * self.n_velocity_bins + np.asarray(velocity_bin)

    def position_centers(self) -> FloatArray:
        return 0.5 * (self.position_edges[:-1] + self.position_edges[1:])

    def velocity_centers(self) -> FloatArray:
        return 0.5 * (self.velocity_edges[:-1] + self.velocity_edges[1:])

    def classify(
        self, position: FloatArray, velocity: FloatArray, *, failed: FloatArray | None = None
    ) -> IntArray:
        """Map continuous ``(position, velocity)`` pairs to grid state indices.

        ``failed`` is the environment's own failure flag, OR'd into the
        boundary check. It matters whenever the environment's actual boundary
        differs from this (shared, fixed) grid's edge -- e.g. a structural
        shift that narrows ``boundary`` below the grid's -- since the
        environment freezes a failed lane's position at whatever value
        triggered *its own* failure, which may sit inside this grid's legal
        range even though the trajectory has, in truth, already terminated.
        """
        position = np.atleast_1d(np.asarray(position, dtype=np.float64))
        velocity = np.atleast_1d(np.asarray(velocity, dtype=np.float64))
        terminal = np.abs(position) >= self.boundary
        if failed is not None:
            terminal = terminal | np.atleast_1d(np.asarray(failed, dtype=bool))
        pos_bin = np.clip(
            np.searchsorted(self.position_edges, position, side="right") - 1,
            0,
            self.n_position_bins - 1,
        )
        vel_bin = np.clip(
            np.searchsorted(self.velocity_edges, velocity, side="right") - 1,
            0,
            self.n_velocity_bins - 1,
        )
        grid_index = (pos_bin * self.n_velocity_bins + vel_bin).astype(np.int64)
        return np.where(terminal, self.terminal_state, grid_index).astype(np.int64)


def default_grid(
    env: InertialIntervention,
    *,
    position_bins: int = 41,
    velocity_bins: int = 61,
    velocity_range: tuple[float, float] = (-1.5, 1.5),
) -> InertialGrid:
    return InertialGrid(
        position_edges=np.linspace(-env.boundary, env.boundary, position_bins + 1),
        velocity_edges=np.linspace(velocity_range[0], velocity_range[1], velocity_bins + 1),
    )


def build_exact_mdp(env: InertialIntervention, grid: InertialGrid, *, gamma: float) -> TabularMDP:
    """Closed-form ``(S, 3, S)`` transition/reward tensor -- no rollout involved.

    For a fixed pre-step cell and action, ``proposed_velocity`` is Gaussian
    (``control.py``'s single additive noise term) and ``proposed_position`` is a
    deterministic affine function of it, so each destination velocity bin's
    probability mass is a Gaussian-CDF difference, routed to a single
    destination position bin by that bin's center (see module docstring).
    """
    if env.boundary > grid.boundary + 1e-12:
        raise ValueError(
            "grid position range must cover the environment's boundary "
            f"(grid={grid.boundary}, env={env.boundary})"
        )
    n_actions = env.n_actions
    n_states = grid.n_states
    P = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    R = np.zeros((n_states, n_actions), dtype=np.float64)

    position_centers = grid.position_centers()
    velocity_centers = grid.velocity_centers()
    std_v = float(np.sqrt(env.dt) * env.noise_std)
    terminal = grid.terminal_state

    for a in range(n_actions):
        command = env.acceleration * float(ACCELERATIONS[a])
        control_cost_term = env.control_cost * abs(float(ACCELERATIONS[a]))
        for pb, position in enumerate(position_centers):
            mean_v = env.drag * velocity_centers + env.dt * command  # (n_vel_bins,)
            for vb, velocity in enumerate(velocity_centers):
                del velocity
                s = grid.index(pb, vb)
                mean = mean_v[vb]
                if std_v > 0:
                    cdf = norm.cdf((grid.velocity_edges - mean) / std_v)
                    mass = np.diff(cdf)
                    mass[0] += cdf[0]
                    mass[-1] += 1.0 - cdf[-1]
                else:
                    mass = np.zeros(grid.n_velocity_bins)
                    dest_vb = int(
                        np.clip(
                            np.searchsorted(grid.velocity_edges, mean, side="right") - 1,
                            0,
                            grid.n_velocity_bins - 1,
                        )
                    )
                    mass[dest_vb] = 1.0
                nonzero = np.flatnonzero(mass > 1e-15)
                for dest_vb in nonzero:
                    weight = float(mass[dest_vb])
                    v_next = float(velocity_centers[dest_vb])
                    position_next = position + env.dt * v_next
                    if abs(position_next) >= env.boundary:
                        dest = terminal
                        clamped = env.boundary if position_next > 0 else -env.boundary
                        progress = abs(position - env.target) - abs(clamped - env.target)
                        reward = (
                            env.progress_scale * progress
                            - control_cost_term
                            - env.failure_penalty
                        )
                    else:
                        dest_pb = int(
                            np.clip(
                                np.searchsorted(grid.position_edges, position_next, side="right")
                                - 1,
                                0,
                                grid.n_position_bins - 1,
                            )
                        )
                        dest = int(grid.index(dest_pb, dest_vb))
                        progress = abs(position - env.target) - abs(position_next - env.target)
                        reward = env.progress_scale * progress - control_cost_term
                    P[s, a, dest] += weight
                    R[s, a] += weight * reward

    P[terminal, :, terminal] = 1.0
    R[terminal, :] = 0.0
    terminal_mask = np.zeros(n_states, dtype=bool)
    terminal_mask[terminal] = True
    mdp = TabularMDP(
        P=P, R=R, gamma=gamma, terminal=terminal_mask, name="inertial_intervention_grid",
        failure=terminal_mask.copy(),
    )
    mdp.validate()
    return mdp


def reset_distribution(
    grid: InertialGrid,
    *,
    position_range: tuple[float, float] = (-0.65, -0.35),
    velocity_range: tuple[float, float] = (0.05, 0.80),
) -> FloatArray:
    """Exact reset mass per grid cell, from the uniform reset rectangle's overlap."""
    dist = np.zeros(grid.n_states, dtype=np.float64)
    total_area = (position_range[1] - position_range[0]) * (velocity_range[1] - velocity_range[0])
    for pb in range(grid.n_position_bins):
        p_lo, p_hi = grid.position_edges[pb], grid.position_edges[pb + 1]
        overlap_p = min(p_hi, position_range[1]) - max(p_lo, position_range[0])
        if overlap_p <= 0:
            continue
        for vb in range(grid.n_velocity_bins):
            v_lo, v_hi = grid.velocity_edges[vb], grid.velocity_edges[vb + 1]
            overlap_v = min(v_hi, velocity_range[1]) - max(v_lo, velocity_range[0])
            if overlap_v <= 0:
                continue
            dist[int(grid.index(pb, vb))] = overlap_p * overlap_v / total_area
    total = dist.sum()
    if total <= 0:
        raise ValueError("reset rectangle does not overlap the grid")
    return dist / total


def exact_prefix_risks_from_distribution(
    matrices: FloatArray, action_schedule: Sequence[int], start_distribution: FloatArray
) -> FloatArray:
    """Cumulative failure mass after each scheduled action, from a start distribution.

    Generalizes :meth:`TransitionRiskModel.exact_prefix_risks` (which starts
    from a one-hot at state 0) to an arbitrary start distribution -- needed
    because the realistic environment resets into a spread of grid cells, not
    one fixed state.
    """
    distribution = np.asarray(start_distribution, dtype=np.float64).copy()
    risks: list[float] = []
    terminal_index = matrices.shape[1] - 1
    for action in action_schedule:
        distribution = distribution @ matrices[action]
        risks.append(float(distribution[terminal_index]))
    return np.asarray(risks, dtype=np.float64)


def rollout_grid_states(
    env: InertialIntervention,
    grid: InertialGrid,
    action_schedule: Sequence[int],
    n_trajectories: int,
    seed: int,
    *,
    group_size: int = 1,
) -> IntArray:
    """Real continuous rollouts, discretized post-hoc into grid state indices.

    ``group_size`` > 1 assigns several trajectory indices the same underlying
    lane id, so they draw *identical* noise (the CRN keys off lane id only) --
    an explicit exchangeability stress cell (fewer independent noise roots than
    trajectories), not a modeling choice for the nominal/shift conditions.
    """
    if n_trajectories < 1:
        raise ValueError("n_trajectories must be positive")
    if group_size < 1:
        raise ValueError("group_size must be positive")
    lane_ids = (np.arange(n_trajectories, dtype=np.int64) // group_size).astype(np.int64)
    scope = SeedScope(seed).child("p6-5-realistic-rollout")
    state = env.reset(lane_ids, scope=scope, episode=0)
    horizon = len(action_schedule)
    grid_states = np.zeros((n_trajectories, horizon + 1), dtype=np.int64)
    grid_states[:, 0] = grid.classify(state["position"], state["velocity"], failed=state["failed"])
    for t, action in enumerate(action_schedule):
        actions = np.full(n_trajectories, int(action), dtype=np.int64)
        result = env.step(state, actions, scope=scope, episode=0, tick=t)
        state = result.state
        grid_states[:, t + 1] = grid.classify(
            state["position"], state["velocity"], failed=state["failed"]
        )
    return grid_states


# -- structural shifts --------------------------------------------------------


@dataclass(frozen=True)
class RealisticShift:
    name: str
    env: InertialIntervention
    group_size: int = 1


def default_shifts(nominal: InertialIntervention, *, clustered_unit_group_size: int = 8) -> tuple[RealisticShift, ...]:
    """Four physically-motivated perturbations of the nominal environment.

    ``clustered_units`` reuses the nominal dynamics but stresses the
    independent-statistical-unit assumption (v4 audit finding B4) rather than
    perturbing the physics.
    """
    return (
        RealisticShift("stable", nominal),
        RealisticShift(
            "increased_noise", replace(nominal, noise_std=nominal.noise_std * 1.5)
        ),
        RealisticShift(
            "reduced_drag", replace(nominal, drag=1.0 - (1.0 - nominal.drag) * 0.5)
        ),
        RealisticShift("narrower_margin", replace(nominal, boundary=nominal.boundary * 0.85)),
        RealisticShift("clustered_units", nominal, group_size=clustered_unit_group_size),
    )


# -- benchmark ----------------------------------------------------------------


@dataclass(frozen=True)
class RealisticFitProfile:
    fit_latency_ms: float
    pilot_trajectories: int
    grid_discretization_l1_error: float


@dataclass(frozen=True)
class RealisticResult:
    condition: str
    samples: int
    method: str
    validity_scope: str
    model_false_certification_rate: float
    true_false_certification_rate: float
    mean_selected_model_risk: float
    mean_selected_true_risk: float
    mean_selected_prefix: float
    nonvacuity_rate: float
    nonabstention_rate: float
    oracle_safe_fraction_recovered: float
    abstention_rate: float
    mean_certificate_latency_us: float
    mean_shift_calibration_latency_us: float
    mean_end_to_end_latency_us: float
    unique_noise_roots: int
    effective_sample_fraction: float
    transitions_evaluated: int


CertificateMethod = str
METHOD_VALIDITY: dict[str, str] = {
    "model_only_anytime_cp": "model-relative anytime process; no structural-shift guarantee",
    "shift_robust_union": "finite exit/prefix union plus independent-pair paired shift envelope",
    "shift_robust_avcrc_specialization": (
        "AVCRC bounded-loss specialization plus independent-pair paired shift envelope"
    ),
    "shift_robust_csa_specialization": (
        "CSA finite-grid e-process specialization plus independent-pair paired shift envelope"
    ),
    "always_abstain": "deterministic abstention",
    "always_execute": "uncertified execution control",
}


def _oracle_safe_prefix(true_risks: FloatArray, episode_risk: float) -> int:
    bounds = {i + 1: float(risk) for i, risk in enumerate(true_risks)}
    return longest_certified(bounds, episode_risk)


def _compact_support(matrices: FloatArray, max_support: int) -> tuple[IntArray, FloatArray]:
    """Pack a dense ``(A, S, S)`` transition tensor into ``(A, S, K)`` sparse form.

    Each grid ``(state, action)`` row's *physically real* mass sits on at most
    ``n_velocity_bins`` destinations (every destination is reached through
    exactly one destination velocity bin -- see ``build_exact_mdp``). But
    :meth:`TransitionRiskModel.fit`'s additive smoothing adds a small floor
    probability to *every* entry, including states never visited by the pilot
    -- so a fitted row is formally dense even though almost all of that mass
    is smoothing noise. This packs the top ``max_support`` entries by
    probability per row and renormalizes onto them, which is exact for a
    row whose real support is <= ``max_support`` (the renormalization factor
    is then ~1) and a small, controlled truncation of negligible smoothing
    floor otherwise.

    This exists because the dense gather :meth:`TransitionRiskModel.
    sample_states_with_uniforms` does per step -- correct and cheap for
    P6.0-P6.4's 3-4-state chain -- is an ``O(n_trajectories * S)`` memory
    traffic blowup once ``S`` is in the thousands. Packing once (outside the
    repetition loop) makes per-repetition sampling ``O(n_trajectories * K)``.
    """
    n_actions, n_states, _ = matrices.shape
    k = min(max_support, n_states)
    idx = np.zeros((n_actions, n_states, k), dtype=np.int64)
    cum = np.ones((n_actions, n_states, k), dtype=np.float64)
    for a in range(n_actions):
        for s in range(n_states):
            row = matrices[a, s]
            top = np.argpartition(row, -k)[-k:] if k < n_states else np.arange(n_states)
            top = top[np.argsort(row[top])]
            probs = row[top]
            total = float(probs.sum())
            if total <= 0.0:
                idx[a, s, :] = s
                continue
            probs = probs / total
            c = np.cumsum(probs)
            c[-1] = 1.0  # guard fp round-off so the inverse-CDF always resolves
            idx[a, s, :] = top
            cum[a, s, :] = c
    return idx, cum


def _sample_states_compact(
    support_idx: IntArray,
    support_cum: FloatArray,
    action_schedule: Sequence[int],
    uniforms: FloatArray,
) -> IntArray:
    """Same contract as :meth:`TransitionRiskModel.sample_states_with_uniforms`,
    against the packed representation from :func:`_compact_support`."""
    n_traj = uniforms.shape[0]
    horizon = len(action_schedule)
    states = np.zeros((n_traj, horizon + 1), dtype=np.int64)
    rows = np.arange(n_traj)
    for step, action in enumerate(action_schedule):
        idx_row = support_idx[action][states[:, step]]
        cum_row = support_cum[action][states[:, step]]
        choice = np.sum(uniforms[:, step, None] > cum_row, axis=1)
        choice = np.clip(choice, 0, idx_row.shape[1] - 1)
        states[:, step + 1] = idx_row[rows, choice]
    return states


def run_realistic_benchmark(
    *,
    env: InertialIntervention,
    grid: InertialGrid,
    gamma: float,
    action_schedule: Sequence[int],
    shifts: Sequence[RealisticShift],
    pilot_seed: int,
    eval_seed: int,
    pilot_trajectories: int,
    calibration_samples: int,
    repetitions: int,
    sample_sizes: Sequence[int],
    episode_risk: float,
    confidence_delta: float,
    csa_bet_margin: float,
) -> tuple[list[RealisticResult], TransitionRiskModel, RealisticFitProfile, dict[str, Any]]:
    sizes = tuple(int(value) for value in sample_sizes)
    if not sizes or sizes[0] < 1 or any(b <= a for a, b in pairwise(sizes)):
        raise ValueError("sample sizes must be positive and strictly increasing")
    schedule = tuple(int(value) for value in action_schedule)
    declared_looks = len(sizes)  # real stopping loop: no unused/phantom exits (v4 audit B1)

    exact_mdp = build_exact_mdp(env, grid, gamma=gamma)
    reset_dist = reset_distribution(grid)
    V, _, _ = exact_mdp.value_iteration()
    optimal_value_at_reset = float(V @ reset_dist)

    fit_start = perf_counter_ns()
    pilot_states = rollout_grid_states(env, grid, schedule, pilot_trajectories, pilot_seed)
    if int(pilot_states.max()) != grid.terminal_state:
        raise ValueError(
            "no pilot trajectory reached the terminal state; the action schedule or pilot "
            "count must change so TransitionRiskModel.fit infers the full grid state space"
        )
    # TransitionRiskModel.fit's default smoothing=0.5 was calibrated for
    # P6.0-P6.4's 3-4-state chain (a total floor mass of ~1-2 pseudo-counts
    # per row); at this grid's ~2,500 states that default would spread
    # ~1,251 pseudo-counts of floor mass across every row, swamping real
    # pilot signal. Scaling by 1/n_states keeps the same total floor mass.
    learned = TransitionRiskModel.fit(
        pilot_states, schedule, n_actions=env.n_actions, smoothing=1.0 / grid.n_states
    )
    fit_ms = (perf_counter_ns() - fit_start) / 1_000_000.0

    nominal_matrices = np.transpose(exact_mdp.P, (1, 0, 2))
    l1_error = float(np.mean(np.abs(learned.matrices - nominal_matrices)))
    profile = RealisticFitProfile(
        fit_latency_ms=fit_ms,
        pilot_trajectories=pilot_trajectories,
        grid_discretization_l1_error=l1_error,
    )
    model_truth = exact_prefix_risks_from_distribution(learned.matrices, schedule, reset_dist)
    n_prefixes = len(model_truth)
    support_idx, support_cum = _compact_support(learned.matrices, grid.n_velocity_bins + 16)
    terminal_index = learned.matrices.shape[1] - 1

    rng = np.random.default_rng(eval_seed)
    results: list[RealisticResult] = []
    methods = tuple(METHOD_VALIDITY)

    for shift in shifts:
        shift_mdp = (
            exact_mdp if shift.name == "stable" else build_exact_mdp(shift.env, grid, gamma=gamma)
        )
        shift_matrices = np.transpose(shift_mdp.P, (1, 0, 2))
        true_truth = exact_prefix_risks_from_distribution(shift_matrices, schedule, reset_dist)
        oracle_safe_prefix = _oracle_safe_prefix(true_truth, episode_risk)
        counters: dict[tuple[int, str], dict[str, float]] = {
            (n, method): {
                "model_false": 0.0, "true_false": 0.0, "model_risk": 0.0, "true_risk": 0.0,
                "selected": 0.0, "abstained": 0.0, "certificate_ns": 0.0, "calibration_ns": 0.0,
                "end_to_end_ns": 0.0,
            }
            for n in sizes
            for method in methods
        }
        unique_roots_total = 0
        rollout_units_total = 0
        for _ in range(repetitions):
            calib_seed = int(rng.integers(0, 2**31 - 1))
            env_calib_start = perf_counter_ns()
            env_calib_states = rollout_grid_states(
                shift.env, grid, schedule, calibration_samples, calib_seed, group_size=shift.group_size
            )
            env_calibration = env_calib_states[:, 1:] == grid.terminal_state
            env_calib_ns = perf_counter_ns() - env_calib_start
            unique_roots_total += min(calibration_samples, calibration_samples // shift.group_size) or 1
            rollout_units_total += calibration_samples

            model_calib_uniforms = rng.random((calibration_samples, n_prefixes))
            model_calibration = (
                _sample_states_compact(support_idx, support_cum, schedule, model_calib_uniforms)[
                    :, 1:
                ]
                == terminal_index
            )

            certificate_rollout_start = perf_counter_ns()
            failure_uniforms = rng.random((sizes[-1], n_prefixes))
            model_outcomes = (
                _sample_states_compact(support_idx, support_cum, schedule, failure_uniforms)[:, 1:]
                == terminal_index
            )
            certificate_rollout_ns = perf_counter_ns() - certificate_rollout_start
            cumulative = np.cumsum(model_outcomes, axis=0)

            for n in sizes:
                look_index = sizes.index(n) + 1
                calibration_start = perf_counter_ns()
                envelope = calibrate_shift_envelope(
                    model_calibration[:n], env_calibration[:n], confidence_delta=confidence_delta / 2.0
                )
                calibration_ns = perf_counter_ns() - calibration_start
                bad_counts = cumulative[n - 1].astype(int)
                for method in methods:
                    start = perf_counter_ns()
                    selected = certify_prefixes(
                        method,  # type: ignore[arg-type]
                        bad_counts,
                        n=n,
                        look_index=look_index,
                        declared_looks=declared_looks,
                        episode_risk=episode_risk,
                        confidence_delta=confidence_delta,
                        shift_envelope=envelope,
                        csa_bet_margin=csa_bet_margin,
                    )
                    certificate_ns = perf_counter_ns() - start
                    model_risk = 0.0 if selected == 0 else float(model_truth[selected - 1])
                    true_risk = 0.0 if selected == 0 else float(true_truth[selected - 1])
                    count = counters[(n, method)]
                    count["model_false"] += float(selected > 0 and model_risk > episode_risk)
                    count["true_false"] += float(selected > 0 and true_risk > episode_risk)
                    count["model_risk"] += model_risk
                    count["true_risk"] += true_risk
                    count["selected"] += selected
                    count["abstained"] += float(selected == 0)
                    count["certificate_ns"] += certificate_ns
                    calib_share = calibration_ns if method.startswith("shift_") else 0
                    count["calibration_ns"] += calib_share
                    # End-to-end: pro-rated share of this repetition's rollout cost
                    # (env calibration + certificate rollout) plus this call's own cost.
                    count["end_to_end_ns"] += (
                        certificate_ns
                        + calib_share
                        + (env_calib_ns + certificate_rollout_ns) / (len(sizes) * len(methods))
                    )

        horizon = len(schedule)
        for n in sizes:
            for method in methods:
                count = counters[(n, method)]
                results.append(
                    RealisticResult(
                        condition=shift.name,
                        samples=n,
                        method=method,
                        validity_scope=METHOD_VALIDITY[method],
                        model_false_certification_rate=count["model_false"] / repetitions,
                        true_false_certification_rate=count["true_false"] / repetitions,
                        mean_selected_model_risk=count["model_risk"] / repetitions,
                        mean_selected_true_risk=count["true_risk"] / repetitions,
                        mean_selected_prefix=count["selected"] / repetitions,
                        nonvacuity_rate=count["selected"] / (repetitions * n_prefixes),
                        nonabstention_rate=1.0 - count["abstained"] / repetitions,
                        oracle_safe_fraction_recovered=(
                            (count["selected"] / repetitions) / oracle_safe_prefix
                            if oracle_safe_prefix > 0
                            else float("nan")
                        ),
                        abstention_rate=count["abstained"] / repetitions,
                        mean_certificate_latency_us=count["certificate_ns"] / (repetitions * 1_000.0),
                        mean_shift_calibration_latency_us=count["calibration_ns"]
                        / (repetitions * 1_000.0),
                        mean_end_to_end_latency_us=count["end_to_end_ns"] / (repetitions * 1_000.0),
                        unique_noise_roots=unique_roots_total // repetitions,
                        effective_sample_fraction=(
                            (unique_roots_total / repetitions) / calibration_samples
                        ),
                        transitions_evaluated=n * n_prefixes,
                    )
                )

    metadata = {
        "optimal_value_at_reset": optimal_value_at_reset,
        "calibration_scope": {
            "model_checkpoint": f"pilot_seed={pilot_seed}, pilot_trajectories={pilot_trajectories}",
            "rollout_policy": "fixed action_schedule, no adaptive control",
            "action_schedule": list(schedule),
            "grid": f"{grid.n_position_bins}x{grid.n_velocity_bins}+1 (v1)",
            "coupling": "independent-pair (not common-random-number)",
            "expiry": (
                "recalibrate if action_schedule, grid resolution, gamma, or environment "
                "parameters change; this envelope is not valid across policy or model updates"
            ),
        },
        "declared_looks": declared_looks,
    }
    return results, learned, profile, metadata


# -- CLI/config wiring ----------------------------------------------------------


def _env(config: ProposalRunConfig) -> InertialIntervention:
    kwargs = {
        key: float(value)
        for key, value in config.params.get("environment", {}).items()
        if key in {"boundary", "target", "dt", "acceleration", "drag", "noise_std", "failure_penalty", "progress_scale", "control_cost"}
    }
    return InertialIntervention(**kwargs)


def _action_schedule(config: ProposalRunConfig) -> tuple[int, ...]:
    raw = config.params.get("action_schedule", [])
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return ()
    try:
        return tuple(int(value) for value in raw)
    except (TypeError, ValueError):
        return ()


def validate_p6_5(config: ProposalRunConfig) -> list[PrerequisiteCheck]:
    checks = common_checks(config, "p6_certification")
    schedule = _action_schedule(config)
    sizes_raw = config.params.get("sample_sizes", [])
    try:
        sizes = tuple(int(value) for value in sizes_raw)
    except (TypeError, ValueError):
        sizes = ()
    gamma = float(config.params.get("gamma", -1.0))
    position_bins = int(config.params.get("position_bins", 0))
    velocity_bins = int(config.params.get("velocity_bins", 0))
    checks += [
        PrerequisiteCheck("P6.5 stage declared", config.params.get("stage") == "p6.5"),
        PrerequisiteCheck("fixed action schedule declared", bool(schedule) and all(0 <= a < 3 for a in schedule)),
        PrerequisiteCheck(
            "scale frontier declared",
            bool(sizes) and sizes[0] > 0 and all(b > a for a, b in pairwise(sizes)),
        ),
        PrerequisiteCheck(
            "declared looks equal actual checked exits (v4 audit B1)",
            int(config.params.get("declared_looks", -1)) == len(sizes),
            f"declared_looks={config.params.get('declared_looks')}, checked exits={len(sizes)}",
        ),
        PrerequisiteCheck("discount factor preregistered explicitly", 0.0 <= gamma < 1.0),
        PrerequisiteCheck(
            "grid resolution preregistered", position_bins > 0 and velocity_bins > 0
        ),
        PrerequisiteCheck(
            "fresh pilot/evaluation seeds",
            bool(config.pilot_seeds and config.eval_seeds)
            and not set(config.pilot_seeds).intersection(config.eval_seeds),
        ),
        PrerequisiteCheck(
            "positive fit/evaluation budgets",
            int(config.params.get("pilot_trajectories", 0)) > 0
            and int(config.params.get("repetitions", 0)) > 0
            and int(config.params.get("calibration_samples", 0)) > 0,
        ),
        PrerequisiteCheck(
            "exact truth remains evaluation-only",
            True,
            "certify_prefixes accepts sampled counts and shift envelopes only; the closed-form "
            "discretized MDP is used solely for post-selection true_risk and oracle-safe-prefix",
        ),
        PrerequisiteCheck(
            "independent-pair calibration coupling declared (v4 audit B2)",
            True,
            "model/environment calibration draws use disjoint randomness, not shared uniforms",
        ),
    ]
    return checks


def _metric_rows(results: Sequence[RealisticResult]) -> list[dict[str, Any]]:
    units = {
        "model_false_certification_rate": "probability",
        "true_false_certification_rate": "probability",
        "mean_selected_model_risk": "probability",
        "mean_selected_true_risk": "probability",
        "mean_selected_prefix": "actions",
        "nonvacuity_rate": "fraction_of_max_prefix",
        "nonabstention_rate": "probability",
        "oracle_safe_fraction_recovered": "fraction",
        "abstention_rate": "probability",
        "mean_certificate_latency_us": "microseconds",
        "mean_shift_calibration_latency_us": "microseconds",
        "mean_end_to_end_latency_us": "microseconds",
        "unique_noise_roots": "trajectories",
        "effective_sample_fraction": "fraction",
        "transitions_evaluated": "transitions",
    }
    rows: list[dict[str, Any]] = []
    for result in results:
        for metric, unit in units.items():
            rows.append(
                {
                    "stage": "p6.5",
                    "condition": result.condition,
                    "samples": result.samples,
                    "method": result.method,
                    "validity_scope": result.validity_scope,
                    "metric_name": metric,
                    "value": getattr(result, metric),
                    "unit": unit,
                }
            )
    return rows


def _upper(rate: float, repetitions: int) -> float:
    failures = round(rate * repetitions)
    if failures >= repetitions:
        return 1.0
    return float(beta.ppf(0.95, failures + 1, repetitions - failures))


def run_p6_5(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    checks = validate_p6_5(config)
    if not all(check.passed for check in checks):
        return VerdictReport(
            question=config.preregistration.primary_hypothesis,
            status=Status.INCONCLUSIVE,
            evidence_level="E0",
            primary_result={"metric": "configuration_valid", "value": 0, "unit": "boolean"},
            controls=checks,
            interpretation="P6.5 did not run because its realistic-simulator protocol is invalid.",
            non_claim=config.preregistration.non_claim,
            decision="Repair instrumentation without inspecting held-out outcomes and rerun.",
        )

    env = _env(config)
    grid = default_grid(
        env,
        position_bins=int(config.params["position_bins"]),
        velocity_bins=int(config.params["velocity_bins"]),
        velocity_range=tuple(float(v) for v in config.params.get("velocity_range", [-1.5, 1.5])),
    )
    shifts = default_shifts(
        env, clustered_unit_group_size=int(config.params.get("clustered_unit_group_size", 8))
    )
    results, learned, profile, metadata = run_realistic_benchmark(
        env=env,
        grid=grid,
        gamma=float(config.params["gamma"]),
        action_schedule=_action_schedule(config),
        shifts=shifts,
        pilot_seed=config.pilot_seeds[0],
        eval_seed=config.eval_seeds[0],
        pilot_trajectories=int(config.params["pilot_trajectories"]),
        calibration_samples=int(config.params["calibration_samples"]),
        repetitions=int(config.params["repetitions"]),
        sample_sizes=tuple(int(v) for v in config.params["sample_sizes"]),
        episode_risk=float(config.params["episode_risk"]),
        confidence_delta=float(config.params["confidence_delta"]),
        csa_bet_margin=float(config.params["csa_bet_margin"]),
    )
    writer.write_metrics(_metric_rows(results))
    for result in results:
        writer.event(stage="p6.5", **result.__dict__)
    writer.event(
        stage="p6.5-fit",
        learned_transition_matrices_shape=list(learned.matrices.shape),
        **profile.__dict__,
    )
    writer.event(stage="p6.5-calibration-scope", **metadata)

    largest = max(result.samples for result in results)
    repetitions = int(config.params["repetitions"])
    tolerance = float(config.params["false_certification_tolerance"])
    gain_target = float(config.params["gain_actions"])
    primary = [
        r for r in results
        if r.samples == largest and r.method == "shift_robust_csa_specialization"
        and r.condition != "clustered_units"
    ]
    clustered = [
        r for r in results
        if r.samples == largest and r.method == "shift_robust_csa_specialization"
        and r.condition == "clustered_units"
    ]
    union = [
        r for r in results
        if r.samples == largest and r.method == "shift_robust_union" and r.condition != "clustered_units"
    ]
    uppers = [_upper(r.true_false_certification_rate, repetitions) for r in primary]
    valid = all(upper <= tolerance for upper in uppers)
    nonvacuous = all(r.nonvacuity_rate > 0 for r in primary)
    gain = float(np.mean([r.mean_selected_prefix for r in primary])) - float(
        np.mean([r.mean_selected_prefix for r in union])
    )
    model_true_gap = max(
        r.true_false_certification_rate - r.model_false_certification_rate
        for r in results
        if r.samples == largest and r.method == "model_only_anytime_cp" and r.condition != "clustered_units"
    )
    clustered_upper = max(
        (_upper(r.true_false_certification_rate, repetitions) for r in clustered), default=0.0
    )
    checks += [
        PrerequisiteCheck(
            "realistic-environment true validity",
            valid,
            f"worst one-sided 95% upper={max(uppers, default=1.0):.3f}",
        ),
        PrerequisiteCheck(
            "realistic-environment nonvacuity",
            nonvacuous,
            f"minimum={min((r.nonvacuity_rate for r in primary), default=0.0):.3f}",
        ),
        PrerequisiteCheck(
            "gain over robust union",
            gain >= gain_target,
            f"gain={gain:.3f}; target={gain_target:.3f}",
        ),
        PrerequisiteCheck(
            "structural shift separates model and true validity",
            model_true_gap > 0,
            f"maximum gap={model_true_gap:.3f}",
        ),
        PrerequisiteCheck(
            "exchangeability stress cell (clustered noise roots) reported separately",
            True,
            f"clustered_units worst one-sided 95% upper={clustered_upper:.3f} (not gated -- descriptive)",
        ),
        PrerequisiteCheck(
            "task-utility metrics beyond nonvacuity reported (v4 audit B5)",
            all(not np.isnan(r.oracle_safe_fraction_recovered) for r in primary),
        ),
        PrerequisiteCheck(
            "end-to-end latency includes rollout/inference (v4 audit B6)",
            all(r.mean_end_to_end_latency_us >= r.mean_certificate_latency_us for r in results),
        ),
    ]

    if not valid:
        status = Status.STOP
        interpretation = (
            "The realistic-environment robust certificate failed true validity; the machinery "
            "does not survive contact with a genuinely continuous, non-toy generative process "
            "at this configuration."
        )
        decision = "Do not advance past P6.5; diagnose the discretization or calibration coupling."
    elif not nonvacuous:
        status = Status.NARROW
        interpretation = "The realistic-environment certificate was valid but vacuous."
        decision = "Remain NARROW and preserve the validity threshold."
    elif gain >= gain_target:
        status = Status.GO
        interpretation = (
            "The certificate/comparator/mismatch core, unmodified, retained true validity and "
            "nonvacuity and met the locked prefix gain over robust union against a discretized "
            "but closed-form-exact continuous environment with real Gaussian rollout noise. "
            "This is bounded CPU realistic-simulator evidence, not deployment evidence."
        )
        decision = (
            "Freeze P6.5. Preserve P6's core certification result across the whole theory track; "
            "any further stage is a separate, freshly preregistered proposal."
        )
    else:
        status = Status.NARROW
        interpretation = (
            "The realistic-environment integration was valid and nonvacuous but did not beat the "
            "robust union by the locked amount."
        )
        decision = "Preserve the integration and remain NARROW on utility."
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=status,
        evidence_level="E2",
        primary_result={
            "metric": "largest_scale_mean_prefix_gain_over_robust_union",
            "value": gain,
            "unit": "actions",
        },
        baselines_run=list(METHOD_VALIDITY),
        controls=checks,
        interpretation=interpretation,
        non_claim=config.preregistration.non_claim,
        decision=decision,
    )
