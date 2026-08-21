"""P1.1 sampled CPU pilot.

The headline surface is the deployable recurring plan/act return.  The named
decomposition is deliberately measured by :mod:`instinct.p1_atlas.sampled` on
one matched handoff, as required by P1 amendment 1.1.  Keeping both tables in
one artifact makes it difficult to accidentally present the local causal
decomposition as a decomposition of the recurring policy value.

This runner exercises three inexpensive CPU families (pursuit, Tetris-lite,
and inertial delayed intervention), all speed/runtime/reflex combinations, and
the full M0--M5 registry.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats as scipy_stats

from instinct.core.configschema import ProposalRunConfig
from instinct.core.env import BatchedEnv, EnvState
from instinct.core.envs.arcade import TetrisLite
from instinct.core.envs.control import InertialIntervention
from instinct.core.envs.gridworld import pillar_field
from instinct.core.holdout import SplitRegistry
from instinct.core.proposal import common_checks
from instinct.core.results import ResultsWriter
from instinct.core.rng import SeedScope
from instinct.core.rollout import PlannerFn, ReflexFn
from instinct.core.verdict import FailureCode, PrerequisiteCheck, Status, VerdictReport
from instinct.p1_atlas.models import (
    RegistryResult,
    evaluate_registry,
    uncertainty_shrunk_atlas,
)
from instinct.p1_atlas.sampled import (
    FailureFn,
    decompose_handoff,
    event_delays,
    matched_failure_delta,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

FAMILIES = ("pursuit", "tetris", "control")
REFLEXES = ("hold", "greedy", "distilled")
_GRID_MOVES: IntArray = np.array(
    [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64
)
_TETRIS_WIDTHS: IntArray = np.array([1, 2, 1, 2, 3, 1, 4, 1], dtype=np.int64)


@dataclass(frozen=True, slots=True)
class RecurringTrace:
    """Per-episode operational return and actual work launched."""

    total_return: FloatArray
    planner_calls: IntArray
    cumulative_hardware_cost: FloatArray
    initial_delay_events: IntArray
    failed: BoolArray


@dataclass(frozen=True, slots=True)
class PilotArtifacts:
    metrics: pd.DataFrame
    contexts: pd.DataFrame
    model_result: RegistryResult
    model_summary: pd.DataFrame
    controls: tuple[PrerequisiteCheck, ...]
    full_family_coverage: bool
    criteria_pass: bool


def _as_int_list(config: ProposalRunConfig, name: str, default: Sequence[int]) -> list[int]:
    value = config.params.get(name, list(default))
    if not isinstance(value, list) or not value or not all(isinstance(v, int) for v in value):
        raise ValueError(f"params.{name} must be a nonempty list of integers")
    return [int(v) for v in value]


def _as_float_list(
    config: ProposalRunConfig, name: str, default: Sequence[float]
) -> list[float]:
    value = config.params.get(name, list(default))
    if not isinstance(value, list) or not value or not all(
        isinstance(v, int | float) and not isinstance(v, bool) for v in value
    ):
        raise ValueError(f"params.{name} must be a nonempty numeric list")
    return [float(v) for v in value]


def validate_sampled(config: ProposalRunConfig) -> list[PrerequisiteCheck]:
    """Checks that can run without opening a run directory or timing a planner."""
    checks = common_checks(config, "p1_atlas")
    budgets = _as_int_list(config, "budgets", range(1, 9))
    speeds = _as_float_list(config, "event_rates_hz", (5_000.0, 20_000.0, 80_000.0))
    regimes = _as_int_list(config, "runtime_work_scales", (1, 4, 12))
    families = config.params.get("families", list(FAMILIES))
    reflexes = config.params.get("reflexes", list(REFLEXES))
    checks.extend(
        (
            PrerequisiteCheck(
                "P1.1 stage",
                config.params.get("stage") in {"p1.1", "p1.1-r1", "p1.1-r2"},
            ),
            PrerequisiteCheck("8-12 budget grid", 8 <= len(set(budgets)) <= 12),
            PrerequisiteCheck("at least three speeds", len(set(speeds)) >= 3),
            PrerequisiteCheck("at least three measured CPU regimes", len(set(regimes)) >= 3),
            PrerequisiteCheck(
                "required reflex endpoints",
                isinstance(reflexes, list) and set(REFLEXES) <= set(reflexes),
            ),
            PrerequisiteCheck(
                "implemented CPU families requested",
                isinstance(families, list) and set(families) <= set(FAMILIES),
                f"requested={families!r}",
            ),
            PrerequisiteCheck(
                "P1.1 amendment selected", "p1-1.1" in config.theory.amendment_id
            ),
        )
    )
    return checks


def _burn_work(state: EnvState, budget: int, work_scale: int) -> None:
    """Small real CPU workload whose elapsed time is measured, never assumed."""
    value = float(state.n_lanes + budget)
    for index in range(max(1, work_scale * budget * 24)):
        value = (value * 1.000000119 + float(index + 1)) % 997.0
    if not np.isfinite(value):  # pragma: no cover - defensive tripwire
        raise FloatingPointError("planner timing workload became nonfinite")


def _grid_action(state: EnvState, budget: int) -> IntArray:
    agent = state["agent"].astype(np.int64)
    target = state["target"].astype(np.int64)
    env_candidates = min(max(budget, 1), len(_GRID_MOVES))
    scores = np.full((state.n_lanes, len(_GRID_MOVES)), np.inf)
    for action in range(env_candidates):
        candidate = agent + _GRID_MOVES[action]
        scores[:, action] = np.abs(candidate - target).sum(axis=1)
    return scores.argmin(axis=1).astype(np.int64)


def _tetris_action(state: EnvState, budget: int) -> IntArray:
    heights = state["heights"].astype(np.int64)
    piece = state["piece"].astype(np.int64)
    width = heights.shape[1]
    candidates = min(max(budget, 1), width)
    score = np.full((state.n_lanes, width), np.inf)
    for action in range(candidates):
        piece_width = _TETRIS_WIDTHS[piece]
        column = np.minimum(action, width - piece_width)
        columns = np.arange(width)[None, :]
        span = (columns >= column[:, None]) & (columns < column[:, None] + piece_width[:, None])
        base = np.max(np.where(span, heights, -1), axis=1)
        score[:, action] = base + 0.05 * np.var(heights, axis=1)
    return score.argmin(axis=1).astype(np.int64)


def _control_action(state: EnvState, budget: int) -> IntArray:
    """Deterministic lookahead whose horizon, not action set, is the budget."""
    position = state["position"].astype(np.float64)
    velocity = state["velocity"].astype(np.float64)
    horizon = min(max(budget, 1), 8)
    accelerations = np.array([-1.0, 0.0, 1.0])
    scores = np.empty((state.n_lanes, 3), dtype=np.float64)
    for action, acceleration in enumerate(accelerations):
        predicted_position = position.copy()
        predicted_velocity = velocity.copy()
        crossed = np.zeros(state.n_lanes, dtype=bool)
        for _ in range(horizon):
            predicted_velocity = 0.985 * predicted_velocity + 0.12 * acceleration
            predicted_position = predicted_position + 0.12 * predicted_velocity
            crossed |= np.abs(predicted_position) >= 1.0
        scores[:, action] = (
            np.abs(predicted_position - 0.65)
            + 0.20 * np.abs(predicted_velocity)
            + 100.0 * crossed
        )
    return scores.argmin(axis=1).astype(np.int64)


def _planner(family: str, work_scale: int) -> PlannerFn:
    action_fns = {
        "pursuit": _grid_action,
        "tetris": _tetris_action,
        "control": _control_action,
    }
    try:
        action_fn = action_fns[family]
    except KeyError as exc:
        raise ValueError(f"unknown P1.1 planner family {family!r}") from exc

    def plan(state: EnvState, budget: int, tick: int) -> IntArray:
        del tick
        result = action_fn(state, budget)
        _burn_work(state, budget, work_scale)
        return result

    return plan


def _reflex(family: str, name: str, max_budget: int) -> ReflexFn:
    if family == "pursuit":
        if name == "hold":
            return lambda state, tick: np.zeros(state.n_lanes, dtype=np.int64)
        if name == "greedy":
            # Direct axis-priority pursuit ignores pillars and is genuinely
            # distinct from the exhaustive distilled action on this family.
            def grid_greedy(state: EnvState, tick: int) -> IntArray:
                del tick
                delta = state["target"].astype(np.int64) - state["agent"].astype(np.int64)
                vertical = np.where(delta[:, 0] < 0, 1, 2)
                horizontal = np.where(delta[:, 1] < 0, 3, 4)
                return np.where(delta[:, 0] != 0, vertical, horizontal).astype(np.int64)

            return grid_greedy
        return lambda state, tick: _grid_action(state, max_budget)

    if family == "tetris":
        if name == "hold":
            return lambda state, tick: np.zeros(state.n_lanes, dtype=np.int64)
        if name == "greedy":
            return lambda state, tick: state["heights"].argmin(axis=1).astype(np.int64)
        return lambda state, tick: _tetris_action(state, max_budget)

    if family == "control":
        if name == "hold":
            return lambda state, tick: np.ones(state.n_lanes, dtype=np.int64)
        if name == "greedy":
            return lambda state, tick: np.where(
                state["position"].astype(np.float64) < 0.65, 2, 0
            ).astype(np.int64)
        return lambda state, tick: _control_action(state, max_budget)
    raise ValueError(f"unknown P1.1 reflex family {family!r}")


def _family_parts(
    family: str, reflex_name: str, max_budget: int
) -> tuple[Any, ReflexFn, ReflexFn, FailureFn, float]:
    if family == "pursuit":
        env = pillar_field(size=7, spacing=3, drift=0.65, evasion=0.25)

        def zero_failure(state: EnvState) -> BoolArray:
            return np.zeros(state.n_lanes, dtype=bool)

        return (
            env,
            _reflex(family, reflex_name, max_budget),
            _reflex(family, "hold", max_budget),
            zero_failure,
            0.0,
        )
    if family == "tetris":
        tetris_env = TetrisLite(width=6, ceiling=8)

        def failure(state: EnvState) -> BoolArray:
            return state["dead"].astype(bool)

        return (
            tetris_env,
            _reflex(family, reflex_name, max_budget),
            _reflex(family, "greedy", max_budget),
            failure,
            1.0,
        )
    if family == "control":
        control_env = InertialIntervention()

        def failure(state: EnvState) -> BoolArray:
            return state["failed"].astype(bool)

        def safe(state: EnvState, tick: int) -> IntArray:
            del tick
            velocity = state["velocity"].astype(np.float64)
            return np.where(velocity > 0.02, 0, np.where(velocity < -0.02, 2, 1)).astype(
                np.int64
            )

        return (
            control_env,
            _reflex(family, reflex_name, max_budget),
            safe,
            failure,
            2.0,
        )
    raise ValueError(f"unknown P1.1 family {family!r}")


def _state_descriptor(family: str, state: EnvState) -> float:
    if family == "pursuit":
        distance = np.abs(
            state["agent"].astype(np.float64) - state["target"].astype(np.float64)
        ).sum(axis=1)
        return float(np.mean(distance / 12.0))
    if family == "tetris":
        return float(np.mean(state["piece"].astype(np.float64) / 7.0))
    position = state["position"].astype(np.float64)
    velocity = state["velocity"].astype(np.float64)
    boundary_margin = np.where(velocity >= 0, 1.0 - position, position + 1.0)
    recovery_margin = boundary_margin - np.square(velocity) / 2.0
    difficulty = np.maximum(-recovery_margin, 0.0) + 0.25 * np.abs(position - 0.65)
    return float(np.mean(difficulty))


def _control_region_check() -> tuple[bool, str]:
    """Known states must span both sides of the deterministic recovery frontier."""
    env = InertialIntervention(noise_std=0.0)
    states = EnvState(
        lane_ids=np.array([0, 1], dtype=np.int64),
        fields={
            "position": np.array([0.50, 0.85], dtype=np.float64),
            "velocity": np.array([0.20, 0.80], dtype=np.float64),
            "failed": np.array([False, False]),
        },
    )
    margins = env.recovery_margin(states)
    passed = bool(np.array_equal(env.recoverable(states), np.array([True, False])))
    return passed, f"margins={margins.tolist()}"


def _failure_rich_control_state(
    state: EnvState, scope: SeedScope, lanes: IntArray
) -> EnvState:
    """Fresh preregistered starts spanning the inertial recovery frontier."""

    draws = scope.stream("failure-rich-control-start").uniform(lanes, count=2)
    velocity = 0.35 + 0.70 * draws[:, 0]
    requested_margin = -0.12 + 0.36 * draws[:, 1]
    position = np.clip(1.0 - np.square(velocity) / 2.0 - requested_margin, -0.95, 0.95)
    return state.replace_fields(
        position=position.astype(np.float64),
        velocity=velocity.astype(np.float64),
        failed=np.zeros(len(lanes), dtype=bool),
    )


def _measure_latencies(planner: PlannerFn, state: EnvState, budget: int) -> FloatArray:
    """One observed CPU duration per episode, including call overhead and jitter."""
    if state.n_lanes:
        planner(state.take(np.array([0], dtype=np.int64)), budget, 0)  # warm this path
    elapsed = np.empty(state.n_lanes, dtype=np.float64)
    for lane in range(state.n_lanes):
        index = np.array([lane], dtype=np.int64)
        started = time.perf_counter_ns()
        planner(state.take(index), budget, 0)
        elapsed[lane] = max(time.perf_counter_ns() - started, 1) * 1e-9
    return elapsed


def simulate_recurring_observed(
    env: BatchedEnv,
    state0: EnvState,
    *,
    budget: int,
    latency_s: FloatArray,
    event_rate_hz: float,
    initial_phase: FloatArray,
    horizon: int,
    gamma: float,
    planner: PlannerFn,
    reflex: ReflexFn,
    failure: FailureFn,
    scope: SeedScope,
    hardware_price_per_s: float,
    episode: int = 0,
) -> RecurringTrace:
    """Recurring stale-plan policy with per-lane observed latency and work.

    A new plan launches after every committed action.  Each launch is charged,
    even when its result remains pending at the horizon.  Dead lanes never
    launch additional work.  Runtime is replayed from that episode's observed
    CPU duration; phase is redrawn counterfactually by absolute tick and lane.
    """
    latency = np.asarray(latency_s, dtype=np.float64)
    if latency.shape != (state0.n_lanes,):
        raise ValueError("recurring latency must contain one observation per lane")
    state = state0.copy()
    alive = np.ones(state.n_lanes, dtype=bool)
    failed = np.asarray(failure(state), dtype=bool).copy()
    total = np.zeros(state.n_lanes)
    calls = np.ones(state.n_lanes, dtype=np.int64)
    cost = latency * hardware_price_per_s
    landed = planner(state, budget, 0)
    remaining = event_delays(latency, event_rate_hz=event_rate_hz, phase=initial_phase)
    initial_delay = remaining.copy()
    discount = 1.0

    for tick in range(horizon):
        committing = alive & (remaining == 0)
        actions = np.where(committing, landed, reflex(state, tick))
        result = env.step(state, actions, scope=scope, episode=episode, tick=tick)
        total += discount * result.reward * alive
        state = result.state
        failed |= np.asarray(failure(state), dtype=bool) & alive
        alive &= ~result.done
        waiting = alive & ~committing
        remaining = np.where(waiting, np.maximum(remaining - 1, 0), remaining)

        relaunch = committing & alive
        if relaunch.any():
            indices = np.flatnonzero(relaunch)
            landed[indices] = planner(state.take(indices), budget, tick + 1)
            phase = scope.stream("recurring-launch-phase").uniform(
                state.lane_ids[indices], count=1, episode=episode, tick=tick + 1
            )[:, 0]
            remaining[indices] = event_delays(
                latency[indices], event_rate_hz=event_rate_hz, phase=phase
            )
            calls[indices] += 1
            cost[indices] += latency[indices] * hardware_price_per_s
        discount *= gamma

    return RecurringTrace(total, calls, cost, initial_delay, failed)


def _context_split(speed_index: int, runtime_index: int) -> str:
    """Frozen 3x3 combination split; no budget point is split independently."""
    pair = (speed_index, runtime_index)
    if pair in {(1, 2), (2, 1)}:
        return "test"
    if pair in {(0, 2), (2, 0)}:
        return "validation"
    return "train"


def _one_sided_t_bound(values: FloatArray, *, upper: bool) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.size < 2 or not np.isfinite(x).all():
        return float("inf") if upper else float("-inf")
    sem = float(np.std(x, ddof=1) / np.sqrt(x.size))
    critical = float(scipy_stats.t.ppf(0.95, x.size - 1))
    mean = float(np.mean(x))
    return mean + critical * sem if upper else mean - critical * sem


def _surface(episodes: pd.DataFrame, contexts: pd.DataFrame) -> pd.DataFrame:
    oracle = episodes.loc[episodes["unit_role"] == "oracle"]
    means = (
        oracle.groupby(["context_id", "budget"])["net_return"]
        .mean()
        .reset_index(name="value")
    )
    base = means.loc[means["budget"] == means["budget"].min()].set_index("context_id")[
        "value"
    ]
    means["sigma"] = means["value"] - means["context_id"].map(base)
    columns = [
        "context_id",
        "split",
        "family",
        "reflex",
        "nu_e",
        "nu_h",
        "state_descriptor",
        "reflex_descriptor",
        "environment_descriptor",
        "hardware_quality",
    ]
    return means.merge(contexts[columns], on="context_id", validate="many_to_one")


def _extra_choices(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Smallest/largest, speed schedule, and a lightweight uncertainty gate."""
    budgets = sorted(int(value) for value in np.asarray(train["budget"].unique()))
    test_context_ids = [str(value) for value in np.asarray(test["context_id"].unique())]
    choices: dict[str, dict[str, int]] = {
        "SMALLEST": {context: budgets[0] for context in test_context_ids},
        "LARGEST": {context: budgets[-1] for context in test_context_ids},
    }
    speed_best = train.groupby(["nu_e", "budget"])["sigma"].mean().unstack("budget")
    choices["SPEED"] = {
        str(row["context_id"]): int(cast(Any, speed_best.loc[row["nu_e"]]).idxmax())
        for row in cast(list[dict[str, Any]], test.to_dict("records"))
    }

    train_context = train.drop_duplicates("context_id").copy()
    test_context = test.drop_duplicates("context_id").copy()
    train_context["difficulty"] = (
        train_context["nu_e"]
        * train_context["nu_h"]
        * (1.0 + train_context["state_descriptor"])
    )
    test_context["difficulty"] = (
        test_context["nu_e"]
        * test_context["nu_h"]
        * (1.0 + test_context["state_descriptor"])
    )
    threshold = float(train_context["difficulty"].median())
    context_bin = {
        str(row["context_id"]): int(float(row["difficulty"]) > threshold)
        for row in cast(list[dict[str, Any]], train_context.to_dict("records"))
    }
    train_with_bin = train.copy()
    train_with_bin["gate_bin"] = train_with_bin["context_id"].map(context_bin)
    best_by_bin = (
        train_with_bin.groupby(["gate_bin", "budget"])["sigma"]
        .mean()
        .unstack("budget")
        .idxmax(axis=1)
    )
    choices["FTT_LITE"] = {
        str(row["context_id"]): int(
            cast(Any, best_by_bin.loc[int(float(row["difficulty"]) > threshold)])
        )
        for row in cast(list[dict[str, Any]], test_context.to_dict("records"))
    }
    return choices


def _regret_table(
    episodes: pd.DataFrame,
    surface: pd.DataFrame,
    result: RegistryResult,
    *,
    additional_choices: Mapping[str, Mapping[str, int]] | None = None,
) -> pd.DataFrame:
    test_surface = surface.loc[surface["split"] == "test"]
    oracle_budget = {
        str(context): int(cast(Any, group.loc[group["value"].idxmax(), "budget"]))
        for context, group in test_surface.groupby("context_id")
    }
    choices: dict[str, dict[str, int]] = {}
    for model_id, group in result.predictions.groupby("model_id"):
        choices[str(model_id)] = {
            str(row["context_id"]): int(row["chosen_budget"])
            for row in cast(list[dict[str, Any]], group.to_dict("records"))
        }
    train_surface = surface.loc[
        (surface["split"] == "train") & (surface["reflex"] != "distilled")
    ]
    choices.update(_extra_choices(train_surface, test_surface))
    if additional_choices is not None:
        overlap = set(choices) & set(additional_choices)
        if overlap:
            raise ValueError(f"additional model choices collide with frozen models: {overlap}")
        choices.update(
            {
                str(model): {str(context): int(budget) for context, budget in by_context.items()}
                for model, by_context in additional_choices.items()
            }
        )

    evaluation = episodes.loc[
        (episodes["unit_role"] == "regret") & (episodes["split"] == "test")
    ]
    rows: list[dict[str, object]] = []
    for context_id, group in evaluation.groupby("context_id"):
        pivot = group.pivot(index="unit_id", columns="budget", values="net_return")
        optimum = oracle_budget[str(context_id)]
        for model_id, by_context in choices.items():
            chosen = by_context[str(context_id)]
            paired = pivot[optimum].to_numpy() - pivot[chosen].to_numpy()
            rows.append(
                {
                    "record_type": "context_regret",
                    "context_id": context_id,
                    "model_id": model_id,
                    "chosen_budget": chosen,
                    "oracle_budget": optimum,
                    "paired_regret": float(np.mean(paired)),
                    "paired_regret_se": float(np.std(paired, ddof=1) / np.sqrt(len(paired)))
                    if len(paired) > 1
                    else float("inf"),
                    "n_regret_units": len(paired),
                }
            )
    return pd.DataFrame(rows)


def _model_summary(regrets: pd.DataFrame, target_range: float) -> pd.DataFrame:
    by_model = {
        str(model): group.set_index("context_id")["paired_regret"]
        for model, group in regrets.groupby("model_id")
    }
    m4 = by_model["M4"]
    rows: list[dict[str, object]] = []
    for model_id, values in by_model.items():
        aligned = values.reindex(m4.index)
        improvement = aligned.to_numpy() - m4.to_numpy()
        rows.append(
            {
                "record_type": "model_summary",
                "model_id": model_id,
                "mean_context_regret": float(values.mean()),
                "context_regret_ucb95": _one_sided_t_bound(values.to_numpy(), upper=True),
                "improvement_over_m4_lcb95": _one_sided_t_bound(improvement, upper=False),
                "n_test_contexts": len(values),
                "target_range": target_range,
            }
        )
    return pd.DataFrame(rows)


def generate_pilot(config: ProposalRunConfig) -> PilotArtifacts:
    """Generate the sampled artifact without writing files, for tests and CLI."""
    initial_checks = validate_sampled(config)
    if any(not check.passed for check in initial_checks):
        failed = [check.name for check in initial_checks if not check.passed]
        raise ValueError(f"P1.1 validation failed closed: {failed}")

    budgets = sorted(set(_as_int_list(config, "budgets", range(1, 9))))
    speeds = _as_float_list(config, "event_rates_hz", (5_000.0, 20_000.0, 80_000.0))
    regimes = _as_int_list(config, "runtime_work_scales", (1, 4, 12))
    families = [str(value) for value in config.params.get("families", list(FAMILIES))]
    reflexes = [str(value) for value in config.params.get("reflexes", list(REFLEXES))]
    n_oracle = int(config.params.get("n_oracle", 4))
    n_regret = int(config.params.get("n_regret", 6))
    horizon = int(config.params.get("horizon", 12))
    handoff_horizon = int(config.params.get("handoff_horizon", horizon))
    gamma = float(config.params.get("gamma", 0.97))
    hardware_price = float(config.params.get("hardware_price_per_s", 0.01))
    failure_rich_control = bool(config.params.get("failure_rich_control", False))
    if min(n_oracle, n_regret, horizon, handoff_horizon) < 1:
        raise ValueError("P1.1 unit counts and horizons must be positive")
    if len(config.pilot_seeds) != n_oracle or len(config.eval_seeds) != n_regret:
        raise ValueError(
            "params.n_oracle/n_regret must match pilot_seeds/eval_seeds; "
            "the declared seed split is the scientific unit registry"
        )
    role_lanes = {
        "oracle": np.asarray(config.pilot_seeds, dtype=np.int64),
        "regret": np.asarray(config.eval_seeds, dtype=np.int64),
    }

    root_scope = SeedScope(config.seed).child("p1.1")
    context_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    decomposition_rows: list[dict[str, object]] = []
    split_registry = SplitRegistry()
    context_number = 0

    for family_index, family in enumerate(families):
        for reflex_index, reflex_name in enumerate(reflexes):
            env, reflex, safe, failure, environment_descriptor = _family_parts(
                family, reflex_name, max(budgets)
            )
            for speed_index, speed in enumerate(speeds):
                for runtime_index, work_scale in enumerate(regimes):
                    split = _context_split(speed_index, runtime_index)
                    context_id = f"p11-{family}-{reflex_name}-s{speed_index}-h{runtime_index}"
                    split_registry.assign(context_id, split)
                    context_number += 1
                    unit_data: dict[str, tuple[EnvState, SeedScope, IntArray]] = {}
                    latency_by_role: dict[str, dict[int, FloatArray]] = {}
                    phase_by_role: dict[str, FloatArray] = {}
                    planner = _planner(family, work_scale)
                    for role, count in (
                        ("oracle", n_oracle),
                        ("regret", n_regret),
                    ):
                        lanes = role_lanes[role]
                        if len(lanes) != count:  # already checked, preserves local invariant
                            raise ValueError(f"{role} unit registry length changed unexpectedly")
                        scope = root_scope.child(context_id, role)
                        state0 = env.reset(lanes, scope=scope)
                        if family == "control" and failure_rich_control:
                            state0 = _failure_rich_control_state(state0, scope, lanes)
                        phase = scope.stream("initial-launch-phase").uniform(lanes, count=1)[:, 0]
                        unit_data[role] = (state0, scope, lanes)
                        phase_by_role[role] = phase
                        latency_by_role[role] = {
                            budget: _measure_latencies(planner, state0, budget)
                            for budget in budgets
                        }

                    base_latency = latency_by_role["oracle"][budgets[0]]
                    base_median = float(np.median(base_latency))
                    state_descriptor = _state_descriptor(family, unit_data["oracle"][0])
                    all_base = np.concatenate(
                        [latency_by_role[role][budgets[0]] for role in ("oracle", "regret")]
                    )
                    jitter = float(np.std(all_base) / max(np.mean(all_base), 1e-12))
                    context_rows.append(
                        {
                            "record_type": "context_registry",
                            "context_id": context_id,
                            "curve_id": context_id,
                            "split": split,
                            "family": family,
                            "reflex": reflex_name,
                            "speed_index": speed_index,
                            "runtime_index": runtime_index,
                            "runtime_condition": f"cpu-work-{work_scale}",
                            "nu_e": speed,
                            "nu_h": base_median,
                            "state_descriptor": state_descriptor,
                            "reflex_descriptor": float(reflex_index) / max(len(reflexes) - 1, 1),
                            "environment_descriptor": environment_descriptor,
                            "hardware_quality": jitter,
                            "oracle_units": n_oracle,
                            "regret_units": n_regret,
                            "family_index": family_index,
                        }
                    )

                    for role in ("oracle", "regret"):
                        state0, scope, lanes = unit_data[role]
                        phase = phase_by_role[role]
                        initial_margin = (
                            cast(InertialIntervention, env).recovery_margin(state0)
                            if family == "control"
                            else np.full(state0.n_lanes, np.nan)
                        )
                        for budget in budgets:
                            latency = latency_by_role[role][budget]
                            recurring = simulate_recurring_observed(
                                env,
                                state0,
                                budget=budget,
                                latency_s=latency,
                                event_rate_hz=speed,
                                initial_phase=phase,
                                horizon=horizon,
                                gamma=gamma,
                                planner=planner,
                                reflex=reflex,
                                failure=failure,
                                scope=scope,
                                hardware_price_per_s=hardware_price,
                            )
                            net = recurring.total_return - recurring.cumulative_hardware_cost
                            for lane_position, unit_id in enumerate(lanes):
                                episode_rows.append(
                                    {
                                        "record_type": "recurring_episode",
                                        "context_id": context_id,
                                        "curve_id": context_id,
                                        "split": split,
                                        "family": family,
                                        "reflex": reflex_name,
                                        "unit_role": role,
                                        "unit_id": int(unit_id),
                                        "budget": budget,
                                        "observed_latency_s": float(latency[lane_position]),
                                        "launch_phase": float(phase[lane_position]),
                                        "initial_delay_events": int(
                                            recurring.initial_delay_events[lane_position]
                                        ),
                                        "planner_calls": int(
                                            recurring.planner_calls[lane_position]
                                        ),
                                        "cumulative_hardware_cost": float(
                                            recurring.cumulative_hardware_cost[lane_position]
                                        ),
                                        "recurring_return": float(
                                            recurring.total_return[lane_position]
                                        ),
                                        "net_return": float(net[lane_position]),
                                        "failed": bool(recurring.failed[lane_position]),
                                        "initial_recovery_margin": float(
                                            initial_margin[lane_position]
                                        ),
                                    }
                                )

                    # Decomposition is evaluated only on the locked regret
                    # units; oracle units identify the frontier and never enter
                    # these causal summaries.
                    state0, scope, lanes = unit_data["regret"]
                    phase = phase_by_role["regret"]
                    base_latency_eval = latency_by_role["regret"][budgets[0]]
                    for budget in budgets:
                        latency = latency_by_role["regret"][budget]
                        decomposition = decompose_handoff(
                            env,
                            state0,
                            budget=budget,
                            base_budget=budgets[0],
                            latency_s=latency,
                            base_latency_s=base_latency_eval,
                            event_rate_hz=speed,
                            phase=phase,
                            horizon=handoff_horizon,
                            gamma=gamma,
                            planner=planner,
                            reflex=reflex,
                            failure=failure,
                            scope=scope,
                            hardware_price_per_s=hardware_price,
                        )
                        irreversible = matched_failure_delta(
                            env,
                            state0,
                            latency_s=latency,
                            base_latency_s=base_latency_eval,
                            event_rate_hz=speed,
                            phase=phase,
                            reflex=reflex,
                            safe=safe,
                            failure=failure,
                            scope=scope,
                        )
                        for lane_position, unit_id in enumerate(lanes):
                            decomposition_rows.append(
                                {
                                    "record_type": "handoff_decomposition",
                                    "context_id": context_id,
                                    "curve_id": context_id,
                                    "split": split,
                                    "family": family,
                                    "reflex": reflex_name,
                                    "unit_role": "regret",
                                    "unit_id": int(unit_id),
                                    "budget": budget,
                                    "sigma": float(decomposition.sigma[lane_position]),
                                    "G_plan": float(decomposition.G_plan[lane_position]),
                                    "R_intermediate": float(
                                        decomposition.R_intermediate[lane_position]
                                    ),
                                    "L_arrival": float(decomposition.L_arrival[lane_position]),
                                    "L_wait": float(decomposition.L_wait[lane_position]),
                                    "C_hw": float(decomposition.C_hw[lane_position]),
                                    "L_base_delay": float(
                                        decomposition.L_base_delay[lane_position]
                                    ),
                                    "epsilon_id": float(decomposition.epsilon_id[lane_position]),
                                    "L_irreversible": float(irreversible[lane_position]),
                                    "J_actual": float(
                                        decomposition.actual.total_return[lane_position]
                                    ),
                                    "J_fresh": float(
                                        decomposition.fresh.total_return[lane_position]
                                    ),
                                    "J_instant": float(
                                        decomposition.instant.total_return[lane_position]
                                    ),
                                    "J_base": float(
                                        decomposition.base.total_return[lane_position]
                                    ),
                                    "J_instant_base": float(
                                        decomposition.instant_base.total_return[lane_position]
                                    ),
                                }
                            )

    contexts = pd.DataFrame(context_rows)
    episodes = pd.DataFrame(episode_rows)
    decomposition_frame = pd.DataFrame(decomposition_rows)
    surface = _surface(episodes, contexts)
    train = surface.loc[
        (surface["split"] == "train") & (surface["reflex"] != "distilled")
    ].reset_index(drop=True)
    test = surface.loc[surface["split"] == "test"].reset_index(drop=True)
    model_result = evaluate_registry(train, test)
    recovery2_metrics = pd.DataFrame()
    additional_choices: dict[str, Mapping[str, int]] = {}
    if config.params.get("stage") == "p1.1-r2":
        shrinkage = uncertainty_shrunk_atlas(
            train,
            test,
            confidence=float(config.params.get("shrinkage_confidence", 0.95)),
        )
        base_choices = _extra_choices(train, test)
        additional_choices = {
            "M4S": shrinkage.choices,
            "R1_SPEED_CHAMPION": base_choices["SPEED"],
        }
        shrinkage_diagnostic = pd.DataFrame(
            [
                {
                    "record_type": "recovery2_shrinkage",
                    "model_id": "M4S",
                    "raw_alpha": shrinkage.raw_alpha,
                    "active_alpha": shrinkage.active_alpha,
                    "training_improvement_lcb95": shrinkage.improvement_lcb95,
                    "distance_scale": shrinkage.distance_scale,
                    "m4_weight_min": float(shrinkage.predictions["m4_weight"].min()),
                    "m4_weight_max": float(shrinkage.predictions["m4_weight"].max()),
                    "fit_uses_evaluation_sigma": False,
                }
            ]
        )
        recovery2_metrics = pd.concat(
            (shrinkage.predictions, shrinkage_diagnostic), ignore_index=True, sort=False
        )
    regrets = _regret_table(
        episodes,
        surface,
        model_result,
        additional_choices=additional_choices,
    )
    ranges = test.groupby("context_id")["value"].agg(lambda values: values.max() - values.min())
    target_range = float(np.median(np.asarray(ranges, dtype=np.float64)))
    model_summary = _model_summary(regrets, target_range)
    m4 = model_summary.set_index("model_id").loc["M4"]
    comparisons = model_summary.set_index("model_id")
    required_models = ("M0", "M2", "M5", "FTT_LITE")
    criteria_pass = bool(
        target_range > float(config.params.get("flat_tolerance", 0.01))
        and float(cast(Any, m4["context_regret_ucb95"])) < 0.2 * target_range
        and all(
            float(cast(Any, comparisons.loc[name, "improvement_over_m4_lcb95"])) > 0
            for name in required_models
        )
    )

    expected_contexts = len(families) * len(reflexes) * len(speeds) * len(regimes)
    expected_episodes = expected_contexts * len(budgets) * (n_oracle + n_regret)
    expected_decompositions = expected_contexts * len(budgets) * n_regret
    test_pairs = {
        (int(row["speed_index"]), int(row["runtime_index"]))
        for row in cast(
            list[dict[str, Any]],
            contexts.loc[contexts["split"] == "test"].to_dict("records"),
        )
    }
    all_test_crossed = all(
        len(
            contexts.loc[
                (contexts["split"] == "test")
                & (contexts["family"] == family)
                & (contexts["reflex"] == reflex_name)
            ]
        )
        == len(test_pairs)
        for family in families
        for reflex_name in reflexes
    )
    finite_arms = decomposition_frame[
        ["J_actual", "J_fresh", "J_instant", "J_base", "J_instant_base"]
    ].notna().all(axis=None)
    identity_max = float(decomposition_frame["epsilon_id"].abs().max())
    unit_disjoint = not (
        set(episodes.loc[episodes["unit_role"] == "oracle", "unit_id"])
        & set(episodes.loc[episodes["unit_role"] == "regret", "unit_id"])
    )
    full_family_coverage = set(FAMILIES) <= set(families)
    region_pass, region_detail = _control_region_check()
    missing_families = sorted(set(FAMILIES) - set(families))
    dynamic_checks = (
        PrerequisiteCheck("whole context registry complete", len(contexts) == expected_contexts),
        PrerequisiteCheck("all held-out pair/family/reflex combinations", all_test_crossed),
        PrerequisiteCheck("oracle and regret units disjoint", unit_disjoint),
        PrerequisiteCheck("recurring rows complete", len(episodes) == expected_episodes),
        PrerequisiteCheck(
            "required handoff arms fail closed",
            len(decomposition_frame) == expected_decompositions and bool(finite_arms),
        ),
        PrerequisiteCheck(
            "episodewise sampled identity", identity_max < 1e-10, f"max={identity_max:.3e}"
        ),
        PrerequisiteCheck(
            "observed CPU latency positive",
            bool((episodes["observed_latency_s"] > 0).all()),
        ),
        PrerequisiteCheck(
            "randomized launch phase",
            episodes["launch_phase"].nunique() > max(n_oracle, n_regret),
        ),
        PrerequisiteCheck("lightweight FTT-style gate included", "FTT_LITE" in comparisons.index),
        PrerequisiteCheck(
            "continuous recoverable/unrecoverable regions",
            region_pass,
            region_detail,
        ),
        PrerequisiteCheck(
            "full frozen three-family coverage",
            full_family_coverage,
            f"implemented={families}; missing={missing_families}",
        ),
    )
    context_metrics = contexts.copy()
    surface_metrics = surface.copy()
    surface_metrics["record_type"] = "oracle_surface"
    prediction_metrics = model_result.predictions.copy()
    prediction_metrics["record_type"] = "model_prediction"
    metrics = pd.concat(
        (
            context_metrics,
            episodes,
            decomposition_frame,
            surface_metrics,
            prediction_metrics,
            recovery2_metrics,
            regrets,
            model_summary,
        ),
        ignore_index=True,
        sort=False,
    )
    return PilotArtifacts(
        metrics=metrics,
        contexts=contexts,
        model_result=model_result,
        model_summary=model_summary,
        controls=tuple((*initial_checks, *dynamic_checks)),
        full_family_coverage=full_family_coverage,
        criteria_pass=criteria_pass,
    )


def run_sampled(config: ProposalRunConfig, writer: ResultsWriter) -> VerdictReport:
    artifacts = generate_pilot(config)
    writer.write_metrics(artifacts.metrics)
    summary = artifacts.model_summary.set_index("model_id")
    m4 = summary.loc["M4"]
    hard_controls = [
        check
        for check in artifacts.controls
        if check.name != "full frozen three-family coverage"
    ]
    hard_pass = all(check.passed for check in hard_controls)
    if not hard_pass:
        status = Status.INCONCLUSIVE
        failure_code = FailureCode.F0_CODE_INVARIANT
        decision = "Repair failed P1.1 instrumentation/control before interpretation."
    elif not artifacts.criteria_pass:
        status = Status.INCONCLUSIVE
        failure_code = FailureCode.F1_UNIDENTIFIABLE
        decision = (
            "Use pilot-only recovery if frontiers are flat or intervals wide; do not inspect final "
            "units to tune the current experiment version."
        )
    elif not artifacts.full_family_coverage:
        status = Status.NARROW
        failure_code = None
        decision = (
            "Add the frozen continuous delayed-intervention family and a fresh experiment version "
            "before any P1.1 GO claim."
        )
    else:
        status = Status.GO
        failure_code = None
        decision = "Proceed to measured T4 transfer with the frozen model and thresholds."

    writer.event(
        stage="p1.1",
        contexts=len(artifacts.contexts),
        status=status.value,
        m4_context_regret_ucb95=float(cast(Any, m4["context_regret_ucb95"])),
        target_range=float(cast(Any, m4["target_range"])),
        full_family_coverage=artifacts.full_family_coverage,
    )
    return VerdictReport(
        question=config.preregistration.primary_hypothesis,
        status=status,
        evidence_level="E2",
        primary_result={
            "metric": "M4_context_regret_ucb95",
            "value": float(cast(Any, m4["context_regret_ucb95"])),
            "unit": "return",
        },
        baselines_run=[
            "M0 global fixed budget",
            "M1 raw budget curve",
            "M2 product collapse",
            "M3 separate axes",
            "M4 conditional descriptors",
            "M5 nearest-context lookup",
            "smallest and largest budgets",
            "state-independent speed schedule",
            "FTT-lite uncertainty gate",
            "hold, greedy, and distilled reflexes",
        ],
        controls=list(artifacts.controls),
        failure_code=failure_code,
        interpretation=(
            "A sampled CPU pilot over pursuit, Tetris-lite, and inertial delayed intervention. "
            "Operational recurring budget regret and the amendment-1.1 matched one-handoff "
            "decomposition are separate records. CPU evidence does not establish T4 transfer "
            "or a universal collapse law."
        ),
        non_claim=config.preregistration.non_claim,
        decision=decision,
    )


def resume_sampled(
    config: ProposalRunConfig,
    writer: ResultsWriter,
    checkpoint_state: Mapping[str, Any],
) -> VerdictReport:
    # The CPU pilot is short and its latency observations are part of the new
    # child run.  Rerunning is explicit; it never splices newly timed rows into
    # the interrupted parent's partially observed timing distribution.
    del checkpoint_state
    return run_sampled(config, writer)
