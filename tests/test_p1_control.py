from __future__ import annotations

import numpy as np

from instinct.core.env import EnvState
from instinct.core.envs.control import ACCELERATIONS, InertialIntervention
from instinct.core.rng import SeedScope


def _naive_step(
    env: InertialIntervention,
    state: EnvState,
    actions: np.ndarray,
    *,
    scope: SeedScope,
    episode: int,
    tick: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions: list[float] = []
    velocities: list[float] = []
    failures: list[bool] = []
    rewards: list[float] = []
    dones: list[bool] = []
    stream = scope.stream("acceleration-noise")
    for lane_index, lane_id in enumerate(state.lane_ids):
        position = float(state["position"][lane_index])
        velocity = float(state["velocity"][lane_index])
        failed = bool(state["failed"][lane_index])
        action = int(actions[lane_index])
        noise = float(
            stream.normal(
                np.array([lane_id], dtype=np.int64),
                count=1,
                episode=episode,
                tick=tick,
                scale=env.noise_std,
            )[0, 0]
        )
        proposed_velocity = (
            env.drag * velocity
            + env.dt * env.acceleration * float(ACCELERATIONS[action])
            + np.sqrt(env.dt) * noise
        )
        proposed_position = position + env.dt * proposed_velocity
        crossed = abs(proposed_position) >= env.boundary
        newly_failed = not failed and crossed
        next_velocity = velocity if failed else proposed_velocity
        next_position = position if failed else proposed_position
        progress = abs(position - env.target) - abs(next_position - env.target)
        reward = (
            0.0
            if failed
            else env.progress_scale * progress
            - env.control_cost * abs(float(ACCELERATIONS[action]))
            - env.failure_penalty * float(newly_failed)
        )
        positions.append(next_position)
        velocities.append(next_velocity)
        failures.append(failed or newly_failed)
        rewards.append(reward)
        dones.append(failed or newly_failed)
    return (
        np.asarray(positions),
        np.asarray(velocities),
        np.asarray(failures),
        np.asarray(rewards),
        np.asarray(dones),
    )


def test_control_vectorized_step_matches_naive_scalar_reference() -> None:
    env = InertialIntervention(noise_std=0.07)
    scope = SeedScope(991).child("equivalence")
    state = EnvState(
        lane_ids=np.array([19, 3, 71, 8], dtype=np.int64),
        fields={
            "position": np.array([-0.4, 0.3, 0.97, -0.99]),
            "velocity": np.array([0.5, -0.3, 0.8, -0.5]),
            "failed": np.array([False, False, False, True]),
        },
    )
    actions = np.array([0, 2, 2, 1], dtype=np.int64)
    expected = _naive_step(env, state, actions, scope=scope, episode=4, tick=11)
    actual = env.step(state, actions, scope=scope, episode=4, tick=11)

    np.testing.assert_allclose(
        np.asarray(actual.state["position"], dtype=np.float64), expected[0]
    )
    np.testing.assert_allclose(
        np.asarray(actual.state["velocity"], dtype=np.float64), expected[1]
    )
    np.testing.assert_array_equal(actual.state["failed"], expected[2])
    np.testing.assert_allclose(actual.reward, expected[3])
    np.testing.assert_array_equal(actual.done, expected[4])


def test_control_counter_randomness_survives_lane_subset_and_reorder() -> None:
    env = InertialIntervention()
    scope = SeedScope(812).child("common-random-numbers")
    state = env.reset(np.array([7, 12, 29, 44], dtype=np.int64), scope=scope, episode=2)
    actions = np.array([2, 0, 1, 2], dtype=np.int64)
    full = env.step(state, actions, scope=scope, episode=2, tick=9)
    selected = np.array([3, 0, 2], dtype=np.int64)
    subset = env.step(
        state.take(selected), actions[selected], scope=scope, episode=2, tick=9
    )

    for field in ("position", "velocity", "failed"):
        np.testing.assert_array_equal(subset.state[field], full.state[field][selected])
    np.testing.assert_array_equal(subset.reward, full.reward[selected])
    np.testing.assert_array_equal(subset.done, full.done[selected])


def test_control_has_recoverable_and_unrecoverable_legal_regions() -> None:
    env = InertialIntervention(noise_std=0.0)
    state = EnvState(
        lane_ids=np.array([0, 1], dtype=np.int64),
        fields={
            "position": np.array([0.50, 0.85]),
            "velocity": np.array([0.20, 0.80]),
            "failed": np.array([False, False]),
        },
    )
    assert np.all(np.abs(state["position"]) < env.boundary)
    np.testing.assert_array_equal(env.recoverable(state), np.array([True, False]))
    assert env.recovery_margin(state)[0] > 0 > env.recovery_margin(state)[1]


def test_control_failure_is_absorbing() -> None:
    env = InertialIntervention(noise_std=0.0)
    scope = SeedScope(4)
    state = EnvState(
        lane_ids=np.array([5], dtype=np.int64),
        fields={
            "position": np.array([0.99]),
            "velocity": np.array([1.0]),
            "failed": np.array([False]),
        },
    )
    failed = env.step(state, np.array([2]), scope=scope, tick=0)
    assert failed.done[0] and failed.state["failed"][0]
    frozen = env.step(failed.state, np.array([0]), scope=scope, tick=1)
    np.testing.assert_array_equal(frozen.state["position"], failed.state["position"])
    np.testing.assert_array_equal(frozen.state["velocity"], failed.state["velocity"])
    np.testing.assert_array_equal(frozen.reward, np.array([0.0]))
    np.testing.assert_array_equal(frozen.done, np.array([True]))
