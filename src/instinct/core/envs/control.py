"""Continuous inertial control with delayed-intervention failure regions.

The state is a one-dimensional position and velocity inside two absorbing
boundaries.  Actions apply left, zero, or right acceleration.  A state can be
inside the legal interval but already dynamically unrecoverable: if its
stopping distance exceeds the remaining boundary margin, even maximum braking
cannot prevent failure.  This supplies P1.1 with both recoverable and
unrecoverable regions without importing a generic RL suite.

All noise is counter-addressed by ``(episode, tick, lane)``.  There is no
mutable generator, rejection sampling, or lane-position dependence, so sampled
counterfactual arms receive the same exogenous acceleration disturbance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from instinct.core.env import EnvState, StepResult
from instinct.core.rng import SeedScope

__all__ = ["InertialIntervention"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

# Action indices are public experiment semantics: brake/left, hold, drive/right.
ACCELERATIONS: FloatArray = np.array([-1.0, 0.0, 1.0], dtype=np.float64)


@dataclass(frozen=True, eq=False)
class InertialIntervention:
    """A noisy inertial point mass between absorbing failure boundaries."""

    boundary: float = 1.0
    target: float = 0.65
    dt: float = 0.12
    acceleration: float = 1.0
    drag: float = 0.985
    noise_std: float = 0.04
    failure_penalty: float = 2.0
    progress_scale: float = 2.0
    control_cost: float = 0.005
    name: str = "inertial_intervention"

    def __post_init__(self) -> None:
        if self.boundary <= 0 or not -self.boundary < self.target < self.boundary:
            raise ValueError("target must lie strictly inside a positive boundary")
        if self.dt <= 0 or self.acceleration <= 0:
            raise ValueError("dt and acceleration must be positive")
        if not 0 <= self.drag <= 1 or self.noise_std < 0:
            raise ValueError("drag must be in [0,1] and noise_std nonnegative")

    @property
    def n_actions(self) -> int:
        return 3

    def reset(self, lane_ids: IntArray, *, scope: SeedScope, episode: int = 0) -> EnvState:
        lanes = np.asarray(lane_ids, dtype=np.int64)
        draws = scope.stream("spawn").uniform(lanes, count=2, episode=episode, tick=0)
        # Starts span easy and high-momentum cases but remain inside the legal set.
        position = -0.65 + 0.30 * draws[:, 0]
        velocity = 0.05 + 0.75 * draws[:, 1]
        return EnvState(
            lane_ids=lanes,
            fields={
                "position": position.astype(np.float64),
                "velocity": velocity.astype(np.float64),
                "failed": np.zeros(len(lanes), dtype=bool),
            },
        )

    def stopping_distance(self, state: EnvState) -> FloatArray:
        velocity = state["velocity"].astype(np.float64)
        return np.square(velocity) / (2.0 * self.acceleration)

    def recovery_margin(self, state: EnvState) -> FloatArray:
        """Boundary margin minus deterministic maximum-braking distance."""
        position = state["position"].astype(np.float64)
        velocity = state["velocity"].astype(np.float64)
        boundary_margin = np.where(
            velocity >= 0,
            self.boundary - position,
            position + self.boundary,
        )
        return boundary_margin - self.stopping_distance(state)

    def recoverable(self, state: EnvState) -> BoolArray:
        failed = state["failed"].astype(bool)
        return ~failed & (self.recovery_margin(state) >= 0.0)

    def step(
        self,
        state: EnvState,
        actions: IntArray,
        *,
        scope: SeedScope,
        episode: int = 0,
        tick: int = 0,
    ) -> StepResult:
        position = state["position"].astype(np.float64)
        velocity = state["velocity"].astype(np.float64)
        failed = state["failed"].astype(bool)
        action = np.asarray(actions, dtype=np.int64)
        if action.shape != (state.n_lanes,) or np.any((action < 0) | (action >= self.n_actions)):
            raise ValueError("actions must contain one value in [0,3) per lane")

        disturbance = scope.stream("acceleration-noise").normal(
            state.lane_ids,
            count=1,
            episode=episode,
            tick=tick,
            scale=self.noise_std,
        )[:, 0]
        command = self.acceleration * ACCELERATIONS[action]
        proposed_velocity = (
            self.drag * velocity
            + self.dt * command
            + np.sqrt(self.dt) * disturbance
        )
        proposed_position = position + self.dt * proposed_velocity
        crossed = np.abs(proposed_position) >= self.boundary
        newly_failed = ~failed & crossed
        active = ~failed

        next_velocity = np.where(active, proposed_velocity, velocity)
        next_position = np.where(active, proposed_position, position)
        next_failed = failed | newly_failed
        progress = np.abs(position - self.target) - np.abs(next_position - self.target)
        reward = np.where(
            failed,
            0.0,
            self.progress_scale * progress
            - self.control_cost * np.abs(ACCELERATIONS[action])
            - self.failure_penalty * newly_failed,
        )
        return StepResult(
            state=state.replace_fields(
                position=next_position.astype(np.float64),
                velocity=next_velocity.astype(np.float64),
                failed=next_failed,
            ),
            reward=reward.astype(np.float64),
            done=next_failed,
        )
