"""Exact 2-D inertial recovery environment for the P4 mechanism gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from instinct.core.mdp import TabularMDP

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

ACCELERATIONS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)


@dataclass(frozen=True, slots=True)
class InertialGrid:
    """Finite MDP plus the coordinate system used to audit recovery states."""

    mdp: TabularMDP
    states: tuple[tuple[int, int, int, int] | None, ...]
    state_index: dict[tuple[int, int, int, int], int]
    width: int
    height: int
    goal: tuple[int, int]
    obstacles: frozenset[tuple[int, int]]
    goal_state: int
    failure_state: int

    def impulse_kernel(self, impulses: tuple[tuple[int, int], ...]) -> FloatArray:
        """Map post-prefix states through each declared velocity impulse."""

        if not impulses:
            raise ValueError("at least one impulse is required")
        kernel = np.zeros((len(impulses), self.mdp.n_states, self.mdp.n_states))
        for disturbance, (dvx, dvy) in enumerate(impulses):
            for state, encoding in enumerate(self.states):
                if encoding is None:
                    kernel[disturbance, state, state] = 1.0
                    continue
                x, y, vx, vy = encoding
                perturbed = (x, y, int(np.clip(vx + dvx, -2, 2)), int(np.clip(vy + dvy, -2, 2)))
                kernel[disturbance, state, self.state_index[perturbed]] = 1.0
        return kernel

    def progress(self) -> FloatArray:
        """Normalized Manhattan progress, with terminal conventions explicit."""

        origin_distance = max(self.goal[0] + self.goal[1], 1)
        out = np.zeros(self.mdp.n_states, dtype=np.float64)
        for state, encoding in enumerate(self.states):
            if encoding is not None:
                x, y, _, _ = encoding
                remaining = abs(self.goal[0] - x) + abs(self.goal[1] - y)
                out[state] = 1.0 - remaining / origin_distance
        out[self.goal_state] = 1.0
        out[self.failure_state] = 0.0
        return out


def inertial_grid(
    *,
    width: int = 4,
    height: int = 4,
    obstacles: tuple[tuple[int, int], ...] = ((1, 1), (2, 1)),
    goal: tuple[int, int] | None = None,
    gamma: float = 0.95,
    actuator_slip: float = 0.1,
) -> InertialGrid:
    """Build a small exact navigation MDP with momentum and collision failure.

    A command changes each velocity component by at most one, while velocity is
    clipped to ``[-2, 2]``. Consequently some legal boundary states are already
    unrecoverable: even maximum braking cannot prevent the next collision.
    ``actuator_slip`` means the requested acceleration is ignored, preserving a
    realistic distinction between certain and probabilistic recovery.
    """

    if width < 3 or height < 3:
        raise ValueError("inertial grid dimensions must be at least three")
    if not 0.0 <= actuator_slip < 1.0:
        raise ValueError("actuator_slip must lie in [0, 1)")
    target = (width - 1, height - 1) if goal is None else goal
    blocked = frozenset((int(x), int(y)) for x, y in obstacles)
    if target in blocked or not (0 <= target[0] < width and 0 <= target[1] < height):
        raise ValueError("goal must be a legal non-obstacle cell")
    if (0, 0) in blocked:
        raise ValueError("origin cannot be an obstacle")

    encodings = tuple(
        (x, y, vx, vy)
        for y in range(height)
        for x in range(width)
        if (x, y) not in blocked and (x, y) != target
        for vy in range(-2, 3)
        for vx in range(-2, 3)
    )
    state_index = {encoding: index for index, encoding in enumerate(encodings)}
    goal_state = len(encodings)
    failure_state = goal_state + 1
    states: tuple[tuple[int, int, int, int] | None, ...] = (*encodings, None, None)
    n_states = len(states)
    n_actions = len(ACCELERATIONS)
    transition = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    reward = np.zeros((n_states, n_actions), dtype=np.float64)

    def successor(encoding: tuple[int, int, int, int], acceleration: tuple[int, int]) -> int:
        x, y, vx, vy = encoding
        ax, ay = acceleration
        next_vx = int(np.clip(vx + ax, -2, 2))
        next_vy = int(np.clip(vy + ay, -2, 2))
        next_position = (x + next_vx, y + next_vy)
        if (
            not 0 <= next_position[0] < width
            or not 0 <= next_position[1] < height
            or next_position in blocked
        ):
            return failure_state
        if next_position == target:
            return goal_state
        return state_index[(*next_position, next_vx, next_vy)]

    for state, encoding in enumerate(encodings):
        for action, acceleration in enumerate(ACCELERATIONS):
            acted = successor(encoding, acceleration)
            slipped = successor(encoding, (0, 0))
            transition[state, action, acted] += 1.0 - actuator_slip
            transition[state, action, slipped] += actuator_slip
            reward[state, action] = (
                -0.01
                + transition[state, action, goal_state]
                - transition[state, action, failure_state]
            )
    for terminal_state in (goal_state, failure_state):
        transition[terminal_state, :, terminal_state] = 1.0

    terminal_mask = np.zeros(n_states, dtype=bool)
    terminal_mask[[goal_state, failure_state]] = True
    failure = np.zeros(n_states, dtype=bool)
    failure[failure_state] = True
    mdp = TabularMDP(
        P=transition,
        R=reward,
        gamma=gamma,
        terminal=terminal_mask,
        name="inertial_grid",
        failure=failure,
    )
    mdp.validate()
    return InertialGrid(
        mdp=mdp,
        states=states,
        state_index=state_index,
        width=width,
        height=height,
        goal=target,
        obstacles=blocked,
        goal_state=goal_state,
        failure_state=failure_state,
    )


def impulse_recovery_values(
    grid: InertialGrid,
    value: FloatArray,
    impulses: tuple[tuple[int, int], ...],
) -> FloatArray:
    """Return ``(state, disturbance)`` exact post-impulse replanning values."""

    values = np.asarray(value, dtype=np.float64)
    if values.shape != (grid.mdp.n_states,):
        raise ValueError("value vector does not match inertial grid")
    kernel = grid.impulse_kernel(impulses)
    return np.einsum("dij,j->id", kernel, values)


def one_step_failure_floor(grid: InertialGrid) -> FloatArray:
    """Minimum immediate collision probability under the best braking action."""

    return grid.mdp.P[:, :, grid.failure_state].min(axis=1)


def braking_policy(grid: InertialGrid) -> IntArray:
    """A conservative reflex that removes the largest velocity component."""

    policy = np.zeros(grid.mdp.n_states, dtype=np.int64)
    acceleration_index = {value: index for index, value in enumerate(ACCELERATIONS)}
    for state, encoding in enumerate(grid.states):
        if encoding is None:
            continue
        _, _, vx, vy = encoding
        if abs(vx) >= abs(vy) and vx:
            acceleration = (-int(np.sign(vx)), 0)
        elif vy:
            acceleration = (0, -int(np.sign(vy)))
        else:
            acceleration = (0, 0)
        policy[state] = acceleration_index[acceleration]
    return policy


def greedy_progress_policy(grid: InertialGrid) -> IntArray:
    """Myopic goal acceleration that ignores collision and momentum."""

    policy = np.zeros(grid.mdp.n_states, dtype=np.int64)
    acceleration_index = {value: index for index, value in enumerate(ACCELERATIONS)}
    for state, encoding in enumerate(grid.states):
        if encoding is None:
            continue
        x, y, _, _ = encoding
        dx, dy = grid.goal[0] - x, grid.goal[1] - y
        if abs(dx) >= abs(dy) and dx:
            acceleration = (int(np.sign(dx)), 0)
        elif dy:
            acceleration = (0, int(np.sign(dy)))
        else:
            acceleration = (0, 0)
        policy[state] = acceleration_index[acceleration]
    return policy


def recovery_policy(grid: InertialGrid, value: FloatArray) -> IntArray:
    """Greedy exact-replanning policy for a frozen finite-horizon value."""

    values = np.asarray(value, dtype=np.float64)
    if values.shape != (grid.mdp.n_states,):
        raise ValueError("value vector does not match inertial grid")
    q_value = grid.mdp.R + grid.mdp.gamma * np.einsum("sat,t->sa", grid.mdp.P, values)
    return np.argmax(q_value, axis=1).astype(np.int64)


def evaluate_impulse_policy(
    grid: InertialGrid,
    policy: IntArray,
    *,
    impulses: tuple[tuple[int, int], ...],
    horizon: int,
    start: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> dict[str, float]:
    """Exact distributional evaluation under a uniform post-step impulse set."""

    actions = np.asarray(policy, dtype=np.int64)
    if actions.shape != (grid.mdp.n_states,) or np.any(
        (actions < 0) | (actions >= grid.mdp.n_actions)
    ):
        raise ValueError("policy must provide one legal action per state")
    if horizon < 1 or start not in grid.state_index:
        raise ValueError("evaluation requires a positive horizon and legal start")
    disturbance = grid.impulse_kernel(impulses).mean(axis=0)
    controlled = grid.mdp.P[np.arange(grid.mdp.n_states), actions]
    transition = controlled @ disturbance
    distribution = np.zeros(grid.mdp.n_states, dtype=np.float64)
    distribution[grid.state_index[start]] = 1.0
    discounted_return = 0.0
    discount = 1.0
    policy_reward = grid.mdp.R[np.arange(grid.mdp.n_states), actions]
    for _ in range(horizon):
        discounted_return += discount * float(distribution @ policy_reward)
        distribution = distribution @ transition
        discount *= grid.mdp.gamma
    return {
        "return": discounted_return,
        "goal_probability": float(distribution[grid.goal_state]),
        "failure_probability": float(distribution[grid.failure_state]),
        "survival_probability": float(1.0 - distribution[grid.failure_state]),
        "progress": float(distribution @ grid.progress()),
    }
