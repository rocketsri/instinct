"""Batched environments, checked against single-lane references.

The headline test in this file is :func:`test_crn_subset_matches_full_batch`, in
its three per-environment forms. Everything downstream — paired arms, forked
rollouts, the oracle's batched evaluation — assumes that a lane's noise follows
its *identity* and not its position in whatever batch happens to be running. If
that assumption breaks, nothing crashes; the pairing quietly evaporates and the
estimators just get noisier. So it is tested directly, on every environment, with
subsets that reorder as well as drop lanes.

The naive references below are the rule-7 counterparts of the vectorized paths:
one lane, Python scalars, obvious control flow, never used outside tests. When a
fast path and its reference disagree, the fast path is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from instinct.core.env import BatchedEnv, EnvState
from instinct.core.envs.arcade import SnakeLite, TetrisLite
from instinct.core.envs.gridworld import GridPursuit, open_field, pillar_field
from instinct.core.rng import SeedScope

SCOPE = SeedScope(root_seed=20240816).child("test_envs")

_MOVES_PY = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
_DRIFTS_PY = _MOVES_PY[1:]
_SNAKE_D_PY = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_OPPOSITE_PY = [1, 0, 3, 2]
_PIECES_PY = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3), (4, 1), (1, 4)]


# -- naive single-lane references -----------------------------------------


def naive_move(env, pos, delta):
    r, c = pos[0] + delta[0], pos[1] + delta[1]
    cr, cc = min(max(r, 0), env.height - 1), min(max(c, 0), env.width - 1)
    if (cr, cc) != (r, c) or bool(env.walls[cr, cc]):
        return (int(pos[0]), int(pos[1]))
    return (cr, cc)


def naive_drift_probs(env, agent, target):
    if env.evasion == 0.0:
        return [0.25, 0.25, 0.25, 0.25]
    dr = target[0] - agent[0]
    dc = target[1] - agent[1]
    if dr == 0 and dc == 0:
        return [0.25, 0.25, 0.25, 0.25]
    # Indices into the drift table: 0 up, 1 down, 2 left, 3 right.
    away = (1 if dr >= 0 else 0) if abs(dr) >= abs(dc) else (3 if dc >= 0 else 2)
    return [(1.0 - env.evasion) / 4 + (env.evasion if i == away else 0.0) for i in range(4)]


def naive_grid_step(env, *, lane_id, agent, target, action, episode, tick, scope=SCOPE):
    u = scope.stream("world").uniform(np.array([lane_id]), count=3, episode=episode, tick=tick)[0]
    agent_next = naive_move(env, agent, _MOVES_PY[action])

    acc, cdf = 0.0, []
    for p in naive_drift_probs(env, agent, target):
        acc += p
        cdf.append(acc)
    direction = min(sum(1 for x in cdf if u[1] >= x), 3)
    delta = _DRIFTS_PY[direction] if u[0] < env.drift else (0, 0)
    target_next = naive_move(env, target, delta)

    caught = agent_next == target_next or (agent_next == target and target_next == agent)
    distance = abs(agent_next[0] - target_next[0]) + abs(agent_next[1] - target_next[1])
    reward = (
        env.catch_reward * caught
        - env.move_cost * (action != 0)
        - env.distance_weight * (0.0 if caught else distance / env.max_distance)
    )
    if caught:
        idx = min(int(u[2] * env.n_free), env.n_free - 1)
        target_next = (int(env.free_cells[idx, 0]), int(env.free_cells[idx, 1]))
    return {"agent": agent_next, "target": target_next}, reward, False


def naive_free_cell(occ, u, n_cells):
    free = [c for c in range(n_cells) if not (occ >> c) & 1]
    if not free:
        return 0
    return free[min(int(u * len(free)), len(free) - 1)]


def naive_snake_step(env, *, lane_id, fields, action, episode, tick, scope=SCOPE):
    cap = env.n_cells
    body = [int(x) for x in fields["body"]]
    ptr = int(fields["head_ptr"])
    length = int(fields["length"])
    heading = int(fields["heading"])
    food = int(fields["food"])
    occ = int(fields["occ"])
    dead = bool(fields["dead"])
    frozen = {
        "body": body,
        "head_ptr": ptr,
        "length": length,
        "heading": heading,
        "food": food,
        "occ": occ,
        "dead": True,
    }
    if dead:
        return frozen, 0.0, True

    new_heading = heading if (action == _OPPOSITE_PY[heading] and length > 1) else action
    head = body[ptr]
    row, col = divmod(head, env.width)
    nrow, ncol = row + _SNAKE_D_PY[new_heading][0], col + _SNAKE_D_PY[new_heading][1]
    off = not (0 <= nrow < env.height and 0 <= ncol < env.width)
    new_head = head if off else nrow * env.width + ncol

    ate = (not off) and new_head == food
    tail = body[(ptr - length + 1) % cap]
    occ_free = occ if ate else occ & ~(1 << tail)
    if off or ((occ_free >> new_head) & 1):
        return frozen, -env.step_cost - env.death_penalty, True

    body = list(body)
    ptr = (ptr + 1) % cap
    body[ptr] = new_head
    occ = occ_free | (1 << new_head)
    reward = -env.step_cost
    if ate:
        length += 1
        reward += env.food_reward
        u = scope.stream("food").uniform(
            np.array([lane_id]), count=1, episode=episode, tick=tick
        )[0, 0]
        food = naive_free_cell(occ, u, cap)
    return (
        {
            "body": body,
            "head_ptr": ptr,
            "length": length,
            "heading": new_heading,
            "food": food,
            "occ": occ,
            "dead": False,
        },
        reward,
        False,
    )


def naive_tetris_step(env, *, lane_id, heights, piece, dead, action, episode, tick, scope=SCOPE):
    heights = [int(h) for h in heights]
    if dead:
        return {"heights": heights, "piece": int(piece), "dead": True}, 0.0, True

    pw, ph = _PIECES_PY[int(piece)]
    col = min(action, env.width - pw)
    base = max(heights[col : col + pw])
    top = base + ph
    if top > env.ceiling:
        return (
            {"heights": heights, "piece": int(piece), "dead": True},
            -env.death_penalty,
            True,
        )

    placed = list(heights)
    for c in range(col, col + pw):
        placed[c] = top
    cleared = min(placed)
    placed = [h - cleared for h in placed]
    drawn = int(
        scope.stream("piece").integers(
            np.array([lane_id]), env.n_pieces, count=1, episode=episode, tick=tick
        )[0, 0]
    )
    reward = env.survive_reward + env.clear_reward * cleared
    return {"heights": placed, "piece": drawn, "dead": False}, reward, False


# -- fixtures / helpers ----------------------------------------------------


def grid_env():
    return open_field(size=7, drift=0.7, evasion=0.35)


def snake_env():
    return SnakeLite(height=6, width=6)


def tetris_env():
    return TetrisLite(width=6, ceiling=8)


ENVS = [
    ("grid_open", grid_env),
    ("grid_pillars", lambda: pillar_field(size=9, spacing=3, drift=0.6)),
    ("snake", snake_env),
    ("tetris", tetris_env),
]


def random_actions(env, n_lanes, rng):
    return rng.integers(0, env.n_actions, size=n_lanes).astype(np.int64)


def same_state(a: EnvState, b: EnvState) -> bool:
    if set(a.fields) != set(b.fields):
        return False
    return np.array_equal(a.lane_ids, b.lane_ids) and all(
        np.array_equal(a[k], b[k]) for k in a.fields
    )


# -- protocol / contracts --------------------------------------------------


@pytest.mark.parametrize("name,make", ENVS)
def test_satisfies_batched_env_protocol(name, make):
    env = make()
    assert isinstance(env, BatchedEnv)
    assert env.n_actions >= 2
    assert isinstance(env.name, str)


@pytest.mark.parametrize("name,make", ENVS)
def test_shapes_and_dtypes(name, make):
    env = make()
    lanes = np.arange(11, dtype=np.int64)
    state = env.reset(lanes, scope=SCOPE, episode=3)
    assert state.n_lanes == 11
    for key, arr in state.fields.items():
        assert arr.shape[0] == 11, key

    res = env.step(state, np.zeros(11, dtype=np.int64), scope=SCOPE, episode=3, tick=0)
    assert res.reward.shape == (11,)
    assert res.reward.dtype == np.float64
    assert res.done.shape == (11,)
    assert res.done.dtype == np.bool_
    assert np.array_equal(res.state.lane_ids, lanes)
    assert set(res.state.fields) == set(state.fields)
    for key in state.fields:
        assert res.state[key].dtype == state[key].dtype, key
        assert res.state[key].shape == state[key].shape, key


@pytest.mark.parametrize("name,make", ENVS)
def test_rejects_out_of_range_actions(name, make):
    env = make()
    state = env.reset(np.arange(4, dtype=np.int64), scope=SCOPE)
    for bad in (-1, env.n_actions):
        with pytest.raises(ValueError):
            env.step(state, np.full(4, bad, dtype=np.int64), scope=SCOPE)


# -- purity ----------------------------------------------------------------


@pytest.mark.parametrize("name,make", ENVS)
def test_step_is_pure(name, make):
    """The same call twice must give the same answer, and must not mutate input."""
    env = make()
    rng = np.random.default_rng(0)
    lanes = np.arange(16, dtype=np.int64)
    state = env.reset(lanes, scope=SCOPE, episode=1)
    for tick in range(6):
        actions = random_actions(env, 16, rng)
        before = state.copy()
        first = env.step(state, actions, scope=SCOPE, episode=1, tick=tick)
        second = env.step(state, actions, scope=SCOPE, episode=1, tick=tick)
        assert same_state(before, state), "step mutated the state it was given"
        assert same_state(first.state, second.state)
        assert np.array_equal(first.reward, second.reward)
        assert np.array_equal(first.done, second.done)
        state = first.state


@pytest.mark.parametrize("name,make", ENVS)
def test_reset_is_pure_and_episode_dependent(name, make):
    env = make()
    lanes = np.arange(32, dtype=np.int64)
    a = env.reset(lanes, scope=SCOPE, episode=0)
    b = env.reset(lanes, scope=SCOPE, episode=0)
    c = env.reset(lanes, scope=SCOPE, episode=1)
    assert same_state(a, b)
    # Different episodes must not hand every lane the same start, or the episode
    # axis would contribute no independent information at all.
    assert not all(np.array_equal(a[k], c[k]) for k in a.fields)


# -- CRN lane identity: the property everything else rests on --------------


@pytest.mark.parametrize("name,make", ENVS)
@pytest.mark.parametrize("subset", [[3, 1], [0], [7, 7, 2], [6, 5, 4, 3, 2, 1, 0]])
def test_crn_subset_matches_full_batch(name, make, subset):
    """Lanes must draw noise by identity, so any subset reproduces the full batch.

    Reordered and repeated subsets are included deliberately: a positional
    implementation passes a straight prefix and fails these.
    """
    env = make()
    rng = np.random.default_rng(11)
    lanes = np.arange(8, dtype=np.int64)
    full = env.reset(lanes, scope=SCOPE, episode=2)

    sub_idx = np.array(subset, dtype=np.int64)
    sub = full.take(sub_idx)
    assert np.array_equal(sub.lane_ids, lanes[sub_idx])

    for tick in range(8):
        actions = random_actions(env, 8, rng)
        full_res = env.step(full, actions, scope=SCOPE, episode=2, tick=tick)
        sub_res = env.step(sub, actions[sub_idx], scope=SCOPE, episode=2, tick=tick)
        for key in full.fields:
            assert np.array_equal(full_res.state[key][sub_idx], sub_res.state[key]), key
        assert np.array_equal(full_res.reward[sub_idx], sub_res.reward)
        assert np.array_equal(full_res.done[sub_idx], sub_res.done)
        full, sub = full_res.state, sub_res.state


@pytest.mark.parametrize("name,make", ENVS)
def test_reset_is_lane_identity_keyed(name, make):
    env = make()
    a = env.reset(np.array([4, 9, 1], dtype=np.int64), scope=SCOPE, episode=5)
    b = env.reset(np.array([9, 1, 4], dtype=np.int64), scope=SCOPE, episode=5)
    order = [1, 2, 0]
    for key in a.fields:
        assert np.array_equal(a[key][order], b[key]), key


@pytest.mark.parametrize("name,make", ENVS)
def test_scope_separates_streams(name, make):
    """A different scope path must change the noise; the same path must not."""
    env = make()
    lanes = np.arange(24, dtype=np.int64)
    other = SeedScope(root_seed=20240816).child("test_envs_other")
    same = SeedScope(root_seed=20240816).child("test_envs")
    a = env.reset(lanes, scope=SCOPE, episode=0)
    assert same_state(a, env.reset(lanes, scope=same, episode=0))
    b = env.reset(lanes, scope=other, episode=0)
    assert not all(np.array_equal(a[k], b[k]) for k in a.fields)


# -- batched vs naive (rule 7) ---------------------------------------------


def test_grid_matches_naive_reference():
    env = grid_env()
    rng = np.random.default_rng(3)
    lanes = np.array([0, 5, 12, 13, 40, 41, 42, 99], dtype=np.int64)
    state = env.reset(lanes, scope=SCOPE, episode=7)
    for tick in range(60):
        actions = random_actions(env, lanes.size, rng)
        res = env.step(state, actions, scope=SCOPE, episode=7, tick=tick)
        for i, lane in enumerate(lanes):
            want, reward, done = naive_grid_step(
                env,
                lane_id=int(lane),
                agent=(int(state["agent"][i, 0]), int(state["agent"][i, 1])),
                target=(int(state["target"][i, 0]), int(state["target"][i, 1])),
                action=int(actions[i]),
                episode=7,
                tick=tick,
            )
            assert tuple(res.state["agent"][i]) == want["agent"]
            assert tuple(res.state["target"][i]) == want["target"]
            assert res.reward[i] == reward
            assert bool(res.done[i]) is done
        state = res.state


def test_snake_matches_naive_reference():
    env = snake_env()
    rng = np.random.default_rng(4)
    lanes = np.array([2, 3, 17, 18, 60, 61, 200, 201], dtype=np.int64)
    state = env.reset(lanes, scope=SCOPE, episode=1)
    ate_at_least_once = False
    for tick in range(120):
        # Biased towards repeating a heading so lanes actually reach food and grow;
        # uniform actions on a 6x6 board mostly just die against a wall.
        actions = np.where(
            rng.random(lanes.size) < 0.7,
            state["heading"],
            random_actions(env, lanes.size, rng),
        ).astype(np.int64)
        res = env.step(state, actions, scope=SCOPE, episode=1, tick=tick)
        for i, lane in enumerate(lanes):
            fields = {k: state[k][i] for k in state.fields}
            want, reward, done = naive_snake_step(
                env, lane_id=int(lane), fields=fields, action=int(actions[i]), episode=1, tick=tick
            )
            ate_at_least_once |= reward > 0
            assert list(res.state["body"][i]) == want["body"]
            for key in ("head_ptr", "length", "heading", "food", "dead"):
                assert res.state[key][i] == want[key], key
            assert int(res.state["occ"][i]) == want["occ"]
            assert res.reward[i] == reward
            assert bool(res.done[i]) is done
        state = res.state
    assert ate_at_least_once, "reference comparison never exercised the eat/respawn path"


def test_tetris_matches_naive_reference():
    env = tetris_env()
    rng = np.random.default_rng(5)
    lanes = np.array([1, 2, 8, 9, 33, 34, 500, 501], dtype=np.int64)
    state = env.reset(lanes, scope=SCOPE, episode=4)
    saw_clear = saw_death = False
    for tick in range(80):
        actions = random_actions(env, lanes.size, rng)
        res = env.step(state, actions, scope=SCOPE, episode=4, tick=tick)
        for i, lane in enumerate(lanes):
            want, reward, done = naive_tetris_step(
                env,
                lane_id=int(lane),
                heights=state["heights"][i],
                piece=state["piece"][i],
                dead=bool(state["dead"][i]),
                action=int(actions[i]),
                episode=4,
                tick=tick,
            )
            saw_clear |= reward > env.survive_reward
            saw_death |= done and not bool(state["dead"][i])
            assert list(res.state["heights"][i]) == want["heights"]
            assert int(res.state["piece"][i]) == want["piece"]
            assert bool(res.state["dead"][i]) is want["dead"]
            assert res.reward[i] == reward
            assert bool(res.done[i]) is done
        state = res.state
    assert saw_clear and saw_death, "reference comparison missed clears or top-outs"


# -- absorption ------------------------------------------------------------


@pytest.mark.parametrize("name,make", [("snake", snake_env), ("tetris", tetris_env)])
def test_terminal_states_absorb(name, make):
    """Once done, the state is frozen bit-for-bit and no further reward accrues."""
    env = make()
    rng = np.random.default_rng(6)
    lanes = np.arange(64, dtype=np.int64)
    state = env.reset(lanes, scope=SCOPE, episode=0)
    dead = np.zeros(64, dtype=bool)
    for tick in range(200):
        actions = random_actions(env, 64, rng)
        res = env.step(state, actions, scope=SCOPE, episode=0, tick=tick)
        for key in state.fields:
            assert np.array_equal(res.state[key][dead], state[key][dead]), key
        assert np.all(res.reward[dead] == 0.0)
        assert np.all(res.done[dead])
        dead = res.done.copy()
        state = res.state
        if dead.all():
            break
    assert dead.any(), "no lane ever terminated, so absorption was never exercised"


def test_gridworld_has_no_terminal_states():
    """The smooth-decay environment is recoverable everywhere, on purpose.

    ``L_irreversible`` must measure as exactly zero here; that is what makes it
    the control against the arcade games' non-zero one.
    """
    env = grid_env()
    rng = np.random.default_rng(7)
    state = env.reset(np.arange(32, dtype=np.int64), scope=SCOPE)
    for tick in range(50):
        res = env.step(state, random_actions(env, 32, rng), scope=SCOPE, tick=tick)
        assert not res.done.any()
        state = res.state


# -- invariants ------------------------------------------------------------


@pytest.mark.parametrize("make", [grid_env, lambda: pillar_field(size=9, spacing=3)])
def test_grid_positions_stay_legal(make):
    env = make()
    rng = np.random.default_rng(8)
    state = env.reset(np.arange(48, dtype=np.int64), scope=SCOPE)
    for tick in range(60):
        for who in ("agent", "target"):
            pos = state[who]
            assert pos.shape == (48, 2)
            assert np.all(pos[:, 0] >= 0) and np.all(pos[:, 0] < env.height)
            assert np.all(pos[:, 1] >= 0) and np.all(pos[:, 1] < env.width)
            assert not env.walls[pos[:, 0], pos[:, 1]].any()
        state = env.step(state, random_actions(env, 48, rng), scope=SCOPE, tick=tick).state


def test_grid_agent_moves_at_most_one_cell():
    """No teleporting agents. The target is exempt: a catch respawns it anywhere."""
    env = grid_env()
    rng = np.random.default_rng(9)
    state = env.reset(np.arange(48, dtype=np.int64), scope=SCOPE)
    for tick in range(40):
        nxt = env.step(state, random_actions(env, 48, rng), scope=SCOPE, tick=tick).state
        assert np.all(np.abs(nxt["agent"] - state["agent"]).sum(axis=1) <= 1)
        state = nxt


def test_snake_invariants_hold():
    env = snake_env()
    rng = np.random.default_rng(10)
    lanes = np.arange(64, dtype=np.int64)
    state = env.reset(lanes, scope=SCOPE)
    for tick in range(80):
        occ = state["occ"].astype(np.uint64)
        length = state["length"]
        popcount = np.array([bin(int(o)).count("1") for o in occ])
        # The bitboard is a derived view of the ring buffer; if they ever
        # disagree the collision test is reading a board that does not exist.
        assert np.array_equal(popcount, length)
        assert np.all(state["body"] >= 0) and np.all(state["body"] < env.n_cells)
        assert np.all(state["food"] >= 0) and np.all(state["food"] < env.n_cells)
        food_bit = (occ >> state["food"].astype(np.uint64)) & np.uint64(1)
        assert np.all(food_bit[~state["dead"]] == 0), "food spawned inside the snake"
        for i in range(len(lanes)):
            ptr, n = int(state["head_ptr"][i]), int(length[i])
            cells = [int(state["body"][i, (ptr - j) % env.n_cells]) for j in range(n)]
            assert len(set(cells)) == n, "ring buffer holds a duplicated cell"
        state = env.step(state, random_actions(env, 64, rng), scope=SCOPE, tick=tick).state


def test_tetris_heights_stay_in_range():
    env = tetris_env()
    rng = np.random.default_rng(12)
    state = env.reset(np.arange(64, dtype=np.int64), scope=SCOPE)
    for tick in range(120):
        assert np.all(state["heights"] >= 0)
        assert np.all(state["heights"] <= env.ceiling)
        assert np.all(state["piece"] >= 0) and np.all(state["piece"] < env.n_pieces)
        state = env.step(state, random_actions(env, 64, rng), scope=SCOPE, tick=tick).state


def test_tetris_clamps_every_action_into_a_legal_placement():
    """Every action in [0, width) must place a piece, never raise, never overhang."""
    env = TetrisLite(width=5, ceiling=9)
    lanes = np.arange(env.n_pieces, dtype=np.int64)
    state = EnvState(
        lane_ids=lanes,
        fields={
            "heights": np.zeros((env.n_pieces, env.width), dtype=np.int64),
            "piece": np.arange(env.n_pieces, dtype=np.int64),
            "dead": np.zeros(env.n_pieces, dtype=bool),
        },
    )
    for action in range(env.n_actions):
        res = env.step(state, np.full(env.n_pieces, action, dtype=np.int64), scope=SCOPE)
        filled = (res.state["heights"] > 0).sum(axis=1)
        assert np.all(filled >= 1)
        assert not res.done.any()


# -- behavioural sanity ----------------------------------------------------


def test_target_actually_drifts():
    env = open_field(size=9, drift=1.0)
    lanes = np.arange(256, dtype=np.int64)
    state = env.reset(lanes, scope=SCOPE, episode=0)
    start = state["target"].copy()
    for tick in range(6):
        # STAY throughout, so any target motion is drift and not a catch/respawn.
        state = env.step(state, np.zeros(256, dtype=np.int64), scope=SCOPE, tick=tick).state
    moved = np.any(state["target"] != start, axis=1)
    assert moved.mean() > 0.8


def test_target_holds_still_when_drift_is_zero():
    env = open_field(size=9, drift=0.0)
    state = env.reset(np.arange(64, dtype=np.int64), scope=SCOPE)
    start = state["target"].copy()
    for tick in range(10):
        res = env.step(state, np.zeros(64, dtype=np.int64), scope=SCOPE, tick=tick)
        # Only a catch can move a stationary target, and the agent is holding still.
        assert np.array_equal(res.state["target"], start)
        state = res.state


def test_evasion_pushes_the_target_away():
    """A paired comparison: same lanes, same seeds, only `evasion` differs."""
    lanes = np.arange(512, dtype=np.int64)
    gaps = []
    for evasion in (0.0, 0.9):
        env = open_field(size=11, drift=1.0, evasion=evasion)
        state = env.reset(lanes, scope=SCOPE, episode=0)
        for tick in range(8):
            state = env.step(state, np.zeros(512, dtype=np.int64), scope=SCOPE, tick=tick).state
        gaps.append(np.abs(state["agent"] - state["target"]).sum(axis=1).mean())
    assert gaps[1] > gaps[0]


def test_catching_the_target_pays_and_respawns_it():
    env = open_field(size=5, drift=0.0)
    lanes = np.arange(8, dtype=np.int64)
    state = EnvState(
        lane_ids=lanes,
        fields={
            "agent": np.tile(np.array([2, 1]), (8, 1)),
            "target": np.tile(np.array([2, 2]), (8, 1)),
        },
    )
    res = env.step(state, np.full(8, 4, dtype=np.int64), scope=SCOPE)  # RIGHT, onto the target
    assert np.all(res.state["agent"] == np.array([2, 2]))
    assert np.all(res.reward > 0)
    assert np.all(np.any(res.state["target"] != np.array([2, 2]), axis=1))


def test_walking_away_costs_more_than_closing_in():
    env = open_field(size=7, drift=0.0, evasion=0.0)
    lanes = np.arange(4, dtype=np.int64)
    fields = {
        "agent": np.tile(np.array([3, 1]), (4, 1)),
        "target": np.tile(np.array([3, 5]), (4, 1)),
    }
    state = EnvState(lane_ids=lanes, fields=fields)
    closer = env.step(state, np.full(4, 4, dtype=np.int64), scope=SCOPE).reward
    further = env.step(state, np.full(4, 3, dtype=np.int64), scope=SCOPE).reward
    assert np.all(closer > further)


def _coiled_snake(env, lanes, cells, heading):
    """A hand-built snake whose body is ``cells`` from tail to head."""
    n = lanes.size
    body = np.zeros((n, env.n_cells), dtype=np.int16)
    occ = np.zeros(n, dtype=np.uint64)
    flat = [r * env.width + c for r, c in cells]
    for j, cell in enumerate(flat):
        body[:, j] = cell
        occ |= np.uint64(1) << np.uint64(cell)
    food = next(c for c in range(env.n_cells) if c not in flat)
    return EnvState(
        lane_ids=lanes,
        fields={
            "body": body,
            "head_ptr": np.full(n, len(flat) - 1, dtype=np.int64),
            "length": np.full(n, len(flat), dtype=np.int64),
            "heading": np.full(n, heading, dtype=np.int64),
            "food": np.full(n, food, dtype=np.int64),
            "occ": occ,
            "dead": np.zeros(n, dtype=bool),
        },
    )


def test_snake_dies_on_itself_but_may_chase_its_tail():
    env = snake_env()
    lanes = np.arange(3, dtype=np.int64)
    # Tail .. head, heading LEFT (2) because the head stepped from (3,3) to (3,2).
    coil = _coiled_snake(env, lanes, [(2, 2), (2, 3), (3, 3), (3, 2)], heading=2)

    into_body = env.step(coil, np.zeros(3, dtype=np.int64), scope=SCOPE)  # UP onto (2,2)?
    # (2,2) is the *tail*, which vacates this tick, so this move is legal.
    assert not into_body.done.any()
    assert np.all(into_body.state["length"] == 4)

    # Now a coil where the cell ahead is body but not the tail.
    coil2 = _coiled_snake(env, lanes, [(1, 2), (2, 2), (2, 3), (3, 3), (3, 2)], heading=2)
    res = env.step(coil2, np.zeros(3, dtype=np.int64), scope=SCOPE)  # UP onto (2,2)
    assert res.done.all()
    assert np.all(res.reward < 0)
    # And it stays dead, frozen, and rewardless.
    after = env.step(res.state, np.ones(3, dtype=np.int64), scope=SCOPE, tick=1)
    assert same_state(after.state, res.state)
    assert np.all(after.reward == 0.0)


def test_snake_dies_on_a_wall():
    env = snake_env()
    lanes = np.arange(4, dtype=np.int64)
    state = _coiled_snake(env, lanes, [(0, 1), (0, 0)], heading=2)  # head at (0,0), facing LEFT
    res = env.step(state, np.zeros(4, dtype=np.int64), scope=SCOPE)  # UP, off the board
    assert res.done.all()
    assert np.all(res.reward == -env.step_cost - env.death_penalty)
    assert np.array_equal(res.state["body"], state["body"])


def test_snake_eating_grows_and_pays():
    env = snake_env()
    lanes = np.arange(4, dtype=np.int64)
    state = _coiled_snake(env, lanes, [(2, 1), (2, 2)], heading=3)  # head (2,2), facing RIGHT
    state = state.replace_fields(food=np.full(4, 2 * env.width + 3, dtype=np.int64))
    res = env.step(state, np.full(4, 3, dtype=np.int64), scope=SCOPE)  # RIGHT onto the food
    assert np.all(res.state["length"] == 3)
    assert np.all(res.reward == env.food_reward - env.step_cost)
    assert np.all(res.state["food"] != 2 * env.width + 3)


def test_tetris_tops_out_and_the_penalty_is_the_only_negative():
    env = TetrisLite(width=4, ceiling=4)
    lanes = np.arange(6, dtype=np.int64)
    state = EnvState(
        lane_ids=lanes,
        fields={
            # Three columns are at the ceiling; only column 0 has room, and only
            # for a piece one unit tall.
            "heights": np.tile(np.array([3, 4, 4, 4]), (6, 1)),
            "piece": np.zeros(6, dtype=np.int64),  # the 1x1
            "dead": np.zeros(6, dtype=bool),
        },
    )
    ok = env.step(state, np.zeros(6, dtype=np.int64), scope=SCOPE)
    assert not ok.done.any()
    # That completes the shortest column, so four rows clear at once.
    assert np.all(ok.state["heights"] == 0)
    assert np.all(ok.reward == env.survive_reward + 4 * env.clear_reward)

    tall = state.replace_fields(piece=np.full(6, 2, dtype=np.int64))  # the 1x2
    dead = env.step(tall, np.zeros(6, dtype=np.int64), scope=SCOPE)
    assert dead.done.all()
    assert np.all(dead.reward == -env.death_penalty)
    assert np.array_equal(dead.state["heights"], tall["heights"])


def test_tetris_survival_reward_is_positive_until_it_is_not():
    """The threshold shape, stated as a test: flat, then a cliff."""
    env = TetrisLite(width=6, ceiling=10)
    rng = np.random.default_rng(13)
    state = env.reset(np.arange(128, dtype=np.int64), scope=SCOPE)
    rewards = []
    for tick in range(60):
        res = env.step(state, random_actions(env, 128, rng), scope=SCOPE, tick=tick)
        live = ~state["dead"]
        if live.any():
            rewards.append(res.reward[live])
        state = res.state
    flat = np.concatenate(rewards)
    assert (flat > 0).mean() > 0.5
    assert (flat <= -env.death_penalty).any()


# -- construction guards ---------------------------------------------------


def test_boards_larger_than_the_bitboard_are_refused():
    with pytest.raises(ValueError):
        SnakeLite(height=9, width=9)


def test_grid_rejects_degenerate_configurations():
    with pytest.raises(ValueError):
        GridPursuit(walls=np.ones((4, 4), dtype=bool))
    with pytest.raises(ValueError):
        GridPursuit(walls=np.zeros((4, 4), dtype=bool), drift=1.5)
    with pytest.raises(ValueError):
        GridPursuit(walls=np.zeros(4, dtype=bool))


def test_tetris_rejects_boards_the_pieces_do_not_fit():
    with pytest.raises(ValueError):
        TetrisLite(width=3)
    with pytest.raises(ValueError):
        TetrisLite(ceiling=2)
