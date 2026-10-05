"""Passive discovery and fruit-spawn measurements."""

import numpy as np
from snake import evaluation as ev


class SearchTracker:
    def __init__(self, env, horizon):
        self.env, self.horizon, self.step = env, horizon, 0
        self.n = env.n_boards
        self.ids = np.arange(self.n)
        self.coords = np.indices((env.board_size, env.board_size))
        self.birth = np.zeros(self.n, dtype=int)
        self.first_seen = np.full(self.n, -1, dtype=int)
        self.first_distance = np.zeros(self.n, dtype=int)
        self.initially_hidden = np.zeros(self.n, dtype=bool)
        self.lost_current = np.zeros(self.n, dtype=bool)
        self.sums = {
            key: np.zeros(self.n)
            for key in (
                "search_actions",
                "post_sight_actions",
                "invisible_actions",
                "visible_actions",
                "visible_reducing_actions",
                "visibility_losses",
                "spawn_count",
                "hidden_spawns",
                "expected_visible_spawns",
                "hidden_spawn_distance",
                "hidden_discoveries",
                "discovery_wait",
                "completed_count",
                "completed_search",
                "completed_pursuit",
                "completed_sight_distance",
                "completed_excess",
                "completed_lost_sight",
            )
        }
        self.spawn(self.ids)
        self.initial_hidden = self.initially_hidden.copy()
        self.initial_wait = np.full(self.n, horizon, dtype=int)
        heads, fruits, _ = self.positions()
        self.initial_distance = np.abs(heads - fruits).sum(axis=1)
        original_state, original_move = env.to_state, env.move

        def state():
            observation = original_state()
            self.observed_visible = np.any(observation[..., 1] > 0, axis=(1, 2))
            return observation

        def move(actions):
            heads, fruits, visible = self.positions()
            assert np.array_equal(visible, self.observed_visible)
            distance = np.abs(heads - fruits).sum(axis=1)
            first = visible & (self.first_seen < 0)
            initial_sighting = first & (self.sums["completed_count"] == 0)
            self.initial_wait[initial_sighting] = self.step
            discovered = first & self.initially_hidden
            self.first_seen[first] = self.step
            self.first_distance[first] = distance[first]
            self.sums["hidden_discoveries"][discovered] += 1
            self.sums["discovery_wait"][discovered] += (
                self.step - self.birth[discovered]
            )
            self.sums["search_actions"] += self.first_seen < 0
            self.sums["post_sight_actions"] += self.first_seen >= 0
            self.sums["invisible_actions"] += ~visible
            self.sums["visible_actions"] += visible
            target = heads + ev.OFFSETS[actions[:, 0]]
            reducing = np.abs(target - fruits).sum(axis=1) < distance
            self.sums["visible_reducing_actions"] += visible & reducing
            eaten = np.flatnonzero(
                self.env.boards[self.ids, target[:, 0], target[:, 1]] == self.env.FRUIT
            )
            rewards = original_move(actions)
            self.step += 1
            _, _, next_visible = self.positions()
            lost = visible & ~next_visible
            lost[eaten] = False  # A new fruit is not a lost previous fruit.
            self.lost_current |= lost
            self.sums["visibility_losses"] += lost
            assert np.all(self.first_seen[eaten] >= 0)
            search = self.first_seen[eaten] - self.birth[eaten]
            pursuit = self.step - self.first_seen[eaten]
            excess = pursuit - self.first_distance[eaten]
            assert np.all(excess >= 0)
            self.sums["completed_count"][eaten] += 1
            self.sums["completed_search"][eaten] += search
            self.sums["completed_pursuit"][eaten] += pursuit
            self.sums["completed_sight_distance"][eaten] += self.first_distance[eaten]
            self.sums["completed_excess"][eaten] += excess
            self.sums["completed_lost_sight"][eaten] += self.lost_current[eaten]
            self.spawn(eaten)
            if self.step % 250 == 0:
                print(f"  {self.step}/{self.horizon}", flush=True)
            return rewards

        env.to_state, env.move = state, move

    def positions(self):
        heads = np.argwhere(self.env.boards == self.env.HEAD)[:, 1:]
        fruits = np.argwhere(self.env.boards == self.env.FRUIT)[:, 1:]
        visible = np.max(np.abs(heads - fruits), axis=1) <= self.env.mask_size
        return heads, fruits, visible

    def spawn(self, ids):
        if not len(ids):
            return
        heads, fruits, visible = self.positions()
        self.birth[ids] = self.step
        self.first_seen[ids] = -1
        self.initially_hidden[ids] = ~visible[ids]
        self.lost_current[ids] = False
        if self.step == self.horizon:
            return  # No remaining decision exposure for this last spawn.
        boards = self.env.boards[ids]
        eligible = (boards == self.env.EMPTY) | (boards == self.env.FRUIT)
        within_view = (
            np.max(np.abs(self.coords[None] - heads[ids, :, None, None]), axis=1)
            <= self.env.mask_size
        )
        self.sums["spawn_count"][ids] += 1
        self.sums["hidden_spawns"][ids] += ~visible[ids]
        self.sums["expected_visible_spawns"][ids] += (eligible & within_view).sum(
            axis=(1, 2)
        ) / eligible.sum(axis=(1, 2))
        self.sums["hidden_spawn_distance"][ids] += np.abs(heads[ids] - fruits[ids]).sum(
            axis=1
        ) * (~visible[ids])

    def finish(self, case, policy, rows):
        s = self.sums
        fruits = np.array([r["fruits"] for r in rows])
        assert np.array_equal(s["completed_count"], fruits)
        assert np.all(s["search_actions"] + s["post_sight_actions"] == self.horizon)
        assert np.all(s["visible_actions"] + s["invisible_actions"] == self.horizon)
        tail = self.horizon - self.birth
        tail_search = np.where(self.first_seen < 0, tail, self.first_seen - self.birth)
        tail_pursuit = tail - tail_search
        assert np.array_equal(s["search_actions"], s["completed_search"] + tail_search)
        assert np.array_equal(
            s["post_sight_actions"], s["completed_pursuit"] + tail_pursuit
        )
        assert np.array_equal(
            s["completed_pursuit"],
            s["completed_sight_distance"] + s["completed_excess"],
        )
        undiscovered = (tail > 0) & self.initially_hidden & (self.first_seen < 0)
        assert np.array_equal(
            s["hidden_spawns"], s["hidden_discoveries"] + undiscovered
        )
        zero = fruits == 0
        never_seen_zero = zero & (self.first_seen < 0)
        if policy == "direct":
            assert (
                s["completed_excess"].sum() == 0 and s["visibility_losses"].sum() == 0
            )

        def ratio(a, b):
            return float(s[a].sum() / s[b].sum()) if s[b].sum() else None

        summary = dict(
            case=case,
            policy=policy,
            episodes=self.n,
            fruits_mean=float(fruits.mean()),
            zero_fruit_episodes=int(zero.sum()),
            zero_fruit_never_seen=int(never_seen_zero.sum()),
            zero_fruit_seen=int((zero & ~never_seen_zero).sum()),
            initially_hidden_episodes=int(self.initial_hidden.sum()),
            initial_hidden_wait_capped_mean=float(
                self.initial_wait[self.initial_hidden].mean()
            ),
            initial_hidden_wait_capped_median=float(
                np.median(self.initial_wait[self.initial_hidden])
            ),
            initial_hidden_undiscovered=int(
                np.count_nonzero(
                    self.initial_hidden & (self.initial_wait == self.horizon)
                )
            ),
            search_actions_percent=float(
                100 * s["search_actions"].sum() / (self.n * self.horizon)
            ),
            invisible_actions_percent=float(
                100 * s["invisible_actions"].sum() / (self.n * self.horizon)
            ),
            hidden_discovery_wait_mean=ratio("discovery_wait", "hidden_discoveries"),
            hidden_spawns=int(s["hidden_spawns"].sum()),
            hidden_discoveries=int(s["hidden_discoveries"].sum()),
            hidden_undiscovered_at_horizon=int(undiscovered.sum()),
            observed_spawn_visible_percent=100
            * (1 - ratio("hidden_spawns", "spawn_count")),
            expected_spawn_visible_percent=100
            * ratio("expected_visible_spawns", "spawn_count"),
            hidden_spawn_distance_mean=ratio("hidden_spawn_distance", "hidden_spawns"),
            completed_search_mean=ratio("completed_search", "completed_count"),
            completed_pursuit_mean=ratio("completed_pursuit", "completed_count"),
            completed_sight_distance_mean=ratio(
                "completed_sight_distance", "completed_count"
            ),
            completed_excess_mean=ratio("completed_excess", "completed_count"),
            completed_lost_sight_percent=100
            * ratio("completed_lost_sight", "completed_count"),
            visible_reducing_percent=100
            * ratio("visible_reducing_actions", "visible_actions"),
            visibility_losses=int(s["visibility_losses"].sum()),
            exact_original_reproduction=True,
        )
        detailed = [
            dict(
                case=case,
                policy=policy,
                board_id=i,
                fruits=int(fruits[i]),
                **{key: value[i] for key, value in s.items()},
                terminal_search=int(tail_search[i]),
                terminal_pursuit=int(tail_pursuit[i]),
                terminal_undiscovered=bool(undiscovered[i]),
                initially_hidden=bool(self.initial_hidden[i]),
                initial_distance=int(self.initial_distance[i]),
                initial_wait_capped=int(self.initial_wait[i]),
                zero_fruit_never_seen=bool(never_seen_zero[i]),
                zero_fruit_seen=bool(zero[i] and not never_seen_zero[i]),
            )
            for i in self.ids
        ]
        return summary, detailed


class SpawnTracker:
    def __init__(self, env):
        self.env = env
        self.step = 0
        self.n = env.n_boards
        self.coords = np.indices((env.board_size, env.board_size))
        self.birth = np.zeros(self.n, dtype=int)
        self.current = {}
        self.sums = {
            key: np.zeros(self.n)
            for key in (
                "spawn_count",
                "spawn_distance",
                "spawn_expected",
                "spawn_empty_board",
                "spawn_body_effect",
                "spawn_residual",
                "spawn_body_cells",
                "completed_count",
                "completed_distance",
                "completed_expected",
                "completed_residual",
                "completed_excess",
                "completed_duration",
            )
        }
        self.spawn(np.arange(self.n))
        original_move = env.move

        def move(actions):
            heads = np.argwhere(env.boards == env.HEAD)[:, 1:]
            targets = heads + ev.OFFSETS[actions[:, 0]]
            eaten = np.flatnonzero(
                env.boards[np.arange(self.n), targets[:, 0], targets[:, 1]] == env.FRUIT
            )
            rewards = original_move(actions)
            self.step += 1
            duration = self.step - self.birth[eaten]
            excess = duration - self.current["distance"][eaten]
            assert np.all(excess >= 0)
            self.sums["completed_count"][eaten] += 1
            self.sums["completed_duration"][eaten] += duration
            self.sums["completed_excess"][eaten] += excess
            for name in ("distance", "expected", "residual"):
                self.sums["completed_" + name][eaten] += self.current[name][eaten]
            self.spawn(eaten)
            if self.step % 250 == 0:
                print(f"  {self.step}/1000 steps", flush=True)
            return rewards

        env.move = move

    def spawn(self, ids):
        if not len(ids):
            return
        boards = self.env.boards[ids]
        heads = np.argwhere(boards == self.env.HEAD)[:, 1:]
        fruits = np.argwhere(boards == self.env.FRUIT)[:, 1:]
        distances = np.abs(self.coords[None] - heads[:, :, None, None]).sum(axis=1)
        # Restore the chosen fruit cell to its eligibility before placement.
        eligible = (boards == self.env.EMPTY) | (boards == self.env.FRUIT)
        empty_board = (boards != self.env.WALL) & (boards != self.env.HEAD)
        expected = (distances * eligible).sum(axis=(1, 2)) / eligible.sum(axis=(1, 2))
        base = (distances * empty_board).sum(axis=(1, 2)) / empty_board.sum(axis=(1, 2))
        actual = np.abs(heads - fruits).sum(axis=1)
        values = dict(
            distance=actual,
            expected=expected,
            empty_board=base,
            body_effect=expected - base,
            residual=actual - expected,
            body_cells=(boards == self.env.BODY).sum(axis=(1, 2)),
        )
        self.birth[ids] = self.step
        self.sums["spawn_count"][ids] += 1
        for key, value in values.items():
            self.sums["spawn_" + key][ids] += value
            self.current.setdefault(key, np.zeros(self.n))[ids] = value
