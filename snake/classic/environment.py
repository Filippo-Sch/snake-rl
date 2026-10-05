"""Terminal Snake physics with public heading and operational truncation."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from ..baselines import OFFSETS
from .config import CONFIG, input_size, rng


def legal_mask(heading):
    mask = np.ones(4, dtype=bool)
    mask[(int(heading) + 2) % 4] = False
    return mask


@dataclass(frozen=True)
class Observation:
    cells: np.ndarray
    heading: int

    @property
    def legal(self):
        return legal_mask(self.heading)

    def encode(self):
        return np.concatenate((self.cells.reshape(-1), np.eye(4, dtype=np.uint8)[self.heading]))


@dataclass(frozen=True)
class Step:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    outcome: str
    fruit: bool


class ClassicSnake:
    """One game. Body positions are head first and never exposed to policies."""

    def __init__(self, regime, role="training", game_id=0, *,
                 limit=CONFIG.train_limit, fruit_free_limit=CONFIG.fruit_free_limit):
        input_size(regime)
        if limit < 1 or (fruit_free_limit is not None and fruit_free_limit < 1):
            raise ValueError("Episode limits must be positive")
        self.regime, self.role, self.game_id = regime, role, int(game_id)
        self.limit, self.fruit_free_limit = limit, fruit_free_limit
        self.rng = rng(role, game_id)
        placements = []
        for row in range(1, 6):
            for col in range(1, 6):
                for heading, (dr, dc) in enumerate(OFFSETS):
                    body = [(row - i * dr, col - i * dc) for i in range(3)]
                    if all(1 <= r <= 5 and 1 <= c <= 5 for r, c in body):
                        placements.append((heading, body))
        self.heading, self.body = placements[int(self.rng.integers(len(placements)))]
        self.steps = self.fruit_free = self.fruits = 0
        self.total_reward = 0.0
        self.done, self.outcome = False, "active"
        self.max_fruit_free = 0
        self.length_counts = np.zeros(26, dtype=np.int64)
        self.length_counts[3] = 1
        self.fruit = self._spawn_fruit()

    def _spawn_fruit(self):
        free = [(r, c) for r in range(1, 6) for c in range(1, 6)
                if (r, c) not in self.body]
        return free[int(self.rng.integers(len(free)))] if free else None

    def observe(self):
        cells = np.zeros((7, 7, 4), dtype=np.uint8)
        cells[1:6, 1:6, 0] = 1
        for index, (row, col) in enumerate(self.body):
            cells[row, col] = 0
            cells[row, col, 3 if index == 0 else 2] = 1
        if self.fruit is not None:
            row, col = self.fruit
            cells[row, col] = (0, 1, 0, 0)
        if self.regime == "partial":
            padded = np.pad(cells, ((2, 2), (2, 2), (0, 0)))
            row, col = self.body[0]
            cells = padded[row:row + 5, col:col + 5].copy()
        return Observation(cells, self.heading)

    def step(self, action):
        if self.done:
            raise ValueError("Game has ended")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError("Action must be an integer")
        if not 0 <= action < 4 or not legal_mask(self.heading)[action]:
            raise ValueError("Invalid or reversing action")
        dr, dc = OFFSETS[action]
        destination = (self.body[0][0] + dr, self.body[0][1] + dc)
        fruit = destination == self.fruit
        wall = not (1 <= destination[0] <= 5 and 1 <= destination[1] <= 5)
        collision = destination in (self.body if fruit else self.body[:-1])
        self.steps += 1
        self.fruit_free += 1
        reward = CONFIG.step_reward
        terminated = False
        if wall or collision:
            reward += CONFIG.death_reward
            self.outcome = "wall_death" if wall else "body_death"
            terminated = True
        else:
            self.heading = int(action)
            self.body.insert(0, destination)
            if fruit:
                self.fruits += 1
                self.fruit_free = 0
                reward += CONFIG.fruit_reward
                self.fruit = self._spawn_fruit()
                if len(self.body) == 25:
                    reward += CONFIG.win_reward
                    self.outcome, terminated = "win", True
            else:
                self.body.pop()
        self.max_fruit_free = max(self.max_fruit_free, self.fruit_free)
        self.length_counts[len(self.body)] += 1
        truncated = False
        if not terminated:
            if self.steps >= self.limit:
                self.outcome, truncated = "total_limit", True
            elif self.fruit_free_limit is not None and self.fruit_free >= self.fruit_free_limit:
                self.outcome, truncated = "fruit_free_limit", True
        self.done = terminated or truncated
        self.total_reward += reward
        return Step(self.observe(), reward, terminated, truncated, self.outcome, fruit)

    def record(self):
        return dict(game_id=self.game_id, outcome=self.outcome, actions=self.steps,
                    fruits=self.fruits, reward=self.total_reward, length=len(self.body),
                    max_fruit_free=self.max_fruit_free, length_counts=self.length_counts.tolist())

    def state_dict(self):
        values = {k: deepcopy(v) for k, v in vars(self).items() if k != "rng"}
        values["rng"] = deepcopy(self.rng.bit_generator.state)
        return values

    @classmethod
    def from_state(cls, values):
        instance = cls.__new__(cls)
        values = deepcopy(values)
        state = values.pop("rng")
        instance.__dict__.update(values)
        instance.rng = rng(instance.role, instance.game_id)
        instance.rng.bit_generator.state = state
        return instance
