"""Observation-only classic baselines, including auditable teacher labels."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..baselines import _neighbors
from .config import rng

MODES = ("student", "bfs_route", "greedy_closer", "safe_search", "blocked")


@dataclass(frozen=True)
class Decision:
    action: int
    mode: str

    @property
    def eligible(self):
        return self.mode == "bfs_route"


class Heuristic:
    def __init__(self, name="bfs", game_id=0):
        if name not in ("bfs", "greedy"):
            raise ValueError("Unknown classic heuristic")
        self.name = name
        self.rng = rng("teacher", game_id)

    def act(self, observation):
        board, legal = observation.cells, observation.legal
        head = tuple(np.argwhere(board[..., 3])[0])
        fruits = np.argwhere(board[..., 1])
        fruit = tuple(fruits[0]) if len(fruits) else None
        walkable = (board[..., 0] != 0) | (board[..., 1] != 0)
        safe = [(a, p) for a, p in _neighbors(head, walkable.shape) if legal[a] and walkable[p]]
        if self.name == "bfs" and fruit is not None:
            queue, visited = deque([(head, None)]), {head}
            while queue:
                position, first = queue.popleft()
                for action, target in _neighbors(position, walkable.shape):
                    if first is None and not legal[action]:
                        continue
                    if target in visited or not walkable[target]:
                        continue
                    first_action = action if first is None else first
                    if target == fruit:
                        return Decision(first_action, "bfs_route")
                    visited.add(target)
                    queue.append((target, first_action))
        if fruit is not None:
            distance = abs(head[0] - fruit[0]) + abs(head[1] - fruit[1])
            closer = [a for a, p in safe if abs(p[0] - fruit[0]) + abs(p[1] - fruit[1]) < distance]
            if closer:
                return Decision(int(self.rng.choice(closer)), "greedy_closer")
        choices = [a for a, _ in safe]
        return Decision(int(self.rng.choice(choices or np.flatnonzero(legal))),
                        "safe_search" if choices else "blocked")
