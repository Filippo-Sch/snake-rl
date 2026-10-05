"""Three observation-only policies, shared by full and partial evaluation.

Channels are EMPTY, FRUIT, BODY, HEAD; walls have all channels zero.
UP increases the row index, following the supplied environments.
"""

from collections import deque

import numpy as np


OFFSETS = ((1, 0), (0, 1), (-1, 0), (0, -1))  # UP, RIGHT, DOWN, LEFT


def _neighbors(position, shape):
    row, column = position
    for action, (dr, dc) in enumerate(OFFSETS):
        target = (row + dr, column + dc)
        if 0 <= target[0] < shape[0] and 0 <= target[1] < shape[1]:
            yield action, target


class GreedyPolicy:
    """Choose a safe step closer to visible fruit, with uniform tie breaking.

    Otherwise sample a safe step, or any direction if all are blocked.
    No memory, heading persistence or preference between collision types.
    """

    device = "cpu"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        actions = np.empty((len(observation), 1), dtype=np.int64)
        for i, board in enumerate(observation):
            head = tuple(np.argwhere(board[..., 3])[0])
            fruits = np.argwhere(board[..., 1])
            fruit = tuple(fruits[0]) if len(fruits) else None
            walkable = (board[..., 0] > 0) | (board[..., 1] > 0)
            safe = [(action, target) for action, target in _neighbors(head, walkable.shape)
                    if walkable[target]]
            actions[i, 0] = self._choose(walkable, head, fruit, safe)
        return actions

    def _choose(self, walkable, head, fruit, safe):
        if fruit is not None:
            distance = abs(head[0] - fruit[0]) + abs(head[1] - fruit[1])
            closer = [action for action, target in safe
                      if abs(target[0] - fruit[0]) + abs(target[1] - fruit[1]) < distance]
            if closer:
                return self.rng.choice(closer)
        return self.rng.choice([action for action, _ in safe] or [0, 1, 2, 3])


class DirectPolicy(GreedyPolicy):
    """Always approach visible fruit, preferring free steps over body collisions.

    On the supplied rectangular boards, distance-reducing steps cannot hit a
    wall. If every such step enters BODY, accept a cut instead of a detour.
    Without visible fruit, use greedy's safe random search and blocked fallback.
    """

    def _choose(self, walkable, head, fruit, safe):
        if fruit is not None:
            distance = abs(head[0] - fruit[0]) + abs(head[1] - fruit[1])
            closer = [action for action, target in _neighbors(head, walkable.shape)
                      if abs(target[0] - fruit[0]) + abs(target[1] - fruit[1]) < distance]
            preferred = [action for action, _ in safe if action in closer]
            return self.rng.choice(preferred or closer)
        return super()._choose(walkable, head, fruit, safe)


class BFSPolicy(GreedyPolicy):
    """Replan a shortest visible path each step; use greedy when none exists.

    The current body is blocked, including the tail. Cells outside a partial
    observation are never traversed by the search. Equal paths are resolved
    in UP, RIGHT, DOWN, LEFT order. This does not guarantee future safety.
    """

    def _choose(self, walkable, head, fruit, safe):
        if fruit is not None:
            queue = deque([(head, None)])
            visited = {head}
            while queue:
                position, first_action = queue.popleft()
                for action, target in _neighbors(position, walkable.shape):
                    if target in visited or not walkable[target]:
                        continue
                    first = action if first_action is None else first_action
                    if target == fruit:
                        return first
                    visited.add(target)
                    queue.append((target, first))
        return super()._choose(walkable, head, fruit, safe)


def make_greedy(*, seed, regime):
    # All factories accept the evaluator interface; no regime-specific state.
    return GreedyPolicy(seed)


def make_bfs(*, seed, regime):
    return BFSPolicy(seed)


def make_direct(*, seed, regime):
    return DirectPolicy(seed)
