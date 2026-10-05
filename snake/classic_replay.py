"""Bounded self-success replay for the isolated policy continuation experiment."""
from copy import deepcopy

import numpy as np

from .classic.config import CONFIG, rng
from .classic.data import Replay

FORMAT = "protected-self-replay-v1"


class ProtectedReplay:
    """Ordinary FIFO plus complete winning episodes; deliberate biased sampling.

    Ordinary replay includes successes too. Duplicate copies count against the
    50k storage budget. Pending collector trajectories are not sampleable replay.
    """
    def __init__(self, ordinary, active=(), protected_capacity=5000, quota=8):
        if not 0 < protected_capacity < ordinary.capacity:
            raise ValueError("Invalid protected capacity")
        self.regime = ordinary.regime
        self.capacity = ordinary.capacity
        self.protected_capacity, self.quota = protected_capacity, quota
        self.ordinary = ordinary
        self.episodes, self.pending, self.ignored = [], {}, set()
        self.rng = rng("sampling")
        self.wins_seen = self.wins_admitted = self.evicted_episodes = 0
        self.protected_samples = self.ordinary_samples = 0
        self.sampled_games = set()
        # Recover inherited active prefixes only when every previous step exists.
        order = self._order(ordinary)
        for env in active:
            indices = order[ordinary.arrays["game_ids"][order] == env.game_id]
            if len(indices) == env.steps:
                self.pending[env.game_id] = [
                    {k: a[i].copy() for k, a in ordinary.arrays.items()} for i in indices]
            else:
                self.ignored.add(env.game_id)

    @staticmethod
    def _order(replay):
        if replay.size < replay.capacity:
            return np.arange(replay.size)
        return np.concatenate((np.arange(replay.position, replay.size),
                               np.arange(replay.position)))

    @property
    def protected_size(self):
        return sum(len(e["actions"]) for e in self.episodes)

    def _resize_ordinary(self):
        capacity = self.capacity - self.protected_size
        if capacity == self.ordinary.capacity:
            return
        old = self.ordinary
        indices = self._order(old)[-capacity:]
        new = Replay(self.regime, capacity)
        new.size = len(indices)
        new.position = new.size % capacity
        for key in new.arrays:
            new.arrays[key][:new.size] = old.arrays[key][indices]
        new.rng.bit_generator.state = deepcopy(old.rng.bit_generator.state)
        self.ordinary = new

    def add(self, row):
        self.ordinary.add(row)
        gid = int(row["game_ids"])
        ended = bool(row["terminated"] or row["truncated"])
        win = bool(row["terminated"] and row["rewards"] > 100)
        self.wins_seen += int(win)
        if gid not in self.ignored:
            self.pending.setdefault(gid, []).append(deepcopy(row))
            if len(self.pending[gid]) > CONFIG.train_limit:
                raise ValueError("Collector episode exceeded the training limit")
        if not ended:
            return
        rows = self.pending.pop(gid, None)
        self.ignored.discard(gid)
        if win and rows:
            if len(rows) > self.protected_capacity:
                raise ValueError("Protected capacity cannot hold a complete winning episode")
            episode = {k: np.asarray([r[k] for r in rows], dtype=a.dtype)
                       for k, a in self.ordinary.arrays.items()}
            self.episodes.append(episode)
            self.wins_admitted += 1
            while self.protected_size > self.protected_capacity:
                self.episodes.pop(0)
                self.evicted_episodes += 1
            self._resize_ordinary()

    def sample(self, count):
        # The scratch collector always requests 64. Reject silent quota changes.
        if count != CONFIG.batch_size:
            raise ValueError("Protected replay requires a 64-transition batch")
        protected_count = self.quota if self.episodes else 0
        ordinary_count = count - protected_count
        ordinary = self.ordinary.sample(ordinary_count)
        self.ordinary_samples += ordinary_count
        if not protected_count:
            return ordinary
        chosen = self.rng.integers(len(self.episodes), size=protected_count)
        rows, probabilities = [], []
        for episode_index in chosen:
            episode = self.episodes[int(episode_index)]
            index = int(self.rng.integers(len(episode["actions"])))
            rows.append({k: a[index] for k, a in episode.items()})
            probabilities.append(protected_count / count /
                                 len(self.episodes) / len(episode["actions"]))
            self.sampled_games.add(int(episode["game_ids"][index]))
        protected = {k: np.asarray([r[k] for r in rows], dtype=a.dtype)
                     for k, a in self.ordinary.arrays.items()}
        protected.update(weights=np.ones(protected_count, dtype=np.float32),
                         probabilities=np.asarray(probabilities),
                         demo=np.zeros(protected_count, dtype=bool))
        protected["eligible"][:] = False
        ordinary["probabilities"] *= ordinary_count / count
        self.protected_samples += protected_count
        # Probabilities describe the source-labelled draw, not a deduplicated
        # transition marginal. Unit weights intentionally retain the 8/64 bias.
        return {k: np.concatenate((ordinary[k], protected[k])) for k in ordinary}

    def summary(self):
        return dict(ordinary_size=self.ordinary.size,
                    ordinary_capacity=self.ordinary.capacity,
                    protected_size=self.protected_size,
                    protected_capacity=self.protected_capacity,
                    protected_episodes=len(self.episodes),
                    protected_game_ids=[int(e["game_ids"][0]) for e in self.episodes],
                    wins_seen=self.wins_seen, wins_admitted=self.wins_admitted,
                    evicted_episodes=self.evicted_episodes,
                    protected_samples=self.protected_samples,
                    ordinary_samples=self.ordinary_samples,
                    distinct_protected_games_sampled=len(self.sampled_games),
                    pending_transitions=sum(map(len, self.pending.values())),
                    incomplete_inherited_games=len(self.ignored),
                    quota_active=bool(self.episodes))

    def state_dict(self):
        return dict(format=FORMAT, regime=self.regime, capacity=self.capacity,
                    protected_capacity=self.protected_capacity, quota=self.quota,
                    ordinary=self.ordinary.state_dict(), episodes=deepcopy(self.episodes),
                    pending=deepcopy(self.pending), ignored=sorted(self.ignored),
                    rng=deepcopy(self.rng.bit_generator.state),
                    wins_seen=self.wins_seen, wins_admitted=self.wins_admitted,
                    evicted_episodes=self.evicted_episodes,
                    protected_samples=self.protected_samples,
                    ordinary_samples=self.ordinary_samples,
                    sampled_games=sorted(self.sampled_games))

    @classmethod
    def from_state(cls, state):
        if state.get("format") != FORMAT:
            raise ValueError("Unknown protected replay format")
        obj = cls.__new__(cls)
        for key in ("regime", "capacity", "protected_capacity", "quota", "episodes",
                    "pending", "wins_seen", "wins_admitted", "evicted_episodes",
                    "protected_samples", "ordinary_samples"):
            setattr(obj, key, deepcopy(state[key]))
        obj.ignored, obj.sampled_games = set(state["ignored"]), set(state["sampled_games"])
        obj.ordinary = Replay.from_state(state["ordinary"])
        obj.rng = rng("sampling")
        obj.rng.bit_generator.state = deepcopy(state["rng"])
        if (obj.protected_size > obj.protected_capacity
                or obj.ordinary.capacity + obj.protected_size != obj.capacity):
            raise ValueError("Protected replay capacity mismatch")
        for e in obj.episodes:
            if (not len(e["actions"]) or not e["terminated"][-1]
                    or e["rewards"][-1] <= 100
                    or (e["terminated"][:-1] | e["truncated"][:-1]).any()
                    or not np.all(e["game_ids"] == e["game_ids"][0])):
                raise ValueError("Invalid protected winning episode")
        return obj

