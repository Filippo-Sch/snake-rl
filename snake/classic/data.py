"""Owned transition storage, whole-game demonstrations and corrected sampling."""

from copy import deepcopy
import json
from pathlib import Path
import os
import time

import numpy as np

from ..training_state import atomic_json, digest
from .config import CONFIG, PROTOCOL, input_size, manifest, rng
from .environment import ClassicSnake
from .policies import Heuristic, MODES

SCALARS = dict(actions=np.int64, rewards=np.float32, terminated=bool, truncated=bool,
               game_ids=np.int64, modes=np.uint8, eligible=bool, lengths=np.uint8)


def allocate(capacity, width):
    result = {key: np.empty(capacity, dtype=dtype) for key, dtype in SCALARS.items()}
    for key in ("states", "next_states"):
        result[key] = np.empty((capacity, width), dtype=np.uint8)
    for key in ("legal", "next_legal"):
        result[key] = np.empty((capacity, 4), dtype=bool)
    return result


def transition(observation, decision, result, game_id, length):
    return dict(states=observation.encode(), actions=decision.action, rewards=result.reward,
                next_states=result.observation.encode(), legal=observation.legal,
                next_legal=result.observation.legal, terminated=result.terminated,
                truncated=result.truncated, game_ids=game_id, modes=MODES.index(decision.mode),
                eligible=decision.eligible and result.outcome not in ("wall_death", "body_death"),
                lengths=length)


class Replay:
    def __init__(self, regime, capacity=CONFIG.replay_capacity):
        self.regime, self.capacity = regime, capacity
        self.arrays = allocate(capacity, input_size(regime))
        self.size = self.position = 0
        self.rng = rng("replay")

    def add(self, row):
        for key, array in self.arrays.items():
            array[self.position] = row[key]
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, count):
        indices = self.rng.integers(self.size, size=count)
        batch = {key: value[indices].copy() for key, value in self.arrays.items()}
        batch.update(weights=np.ones(count, dtype=np.float32),
                     probabilities=np.full(count, 1 / self.size),
                     demo=np.zeros(count, dtype=bool))
        batch["eligible"][:] = False
        return batch

    def state_dict(self):
        return dict(regime=self.regime, capacity=self.capacity, size=self.size,
                    position=self.position, rng=deepcopy(self.rng.bit_generator.state),
                    arrays={k: v[:self.size].copy() for k, v in self.arrays.items()})

    @classmethod
    def from_state(cls, state):
        instance = cls(state["regime"], state["capacity"])
        instance.size, instance.position = state["size"], state["position"]
        if not 0 <= instance.size <= instance.capacity or not 0 <= instance.position < instance.capacity:
            raise ValueError("Invalid classic replay cursor")
        for key in instance.arrays:
            instance.arrays[key][:instance.size] = state["arrays"][key]
        instance.rng.bit_generator.state = state["rng"]
        return instance


class Demonstrations:
    def __init__(self, arrays, metadata, file_hash=None):
        self.arrays, self.metadata, self.file_hash = arrays, metadata, file_hash
        self.regime = metadata["regime"]
        n = len(arrays["actions"])
        if metadata.get("protocol") != PROTOCOL or arrays["states"].shape != (n, input_size(self.regime)):
            raise ValueError("Incompatible demonstration format/regime")
        if set(arrays) != set(allocate(0, input_size(self.regime))):
            raise ValueError("Missing demonstration fields")
        if any(len(a) != n for a in arrays.values()):
            raise ValueError("Inconsistent demonstration array lengths")
        self.training = np.flatnonzero(arrays["game_ids"] % 5 != 4)
        self.holdout = np.flatnonzero(arrays["game_ids"] % 5 == 4)
        if not len(self.training):
            raise ValueError("Empty demonstration training split")
        self.winners = []
        cursor = 0
        for game in metadata["games"]:
            start, stop = game["start"], game["stop"]
            if start != cursor or stop <= start or stop > n:
                raise ValueError("Demonstrations must be contiguous complete games")
            if not np.all(arrays["game_ids"][start:stop] == game["game_id"]):
                raise ValueError("Demonstration game ID mismatch")
            ended = arrays["terminated"][start:stop] | arrays["truncated"][start:stop]
            if not ended[-1] or ended[:-1].any():
                raise ValueError("Invalid demonstration episode boundary")
            if game["outcome"] == "win" and game["game_id"] % 5 != 4:
                self.winners.append((start, stop))
            cursor = stop
        if cursor != n:
            raise ValueError("Unindexed demonstration rows")
        self.q = np.zeros(n, dtype=np.float64)
        self.q[self.training] = (0.75 if self.winners else 1.0) / len(self.training)
        for start, stop in self.winners:
            self.q[start:stop] += 0.25 / (len(self.winners) * (stop - start))
        self.rng = rng("sampling")
        for array in self.arrays.values():
            array.flags.writeable = False

    def sample(self, count):
        indices = self.rng.choice(self.training, size=count, replace=True)
        if self.winners:
            selected = np.flatnonzero(self.rng.random(count) < 0.25)
            for position in selected:
                start, stop = self.winners[int(self.rng.integers(len(self.winners)))]
                indices[position] = self.rng.integers(start, stop)
        batch = {k: v[indices].copy() for k, v in self.arrays.items()}
        probabilities = self.q[indices]
        batch.update(weights=((1 / len(self.training)) / probabilities).astype(np.float32),
                     probabilities=probabilities, demo=np.ones(count, dtype=bool))
        return batch

    def summary(self):
        return dict(transitions=len(self.arrays["actions"]), training=len(self.training),
                    holdout=len(self.holdout), games=len(self.metadata["games"]),
                    winning_training_games=len(self.winners),
                    navigation_only=not bool(self.winners),
                    eligible_fraction=float(self.arrays["eligible"][self.training].mean()),
                    modes={name: int((self.arrays["modes"] == i).sum()) for i, name in enumerate(MODES)},
                    length_counts={str(i): int((self.arrays["lengths"] == i).sum()) for i in range(3, 26)})

    @classmethod
    def load(cls, path, regime):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata["regime"] != regime:
                raise ValueError("Partial learners require partial BFS data; full learners require full data")
            arrays = {k: archive[k].copy() for k in archive.files if k != "metadata"}
        return cls(arrays, metadata, digest(path))


def collect_demonstrations(path, regime):
    """Called only by the explicit collect command, never by import or tests."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite demonstrations: {path}")
    started = time.perf_counter()
    arrays = allocate(CONFIG.demo_budget + CONFIG.train_limit, input_size(regime))
    size, games = 0, []
    while size < CONFIG.demo_budget:
        game_id, start = len(games), size
        env = ClassicSnake(regime, "demos", game_id)
        teacher = Heuristic("bfs", game_id)
        while not env.done:
            observation = env.observe()
            decision = teacher.act(observation)
            result = env.step(decision.action)
            row = transition(observation, decision, result, game_id, len(env.body))
            for key, array in arrays.items():
                array[size] = row[key]
            size += 1
        games.append(dict(env.record(), start=start, stop=size))
    metadata = dict(protocol=PROTOCOL, regime=regime, teacher="bfs", games=games,
                    manifest=manifest(), transitions=size, requested=CONFIG.demo_budget,
                    excess=size - CONFIG.demo_budget, collection_seconds=time.perf_counter() - started)
    arrays = {k: v[:size] for k, v in arrays.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata), **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    dataset = Demonstrations(arrays, metadata, digest(path))
    atomic_json(path.with_suffix(".json"), dict(dataset.summary(), file_hash=dataset.file_hash,
                                               manifest=metadata["manifest"]))
    return dataset


def mixed_batch(replay, demos, probability, mixing_rng, count=CONFIG.batch_size):
    use_demo = mixing_rng.random(count) < probability
    if use_demo.any() and demos is None:
        raise ValueError("Demonstrations required for assisted batches")
    # Avoid consuming the unused source's RNG, especially after the handoff.
    student = replay.sample(int((~use_demo).sum())) if (~use_demo).any() else None
    teacher = demos.sample(int(use_demo.sum())) if use_demo.any() else None
    prototype = student if student is not None else teacher
    batch = {k: np.empty((count,) + v.shape[1:], dtype=v.dtype) for k, v in prototype.items()}
    for mask, source in ((~use_demo, student), (use_demo, teacher)):
        if source is not None:
            for key in batch:
                batch[key][mask] = source[key]
    return batch
