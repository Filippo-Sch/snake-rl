"""Record genuine held-out trajectories without changing policy inputs or physics."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from snake import evaluation as ev
from snake.dqn import load_policy
from snake.classic.config import CONFIG, configure_torch
from snake.classic.environment import ClassicSnake
from snake.classic.learner import choose_actions, load_policy as load_classic

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def cells(observation):
    occupied = observation.any(axis=-1)
    return np.where(occupied, observation.argmax(axis=-1) + 1, 0).astype(np.uint8)


def packed(board):
    return "".join(str(int(value)) for value in board.reshape(-1))


class Recorder:
    def __init__(self, environment):
        self.environment = environment
        self.steps = self.fruits = self.walls = self.hits = self.completions = 0
        self.score = 0.0
        self.frames = [self.snapshot()]

    def __getattr__(self, name):
        return getattr(self.environment, name)

    def snapshot(self):
        board = self.environment.boards[0]
        head = np.argwhere(board == 4)[0].tolist()
        return dict(
            t=self.steps,
            board=packed(board),
            head=head,
            view="",
            action=None,
            fruits=self.fruits,
            walls=self.walls,
            hits=self.hits,
            completions=self.completions,
            score=round(self.score, 6),
            length=1 + len(self.environment.bodies[0]),
            outcome="active",
        )

    def move(self, actions):
        head = np.asarray(self.frames[-1]["head"])
        position = head + ev.OFFSETS[int(actions[0, 0])]
        destination = self.environment.boards[0, position[0], position[1]]
        rewards = self.environment.move(actions)
        self.steps += 1
        self.fruits += int(destination == 2)
        self.walls += int(destination == 0)
        self.hits += int(destination == 3)
        self.completions += int(destination == 2 and not self.environment.bodies[0])
        self.score += float(rewards[0, 0])
        self.frames.append(self.snapshot())
        return rewards


class RecordedPolicy:
    device = "cpu"

    def __init__(self, policy, recorder):
        self.policy = policy
        self.recorder = recorder

    def act(self, observation):
        recorder = self.recorder()
        recorder.frames[-1]["view"] = packed(cells(observation[0]))
        actions = self.policy.act(observation)
        recorder.frames[-1]["action"] = int(actions[0, 0])
        return actions


def native(size, regime):
    case = f"b{size}_{regime}"
    selected = "dqn_2048000" if size == 7 else "dqn_C_selected"
    weights = ROOT / "weights" / f"native_{case}.pt"
    config = ev.EvaluationConfig(
        board_size=size, n_boards=500, steps=1000, split="test", test_stream=3
    )
    original = ev.make_environment
    recorder = None

    def factory(configuration, view):
        nonlocal recorder
        recorder = Recorder(original(configuration, view))
        return recorder

    def policy_factory(*, seed, regime):
        policy, _ = load_policy(weights, config, regime)
        return RecordedPolicy(policy, lambda: recorder)

    try:
        ev.make_environment = factory
        metadata, records, _ = ev.evaluate_policy(
            ev.PolicySpec("dqn", policy_factory, weights=weights), config, regime
        )
    finally:
        ev.make_environment = original

    with (ROOT / f"evidence/native/{case}/test.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        reference = {
            int(r["board_id"]): r
            for r in csv.DictReader(stream)
            if r["policy"] == selected
        }
    for record in records:
        for key in ev.COUNTS + (
            "return",
            "body_cells_removed",
            "body_length_before_hits",
        ):
            if (
                abs(float(record[key]) - float(reference[record["board_id"]][key]))
                > 1e-9
            ):
                raise ValueError(f"Native replay differs from the report: {case}/{key}")
    archived_metadata = read(ROOT / f"evidence/native/{case}/test_metadata.json")
    old = next(r for r in archived_metadata["runs"] if r["policy"] == selected)
    assert metadata["initial_boards_sha256"] == old["initial_boards_sha256"]
    recorder.frames[-1]["view"] = packed(cells(recorder.environment.to_state()[0]))
    recorder.frames[-1]["outcome"] = "horizon"
    final = recorder.frames[-1]
    for name, metric in (
        ("fruits", "fruits"),
        ("walls", "wall_hits"),
        ("hits", "body_hits"),
        ("completions", "board_completions"),
    ):
        assert final[name] == records[0][metric]
    return dict(
        id=case,
        task="native",
        regime=regime,
        size=size,
        episode_id=0,
        horizon=1000,
        observation_side=size if regime == "full" else 5,
        bank_size=500,
        test_stream=3,
        checked_records=500,
        initial_boards_sha256=metadata["initial_boards_sha256"],
        final_record=records[0],
        frames=recorder.frames,
    )


def classic(regime):
    network, _ = load_classic(ROOT / f"weights/classic_{regime}.pt", regime)
    environment = ClassicSnake(
        regime, "test", 0, limit=CONFIG.eval_limit, fruit_free_limit=None
    )
    frames = []
    seen, first_repeat, period = {}, None, None

    def snapshot():
        board = np.zeros((7, 7), dtype=np.uint8)
        board[1:6, 1:6] = 1
        for index, position in enumerate(environment.body):
            board[position] = 4 if index == 0 else 3
        if environment.fruit is not None:
            board[environment.fruit] = 2
        observation = environment.observe()
        return dict(
            t=environment.steps,
            board=packed(board),
            head=list(environment.body[0]),
            view=packed(cells(observation.cells)),
            action=None,
            fruits=environment.fruits,
            walls=0,
            hits=0,
            completions=int(environment.outcome == "win"),
            length=len(environment.body),
            score=round(environment.total_reward, 6),
            outcome=environment.outcome,
            cycle=first_repeat is not None,
        )

    frames.append(snapshot())
    while not environment.done:
        key = (tuple(environment.body), environment.heading, environment.fruit)
        if key in seen and first_repeat is None:
            first_repeat, period = environment.steps, environment.steps - seen[key]
        seen.setdefault(key, environment.steps)
        frames[-1]["cycle"] = first_repeat is not None
        observation = environment.observe()
        action = int(
            choose_actions(
                network, observation.encode()[None], observation.legal[None]
            )[0]
        )
        frames[-1]["action"] = action
        environment.step(action)
        frames.append(snapshot())
    reference = read(ROOT / f"evidence/classic/test/{regime}/dqn/results.json")[
        "records"
    ][0]
    actual = environment.record()
    for name, value in actual.items():
        if value != reference[name]:
            raise ValueError(f"Classic replay differs: {regime}/{name}")
    assert first_repeat == reference["first_repeat_step"]
    assert period == reference["cycle_period"]
    return dict(
        id=f"classic_{regime}",
        task="classic",
        regime=regime,
        size=7,
        episode_id=0,
        horizon=5000,
        observation_side=7 if regime == "full" else 5,
        bank_size=500,
        checked_records=1,
        first_repeat_step=first_repeat,
        cycle_period=period,
        final_record=actual,
        frames=frames,
    )


def record():
    configure_torch()
    manifest = read(ROOT / "weights/manifest.json")
    hashes = {}
    for name, item in manifest["policies"].items():
        digest = hashlib.sha256(
            (ROOT / "weights" / item["file"]).read_bytes()
        ).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
        hashes[name] = digest
    episodes = []
    for size in (7, 11):
        for regime in ("full", "partial"):
            print(
                f"Recording native {size} x {size}/{regime}, original 500-board test bank...",
                flush=True,
            )
            episodes.append(native(size, regime))
    for regime in ("full", "partial"):
        print(f"Recording Classic/{regime}, held-out game 0...", flush=True)
        episodes.append(classic(regime))
    return dict(
        format="snake-visual-replay-v1",
        seed=0,
        policy="DQN argmax, frozen weights",
        episode_selection="Test episode 0 in every configuration, without selection by outcome",
        checkpoints=hashes,
        training=False,
        matched_records=sum(e["checked_records"] for e in episodes),
        episodes=episodes,
    )
