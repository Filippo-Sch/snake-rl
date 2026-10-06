"""Record genuine held-out trajectories without changing policy inputs or physics."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from snake import evaluation as ev
from snake.dqn import load_policy
from snake.baselines import make_direct, OFFSETS
from snake.classic.config import CONFIG, configure_torch
from snake.classic.environment import ClassicSnake
from snake.classic.learner import choose_actions, load_policy as load_classic
from snake.classic.policies import Heuristic

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
            body=[head]
            + [
                np.asarray(position, dtype=int).tolist()
                for position in self.environment.bodies[0]
            ],
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
        self.frames[-1]["move_note"] = (
            "Wall contact: penalized, nonfatal"
            if destination == 0
            else "Body contact: the body is cut, nonfatal" if destination == 3 else ""
        )
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


def native(size, regime, policy="dqn"):
    case = f"b{size}_{regime}"
    selected = (
        ("dqn_2048000" if size == 7 else "dqn_C_selected")
        if policy == "dqn"
        else "direct"
    )
    weights = ROOT / "weights" / f"native_{case}.pt" if policy == "dqn" else None
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
        actor = (
            load_policy(weights, config, regime)[0]
            if policy == "dqn"
            else make_direct(seed=seed, regime=regime)
        )
        return RecordedPolicy(actor, lambda: recorder)

    try:
        ev.make_environment = factory
        metadata, records, _ = ev.evaluate_policy(
            ev.PolicySpec(policy, policy_factory, weights=weights), config, regime
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
        id=f"{case}_{policy}",
        policy=policy,
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
        test_summary=dict(mean_return=metadata["summary"]["return"]["mean"]),
        frames=recorder.frames,
    )


def classic(regime, game_id=0, policy="dqn"):
    network = (
        load_classic(ROOT / f"weights/classic_{regime}.pt", regime)[0]
        if policy == "dqn"
        else None
    )
    heuristic = Heuristic(policy, game_id) if policy != "dqn" else None
    environment = ClassicSnake(
        regime, "test", game_id, limit=CONFIG.eval_limit, fruit_free_limit=None
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
            body=[list(position) for position in environment.body],
            heading=environment.heading,
            view=packed(cells(observation.cells)),
            action=None,
            fruits=environment.fruits,
            walls=0,
            hits=0,
            completions=int(environment.outcome == "win"),
            length=len(environment.body),
            score=round(environment.total_reward, 6),
            outcome=environment.outcome,
            cycle=first_repeat is not None and policy == "dqn",
            tail_entry=False,
            move_note="",
            collision=None,
        )

    frames.append(snapshot())
    while not environment.done:
        key = (tuple(environment.body), environment.heading, environment.fruit)
        if key in seen and first_repeat is None:
            first_repeat, period = environment.steps, environment.steps - seen[key]
        seen.setdefault(key, environment.steps)
        frames[-1]["cycle"] = first_repeat is not None and policy == "dqn"
        observation = environment.observe()
        action = (
            heuristic.act(observation).action
            if heuristic
            else int(
                choose_actions(
                    network, observation.encode()[None], observation.legal[None]
                )[0]
            )
        )
        frames[-1]["action"] = action
        dr, dc = OFFSETS[action]
        target = (environment.body[0][0] + dr, environment.body[0][1] + dc)
        growing = target == environment.fruit
        body_collision = target in (
            environment.body if growing else environment.body[:-1]
        )
        tail_entry = target == environment.body[-1] and not growing
        frames[-1]["tail_entry"] = tail_entry
        if tail_entry:
            frames[-1][
                "move_note"
            ] = "Legal tail entry: the tail vacates this cell on the next move"
        elif body_collision:
            frames[-1]["move_note"] = "Fatal collision: this body cell is not vacated"
        environment_before = list(environment.body)
        environment.step(action)
        frames.append(snapshot())
        if tail_entry:
            assert not environment.done or environment.outcome == "total_limit"
            assert environment.body[0] == environment_before[-1]
        if body_collision:
            assert environment.done and environment.outcome == "body_death"
            frames[-1]["collision"] = list(target)
        elif environment.outcome == "wall_death":
            frames[-1]["collision"] = list(target)
    reference = read(ROOT / f"evidence/classic/test/{regime}/{policy}/results.json")[
        "records"
    ][game_id]
    actual = environment.record()
    for name, value in actual.items():
        if value != reference[name]:
            raise ValueError(f"Classic replay differs: {regime}/{name}")
    assert first_repeat == reference["first_repeat_step"]
    assert period == reference["cycle_period"]
    return dict(
        id=f"classic_{regime}_{policy}_{game_id}",
        policy=policy,
        task="classic",
        regime=regime,
        size=7,
        episode_id=game_id,
        horizon=5000,
        observation_side=7 if regime == "full" else 5,
        bank_size=500,
        checked_records=1,
        first_repeat_step=first_repeat,
        cycle_period=period,
        final_record=actual,
        test_summary=read(
            ROOT / f"evidence/classic/test/{regime}/{policy}/results.json"
        )["summary"],
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
            for policy in ("dqn", "direct"):
                print(f"  {policy}", flush=True)
                episodes.append(native(size, regime, policy))
    partial_test = read(ROOT / "evidence/classic/test/partial/dqn/results.json")
    winning_id = next(
        r["game_id"] for r in partial_test["records"] if r["outcome"] == "win"
    )
    best_classic = {}
    for regime in ("full", "partial"):
        candidates = [
            (
                read(ROOT / f"evidence/classic/test/{regime}/{policy}/results.json")[
                    "summary"
                ],
                policy,
            )
            for policy in ("greedy", "bfs")
        ]
        best_classic[regime] = max(
            candidates, key=lambda pair: (pair[0]["wins"], pair[0]["mean_fruits"])
        )[1]
    for game_id in (0, winning_id):
        for regime in ("full", "partial"):
            for policy in ("dqn", best_classic[regime]):
                print(
                    f"Recording Classic/{regime}/{policy}, held-out game {game_id}...",
                    flush=True,
                )
                episodes.append(classic(regime, game_id, policy))
    return dict(
        format="snake-visual-replay-v2",
        seed=0,
        policy="Frozen DQN argmax and original heuristic policies",
        episode_selection="Episode 0 is the default. The success example is the first archived partial-DQN win; all displayed policies share its episode ID and initial board.",
        partial_success_id=winning_id,
        best_classic=best_classic,
        baseline_ranking="Native: highest mean archived test return (Direct). Classic: most archived test wins, then mean fruit. This ranks the reported heuristics, not DQN checkpoints.",
        checkpoints=hashes,
        training=False,
        matched_records=sum(e["checked_records"] for e in episodes),
        episodes=episodes,
    )
