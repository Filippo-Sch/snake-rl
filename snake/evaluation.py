"""Shared, fixed-horizon evaluation. Policies see observations, never the environment.

Factories implement ``factory(*, seed, regime) -> policy``; policies implement
``act(observation) -> integer array (n_boards, 1)``. Each run creates a fresh policy.
RL adapters must load frozen weights and declare their inference action rule.
Run evaluation in a separate process during training: global RNGs are reseeded.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import re
import sys
from typing import Callable, Protocol

import numpy as np

from . import environments_fully_observable as full
from . import environments_partially_observable as partial


PROTOCOL = "snake-eval-v3"
REWARD_PROFILES = {
    "R0": {"HIT_WALL_REWARD": -0.1, "ATE_HIMSELF_REWARD": -0.2, "WIN_REWARD": 1.0},
    "R1": {"HIT_WALL_REWARD": -5.0, "ATE_HIMSELF_REWARD": -5.0, "WIN_REWARD": 1.0},
    "R2": {"HIT_WALL_REWARD": -5.0, "ATE_HIMSELF_REWARD": -5.0, "WIN_REWARD": 100.0},
}
OFFSETS = np.array(((1, 0), (0, 1), (-1, 0), (0, -1)))
EVENTS = ("fruit", "wall_hit", "body_hit", "board_completed")
COUNTS = ("fruits", "wall_hits", "body_hits", "board_completions")
PRIMARY_METRICS = ("fruits", "wall_hits", "body_hits", "board_completions", "return")


class Policy(Protocol):
    def act(self, observation: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class EvaluationConfig:
    board_size: int = 7
    mask_size: int = 2
    n_boards: int = 100
    steps: int = 1000
    seed: int = 0
    record_steps: bool = False
    reward_profile: str = "R0"
    split: str = "validation"
    test_stream: int = 1

    def __post_init__(self):
        for name, minimum in (("board_size", 4), ("mask_size", 1),
                              ("n_boards", 1), ("steps", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if type(self.seed) is not int or self.seed != 0:
            raise ValueError("The project requires seed=0")
        if type(self.record_steps) is not bool:
            raise ValueError("record_steps must be boolean")
        if self.reward_profile not in REWARD_PROFILES:
            raise ValueError("Unknown reward profile")
        if self.split not in ("validation", "test"):
            raise ValueError("split must be validation or test")
        if type(self.test_stream) is not int or self.test_stream < 1:
            raise ValueError("test_stream must be a positive integer")
        if self.split == "validation" and self.test_stream != 1:
            raise ValueError("test_stream applies only to the test split")


@dataclass(frozen=True)
class PolicySpec:
    name: str
    factory: Callable[..., Policy]
    parameters: dict = field(default_factory=dict)
    weights: Path | dict[str, Path] | None = None

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.name):
            raise ValueError("Policy names may contain letters, numbers, '_' and '-'")
        if not callable(self.factory):
            raise TypeError("factory must be callable")


def seed_everything():
    """Seed installed frameworks only if a policy has imported them."""
    random.seed(0)
    np.random.seed(0)
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
    if "tensorflow" in sys.modules:
        sys.modules["tensorflow"].random.set_seed(0)


def make_environment(config: EvaluationConfig, regime: str):
    if regime == "full":
        env = full.OriginalSnakeEnvironment(config.n_boards, config.board_size)
    elif regime == "partial":
        env = partial.OriginalSnakeEnvironment(
            config.n_boards, config.board_size, config.mask_size)
    else:
        raise ValueError(f"Unknown observation regime: {regime}")
    for name, value in REWARD_PROFILES[config.reward_profile].items():
        setattr(env, name, value)
    return env


def set_evaluation_stream(split, test_stream=1):
    """Test uses numbered 2**128 jumps from the legacy seed-0 MT19937 state.

    Validation keeps its original stream. This separates test random numbers
    without changing the required seed or discarding a large random array.
    """
    if split == "test":
        legacy = np.random.RandomState(0).get_state()
        generator = np.random.MT19937(0)
        generator.state = {"bit_generator": "MT19937",
                           "state": {"key": legacy[1], "pos": legacy[2]}}
        state = generator.jumped(test_stream).state["state"]
        np.random.set_state(("MT19937", state["key"], state["pos"], 0, 0.0))


def score_events(metrics, profile):
    """Rescore unchanged trajectories; completion replaces the fruit reward."""
    rewards = REWARD_PROFILES[profile]
    return (0.5 * np.asarray(metrics["fruits"])
            + rewards["HIT_WALL_REWARD"] * np.asarray(metrics["wall_hits"])
            + rewards["ATE_HIMSELF_REWARD"] * np.asarray(metrics["body_hits"])
            + (rewards["WIN_REWARD"] - 0.5) * np.asarray(metrics["board_completions"]))


def _positions(env, cell):
    positions = np.argwhere(env.boards == cell)
    if (positions.shape != (env.n_boards, 3)
            or not np.array_equal(positions[:, 0], np.arange(env.n_boards))):
        raise RuntimeError(f"Expected exactly one cell of type {cell} per board")
    return positions[:, 1:]


def _lengths(env):
    occupied = 1 + np.count_nonzero(env.boards == env.BODY, axis=(1, 2))
    internal = np.array([1 + len(body) for body in env.bodies])
    return occupied, internal


def _actions(value, n_boards):
    actions = np.asarray(value)
    if (actions.shape != (n_boards, 1)
            or not np.issubdtype(actions.dtype, np.integer)
            or np.any((actions < 0) | (actions > 3))):
        raise ValueError(f"Policy must return integer actions (0..3), shape ({n_boards}, 1)")
    return actions.astype(np.int64, copy=True)


def _reward_settings(env):
    return {name: float(getattr(env, name)) for name in (
        "STEP_REWARD", "FRUIT_REWARD", "HIT_WALL_REWARD",
        "ATE_HIMSELF_REWARD", "WIN_REWARD")}


def _fingerprint(path):
    path = Path(path).resolve()
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _runtime():
    result = {"python": platform.python_version(), "numpy": np.__version__,
              "platform": platform.platform(), "environment_device": "cpu"}
    for name in ("torch", "tensorflow"):
        if name in sys.modules:
            result[name] = sys.modules[name].__version__
    return result


def _summarize(metrics):
    summary = {name: {"mean": float(np.mean(metrics[name])),
                      "std": float(np.std(metrics[name], ddof=1))
                      if len(metrics[name]) > 1 else None}
               for name in PRIMARY_METRICS}
    hits = int(metrics["body_hits"].sum())
    summary["cells_removed_per_body_hit"] = (
        float(metrics["body_cells_removed"].sum() / hits) if hits else None)
    summary["body_length_before_body_hit"] = (
        float(metrics["body_length_before_hits"].sum() / hits) if hits else None)
    summary["completion_rate"] = float(np.mean(metrics["board_completions"] > 0))
    summary["total_completions"] = int(metrics["board_completions"].sum())
    summary["returns_by_reward_profile"] = {
        profile: {"mean": float(np.mean(values)),
                  "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        for profile in REWARD_PROFILES
        for values in (score_events(metrics, profile),)}
    return summary


def evaluate_policy(spec: PolicySpec, config: EvaluationConfig, regime: str):
    """Return metadata, per-board rows and optional step arrays for one fresh run.

    No outputs are written here. Only the logger reads complete boards. Diagnostic
    bookkeeping never samples randomness. Declared environment rewards are checked
    against observed events; dynamics errors are not silently repaired.
    """
    if regime not in ("full", "partial"):
        raise ValueError(f"Unknown observation regime: {regime}")
    weight_path = spec.weights.get(regime) if isinstance(spec.weights, dict) else spec.weights
    weights = _fingerprint(weight_path) if weight_path is not None else None
    seed_everything()
    policy = spec.factory(seed=config.seed, regime=regime)
    # Loading a model may consume random numbers. Reset before board creation.
    seed_everything()
    set_evaluation_stream(config.split, config.test_stream)
    env = make_environment(config, regime)
    initial_digest = hashlib.sha256(env.boards.tobytes()).hexdigest()
    n, horizon = config.n_boards, config.steps
    indices = np.arange(n)
    totals = {key: np.zeros(n, dtype=np.int64) for key in COUNTS}
    totals["body_cells_removed"] = np.zeros(n, dtype=np.int64)
    totals["body_length_before_hits"] = np.zeros(n, dtype=np.int64)
    returns = np.zeros(n, dtype=np.float64)
    length, internal = _lengths(env)
    trace = {}
    if config.record_steps:
        for key in ("length", "internal_length"):
            trace[key] = np.empty((horizon + 1, n), dtype=np.int32)
        for key in EVENTS:
            trace[key] = np.empty((horizon, n), dtype=bool)
        trace["action"] = np.empty((horizon, n), dtype=np.int8)
        trace["body_cells_removed"] = np.empty((horizon, n), dtype=np.int32)
        trace["reward"] = np.empty((horizon, n), dtype=np.float32)
        trace["length"][0], trace["internal_length"][0] = length, internal
    mismatches = int(np.count_nonzero(length != internal))

    for step in range(horizon):
        heads = _positions(env, env.HEAD)
        observation = env.to_state()
        # Policies must use their own generator. Detect accidental use of the
        # environment's global NumPy RNG rather than corrupting comparisons.
        rng_state = np.random.get_state()
        actions = _actions(policy.act(observation), n)
        new_rng_state = np.random.get_state()
        if (rng_state[0] != new_rng_state[0]
                or not np.array_equal(rng_state[1], new_rng_state[1])
                or rng_state[2:] != new_rng_state[2:]):
            raise ValueError("Policy consumed the environment NumPy RNG; use np.random.default_rng(seed)")
        target = heads + OFFSETS[actions[:, 0]]
        destination = env.boards[indices, target[:, 0], target[:, 1]]
        events = {"fruit": destination == env.FRUIT,
                  "wall_hit": destination == env.WALL,
                  "body_hit": destination == env.BODY}
        rewards = np.asarray(env.move(actions), dtype=np.float64)
        if rewards.shape != (n, 1) or not np.all(np.isfinite(rewards)):
            raise RuntimeError("Environment returned invalid rewards")
        rewards = rewards[:, 0]
        next_length, next_internal = _lengths(env)
        # Collecting fruit grows the body, unless the environment completed and
        # reset this board. Detect that event independently of reward constants.
        events["board_completed"] = events["fruit"] & (next_internal == 1)
        expected = np.full(n, env.STEP_REWARD, dtype=float)
        for event, reward in (("wall_hit", env.HIT_WALL_REWARD),
                              ("body_hit", env.ATE_HIMSELF_REWARD),
                              ("fruit", env.FRUIT_REWARD),
                              ("board_completed", env.WIN_REWARD)):
            expected[events[event]] = reward
        if not np.allclose(rewards, expected, rtol=1e-6, atol=1e-7):
            raise RuntimeError(f"Reward/event mismatch at step {step}; verify environment dynamics")
        # Measure net occupied cells lost, not entries removed from bodies:
        # the supplied environment can contain overlapping internal segments.
        removed = np.where(events["body_hit"], length - next_length, 0)
        if np.any(removed < 0):
            raise RuntimeError("A body collision unexpectedly increased body length")
        for event, count in zip(EVENTS, COUNTS):
            totals[count] += events[event]
        totals["body_cells_removed"] += removed
        totals["body_length_before_hits"] += np.where(events["body_hit"], length - 1, 0)
        mismatches += int(np.count_nonzero(next_length != next_internal))
        returns += rewards
        if trace:
            for key, values in events.items():
                trace[key][step] = values
            for key, values in (("action", actions[:, 0]), ("reward", rewards),
                                ("body_cells_removed", removed)):
                trace[key][step] = values
            trace["length"][step + 1] = next_length
            trace["internal_length"][step + 1] = next_internal
        length, internal = next_length, next_internal

    metrics = {**totals, "return": returns}
    rows = [{"policy": spec.name, "regime": regime, "board_id": i,
             **{name: values[i].item() for name, values in metrics.items()}}
            for i in range(n)]
    metadata = {"policy": spec.name, "regime": regime, "parameters": spec.parameters,
                "weights": weights, "rewards": _reward_settings(env),
                "initial_boards_sha256": initial_digest, "summary": _summarize(metrics),
                "diagnostics": {"length_mismatch_states": mismatches},
                "policy_device": str(getattr(policy, "device", "unspecified")),
                "runtime": _runtime()}
    return metadata, rows, trace


def run_evaluation(specs, config=EvaluationConfig(), regimes=("full", "partial"),
                   output_root="results"):
    """Evaluate all pairs and save a single, new results directory on success."""
    specs, regimes = tuple(specs), tuple(regimes)
    if not specs or len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Provide at least one policy, with unique names")
    if (not regimes or len(set(regimes)) != len(regimes)
            or any(regime not in ("full", "partial") for regime in regimes)):
        raise ValueError("Regimes must be a nonempty, unique subset of full/partial")
    runs, rows, arrays = [], [], {}
    for regime in regimes:
        for spec in specs:
            metadata, board_rows, trace = evaluate_policy(spec, config, regime)
            runs.append(metadata)
            rows.extend(board_rows)
            arrays.update({f"{spec.name}__{regime}__{key}": value
                           for key, value in trace.items()})
    payload = {"protocol": PROTOCOL, "config": asdict(config), "regimes": list(regimes),
               "environment_rng": ("seed-0 legacy MT19937; "
                                   f"test stream {config.test_stream} jumps {config.test_stream} * 2**128 outputs"),
               "created_utc": datetime.now(timezone.utc).isoformat(), "runs": runs}
    # Serialize before creating a directory, so invalid metadata leaves no run.
    serialized = json.dumps(payload, indent=2, allow_nan=False)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = root / stamp
    directory.mkdir(exist_ok=False)
    (directory / "results.json").write_text(serialized + "\n", encoding="utf-8")
    with (directory / "per_board.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if config.record_steps:
        np.savez_compressed(directory / "steps.npz", **arrays)
    return directory, payload


def print_summary(payload):
    print("policy / regime | fruits | wall hits | body hits | board completions | return (mean +/- std)"
          " | cells removed/body hit | body before hit | episodes completed (%)")
    for run in payload["runs"]:
        fields = []
        for key in PRIMARY_METRICS:
            metric = run["summary"][key]
            std = "n/a" if metric["std"] is None else f"{metric['std']:.3f}"
            fields.append(f"{metric['mean']:.3f} +/- {std}")
        removed = run["summary"]["cells_removed_per_body_hit"]
        fields.append("n/a" if removed is None else f"{removed:.3f}")
        before = run["summary"]["body_length_before_body_hit"]
        fields.append("n/a" if before is None else f"{before:.3f}")
        fields.append(f"{100 * run['summary']['completion_rate']:.1f}")
        print(f"{run['policy']} / {run['regime']} | " + " | ".join(fields))
