"""Train standard DQN on complete finite-horizon episodes."""

import argparse
from copy import deepcopy
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from snake.dqn import (QNetwork, ReplayBuffer, choose_actions, encode, epsilon, input_side,
                       optimize, save_checkpoint)
from snake.evaluation import REWARD_PROFILES, EvaluationConfig, make_environment, seed_everything
from snake.training_state import (FORMAT as STATE_FORMAT, atomic_bytes, atomic_json,
    atomic_save, capture, restore, load_snapshot, digest)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrainingConfig:
    budget: int = 1_024_000
    evaluation_interval: int = 128_000
    replay_capacity: int = 50_000
    batch_size: int = 64
    warmup: int = 4096
    update_every: int = 16
    target_every: int = 500
    learning_rate: float = 1e-4
    exploration_budget: int | None = None
    exploration_decay_end: int | None = None
    lr_drop_at: int | None = None
    lr_after_drop: float | None = None
    selection_budgets: tuple[int, ...] = ()
    lr_steps: tuple[tuple[int, float], ...] = ()

    def learning_rate_at(self, update_transitions):
        """Keep the update at the boundary unchanged; drop on later updates."""
        if self.lr_steps:
            rate = self.learning_rate
            for boundary, value in self.lr_steps:
                if update_transitions > boundary:
                    rate = value
            return rate
        if self.lr_drop_at is not None and update_transitions > self.lr_drop_at:
            return self.lr_after_drop
        return self.learning_rate

    def validate(self, env_config):
        for key, value in asdict(self).items():
            if key not in ("learning_rate", "exploration_budget", "selection_budgets",
                           "exploration_decay_end", "lr_drop_at", "lr_after_drop", "lr_steps") and (type(value) is not int or value < 1):
                raise ValueError(f"{key} must be a positive integer")
        episode_batch = env_config.n_boards * env_config.steps
        if self.budget % episode_batch or self.evaluation_interval % episode_batch:
            raise ValueError("Budget and evaluation interval must be multiples of boards * steps")
        if self.budget % self.evaluation_interval:
            raise ValueError("Budget must be a multiple of evaluation interval")
        if not self.batch_size <= self.warmup < self.budget:
            raise ValueError("Require batch_size <= warmup < budget")
        if self.replay_capacity < max(self.batch_size, env_config.n_boards):
            raise ValueError("Replay must hold a minibatch and a collected batch")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive and finite")
        if self.exploration_budget is not None and (
                type(self.exploration_budget) is not int or self.exploration_budget <= self.warmup):
            raise ValueError("exploration_budget must be an integer above warmup")
        if self.exploration_decay_end is not None:
            if self.exploration_budget is not None:
                raise ValueError("Use exploration_budget or exploration_decay_end, not both")
            if (type(self.exploration_decay_end) is not int
                    or not self.warmup < self.exploration_decay_end <= self.budget):
                raise ValueError("exploration_decay_end must be above warmup and within budget")
        if (self.lr_drop_at is None) != (self.lr_after_drop is None):
            raise ValueError("lr_drop_at and lr_after_drop must be provided together")
        if self.lr_drop_at is not None:
            if (type(self.lr_drop_at) is not int
                    or not self.warmup < self.lr_drop_at < self.budget
                    or self.lr_drop_at % self.evaluation_interval):
                raise ValueError("lr_drop_at must be a validation boundary above warmup and below budget")
            if not np.isfinite(self.lr_after_drop) or not 0 < self.lr_after_drop < self.learning_rate:
                raise ValueError("lr_after_drop must be positive, finite and below the initial learning rate")
        if self.lr_steps and self.lr_drop_at is not None:
            raise ValueError("Use lr_steps or the legacy single drop, not both")
        previous_step, previous_rate = self.warmup, self.learning_rate
        for step, rate in self.lr_steps:
            if (type(step) is not int or not previous_step < step < self.budget
                    or step % self.evaluation_interval or not np.isfinite(rate)
                    or not 0 < rate < previous_rate):
                raise ValueError("LR steps must be ordered validation boundaries with decreasing positive rates")
            previous_step, previous_rate = step, rate
        if (len(set(self.selection_budgets)) != len(self.selection_budgets)
                or any(type(b) is not int or b <= 0 or b > self.budget
                       or b % self.evaluation_interval for b in self.selection_budgets)):
            raise ValueError("selection budgets must be unique scheduled validation points within budget")


def validate_checkpoint(checkpoint, config, regime, output):
    """The subprocess owns all evaluation RNGs; no training state is reseeded."""
    command = [sys.executable, "-m", "snake.cli", "--policy", "dqn",
               "--regimes", regime, f"--weights-{regime}", str(checkpoint),
               "--board-size", str(config.board_size), "--mask-size", str(config.mask_size),
               "--boards", str(config.n_boards), "--steps", str(config.steps),
               "--reward-profile", config.reward_profile, "--split", config.split,
               "--output", str(output)]
    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=ROOT)
    directory = Path(next(line.removeprefix("Results: ") for line in result.stdout.splitlines()
                          if line.startswith("Results: ")))
    payload = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    return payload["runs"][0]["summary"], directory, time.perf_counter() - started


def collect_step(env, network, replay, rng, step, horizon, exploration, canvas_side=None):
    """Only the deadline terminates; collisions and board regeneration continue."""
    states = encode(env.to_state(), step, horizon, canvas_side)
    actions = choose_actions(network, states, exploration, rng)
    rewards = env.move(actions)
    successor = encode(env.to_state(), step + 1, horizon, canvas_side)
    replay.add(states, actions, rewards, successor, step + 1 == horizon)
    return rewards[:, 0]


def source_fingerprints():
    files = ('train.py', 'snake/dqn.py', 'snake/training_state.py', 'snake/evaluation.py',
             'snake/environments_fully_observable.py', 'snake/environments_partially_observable.py')
    return {name: digest(ROOT / name) for name in files}


def compatible_resume(saved, settings, env_config, regime, evaluation_boards, fork):
    expected = json.loads(json.dumps(asdict(settings)))
    old = json.loads(json.dumps(saved['metadata']['training']))
    if (saved['metadata']['environment'] != asdict(env_config)
            or saved['metadata']['regime'] != regime
            or saved['metadata']['evaluation']['n_boards'] != evaluation_boards
            or saved['metadata']['source_sha256'] != source_fingerprints()
            or saved['metadata']['python'] != platform.python_version()
            or saved['metadata']['numpy'] != np.__version__
            or saved['metadata']['torch'] != str(torch.__version__)):
        raise ValueError('Resume environment, sources or software differs')
    if not fork:
        if old != expected:
            raise ValueError('Resume must keep the exact training configuration')
    else:
        schedule_keys = {'lr_steps', 'lr_drop_at', 'lr_after_drop'}
        if {k: v for k, v in old.items() if k not in schedule_keys} != {
                k: v for k, v in expected.items() if k not in schedule_keys}:
            raise ValueError('Fork may change only future LR steps')
        previous = TrainingConfig(**old)
        boundaries = {settings.warmup, saved['transitions']}
        for config in (previous, settings):
            boundaries.update(step for step, _ in config.lr_steps)
            if config.lr_drop_at is not None:
                boundaries.add(config.lr_drop_at)
        for point in boundaries:
            for t in (point, point + 1):
                if t <= saved['transitions'] and previous.learning_rate_at(t) != settings.learning_rate_at(t):
                    raise ValueError('Fork changes the already executed LR history')
    if saved['transitions'] % (env_config.n_boards * env_config.steps):
        raise ValueError('Resume is not at an episode-batch boundary')


def train(regime, env_config, settings, output, evaluation_boards=100,
          *, resume=None, fork=False, stop_at=None):
    settings.validate(env_config)
    if env_config.split != "validation":
        raise ValueError("Training cannot use the held-out test stream")
    evaluation_config = EvaluationConfig(board_size=env_config.board_size,
                                         mask_size=env_config.mask_size,
                                         n_boards=evaluation_boards, steps=env_config.steps,
                                         reward_profile=env_config.reward_profile)
    output = Path(output).resolve()
    if stop_at is None:
        stop_at = settings.budget
    if not 0 <= stop_at <= settings.budget or stop_at % settings.evaluation_interval:
        raise ValueError('stop_at must be a scheduled validation boundary within budget')
    saved = load_snapshot(resume) if resume is not None else None
    if fork and saved is None:
        raise ValueError('Fork requires a full snapshot')
    if saved is not None:
        compatible_resume(saved, settings, env_config, regime, evaluation_boards, fork)
        if saved['transitions'] > stop_at:
            raise ValueError('Resume would move backwards')
        same_directory = Path(resume).resolve().parent == output
        if same_directory == fork:
            raise ValueError('Resume stays in its directory; fork needs a new directory')
    output.mkdir(parents=True, exist_ok=saved is not None and not fork)
    # One thread avoids parallel overhead on these small CPU networks.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed_everything()
    rng = np.random.default_rng(0)
    side = input_side(env_config, regime)
    network = QNetwork(side * side * 4 + 1)
    target = deepcopy(network).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(network.parameters(), lr=settings.learning_rate)
    replay = ReplayBuffer(settings.replay_capacity, side * side * 4 + 1)
    exploration_budget = settings.exploration_budget or settings.budget
    metadata = {"algorithm": "DQN", "stage": output.parent.name,
                "regime": regime, "environment": asdict(env_config),
                "training": asdict(settings), "evaluation": asdict(evaluation_config),
                "gamma": 1, "hidden_sizes": [128, 128], "gradient_norm_clip": 10,
                "input_canvas_side": side,
                "parameter_count": sum(p.numel() for p in network.parameters()),
                "loss": "Huber threshold 1", "optimizer": "Adam",
                "adam_betas": [0.9, 0.999], "adam_eps": 1e-8, "weight_decay": 0,
                "learning_rate_schedule": {
                    "initial": settings.learning_rate, "steps": settings.lr_steps,
                    "drop_after_transitions": settings.lr_drop_at,
                    "after_drop": settings.lr_after_drop,
                    "boundary": "updates at the boundary retain the initial rate"},
                "epsilon": {"start": 1.0, "end": 0.05,
                            "decay_fraction": 0.30 if settings.exploration_decay_end is None else None,
                            "reference_budget": exploration_budget if settings.exploration_decay_end is None else None,
                            "decay_end_transitions": (settings.exploration_decay_end
                                if settings.exploration_decay_end is not None else
                                settings.warmup + 0.30 * (exploration_budget - settings.warmup))},
                "selection": "highest scheduled trained validation return; earlier exact tie",
                "validation_is_independent_test": False,
                "replay_bytes": replay.nbytes, "python": platform.python_version(),
                "numpy": np.__version__, "torch": str(torch.__version__),
                "device": "cpu", "torch_threads": 1, "source_sha256": source_fingerprints(),
                "created_utc": datetime.now(timezone.utc).isoformat()}
    config_path = output / "run.json"
    atomic_json(config_path, {**metadata, "status": "running"})
    transitions, updates, next_update = 0, 0, settings.warmup + settings.update_every
    best_return, best_step = -float("inf"), None
    selections = {}
    started = time.perf_counter()
    checkpoint = output / "last.pt"
    elapsed_before = 0.0
    if saved is not None:
        transitions, updates, next_update = (saved[k] for k in ('transitions', 'updates', 'next_update'))
        best_return, best_step = saved['best_return'], saved['best_step']
        selections = saved['selections']
        elapsed_before = saved['elapsed_seconds']
        for existing in output.glob('best*.pt'):
            if existing.name not in saved['artifacts']:
                existing.unlink()
        for name, data in saved['artifacts'].items():
            if Path(name).name != name:
                raise ValueError('Invalid snapshot artifact name')
            atomic_bytes(output / name, data)
        metadata['resumed_from'] = {'path': str(Path(resume).resolve()), 'sha256': digest(resume),
                                    'transitions': transitions, 'fork': fork}
        restore(saved['state'], network, target, optimizer, replay, rng)
    log_mode = 'a' if saved is not None else 'w'


    with (output / "training.csv").open(log_mode, newline="", encoding="utf-8") as training_file, \
            (output / "validation.csv").open(log_mode, newline="", encoding="utf-8") as validation_file:
        training_log = csv.DictWriter(training_file, fieldnames=(
            "transitions", "updates", "epsilon", "learning_rate", "mean_episode_return", "mean_loss",
            "max_abs_sampled_q", "board_completions", "elapsed_seconds"))
        validation_log = csv.DictWriter(validation_file, fieldnames=(
            "transitions", "return_mean", "return_std", "fruits_mean", "wall_hits_mean",
            "body_hits_mean", "board_completions_mean", "cells_removed_per_body_hit", "seconds", "results"))
        if saved is None:
            training_log.writeheader()
            validation_log.writeheader()

        def commit():
            training_file.flush()
            validation_file.flush()
            artifacts = {name: (output / name).read_bytes() for name in
                         ('training.csv', 'validation.csv', 'last.pt')}
            for name in ['best.pt'] + [f'best_{b}.pt' for b in settings.selection_budgets]:
                if (output / name).exists():
                    artifacts[name] = (output / name).read_bytes()
            status = 'complete' if transitions == settings.budget else 'paused'
            elapsed = elapsed_before + time.perf_counter() - started
            metadata.update(transitions=transitions, updates=updates,
                selected_transitions=best_step, selected_return=None if best_step is None else best_return,
                selections=selections, duration_seconds=elapsed, status=status)
            snapshot = dict(format=STATE_FORMAT, boundary='before_next_episode_batch',
                metadata=deepcopy(metadata), transitions=transitions, updates=updates,
                next_update=next_update, best_return=best_return, best_step=best_step,
                selections=deepcopy(selections), elapsed_seconds=elapsed, artifacts=artifacts,
                state=capture(network, target, optimizer, replay, rng))
            atomic_save(output / 'resume_latest.pt', snapshot, rotate=True)
            if transitions in settings.selection_budgets or transitions == settings.budget:
                atomic_save(output / f'resume_{transitions}.pt', snapshot)
            atomic_json(config_path, metadata)


        def evaluate():
            nonlocal best_return, best_step
            save_checkpoint(checkpoint, network, env_config, regime, transitions)
            summary, directory, seconds = validate_checkpoint(
                checkpoint, evaluation_config, regime, output / "evaluations")
            score = summary["return"]["mean"]
            validation_log.writerow({"transitions": transitions, "return_mean": score,
                "return_std": summary["return"]["std"],
                **{f"{key}_mean": summary[key]["mean"]
                   for key in ("fruits", "wall_hits", "body_hits", "board_completions")},
                "cells_removed_per_body_hit": summary["cells_removed_per_body_hit"],
                "seconds": seconds, "results": str(directory.resolve())})
            validation_file.flush()
            if transitions > 0 and score > best_return:
                best_return, best_step = score, transitions
                shutil.copyfile(checkpoint, output / "best.pt")
            if transitions in settings.selection_budgets:
                filename = "best.pt" if transitions == settings.budget else f"best_{transitions}.pt"
                if filename != "best.pt":
                    shutil.copyfile(output / "best.pt", output / filename)
                selections[str(transitions)] = {"file": filename, "transitions": best_step,
                                                "validation_return": best_return}
            print(f"{regime}: {transitions:,} transitions, validation return {score:.3f}, "
                  f"evaluation {seconds:.1f}s", flush=True)

        if saved is None:
            evaluate()
            commit()
        else:
            commit()  # Also repairs a milestone/metadata write interrupted after latest was committed.
        while transitions < stop_at:
            env = make_environment(env_config, regime)
            returns = np.zeros(env_config.n_boards, dtype=np.float64)
            completions = 0
            losses, max_q = [], 0.0
            for step in range(env_config.steps):
                exploration = epsilon(transitions, exploration_budget, settings.warmup,
                                      decay_end=settings.exploration_decay_end)
                rewards = collect_step(env, network, replay, rng, step,
                                       env_config.steps, exploration, side)
                returns += rewards
                # All declared profiles give completion a distinct reward.
                completions += int(np.count_nonzero(rewards == env.WIN_REWARD))
                transitions += env_config.n_boards
                # Account in individual transitions even when board count changes.
                while transitions >= next_update:
                    for group in optimizer.param_groups:
                        group["lr"] = settings.learning_rate_at(next_update)
                    loss, sampled_q = optimize(network, target, optimizer,
                                               replay.sample(settings.batch_size))
                    losses.append(loss)
                    max_q = max(max_q, sampled_q)
                    updates += 1
                    if updates % settings.target_every == 0:
                        target.load_state_dict(network.state_dict())
                    next_update += settings.update_every
            training_log.writerow({"transitions": transitions, "updates": updates,
                "epsilon": exploration, "learning_rate": optimizer.param_groups[0]["lr"],
                "mean_episode_return": float(returns.mean()),
                "mean_loss": float(np.mean(losses)) if losses else None,
                "max_abs_sampled_q": max_q if losses else None,
                "board_completions": completions,
                "elapsed_seconds": elapsed_before + time.perf_counter() - started})
            training_file.flush()
            if transitions % settings.evaluation_interval == 0:
                evaluate()
                commit()

    metadata.update(transitions=transitions, updates=updates,
                    selected_transitions=best_step, selected_return=None if best_step is None else best_return,
                    selections=selections,
                    duration_seconds=elapsed_before + time.perf_counter() - started,
                    status="complete" if transitions == settings.budget else "paused")
    atomic_json(config_path, metadata)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "main"), default="pilot")
    parser.add_argument("--regimes", nargs="+", choices=("full", "partial"),
                        default=["full", "partial"])
    parser.add_argument("--board-size", type=int, default=7)
    parser.add_argument("--mask-size", type=int, default=2)
    parser.add_argument("--reward-profile", choices=REWARD_PROFILES, default="R0")
    parser.add_argument("--boards", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--evaluation-interval", type=int)
    parser.add_argument("--evaluation-boards", type=int, default=100)
    parser.add_argument("--exploration-budget", type=int,
                        help="Reference budget for epsilon decay; defaults to the training budget")
    parser.add_argument("--exploration-decay-end", type=int,
                        help="Explicit transition at which epsilon reaches 0.05; exclusive with --exploration-budget")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-drop-at", type=int,
                        help="Keep the initial LR through this validation boundary, then reduce it once")
    parser.add_argument("--lr-after-drop", type=float,
                        help="Learning rate after --lr-drop-at (both options required)")
    parser.add_argument("--lr-step", action="append", default=[], metavar="TRANSITIONS:RATE")
    parser.add_argument("--resume", type=Path, help="Trusted complete local snapshot")
    parser.add_argument("--fork", action="store_true", help="Resume into a new run, changing only future LR steps")
    parser.add_argument("--stop-at", type=int, help="Pause at a validation boundary; budget remains unchanged")
    parser.add_argument("--run-directory", type=Path, help="Exact run directory (required for resume/fork)")
    parser.add_argument("--selection-budgets", nargs="+", type=int, default=[],
                        help="Also retain the best checkpoint within each declared transition budget")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "training")
    args = parser.parse_args(argv)
    budget = args.budget if args.budget is not None else (
        128_000 if args.stage == "pilot" else 1_024_000)
    interval = args.evaluation_interval if args.evaluation_interval is not None else (
        budget if args.stage == "pilot" else 128_000)
    try:
        config = EvaluationConfig(board_size=args.board_size, mask_size=args.mask_size,
                                  n_boards=args.boards, steps=args.steps,
                                  reward_profile=args.reward_profile)
        settings = TrainingConfig(budget=budget, evaluation_interval=interval,
                                  learning_rate=args.learning_rate,
                                  exploration_budget=args.exploration_budget,
                                  exploration_decay_end=args.exploration_decay_end,
                                  lr_drop_at=args.lr_drop_at, lr_after_drop=args.lr_after_drop,
                                  selection_budgets=tuple(args.selection_budgets),
                                  lr_steps=tuple((int(v.split(':')[0]), float(v.split(':')[1]))
                                                 for v in args.lr_step))
        settings.validate(config)
        if len(set(args.regimes)) != len(args.regimes):
            raise ValueError("Regimes must be unique")
        if args.evaluation_boards < 1:
            raise ValueError("Evaluation boards must be positive")
    except (ValueError, IndexError) as error:
        parser.error(str(error))
    if (args.resume or args.fork) and args.run_directory is None:
        parser.error('resume/fork requires --run-directory')
    if args.run_directory is not None and len(args.regimes) != 1:
        parser.error('--run-directory requires exactly one regime')
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for regime in args.regimes:
        directory = train(regime, config, settings, args.run_directory or args.output / stamp / args.stage / regime,
                          args.evaluation_boards, resume=args.resume, fork=args.fork, stop_at=args.stop_at)
        print(f"Training results: {directory}", flush=True)


if __name__ == "__main__":
    main()
