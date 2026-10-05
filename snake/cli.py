"""Run an individual native or Classic policy evaluation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from snake.baselines import make_bfs, make_direct, make_greedy
from snake.evaluation import (
    REWARD_PROFILES,
    EvaluationConfig,
    PolicySpec,
    print_summary,
    run_evaluation,
    seed_everything,
)

ROOT = Path(__file__).resolve().parents[1]
POLICIES = {"greedy": make_greedy, "bfs": make_bfs, "direct": make_direct}


def selected_weights(environment, board, regime):
    key = (
        f"native_b{board}_{regime}" if environment == "native" else f"classic_{regime}"
    )
    records = json.loads((ROOT / "weights/manifest.json").read_text(encoding="utf-8"))
    if key not in records["policies"]:
        raise ValueError(
            f"No delivered weights for {key}; supply an explicit checkpoint"
        )
    record = records["policies"][key]
    path = ROOT / "weights" / record["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"Delivered checkpoint checksum mismatch: {path.name}")
    return path


def classic_evaluation(args, names, paths):
    from snake.classic.config import configure_torch
    from snake.classic.learner import load_policy
    from snake.classic.submission_evaluation import evaluate_with_revisits
    from snake.classic.evaluation import write_evaluation

    configure_torch()
    seed_everything()
    # Validate all requested weights before any episodes are simulated.
    networks = (
        {r: load_policy(paths[r], r)[0] for r in args.regimes} if "dqn" in names else {}
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = args.output / stamp
    for regime in args.regimes:
        for name in names:
            result = evaluate_with_revisits(
                regime,
                role=args.split,
                games=args.boards,
                network=networks.get(regime) if name == "dqn" else None,
                heuristic=None if name == "dqn" else name,
            )
            result["checkpoint_hash"] = (
                hashlib.sha256(paths[regime].read_bytes()).hexdigest()
                if name == "dqn"
                else None
            )
            write_evaluation(directory / regime / name, result)
            s = result["summary"]
            print(
                f"{regime}/{name}: wins={s['wins']}/{args.boards}, fruits={s['mean_fruits']:.3f}, "
                f"return={s['mean_reward']:.3f}, walls={s['outcomes']['wall_death']}, "
                f"state_revisits={s['physical_revisits']}",
                flush=True,
            )
    print(f"Results: {directory}")
    return directory


def main(argv=None, *, baseline_only=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment", choices=("native", "classic"), default="native"
    )
    parser.add_argument("--policy", action="append", choices=(*POLICIES, "dqn"))
    parser.add_argument(
        "--regimes", nargs="+", choices=("full", "partial"), default=["full", "partial"]
    )
    parser.add_argument("--board-size", type=int, default=7)
    parser.add_argument(
        "--mask-size",
        type=int,
        default=2,
        help="Native observation radius (2 means 5x5)",
    )
    parser.add_argument(
        "--boards",
        type=int,
        default=100,
        help="Number of games; default validation demonstration: 100",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Native horizon; classic cap is fixed at 5000",
    )
    parser.add_argument("--reward-profile", choices=REWARD_PROFILES, default="R0")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--test-stream",
        type=int,
        default=None,
        help="Native test substream; final study uses 3",
    )
    parser.add_argument(
        "--record-steps", action="store_true", help="Native per-step diagnostics"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/evaluation")
    for regime in ("full", "partial"):
        parser.add_argument(f"--weights-{regime}", type=Path)
    args = parser.parse_args(argv)
    try:
        if len(set(args.regimes)) != len(args.regimes):
            raise ValueError("Regimes must be unique")
        if args.boards < 1:
            raise ValueError("--boards must be positive")
        defaults = list(POLICIES) if args.environment == "native" else ["greedy", "bfs"]
        names = args.policy or defaults + ([] if baseline_only else ["dqn"])
        if len(set(names)) != len(names):
            raise ValueError("Policies must be unique")
        if baseline_only and "dqn" in names:
            raise ValueError("baseline.py only runs non-learning policies")
        if args.environment == "classic":
            if "direct" in names:
                raise ValueError("Classic supports BFS, Greedy and DQN")
            if (
                args.board_size != 7
                or args.mask_size != 2
                or args.steps not in (None, 5000)
                or args.reward_profile != "R0"
                or args.test_stream is not None
                or args.record_steps
            ):
                raise ValueError(
                    "Classic uses fixed board/window/rewards, cap 5000 and its own test bank; native-only options cannot be applied"
                )
        paths = {}
        if "dqn" in names:
            for regime in args.regimes:
                explicit = getattr(args, f"weights_{regime}")
                paths[regime] = (
                    explicit
                    if explicit is not None
                    else selected_weights(args.environment, args.board_size, regime)
                )
                if not paths[regime].is_file():
                    raise FileNotFoundError(
                        f"Required DQN checkpoint missing: {paths[regime]}"
                    )
        if args.environment == "classic":
            return classic_evaluation(args, names, paths)
        config = EvaluationConfig(
            board_size=args.board_size,
            mask_size=args.mask_size,
            n_boards=args.boards,
            steps=1000 if args.steps is None else args.steps,
            record_steps=args.record_steps,
            reward_profile=args.reward_profile,
            split=args.split,
            test_stream=(
                args.test_stream
                if args.test_stream is not None
                else (3 if args.split == "test" else 1)
            ),
        )
        specs = []
        for name in names:
            if name == "dqn":
                import torch
                from snake.dqn import make_dqn_spec

                torch.set_num_threads(1)
                torch.use_deterministic_algorithms(True)
                specs.append(make_dqn_spec(config, paths))
            else:
                specs.append(PolicySpec(name, POLICIES[name]))
        directory, payload = run_evaluation(specs, config, args.regimes, args.output)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print_summary(payload)
    print(f"Results: {directory}")
    return directory


if __name__ == "__main__":
    main()
