"""Plan/check by default; training requires an explicit --phase train command."""
import argparse
import json
from pathlib import Path

from snake.classic_policy import (CHECKPOINT, START, checked_parent, report,
                                  run_experiment, settings)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "check", "train", "report"), default="plan")
    parser.add_argument("--regime", choices=("full", "partial"))
    parser.add_argument("--arm", choices=("control", "variant"))
    parser.add_argument("--parent-root", type=Path, default=Path("results/phase2_extended"))
    parser.add_argument("--output", type=Path, default=Path("results/phase2_policy"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.resume and args.phase != "train":
            raise ValueError("--resume requires --phase train")
        regimes = [args.regime] if args.regime else list(START)
        arms = [args.arm] if args.arm else ["control", "variant"]
        if args.phase == "plan":
            result = dict(runs=[settings(r, a) for r in regimes for a in arms],
                          checkpoints=CHECKPOINT,
                          note="Plan only: no checkpoint writes, evaluation or training.")
        elif args.phase == "check":
            result = {}
            for regime in regimes:
                folder, state, hashes = checked_parent(args.parent_root, regime)
                replay = state["replay"]
                arrays = replay["arrays"]
                wins = arrays["terminated"] & (arrays["rewards"] > 100)
                result[regime] = dict(parent=str(folder), counter=state["summary"]["counter"],
                                      replay_size=replay["size"], winning_terminals=int(wins.sum()),
                                      hashes=hashes, updates_performed=0)
        elif args.phase == "report":
            result = report(args.output)
        else:
            if not args.regime or not args.arm:
                raise ValueError("Training requires explicit --regime and --arm (one run at a time)")
            result = run_experiment(args.parent_root, args.output, args.regime,
                                    args.arm, args.resume)
        print(json.dumps(result, indent=2))
        return result
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

