"""Full-only Watkins Q(lambda): plan/check by default; explicit training."""
import argparse
import json
from pathlib import Path

from snake.classic_watkins import SequenceReplay
from snake.classic_watkins_run import preflight, report, settings, train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "check", "train", "report"), default="plan")
    parser.add_argument("--parent-root", type=Path, default=Path("results/phase2_extended"))
    parser.add_argument("--control-root", type=Path, default=Path("results/phase2_policy"))
    parser.add_argument("--output", type=Path, default=Path("results/phase2_watkins"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.resume and args.phase != "train":
            raise ValueError("--resume requires --phase train")
        if args.phase == "plan":
            result = dict(settings=settings(), status="plan-only", control="existing-full-DQN")
        elif args.phase == "check":
            parent, state, hashes, control = preflight(args.parent_root, args.control_root)
            replay = SequenceReplay.from_state(state["replay"])
            result = dict(settings=settings(), parent=str(parent), hashes=hashes,
                          replay_size=replay.size, successor_links=int((replay.successor >= 0).sum()),
                          control=control, optimizer_updates=0)
        elif args.phase == "report":
            result = report(args.output, args.control_root)
        else:
            summary = train(args.parent_root, args.control_root, args.output, args.resume)
            result = dict(complete=summary["complete"], counter=summary["counter"],
                          selected_counter=summary["selected_counter"])
        print(json.dumps(result, indent=2))
        return result
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

