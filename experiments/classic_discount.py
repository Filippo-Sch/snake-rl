"""Final planned full DQN discount probe. Training requires an explicit command."""
import argparse
import json
from pathlib import Path

from snake.classic_discount import preflight, report, settings, train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "check", "train", "report"), default="plan")
    parser.add_argument("--parent-root", type=Path, default=Path("results/phase2_extended"))
    parser.add_argument("--control-root", type=Path, default=Path("results/phase2_policy"))
    parser.add_argument("--output", type=Path, default=Path("results/phase2_discount"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.resume and args.phase != "train":
            raise ValueError("--resume requires --phase train")
        if args.phase == "plan":
            result = dict(settings=settings(), status="plan-only", control="existing-full-DQN-gamma-1")
        elif args.phase == "check":
            parent, state, hashes, control = preflight(args.parent_root, args.control_root)
            result = dict(settings=settings(), parent=str(parent), hashes=hashes, control=control,
                          replay_size=state["replay"]["size"], optimizer_updates_performed=0)
        elif args.phase == "report":
            result = report(args.output, args.control_root)
        else:
            summary = train(args.parent_root, args.control_root, args.output, args.resume)
            result = dict(complete=summary["complete"], counter=summary["counter"],
                          selected_counter=summary["selected_counter"], automatic_followup=False)
        print(json.dumps(result, indent=2))
        return result
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

