"""Train every experiment and reconstruct its results."""

import argparse
from pathlib import Path

from analysis.campaign import plan, train
from analysis.common import ROOT, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/from_scratch")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume committed snapshots in the same directory",
    )
    parser.add_argument(
        "--plan", action="store_true", help="Write the command plan without training"
    )
    args = parser.parse_args(argv)
    try:
        if args.plan:
            write_json(args.output / "plan.json", plan(args.output))
            print(f"Plan: {args.output / 'plan.json'}")
        else:
            train(args.output, resume=args.resume, figures=True)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
