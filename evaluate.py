"""Verify the report and evaluate the submitted Snake agents."""

import argparse
from pathlib import Path

from analysis.common import ROOT
from analysis.runner import evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="By default, replay all 11,000 final-policy test episodes (500 per row).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Run a short functional sample instead of the final test bank",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Also replay 4,000 native diagnostic trajectories after the final tests",
    )
    mode.add_argument(
        "--archive-only",
        action="store_true",
        help="Reconstruct and verify the report without simulating episodes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results",
        help="Results directory (default: results)",
    )
    args = parser.parse_args(argv)
    try:
        evaluate(
            args.output.resolve(),
            quick=args.quick,
            full=args.full,
            archive_only=args.archive_only,
        )
    except (ValueError, KeyError, FileNotFoundError, ImportError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
