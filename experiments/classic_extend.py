"""Continue the current scratch configurations to 4.096M. Default: plan only."""

import argparse
import json
from pathlib import Path

from snake.classic.config import ROOT
from snake.classic_extension import (BASE_BUDGET, TOTAL_BUDGET, checked_parent,
                                     extension_report, run_extension)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "train", "report"), default="plan")
    parser.add_argument("--parent-root", type=Path, default=ROOT / "results/phase2")
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase2_extended")
    parser.add_argument("--regimes", nargs="+", choices=("full", "partial"), default=["full", "partial"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if len(args.regimes) != len(set(args.regimes)):
            raise ValueError("Duplicate regimes")
        if args.resume and args.phase != "train":
            raise ValueError("--resume applies only to --phase train")
        if args.phase == "plan":
            plan = dict(base_budget=BASE_BUDGET, total_budget=TOTAL_BUDGET,
                        additional_per_regime=TOTAL_BUDGET - BASE_BUDGET,
                        regimes=args.regimes, parent_root=str(args.parent_root),
                        output=str(args.output), learning_rate=1e-4, epsilon=0.01,
                        validation_interval=128000,
                        note="Plan only. Continue final full states; no exploration reset or automatic test.")
            print(json.dumps(plan, indent=2))
            return plan
        if args.phase == "report":
            report = extension_report(args.parent_root, args.output)
            print(json.dumps(report, indent=2))
            return report
        # Validate all requested parents/destinations before launching any run.
        for regime in args.regimes:
            folder = args.output.resolve() / regime / "scratch"
            if args.resume:
                from snake.classic_extension import ExtendedScratchRun
                from snake.classic.training import read_snapshot
                run = ExtendedScratchRun(folder, regime)
                run.restore(read_snapshot(folder / "resume.pt"))
            else:
                parent, _, _ = checked_parent(args.parent_root, regime)
                if folder == parent or parent in folder.parents:
                    raise ValueError("Use a separate extension output directory")
                if folder.exists() and any(folder.iterdir()):
                    raise FileExistsError(f"Extension already exists; use --resume: {folder}")
        for regime in args.regimes:
            run_extension(args.parent_root, args.output, regime, args.resume)
        extension_report(args.parent_root, args.output)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
