"""Classic scratch campaign. Importing or printing its plan never trains."""
import argparse
import json
from pathlib import Path

from .config import CONFIG, ROOT, configure_torch, manifest
from .evaluation import evaluate, write_evaluation
from .training import assert_compatible, run_stage


def run_audit(root, regime):
    for label, policy, role in (("audit", "bfs", "audit"),
                                ("baselines/bfs", "bfs", "validation"),
                                ("baselines/greedy", "greedy", "validation")):
        folder = Path(root) / regime / label
        if (folder / "results.json").exists():
            assert_compatible(json.loads((folder / "results.json").read_text())["manifest"])
            continue
        result = evaluate(regime, heuristic=policy, role=role)
        write_evaluation(folder, result)
        print(f"{regime}/{label}: {result['summary']['wins']} wins", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "audit", "scratch"), default="plan")
    parser.add_argument("--regimes", nargs="+", choices=("full", "partial"), default=["full", "partial"])
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase2")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if len(set(args.regimes)) != len(args.regimes):
            raise ValueError("Duplicate regimes")
        if args.resume and args.phase != "scratch":
            raise ValueError("--resume applies only to scratch training")
        if args.phase == "plan":
            result = dict(manifest=manifest(), sequence=["audit", "scratch"],
                budget_per_regime=CONFIG.online_budget, note="Plan only; explicit --phase scratch starts training")
            print(json.dumps(result, indent=2))
            return result
        configure_torch()
        for regime in args.regimes:
            if args.phase == "audit":
                run_audit(args.output, regime)
            else:
                record = json.loads((args.output / regime / "audit/results.json").read_text())
                assert_compatible(record["manifest"])
                if record["role"] != "audit" or record["summary"]["games"] != CONFIG.audit_games:
                    raise ValueError("Complete --phase audit before scratch training")
                run_stage(args.output, regime, "scratch", args.resume)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
