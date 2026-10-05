"""Explicit long-running campaign and independent final-policy verification."""

from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys

from .common import (
    ROOT,
    CASES,
    SELECTED,
    near,
    read,
    require,
    rows,
    sha,
    write_json,
)


def plan(output):
    output = Path(output).resolve()
    py = sys.executable
    native = output / "native"
    parent = output / "phase2"
    extended = output / "phase2_extended"
    policy = output / "phase2_policy"
    stages = [
        (
            "native",
            [
                py,
                "-m",
                "experiments.native",
                "--phase",
                "train",
                "--output",
                str(native),
            ],
            None,
        ),
        (
            "classic_audit",
            [
                py,
                "-m",
                "experiments.classic",
                "--phase",
                "audit",
                "--output",
                str(parent),
            ],
            None,
        ),
        (
            "classic_scratch",
            [
                py,
                "-m",
                "experiments.classic",
                "--phase",
                "scratch",
                "--output",
                str(parent),
            ],
            "phase2",
        ),
        (
            "classic_extend",
            [
                py,
                "-m",
                "experiments.classic_extend",
                "--phase",
                "train",
                "--parent-root",
                str(parent),
                "--output",
                str(extended),
            ],
            "phase2_extended",
        ),
        (
            "full_control",
            [
                py,
                "-m",
                "experiments.classic_policy",
                "--phase",
                "train",
                "--regime",
                "full",
                "--arm",
                "control",
                "--parent-root",
                str(extended),
                "--output",
                str(policy),
            ],
            "phase2_policy/full/control",
        ),
        (
            "full_lower_lr",
            [
                py,
                "-m",
                "experiments.classic_policy",
                "--phase",
                "train",
                "--regime",
                "full",
                "--arm",
                "variant",
                "--parent-root",
                str(extended),
                "--output",
                str(policy),
            ],
            "phase2_policy/full/variant",
        ),
        (
            "partial_protected",
            [
                py,
                "-m",
                "experiments.classic_policy",
                "--phase",
                "train",
                "--regime",
                "partial",
                "--arm",
                "variant",
                "--parent-root",
                str(extended),
                "--output",
                str(policy),
            ],
            "phase2_policy/partial/variant",
        ),
        (
            "full_watkins",
            [
                py,
                "-m",
                "experiments.classic_watkins",
                "--phase",
                "train",
                "--parent-root",
                str(extended),
                "--control-root",
                str(policy),
                "--output",
                str(output / "phase2_watkins"),
            ],
            "phase2_watkins/full/watkins",
        ),
        (
            "full_discount",
            [
                py,
                "-m",
                "experiments.classic_discount",
                "--phase",
                "train",
                "--parent-root",
                str(extended),
                "--control-root",
                str(policy),
                "--output",
                str(output / "phase2_discount"),
            ],
            "phase2_discount/full/discount",
        ),
    ]
    return dict(
        seed=0,
        root=str(output),
        unique_native_transitions=22528000,
        unique_classic_transitions=13312000,
        stages=[
            dict(name=name, command=cmd, resume_directory=resume)
            for name, cmd, resume in stages
        ],
        after_training=[
            "validation-only selection",
            "evaluation of selected and comparison checkpoints",
            "search/spawn replays, including the earlier partial checkpoint",
            "replay terminal counts",
            "export compact evidence",
            "reconstruct tables and optional figures",
        ],
        note="Plan only. Run python -m experiments.campaign without --plan to execute these stages and then the reconstruction pipeline.",
    )


def native_specs(folder, case):
    from snake.baselines import make_greedy, make_bfs, make_direct
    from snake.dqn import make_dqn_spec
    from snake.evaluation import EvaluationConfig, PolicySpec

    regime = case.split("_")[1]
    config = EvaluationConfig(
        board_size=int(case.split("_")[0][1:]),
        n_boards=500,
        steps=1000,
        split="test",
        test_stream=3,
    )
    specs = [
        PolicySpec(name, factory)
        for name, factory in [
            ("greedy", make_greedy),
            ("bfs", make_bfs),
            ("direct", make_direct),
        ]
    ]
    a = folder / "training" / case / "A"
    for budget in (1024000, 2048000):
        path = a / f"best_{budget}.pt"
        if not path.exists() and read(a / "run.json")["training"]["budget"] == budget:
            path = a / "best.pt"
        specs.append(
            replace(make_dqn_spec(config, {regime: path}), name=f"dqn_{budget}")
        )
    if case.startswith("b11"):
        for variant in ("A", "B", "C"):
            for role, filename in [("selected", "best.pt"), ("final", "last.pt")]:
                path = folder / "training" / case / variant / filename
                specs.append(
                    replace(
                        make_dqn_spec(config, {regime: path}),
                        name=f"dqn_{variant}_{role}",
                    )
                )
    return config, specs


def freeze_weights(output):
    # Select the LR schedule before any test data are generated.
    from snake.classic.learner import load_policy
    from snake.classic.evaluation import selection_key
    import torch

    weights = output / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    selected = {}
    selection = {}
    for case in CASES:
        candidates = []
        for variant in ("A",) if case.startswith("b7") else ("A", "B", "C"):
            folder = output / "native/training" / case / variant
            records = rows(folder / "validation.csv")
            candidates.append(
                (
                    sum(float(r["return_mean"]) for r in records[-8:]) / 8,
                    variant,
                    folder,
                )
            )
        _, variant, folder = max(candidates, key=lambda r: r[0])
        # The shipped experiment continuations implement the declared C schedule.
        # If a retraining realization changes selection, stop instead of labelling
        # a different policy as the policy described in the historical report.
        require(
            case.startswith("b7") or variant == "C",
            f"{case}: retraining selected {variant}, historical report selected C",
        )
        path = folder / "best.pt"
        meta = torch.load(path, map_location="cpu", weights_only=True)["metadata"]
        key = "native_" + case
        dest = weights / (key + ".pt")
        shutil.copyfile(path, dest)
        selected[key] = dict(
            file=dest.name,
            environment="native",
            regime=case.split("_")[1],
            board_size=int(case.split("_")[0][1:]),
            selected_transitions=meta["transitions"],
            sha256=sha(dest),
            seed=0,
        )
        selection[case] = dict(
            schedule=variant,
            selected_transitions=meta["transitions"],
            criterion="last-eight validation mean, then maximum checkpoint return",
        )
    for regime in ("full", "partial"):
        folder = output / "phase2_extended" / regime / "scratch"
        record = read(folder / "run.json")
        eligible = [h for h in record["history"] if h["counter"] > 0]
        best = max(eligible, key=lambda h: selection_key(h["summary"]))
        require(
            best["counter"] == record["selected_counter"],
            "Classic recorded selection differs",
        )
        path = folder / "best.pt"
        _, meta = load_policy(path, regime)
        key = "classic_" + regime
        dest = weights / (key + ".pt")
        shutil.copyfile(path, dest)
        selected[key] = dict(
            file=dest.name,
            environment="classic",
            regime=regime,
            selected_transitions=record["selected_counter"],
            sha256=sha(dest),
            seed=0,
        )
    manifest = dict(format="snake-submission-weights-v1", policies=selected)
    write_json(weights / "manifest.json", manifest)
    write_json(output / "selection.json", selection)
    return manifest


def collect_campaign(output):
    """Evaluate fresh campaign weights and passively reconstruct its diagnostics."""
    import torch
    from snake.evaluation import EvaluationConfig, PolicySpec, run_evaluation
    from snake.baselines import make_direct
    from snake.classic.config import configure_torch
    from snake.classic.learner import load_policy
    from snake.classic.submission_evaluation import evaluate_with_revisits
    from snake.classic.evaluation import write_evaluation
    from .diagnostics import replay
    from .export_evidence import export
    from .verification import verify

    configure_torch()
    manifest = freeze_weights(output)
    native = output / "native"
    for size in (7, 11):
        directory, _ = run_evaluation(
            [PolicySpec("direct", make_direct)],
            EvaluationConfig(board_size=size),
            ("full", "partial"),
            native / "validation_baselines" / f"b{size}" / "runs",
        )
        for filename in ("results.json", "per_board.csv"):
            shutil.copyfile(
                directory / filename,
                native / "validation_baselines" / f"b{size}" / filename,
            )
    for case in CASES:
        config, specs = native_specs(native, case)
        directory, _ = run_evaluation(
            specs, config, (case.split("_")[1],), native / "test" / case / "runs"
        )
        for filename in ("results.json", "per_board.csv"):
            shutil.copyfile(directory / filename, native / "test" / case / filename)
    for regime in ("full", "partial"):
        network, _ = load_policy(output / "weights" / f"classic_{regime}.pt", regime)
        for name in ("bfs", "greedy", "dqn"):
            result = evaluate_with_revisits(
                regime,
                role="test",
                games=500,
                network=network if name == "dqn" else None,
                heuristic=None if name == "dqn" else name,
            )
            write_evaluation(output / "classic_test" / regime / name, result)
    # Provide the minimal native reference paths needed by the passive replay.
    replay_refs = output / "diagnostic_references"
    for case in CASES:
        folder = replay_refs / "native" / case
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(native / "test" / case / "per_board.csv", folder / "test.csv")
        d = read(native / "test" / case / "results.json")
        write_json(folder / "test_metadata.json", d)
    weights = {case: output / "weights" / ("native_" + case + ".pt") for case in CASES}
    replay(
        replay_refs,
        output / "diagnostics",
        weights=weights,
        historical_base=native / "training/b11_partial/A/best_2048000.pt",
    )
    snapshots = []
    for regime in ("full", "partial"):
        path = output / "phase2_extended" / regime / "scratch/last.resume.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        arrays = state["replay"]["arrays"]
        n = state["replay"]["size"]
        terminal = arrays["terminated"][:n]
        reward = arrays["rewards"][:n]
        snapshots.append(
            dict(
                regime=regime,
                counter=state["summary"]["counter"],
                replay_size=n,
                terminal_wins=int((terminal & (reward > 100)).sum()),
                terminal_deaths=int((terminal & (reward < 0)).sum()),
                path=str(path.relative_to(output)),
                sha256=sha(path),
            )
        )
    write_json(output / "diagnostics/replay.json", dict(seed=0, snapshots=snapshots))
    export(output, output / "evidence", weights_manifest=manifest)
    verify(output / "evidence", output / "reconstruction", check_delivered=False)


def train(output, *, resume=False, figures=False):
    output = Path(output).resolve()
    require(
        ROOT / "evidence" != output and ROOT / "weights" != output,
        "Use a separate campaign output directory",
    )
    require(
        not output.exists() or resume or not any(output.iterdir()),
        "Campaign directory is non-empty; use --resume or a new --output",
    )
    output.mkdir(parents=True, exist_ok=True)
    stages = plan(output)
    write_json(output / "plan.json", stages)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    for stage in stages["stages"]:
        command = list(stage["command"])
        relative = stage["resume_directory"]
        if resume and relative:
            if relative in ("phase2", "phase2_extended"):
                # Each regime may have a different completion/resume state.
                for regime in ("full", "partial"):
                    folder = output / relative / regime / "scratch"
                    if (folder / "run.json").exists() and read(folder / "run.json")[
                        "complete"
                    ]:
                        continue
                    cmd = command + ["--regimes", regime]
                    if (folder / "resume.pt").exists():
                        cmd += ["--resume"]
                    print("Training stage:", stage["name"], regime, flush=True)
                    with (logs / (stage["name"] + "_" + regime + ".log")).open(
                        "a", encoding="utf-8"
                    ) as stream:
                        subprocess.run(
                            cmd,
                            cwd=ROOT,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                continue
            folder = output / relative
            if (folder / "run.json").exists() and read(folder / "run.json")["complete"]:
                continue
            if (folder / "resume.pt").exists():
                command += ["--resume"]
        print("Training stage:", stage["name"], "(log in", logs, ")", flush=True)
        with (logs / (stage["name"] + ".log")).open("a", encoding="utf-8") as stream:
            subprocess.run(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True
            )
    collect_campaign(output)
    if figures:
        from .figures import build

        build(output / "evidence", output / "reconstruction/figures")
    print("Completed independent campaign:", output, flush=True)


def evaluate_delivered(data, output, *, progress=None):
    from snake.cli import main

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    matched = 0
    for size in (7, 11):
        if progress:
            progress(f"Native {size} x {size}: running 4,000 final test episodes...")
        directory = main(
            [
                "--board-size",
                str(size),
                "--split",
                "test",
                "--test-stream",
                "3",
                "--boards",
                "500",
                "--output",
                str(output / "native"),
            ]
        )
        actual = rows(directory / "per_board.csv")
        for regime in ("full", "partial"):
            case = f"b{size}_{regime}"
            reference = {
                (r["policy"], int(r["board_id"])): r
                for r in rows(data / "native" / case / "test.csv")
            }
            for row in (r for r in actual if r["regime"] == regime):
                name = SELECTED[case] if row["policy"] == "dqn" else row["policy"]
                old = reference[name, int(row["board_id"])]
                for metric in (
                    "fruits",
                    "wall_hits",
                    "body_hits",
                    "board_completions",
                    "body_cells_removed",
                    "body_length_before_hits",
                    "return",
                ):
                    near(row[metric], old[metric], case + "/" + metric, tolerance=1e-9)
                matched += 1
        if progress:
            progress(f"Native {size} x {size}: all 4,000 episode records matched.")
    if progress:
        progress("Classic: running 3,000 final test episodes...")
    directory = main(
        [
            "--environment",
            "classic",
            "--split",
            "test",
            "--boards",
            "500",
            "--output",
            str(output / "classic"),
        ]
    )
    for regime in ("full", "partial"):
        for policy in ("greedy", "bfs", "dqn"):
            got = read(directory / regime / policy / "results.json")["records"]
            old = read(data / "classic/test" / regime / policy / "results.json")[
                "records"
            ]
            require(got == old, f"Classic episode records differ: {regime}/{policy}")
            matched += len(got)
    if progress:
        progress("Classic: all 3,000 episode records matched.")
    write_json(
        output / "verification.json",
        dict(
            status="passed",
            records_compared=matched,
            seed=0,
            scope="11,000 final-policy episode records compared with the delivered historical evidence. No training.",
        ),
    )
    print(f"Passed: {matched} independently replayed episode records.", flush=True)
