"""Report reconstruction and independent policy checks."""

from contextlib import redirect_stdout
from pathlib import Path
import sys
import time

from snake.cli import main as evaluate_policies
from .common import ROOT, near, read, require, rows, write_json
from .verification import verify
from .figures import build as plot_figures
from .presentation import build as present

NATIVE_SAMPLE = 16
CLASSIC_SAMPLE = 10


def print_sample(live):
    """Show simulated performance without mixing it with the report test bank."""
    print(f"\nNative live sample ({NATIVE_SAMPLE} validation episodes per row):")
    print("Board  View     Policy     Fruits   Walls    Body   Return")
    for row in live["native"]:
        s = row["summary"]
        print(
            f"{row['board']:>5}  {row['regime']:<7}  {row['policy']:<7}"
            f"  {s['fruits']['mean']:>7.2f}  {s['wall_hits']['mean']:>6.2f}"
            f"  {s['body_hits']['mean']:>6.2f}  {s['return']['mean']:>7.2f}"
        )
    print(f"\nClassic live sample ({CLASSIC_SAMPLE} held-out games per row):")
    print("View     Policy    Wins    Fruits   Return  Deaths  Timeouts")
    for row in live["classic"]:
        s = row["summary"]
        deaths = s["outcomes"]["wall_death"] + s["outcomes"]["body_death"]
        timeouts = s["outcomes"]["total_limit"] + s["outcomes"]["fruit_free_limit"]
        print(
            f"{row['regime']:<7}  {row['policy']:<7}  {s['wins']:>2}/{CLASSIC_SAMPLE}"
            f"  {s['mean_fruits']:>7.2f}  {s['mean_reward']:>7.2f}"
            f"  {deaths:>6}  {timeouts:>8}"
        )
    print(
        "\nReport tables use 500 test episodes per row; see the results page.",
        flush=True,
    )


def quick_rollouts(output):
    """Run every submitted agent and baseline on a bounded evaluation sample."""
    native_rows = []
    for size in (7, 11):
        directory = evaluate_policies(
            [
                "--board-size",
                str(size),
                "--boards",
                str(NATIVE_SAMPLE),
                "--output",
                str(output / "native"),
            ]
        )
        payload = read(directory / "results.json")
        records = rows(directory / "per_board.csv")
        for row in records:
            expected = (
                0.5 * float(row["fruits"])
                - 0.1 * float(row["wall_hits"])
                - 0.2 * float(row["body_hits"])
                + 0.5 * float(row["board_completions"])
            )
            near(row["return"], expected, "Live native reward", tolerance=3e-6)
        for run in payload["runs"]:
            native_rows.append(
                dict(
                    board=size,
                    regime=run["regime"],
                    policy=run["policy"],
                    summary=run["summary"],
                )
            )
    directory = evaluate_policies(
        [
            "--environment",
            "classic",
            "--split",
            "test",
            "--boards",
            str(CLASSIC_SAMPLE),
            "--output",
            str(output / "classic"),
        ]
    )
    classic_rows = []
    matched = 0
    for regime in ("full", "partial"):
        for policy in ("bfs", "greedy", "dqn"):
            result = read(directory / regime / policy / "results.json")
            reference = read(
                ROOT / "evidence/classic/test" / regime / policy / "results.json"
            )
            require(
                result["records"] == reference["records"][:CLASSIC_SAMPLE],
                f"Live Classic records differ: {regime}/{policy}",
            )
            matched += CLASSIC_SAMPLE
            classic_rows.append(
                dict(regime=regime, policy=policy, summary=result["summary"])
            )
    return dict(
        status="passed",
        seed=0,
        native_games=16 * NATIVE_SAMPLE,
        classic_records_compared=matched,
        native=native_rows,
        classic=classic_rows,
        note=f"Native: {NATIVE_SAMPLE} validation boards per policy, a separate RNG allocation from the report bank. Classic: first {CLASSIC_SAMPLE} held-out games per policy, compared exactly.",
    )


def evaluate(output, *, quick=False, full=False, archive_only=False):
    require(sum((quick, full, archive_only)) <= 1, "Choose one evaluation mode")
    output = Path(output)
    require(
        output != ROOT
        and not any(
            output.is_relative_to(ROOT / name)
            for name in (
                "evidence",
                "weights",
                "report",
                "snake",
                "analysis",
                "experiments",
                "tests",
            )
        ),
        "Choose a separate results directory",
    )
    output.mkdir(parents=True, exist_ok=True)
    # Remove a prior success marker so an interrupted run cannot appear complete.
    for name in (
        "index.html",
        "verification.json",
        "run.json",
        "live_checks.json",
        "evaluation.log",
    ):
        (output / name).unlink(missing_ok=True)
    started = time.monotonic()
    data = ROOT / "evidence"
    print("Checking evidence, checkpoints and all four report tables...", flush=True)
    result = verify(data, output)
    print(
        f"Report: {result['report']['printed_values_checked']} printed values matched.",
        flush=True,
    )
    plot_figures(data, output / "figures")
    mode = (
        "archive" if archive_only else "quick" if quick else "full" if full else "final"
    )
    live = None
    if not archive_only:
        print(
            (
                "Running all agents and baselines on a short sample..."
                if quick
                else "Replaying final tests: 500 episodes per row in both environments..."
            ),
            flush=True,
        )
        console = sys.stdout

        def progress(message):
            print(message, file=console, flush=True)

        with (output / "evaluation.log").open(
            "w", encoding="utf-8"
        ) as log, redirect_stdout(log):
            if quick:
                live = quick_rollouts(output / "sample")
            else:
                from .campaign import evaluate_delivered

                evaluate_delivered(data, output / "rollouts", progress=progress)
                live = dict(
                    evaluation=read(output / "rollouts/verification.json"),
                    status="passed",
                )
                if full:
                    from .diagnostics import replay

                    progress(
                        "Replaying the additional native diagnostic trajectories..."
                    )
                    replay(data, output / "diagnostics")
                    live["diagnostics"] = read(output / "diagnostics/verification.json")
        if quick:
            print_sample(live)
        else:
            print(
                f"Final tests: {live['evaluation']['records_compared']:,} episode records matched.",
                flush=True,
            )
        write_json(output / "live_checks.json", live)
    result.update(
        mode=mode, live_checks=live, seconds=round(time.monotonic() - started, 2)
    )
    write_json(output / "verification.json", result)
    present(output, result)
    write_json(
        output / "run.json",
        dict(status="passed", mode=mode, seed=0, seconds=result["seconds"]),
    )
    print(
        f"Passed ({result['seconds']:.1f} s). Results: {output / 'index.html'}",
        flush=True,
    )
