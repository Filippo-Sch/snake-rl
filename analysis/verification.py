"""Validate evidence integrity, rewards, selections and report-linked diagnostics."""

from collections import Counter

from .common import (
    ROOT,
    CASES,
    SELECTED,
    contained,
    near,
    read,
    require,
    rows,
    sha,
    write_json,
)
from .tables import build


def verify(data, output, *, check_delivered=True):
    manifest = read(data / "manifest.json")
    require(
        manifest["format"] == "snake-report-evidence-v1" and manifest["seed"] == 0,
        "Unsupported evidence manifest",
    )
    files = manifest["files"]
    actual = {
        p.relative_to(data).as_posix()
        for p in data.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    require(actual == set(files), "Evidence inventory mismatch")
    for name, record in files.items():
        path = contained(data, name)
        require(
            path.stat().st_size == record["bytes"] and sha(path) == record["sha256"],
            f"Evidence checksum mismatch: {name}",
        )
    tables, choices = build(data, output)
    weights = read(data / "weights.json")["policies"]
    if check_delivered:
        delivered = read(ROOT / "weights/manifest.json")["policies"]
        for name, item in delivered.items():
            require(
                weights[name]["sha256"] == item["sha256"],
                f"Data refer to different weights: {name}",
            )
            require(
                sha(ROOT / "weights" / item["file"]) == item["sha256"],
                f"Weight checksum mismatch: {name}",
            )
    for case in CASES:
        meta = read(data / "native" / case / "test_metadata.json")
        require(
            meta["config"]["seed"] == 0
            and meta["config"]["split"] == "test"
            and meta["config"]["test_stream"] == 3,
            "Native test stream mismatch",
        )
        hashes = {r["initial_boards_sha256"] for r in meta["runs"]}
        require(len(hashes) == 1, f"Policies do not share initial boards: {case}")
        item = next(r for r in meta["runs"] if r["policy"] == SELECTED[case])
        require(
            item["weights"]["sha256"] == weights["native_" + case]["sha256"],
            f"Checkpoint provenance mismatch: {case}",
        )
    # Recompute baseline horizontal lines from validation episode records.
    for size in (7, 11):
        metadata = read(data / f"baselines/native_b{size}.json")
        for regime in ("full", "partial"):
            records = [
                r
                for r in rows(data / f"baselines/native_b{size}.csv")
                if r["regime"] == regime
            ]
            require(len(records) == 100, "Validation baseline needs 100 boards")
            reference = next(r for r in metadata["runs"] if r["regime"] == regime)
            near(
                sum(float(r["return"]) for r in records) / 100,
                reference["summary"]["return"]["mean"],
                "Native baseline mean",
            )
    # Check diagnostic episode identities against the original corresponding test.
    for kind in ("search", "spawn"):
        diagnostic = rows(data / f"diagnostics/{kind}_per_board.csv")
        for case in {r["case"] for r in diagnostic}:
            reference = {
                (r["policy"], int(r["board_id"])): r
                for r in rows(data / "native" / case / "test.csv")
            }
            for row in (r for r in diagnostic if r["case"] == case):
                old = reference[row["policy"], int(row["board_id"])]
                near(row["fruits"], old["fruits"], "Diagnostic trajectory fruit count")
                if kind == "spawn":
                    near(row["return_value"], old["return"], "Spawn trajectory return")
        if kind == "search":
            for case in ("b7_partial", "b11_partial"):
                selected = [
                    r
                    for r in diagnostic
                    if r["case"] == case and r["policy"] == SELECTED[case]
                ]
                direct = [
                    r
                    for r in diagnostic
                    if r["case"] == case and r["policy"] == "direct"
                ]
                require(
                    [
                        (r["board_id"], r["initially_hidden"], r["initial_distance"])
                        for r in selected
                    ]
                    == [
                        (r["board_id"], r["initially_hidden"], r["initial_distance"])
                        for r in direct
                    ],
                    "Unmatched hidden initial starts",
                )
    for regime in ("full", "partial"):
        for policy in ("bfs", "greedy"):
            baseline = read(data / f"baselines/classic_{regime}_{policy}.json")
            require(
                baseline["role"] == "validation" and len(baseline["records"]) == 100,
                "Classic baseline validation bank mismatch",
            )
            near(
                sum(r["outcome"] == "win" for r in baseline["records"]),
                baseline["summary"]["wins"],
                "Classic baseline wins",
            )
            near(
                sum(r["reward"] for r in baseline["records"]) / 100,
                baseline["summary"]["mean_reward"],
                "Classic baseline return",
            )
        records = read(data / "classic/test" / regime / "dqn/results.json")["records"]
        require(
            all(
                bool(r["exact_cycle"]) == (r["outcome"] == "total_limit")
                for r in records
            ),
            "DQN cycle/timeout mismatch",
        )
    replay = read(data / "diagnostics/replay.json")
    require(
        replay["seed"] == 0
        and all(s["counter"] == 4096000 for s in replay["snapshots"]),
        "Unexpected replay diagnostic",
    )
    for snapshot in replay["snapshots"]:
        require(
            0
            <= snapshot["terminal_wins"] + snapshot["terminal_deaths"]
            <= snapshot["replay_size"],
            "Invalid replay terminal counts",
        )
        events = rows(data / f"diagnostics/replay_{snapshot['regime']}.csv")
        require(len(events) == snapshot["replay_size"], "Replay event count mismatch")
        require(
            [int(r["slot"]) for r in events] == list(range(len(events))),
            "Replay slots mismatch",
        )
        require(
            all(
                r["terminated"] in ("0", "1") and r["truncated"] in ("0", "1")
                for r in events
            ),
            "Replay flags invalid",
        )
        wins = sum(int(r["terminated"]) and float(r["reward"]) > 100 for r in events)
        deaths = sum(int(r["terminated"]) and float(r["reward"]) < 0 for r in events)
        near(wins, snapshot["terminal_wins"], "Replay terminal wins")
        near(deaths, snapshot["terminal_deaths"], "Replay terminal deaths")
        if "truncations" in snapshot:
            near(
                sum(int(r["truncated"]) for r in events),
                snapshot["truncations"],
                "Replay truncations",
            )
        if "fruit_steps" in snapshot:
            near(
                sum(
                    float(r["reward"]) > 0
                    and not (int(r["terminated"]) and float(r["reward"]) > 100)
                    for r in events
                ),
                snapshot["fruit_steps"],
                "Replay fruit steps",
            )
        if "lengths" in snapshot:
            near(
                sum(snapshot["lengths"].values()),
                snapshot["replay_size"],
                "Replay length partition",
            )
            counts = Counter(int(r["length"]) for r in events)
            for length, count in snapshot["lengths"].items():
                near(counts[int(length)], count, "Replay length distribution")
    result = dict(
        status="passed",
        evidence_files=len(files),
        native_final_test_records=8000,
        classic_test_records=3000,
        search_diagnostic_records=len(rows(data / "diagnostics/search_per_board.csv")),
        spawn_diagnostic_records=len(rows(data / "diagnostics/spawn_per_board.csv")),
        selected=choices,
        delivered_weights_checked=check_delivered,
        replay_terminal_wins={
            s["regime"]: s["terminal_wins"] for s in replay["snapshots"]
        },
        replay_event_records=sum(s["replay_size"] for s in replay["snapshots"]),
        scope="Integrity and arithmetic on archived evidence; no rollout or training.",
    )
    if check_delivered:
        from .report import check_report

        result["report"] = check_report(tables, choices)
    write_json(output / "verification.json", result)
    return result
