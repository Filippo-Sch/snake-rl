"""Compute report tables from episode records, not from rounded paper entries."""

from collections import defaultdict
import math

import numpy as np

from .common import CASES, SELECTED, near, read, require, rows, write_csv, write_json


def native_tables(data):
    complete, selected = [], []
    for case in CASES:
        grouped = defaultdict(list)
        for row in rows(data / "native" / case / "test.csv"):
            grouped[row["policy"]].append(row)
        for policy, records in grouped.items():
            require(len(records) == 500, f"{case}/{policy}: expected 500 episodes")
            result = dict(case=case, policy=policy, games=len(records), horizon=1000)
            for metric in (
                "fruits",
                "wall_hits",
                "body_hits",
                "board_completions",
                "return",
            ):
                values = np.array([float(r[metric]) for r in records])
                result[metric + "_mean"] = float(values.mean())
                result[metric + "_std"] = float(values.std(ddof=1))
            for row in records:
                reward = (
                    0.5 * float(row["fruits"])
                    - 0.1 * float(row["wall_hits"])
                    - 0.2 * float(row["body_hits"])
                    + 0.5 * float(row["board_completions"])
                )
                near(row["return"], reward, f"{case}/{policy}/reward", tolerance=3e-6)
            complete.append(result)
            if policy in ("greedy", "bfs", "direct", SELECTED[case]):
                selected.append(result)
    return selected, complete


def paired_returns(data):
    comparisons = []
    for case in CASES:
        records = rows(data / "native" / case / "test.csv")
        policies = {}
        for policy in ("direct", SELECTED[case]):
            selected = [r for r in records if r["policy"] == policy]
            require(len(selected) == 500, "Paired comparison requires 500 episodes")
            by_id = {int(r["board_id"]): float(r["return"]) for r in selected}
            require(set(by_id) == set(range(500)), "Paired episode IDs differ")
            policies[policy] = by_id
        differences = np.array(
            [policies[SELECTED[case]][i] - policies["direct"][i] for i in range(500)]
        )
        mean = float(differences.mean())
        standard_error = float(differences.std(ddof=1) / math.sqrt(len(differences)))
        comparisons.append(
            dict(
                case=case,
                games=500,
                difference=mean,
                standard_error=standard_error,
                ci95_low=mean - 1.96 * standard_error,
                ci95_high=mean + 1.96 * standard_error,
            )
        )
    return comparisons


def classic_tables(data):
    values = []
    for regime in ("full", "partial"):
        for policy in ("bfs", "greedy", "dqn"):
            records = read(data / "classic/test" / regime / policy / "results.json")[
                "records"
            ]
            require(len(records) == 500, "Classic test requires 500 games")
            wins = sum(r["outcome"] == "win" for r in records)
            wall = sum(r["outcome"] == "wall_death" for r in records)
            body = sum(r["outcome"] == "body_death" for r in records)
            timeouts = sum(r["outcome"] == "total_limit" for r in records)
            require(
                wins + wall + body + timeouts == 500,
                "Classic outcomes do not partition games",
            )
            for r in records:
                reward = (
                    r["fruits"]
                    + 100 * (r["outcome"] == "win")
                    - (r["outcome"] in ("wall_death", "body_death"))
                    - 0.001 * r["actions"]
                )
                near(r["reward"], reward, "Classic per-game reward")
            n, p, z = len(records), wins / len(records), 1.959963984540054
            denominator = 1 + z * z / n
            center = (p + z * z / (2 * n)) / denominator
            half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
            values.append(
                dict(
                    regime=regime,
                    policy=policy,
                    games=n,
                    wins=wins,
                    fruits_mean=float(np.mean([r["fruits"] for r in records])),
                    deaths=wall + body,
                    timeouts=timeouts,
                    exact_cycles=(
                        sum(bool(r["exact_cycle"]) for r in records)
                        if policy == "dqn"
                        else ""
                    ),
                    return_mean=float(np.mean([r["reward"] for r in records])),
                    completion_rate=p,
                    completion_ci95_low=max(0, center - half),
                    completion_ci95_high=min(1, center + half),
                )
            )
    return values


def search_tables(data):
    grouped = defaultdict(list)
    for row in rows(data / "diagnostics/search_per_board.csv"):
        grouped[row["case"], row["policy"]].append(row)
    values = []
    for (case, policy), records in grouped.items():
        require(len(records) == 500, "Search diagnostic requires 500 boards")
        hidden = [r for r in records if r["initially_hidden"] == "True"]
        collected = sum(float(r["completed_count"]) for r in records)
        for r in records:
            near(
                float(r["search_actions"]) + float(r["post_sight_actions"]),
                1000,
                "Search action accounting",
            )
            near(
                float(r["completed_pursuit"]),
                float(r["completed_sight_distance"]) + float(r["completed_excess"]),
                "Pursuit accounting",
            )
            near(
                float(r["completed_search"]) + float(r["terminal_search"]),
                r["search_actions"],
                "Censored search",
            )
            near(
                float(r["completed_pursuit"]) + float(r["terminal_pursuit"]),
                r["post_sight_actions"],
                "Censored pursuit",
            )
        values.append(
            dict(
                case=case,
                policy=policy,
                games=500,
                hidden_starts=len(hidden),
                initial_wait_mean=float(
                    np.mean([float(r["initial_wait_capped"]) for r in hidden])
                ),
                pursuit_excess=(
                    sum(float(r["completed_excess"]) for r in records) / collected
                    if collected
                    else None
                ),
                search_actions_percent=sum(float(r["search_actions"]) for r in records)
                / 5000,
                zero_fruit_unseen=sum(
                    r["zero_fruit_never_seen"] == "True" for r in records
                ),
                zero_fruit_seen=sum(r["zero_fruit_seen"] == "True" for r in records),
            )
        )
    return values


def spawn_tables(data):
    grouped = defaultdict(list)
    for r in rows(data / "diagnostics/spawn_per_board.csv"):
        grouped[r["case"], r["policy"]].append(r)
    values = []
    for (case, policy), records in grouped.items():
        for r in records:
            near(
                float(r["completed_duration"]) + float(r["terminal_elapsed"]),
                1000,
                "Spawn time accounting",
            )
            near(
                float(r["completed_expected"])
                + float(r["completed_residual"])
                + float(r["completed_excess"]),
                r["completed_duration"],
                "Spawn duration decomposition",
            )
            near(r["spawn_count"], float(r["fruits"]) + 1, "Spawn count")
        total = sum(float(r["spawn_count"]) for r in records)
        values.append(
            dict(
                case=case,
                policy=policy,
                games=len(records),
                expected_distance=sum(float(r["spawn_expected"]) for r in records)
                / total,
                empty_board_distance=sum(float(r["spawn_empty_board"]) for r in records)
                / total,
                body_effect=sum(float(r["spawn_body_effect"]) for r in records) / total,
                draw_residual=sum(float(r["spawn_residual"]) for r in records) / total,
            )
        )
    return values


def selection(data):
    curves, choices = [], {}
    manifest = read(data / "weights.json")["policies"]
    for case in CASES:
        scores = []
        for variant in ("A",) if case.startswith("b7") else ("A", "B", "C"):
            records = rows(data / "native" / case / variant / "validation.csv")
            require(
                [int(r["transitions"]) for r in records]
                == list(
                    range(
                        0, (2048000 if case.startswith("b7") else 4096000) + 1, 128000
                    )
                ),
                "Incomplete native curve",
            )
            curves.extend(
                dict(
                    case=case,
                    variant=variant,
                    transitions=int(r["transitions"]),
                    return_mean=float(r["return_mean"]),
                )
                for r in records
            )
            best = max(records[1:], key=lambda r: float(r["return_mean"]))
            scores.append(
                dict(
                    variant=variant,
                    final_eight_mean=float(
                        np.mean([float(r["return_mean"]) for r in records[-8:]])
                    ),
                    selected_step=int(best["transitions"]),
                    selected_return=float(best["return_mean"]),
                )
            )
        winner = max(scores, key=lambda r: r["final_eight_mean"])
        key = "native_" + case
        require(
            winner["selected_step"] == manifest[key]["selected_transitions"],
            f"Native selected checkpoint mismatch: {case}",
        )
        choices[case] = dict(
            winner=winner["variant"],
            schedules=scores,
            selected_transitions=winner["selected_step"],
        )
    for regime in ("full", "partial"):
        history = read(data / "classic/runs" / ("extended_" + regime) / "run.json")[
            "history"
        ]
        best = max(
            (h for h in history if h["counter"] > 0),
            key=lambda h: (h["summary"]["wins"], h["summary"]["mean_fruits"]),
        )
        require(
            best["counter"] == manifest["classic_" + regime]["selected_transitions"],
            "Classic selection mismatch",
        )
        choices["classic_" + regime] = dict(
            selected_transitions=best["counter"],
            wins=best["summary"]["wins"],
            fruits=best["summary"]["mean_fruits"],
        )
    return curves, choices


def classic_curves(data):
    values = []
    for folder in sorted((data / "classic/runs").iterdir()):
        d = read(folder / "run.json")
        for h in d["history"]:
            s = h["summary"]
            near(sum(s["outcomes"].values()), s["games"], "Classic validation outcomes")
            near(
                s["mean_reward"],
                s["mean_fruits"]
                + 100 * s["wins"] / s["games"]
                - (s["outcomes"]["wall_death"] + s["outcomes"]["body_death"])
                / s["games"]
                - 0.001 * s["actions"] / s["games"],
                "Classic validation return",
            )
            values.append(
                dict(
                    run=folder.name,
                    regime=d["regime"],
                    counter=h["counter"],
                    wins=s["wins"],
                    mean_fruits=s["mean_fruits"],
                    mean_reward=s["mean_reward"],
                    timeouts=s["outcomes"]["total_limit"],
                    exact_cycles=s.get("exact_cycles", ""),
                    selected=h["counter"] == d["selected_counter"],
                )
            )
    return values


def interventions(data):
    values = []
    for name, start in [
        ("full_control", 4096000),
        ("full_lower_lr", 4096000),
        ("full_watkins", 4096000),
        ("full_discount", 4096000),
        ("extended_partial", 2432000),
        ("partial_protected", 2432000),
    ]:
        d = read(data / "classic/runs" / name / "run.json")
        hist = [h for h in d["history"] if start <= h["counter"] <= start + 1024000]
        require(
            [h["counter"] for h in hist] == list(range(start, start + 1024001, 128000)),
            "Incomplete intervention window",
        )
        settings = d.get("experiment", {}).get("settings", {})
        best = max(
            hist[1:], key=lambda h: (h["summary"]["wins"], h["summary"]["mean_fruits"])
        )
        values.append(
            dict(
                run=name,
                start=start,
                stop=start + 1024000,
                control=(
                    "extended_partial"
                    if name in ("extended_partial", "partial_protected")
                    else "full_control"
                ),
                learning_rate=settings.get("learning_rate", 1e-4),
                gamma=settings.get("gamma", 1),
                protected_capacity=settings.get("protected_capacity", 0),
                protected_per_batch=settings.get("protected_per_batch", 0),
                trace_lambda=settings.get("trace_lambda", ""),
                trace_horizon=settings.get("horizon", ""),
                initial_wins=hist[0]["summary"]["wins"],
                final_wins=hist[-1]["summary"]["wins"],
                initial_fruits=hist[0]["summary"]["mean_fruits"],
                final_fruits=hist[-1]["summary"]["mean_fruits"],
                best_new_wins=best["summary"]["wins"],
                best_new_fruits=best["summary"]["mean_fruits"],
                best_new_counter=best["counter"],
                last_four_wins=float(
                    np.mean([h["summary"]["wins"] for h in hist[-4:]])
                ),
                last_four_fruits=float(
                    np.mean([h["summary"]["mean_fruits"] for h in hist[-4:]])
                ),
            )
        )
    return values


def build(data, output):
    selected, complete = native_tables(data)
    curves, choices = selection(data)
    tables = {
        "table1_native": selected,
        "native_all_policies": complete,
        "native_paired_returns": paired_returns(data),
        "table2_search_all": search_tables(data),
        "table3_classic": classic_tables(data),
        "spawn_geometry": spawn_tables(data),
        "native_learning": curves,
        "classic_learning": classic_curves(data),
        "classic_interventions": interventions(data),
    }
    tables["table4_interventions"] = tables["classic_interventions"]
    tables["table2_search"] = [
        r
        for r in tables["table2_search_all"]
        if r["policy"] in ("direct", SELECTED[r["case"]])
    ]
    for name, values in tables.items():
        write_csv(output / "tables" / (name + ".csv"), values)
    write_json(output / "selection.json", choices)
    return tables, choices
