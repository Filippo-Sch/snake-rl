"""Isolated classic gameplay, completion-first summaries and development reports."""

import csv
from io import StringIO
import math
from pathlib import Path

import numpy as np

from ..training_state import atomic_bytes, atomic_json
from .config import CONFIG, manifest
from .environment import ClassicSnake
from .learner import choose_actions, load_policy
from .policies import Heuristic, MODES


def summarize(records):
    n = len(records)
    if not n:
        raise ValueError("Cannot summarize an empty evaluation")
    outcomes = {name: sum(r["outcome"] == name for r in records)
                for name in ("win", "wall_death", "body_death", "total_limit", "fruit_free_limit")}
    if sum(outcomes.values()) != n:
        raise ValueError("Unfinished or unknown evaluation outcomes")
    wins = outcomes["win"]
    censored = outcomes["total_limit"] + outcomes["fruit_free_limit"]
    p, z = wins / n, 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    fruits = np.asarray([r["fruits"] for r in records])
    times = [r["actions"] for r in records if r["outcome"] == "win"]
    length_counts = np.sum([r["length_counts"] for r in records], axis=0)
    for record in records:
        expected = record["fruits"] + 100 * (record["outcome"] == "win")
        expected -= record["outcome"] in ("wall_death", "body_death")
        expected -= 0.001 * record["actions"]
        if not math.isclose(record["reward"], expected, abs_tol=1e-8):
            raise ValueError("Classic reward reconstruction failed")
    return dict(games=n, outcomes=outcomes, wins=wins, completion_rate=p,
                completion_wilson95=[max(0.0, center - half), min(1.0, center + half)],
                eventual_sample_bounds=[p, (wins + censored) / n],
                mean_fruits=float(fruits.mean()), median_fruits=float(np.median(fruits)),
                std_fruits=float(fruits.std()), zero_fruit_fraction=float((fruits == 0).mean()),
                mean_reward=float(np.mean([r["reward"] for r in records])),
                actions=sum(r["actions"] for r in records),
                mean_actions=float(np.mean([r["actions"] for r in records])),
                mean_completion_actions=float(np.mean(times)) if times else None,
                median_completion_actions=float(np.median(times)) if times else None,
                winning_games_interrupted_by_training_limits=sum(
                    r["outcome"] == "win" and (r["actions"] > CONFIG.train_limit
                    or r["max_fruit_free"] >= CONFIG.fruit_free_limit) for r in records),
                length_counts={str(i): int(length_counts[i]) for i in range(3, 26)})


def evaluate(regime, *, network=None, heuristic=None, role="validation", games=None):
    if (network is None) == (heuristic is None):
        raise ValueError("Choose exactly one network or heuristic")
    if role not in ("validation", "test", "audit"):
        raise ValueError("Evaluation requires its own random bank")
    if games is None:
        games = {"validation": CONFIG.validation_games, "test": CONFIG.test_games,
                 "audit": CONFIG.audit_games}[role]
    records, modes = [], {name: 0 for name in MODES}
    for game_id in range(games):
        env = ClassicSnake(regime, role, game_id, limit=CONFIG.eval_limit, fruit_free_limit=None)
        policy = Heuristic(heuristic, game_id) if heuristic else None
        while not env.done:
            obs = env.observe()
            if policy:
                decision = policy.act(obs)
                action, mode = decision.action, decision.mode
            else:
                action = int(choose_actions(network, obs.encode()[None], obs.legal[None])[0])
                mode = "student"
            modes[mode] += 1
            env.step(action)
        records.append(env.record())
    return dict(regime=regime, role=role, cap=CONFIG.eval_limit, policy=heuristic or "dqn",
                summary=summarize(records), modes=modes, records=records)


def write_evaluation(folder, result):
    folder = Path(folder)
    atomic_json(folder / "results.json", dict(result, manifest=manifest()))
    buffer = StringIO(newline="")
    fields = ("game_id", "outcome", "actions", "fruits", "reward", "length", "max_fruit_free")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(result["records"])
    atomic_bytes(folder / "per_game.csv", buffer.getvalue().encode())


def selection_key(summary):
    return summary["wins"], summary["mean_fruits"]


def evaluate_file(path, regime, role="validation"):
    network, _ = load_policy(path, regime)
    return evaluate(regime, network=network, role=role)


def development_report(root):
    """Read saved development artifacts only; never launch evaluation or training."""
    import json
    root = Path(root)
    rows, runs, missing, teachers, curves = [], [], [], {}, []
    baseline_actions = 0
    for regime in ("full", "partial"):
        teachers[regime] = {}
        for label, relative in (("audit", "audit/results.json"), ("demonstrations", "demonstrations.json")):
            path = root / regime / relative
            if path.exists():
                teachers[regime][label] = json.loads(path.read_text(encoding="utf-8"))
            else:
                missing.append(str(path.relative_to(root)))
        for name in ("greedy", "bfs"):
            path = root / regime / "baselines" / name / "results.json"
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
                rows.append(dict(regime=regime, policy=name, checkpoint="baseline", **result["summary"]))
                baseline_actions += result["summary"]["actions"]
            else:
                missing.append(str(path.relative_to(root)))
        for stage in ("scratch", "offline", "online"):
            path = root / regime / stage / "run.json"
            if not path.exists():
                missing.append(str(path.relative_to(root)))
                continue
            run = json.loads(path.read_text(encoding="utf-8"))
            runs.append(dict(run, regime=regime, stage=stage))
            history = run["history"]
            for item in history:
                s = item["summary"]
                curves.append(dict(regime=regime, stage=stage, counter=item["counter"],
                                   counter_unit="updates" if stage == "offline" else "transitions",
                                   wins=s["wins"], completion_rate=s["wins"] / s["games"],
                                   mean_fruits=s["mean_fruits"],
                                   wall_deaths=s["outcomes"].get("wall_death", 0),
                                   body_deaths=s["outcomes"].get("body_death", 0),
                                   truncated=s["outcomes"]["total_limit"]))
            for label, counter in (("selected", run["selected_counter"]), ("final", run["counter"])):
                item = next((h for h in history if h["counter"] == counter), None)
                if item:
                    rows.append(dict(regime=regime, policy=stage, checkpoint=label, **item["summary"]))
            if not run["complete"]:
                missing.append(f"{regime}/{stage}: incomplete budget")
    costs = dict(student_transitions=sum(r["counter"] for r in runs if r["stage"] != "offline"),
                 optimizer_updates=sum(r.get("stage_updates", 0) for r in runs),
                 validation_actions=baseline_actions + sum(r.get("evaluation_actions", 0) for r in runs),
                 teacher_audit_actions=sum(t.get("audit", {}).get("summary", {}).get("actions", 0)
                                          for t in teachers.values()),
                 teacher_transitions=sum(t.get("demonstrations", {}).get("transitions", 0)
                                         for t in teachers.values()),
                 discarded_offline_updates=sum((r.get("parent") or {}).get("discarded_offline_updates", 0)
                                               for r in runs))
    payload = dict(manifest=manifest(), results=rows, runs=runs, missing=missing,
                   teachers=teachers, costs=costs, learning_curves=curves,
                   status="development only; discuss results before tuning or final test")
    atomic_json(root / "development.json", payload)
    if curves:
        buffer = StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
        atomic_bytes(root / "learning_curves.csv", buffer.getvalue().encode())
    lines = ["# Classic Snake development comparison", "",
             "Validation only; one training seed. Selected and final checkpoints are both shown.", "",
             "| Regime | Policy | Checkpoint | Wins | Mean fruit | Truncated |",
             "| --- | --- | --- | ---: | ---: | ---: |"]
    for r in rows:
        truncated = r["outcomes"]["total_limit"] + r["outcomes"]["fruit_free_limit"]
        lines.append(f"| {r['regime']} | {r['policy']} | {r['checkpoint']} | "
                     f"{r['wins']}/{r['games']} | {r['mean_fruits']:.3f} | {truncated} |")
    lines += ["", "Complete curves, collection outcomes and compute costs: `development.json`.",
              "Curve data for plotting: `learning_curves.csv`.",
              "Teacher coverage: each regime's `demonstrations.json` and `audit/results.json`.",
              "", "No automatic tuning, budget extensions or final testing."]
    if missing:
        lines += ["", "Missing work:", ""] + [f"- {p}" for p in missing]
    atomic_bytes(root / "development.md", ("\n".join(lines) + "\n").encode())
    return payload
