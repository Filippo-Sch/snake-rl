"""Replay final policies with the original passive search/spawn trackers."""

from dataclasses import replace

import numpy as np
import torch

from snake import evaluation as ev
from snake.baselines import make_direct
from snake.dqn import make_dqn_spec

from .common import ROOT, SELECTED, near, read, require, rows, write_csv, write_json
from .trackers import SearchTracker, SpawnTracker


def diagnose(case, policy, weights, *, games=500):
    size, regime = int(case.split("_")[0][1:]), case.split("_")[1]
    config = ev.EvaluationConfig(
        board_size=size,
        mask_size=2,
        n_boards=games,
        steps=1000,
        split="test",
        test_stream=3,
    )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    original = ev.make_environment
    tracker = None

    def factory(config, regime):
        nonlocal tracker
        env = original(config, regime)
        tracker = (
            SearchTracker(env, config.steps)
            if regime == "partial"
            else SpawnTracker(env)
        )
        return env

    try:
        ev.make_environment = factory
        spec = (
            ev.PolicySpec("direct", make_direct)
            if policy == "direct"
            else replace(make_dqn_spec(config, {regime: weights}), name=policy)
        )
        meta, records, _ = ev.evaluate_policy(spec, config, regime)
    finally:
        ev.make_environment = original
    if regime == "partial":
        summary, detailed = tracker.finish(case, policy, records)
    else:
        sums = tracker.sums
        fruits = np.array([r["fruits"] for r in records])
        require(
            np.array_equal(sums["completed_count"], fruits),
            "Spawn collected count mismatch",
        )
        tail = config.steps - tracker.birth
        require(
            np.array_equal(
                sums["completed_duration"] + tail, np.full(games, config.steps)
            ),
            "Spawn horizon accounting",
        )
        detailed = [
            dict(
                case=case,
                policy=policy,
                board_id=i,
                **{k: v[i] for k, v in sums.items()},
                terminal_elapsed=int(tail[i]),
                terminal_start_distance=tracker.current["distance"][i],
                fruits=r["fruits"],
                return_value=r["return"],
            )
            for i, r in enumerate(records)
        ]
        summary = dict(
            case=case, policy=policy, episodes=games, fruits_mean=float(fruits.mean())
        )
        for prefix in ("spawn", "completed"):
            count = sums[prefix + "_count"].sum()
            for key, values in sums.items():
                if key.startswith(prefix + "_") and not key.endswith("_count"):
                    summary[key + "_mean"] = (
                        float(values.sum() / count) if count else None
                    )
    return meta, records, summary, detailed


def replay(data, output, *, weights=None, historical_base=None, games=500, cases=None):
    weights = weights or {
        case: ROOT / "weights" / ("native_" + case + ".pt") for case in SELECTED
    }
    all_data = {"search": [], "spawn": []}
    summaries = {"search": [], "spawn": []}
    cases = cases or list(SELECTED)
    compared = 0
    for case in cases:
        policies = [("direct", None), (SELECTED[case], weights[case])]
        if case == "b11_partial" and historical_base is not None:
            policies.insert(1, ("dqn_2048000", historical_base))
        old = {
            (r["policy"], int(r["board_id"])): r
            for r in rows(data / "native" / case / "test.csv")
        }
        metadata = read(data / "native" / case / "test_metadata.json")
        for policy, path in policies:
            print(f"Diagnostic {case}/{policy}: {games} fixed test boards", flush=True)
            meta, records, summary, detailed = diagnose(case, policy, path, games=games)
            # A smaller vectorized batch changes RNG allocation: it is a smoke test,
            # never a prefix of the report's 500-board native evaluation.
            if games == 500:
                ref_meta = next(r for r in metadata["runs"] if r["policy"] == policy)
                require(
                    meta["initial_boards_sha256"] == ref_meta["initial_boards_sha256"],
                    "Diagnostic initial boards differ",
                )
                for r in records:
                    ref = old[policy, r["board_id"]]
                    for metric in ev.COUNTS + (
                        "return",
                        "body_cells_removed",
                        "body_length_before_hits",
                    ):
                        near(
                            r[metric],
                            ref[metric],
                            "Diagnostic replay/" + metric,
                            tolerance=1e-9,
                        )
                kind = "search" if case.endswith("partial") else "spawn"
                archive_path = data / f"diagnostics/{kind}_per_board.csv"
                if archive_path.exists():
                    archived = {
                        (r["policy"], int(r["board_id"])): r
                        for r in rows(archive_path)
                        if r["case"] == case
                    }
                    for r in detailed:
                        expected = archived[policy, int(r["board_id"])]
                        for key in r.keys() & expected.keys() - {
                            "case",
                            "policy",
                            "board_id",
                        }:
                            if isinstance(r[key], (bool, np.bool_)):
                                require(
                                    str(bool(r[key])) == expected[key],
                                    f"Diagnostic boolean differs: {key}",
                                )
                            else:
                                near(
                                    r[key],
                                    expected[key],
                                    f"Diagnostic statistic/{key}",
                                    tolerance=1e-7,
                                )
                compared += len(records)
            kind = "search" if case.endswith("partial") else "spawn"
            summaries[kind].append(summary)
            all_data[kind].extend(detailed)
            write_json(output / kind / "summary.json", summaries[kind])
            write_csv(output / kind / "per_board.csv", all_data[kind])
    write_json(
        output / "verification.json",
        dict(
            status="passed",
            report_bank=games == 500,
            reference_records_compared=compared,
            seed=0,
            training=False,
            note="The earlier b11 partial checkpoint is regenerated by the from-scratch campaign; its archived diagnostic is supplied separately.",
        ),
    )
    return compared
