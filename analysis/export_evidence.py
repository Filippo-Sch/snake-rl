"""Export experiment histories and diagnostic fields as portable evidence."""

from pathlib import Path
import shutil

from .common import CASES, RUNS, read, rows, sha, write_csv, write_json, require


def export(source_root, output, *, historical=False, weights_manifest=None):
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = {}

    def relative_paths(value):
        if isinstance(value, dict):
            return {key: relative_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relative_paths(item) for item in value]
        if isinstance(value, str):
            path = value.replace("\\", "/")
            prefix = str(source_root).replace("\\", "/") + "/"
            if path.casefold().startswith(prefix.casefold()):
                return path[len(prefix) :]
            if value.startswith("results\\"):
                return path
        return value

    def copy(source, relative):
        source = Path(source)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        inputs[relative] = dict(
            source=str(source.relative_to(source_root)).replace("\\", "/"),
            source_sha256=sha(source),
        )
        return target

    def generated(source, relative, value):
        write_json(output / relative, relative_paths(value))
        inputs[relative] = dict(
            source=str(source.relative_to(source_root)).replace("\\", "/"),
            source_sha256=sha(source),
            transformation="compact JSON: selected fields, relative source paths",
        )

    native = source_root / (
        "results/experiments/20260921_final_lr" if historical else "native"
    )
    for case in CASES:
        for variant in ("A",) if case.startswith("b7") else ("A", "B", "C"):
            folder = native / "training" / case / variant
            meta = read(folder / "run.json")
            fields = (
                "regime",
                "environment",
                "training",
                "evaluation",
                "gamma",
                "hidden_sizes",
                "optimizer",
                "learning_rate_schedule",
                "epsilon",
                "selection",
                "transitions",
                "updates",
                "selected_transitions",
                "selected_return",
                "selections",
                "duration_seconds",
                "status",
                "python",
                "numpy",
                "torch",
                "device",
            )
            generated(
                folder / "run.json",
                f"native/{case}/{variant}/run.json",
                {k: meta[k] for k in fields if k in meta},
            )
            validation_source = folder / "validation.csv"
            relative = f"native/{case}/{variant}/validation.csv"
            write_csv(output / relative, [relative_paths(r) for r in rows(validation_source)])
            inputs[relative] = dict(
                source=validation_source.relative_to(source_root).as_posix(),
                source_sha256=sha(validation_source),
                transformation="Validation records unchanged; source paths made relative",
            )
        if historical:
            paths = list((native / "test" / case).glob("*/results.json"))
            require(len(paths) == 1, f"Ambiguous historical native test: {case}")
            test_source = paths[0]
        else:
            test_source = native / "test" / case / "results.json"
        copy(test_source.with_name("per_board.csv"), f"native/{case}/test.csv")
        d = read(test_source)
        generated(
            test_source,
            f"native/{case}/test_metadata.json",
            dict(
                config=d["config"],
                runs=[
                    {
                        k: r[k]
                        for k in (
                            "policy",
                            "regime",
                            "parameters",
                            "weights",
                            "initial_boards_sha256",
                            "runtime",
                        )
                        if k in r
                    }
                    for r in d["runs"]
                ],
            ),
        )

    native_baseline_sources = {}
    if historical:
        native_baseline_sources[7] = (
            source_root / "results/20260920T091823285066Z/results.json"
        )
        index = read(
            source_root
            / "results/submission/diagnostic_completion/native11_baselines.json"
        )
        native_baseline_sources[11] = source_root / index["path"]
    else:
        for size in (7, 11):
            native_baseline_sources[size] = (
                native / f"validation_baselines/b{size}/results.json"
            )
    for size, path in native_baseline_sources.items():
        d = read(path)
        generated(
            path,
            f"baselines/native_b{size}.json",
            dict(
                config=d["config"],
                runs=[r for r in d["runs"] if r["policy"] == "direct"],
            ),
        )
        records = [
            r for r in rows(path.with_name("per_board.csv")) if r["policy"] == "direct"
        ]
        write_csv(output / f"baselines/native_b{size}.csv", records)
        inputs[f"baselines/native_b{size}.csv"] = dict(
            source=str(
                path.with_name("per_board.csv").relative_to(source_root)
            ).replace("\\", "/"),
            source_sha256=sha(path.with_name("per_board.csv")),
            transformation="Direct rows only",
        )

    classic_root = source_root / "results" if historical else source_root
    for name, relative in RUNS.items():
        path = classic_root / relative / "run.json"
        d = read(path)
        fields = (
            "regime",
            "stage",
            "counter",
            "budget",
            "complete",
            "selected_counter",
            "selection",
            "history",
            "elapsed_seconds",
            "extension",
            "experiment",
            "additional_transitions",
            "additional_updates",
        )
        generated(
            path, f"classic/runs/{name}/run.json", {k: d[k] for k in fields if k in d}
        )
    for regime in ("full", "partial"):
        for policy in ("bfs", "greedy"):
            copy(
                classic_root
                / "phase2"
                / regime
                / "baselines"
                / policy
                / "results.json",
                f"baselines/classic_{regime}_{policy}.json",
            )
        for policy in ("bfs", "greedy", "dqn"):
            path = (
                (
                    source_root / "results/submission/test"
                    if historical
                    else source_root / "classic_test"
                )
                / regime
                / policy
                / "results.json"
            )
            d = read(path)
            generated(
                path,
                f"classic/test/{regime}/{policy}/results.json",
                {k: v for k, v in d.items() if k != "manifest"},
            )

    for name, original in [("search", "partial_search"), ("spawn", "fruit_mechanism")]:
        folder = source_root / (
            "analysis/20260921_final_lr/" + original
            if historical
            else "diagnostics/" + name
        )
        copy(folder / "per_board.csv", f"diagnostics/{name}_per_board.csv")
        copy(folder / "summary.json", f"diagnostics/{name}_summary.json")
    if historical:
        path = (
            source_root
            / "analysis/phase2_development/20260924_learning_diagnostics.json"
        )
        d = read(path)
        generated(
            path,
            "diagnostics/replay.json",
            dict(
                seed=d["seed"],
                snapshots=[
                    {
                        k: s[k]
                        for k in (
                            "regime",
                            "counter",
                            "replay_size",
                            "terminal_wins",
                            "terminal_deaths",
                            "truncations",
                            "fruit_steps",
                            "lengths",
                            "path",
                            "sha256",
                        )
                    }
                    for s in d["snapshots"]
                ],
            ),
        )
    else:
        copy(source_root / "diagnostics/replay.json", "diagnostics/replay.json")
    # Keep the four scalar fields needed to independently check every replay
    # event, rather than requiring the large state/optimizer snapshots.
    replay = read(output / "diagnostics/replay.json")
    for snapshot in replay["snapshots"]:
        regime = snapshot["regime"]
        relative = f"diagnostics/replay_{regime}.csv"
        existing = source_root / relative
        if not historical and existing.exists():
            copy(existing, relative)
            continue
        import torch

        path = source_root / snapshot["path"].replace("\\", "/")
        require(
            sha(path) == snapshot["sha256"],
            f"Replay source checksum mismatch: {regime}",
        )
        state = torch.load(path, map_location="cpu", weights_only=False)
        arrays = state["replay"]["arrays"]
        n = state["replay"]["size"]
        require(n == snapshot["replay_size"], "Replay source size mismatch")
        records = [
            dict(
                slot=i,
                reward=float(arrays["rewards"][i]),
                terminated=int(arrays["terminated"][i]),
                truncated=int(arrays["truncated"][i]),
                length=int(arrays["lengths"][i]),
            )
            for i in range(n)
        ]
        write_csv(output / relative, records)
        inputs[relative] = dict(
            source=str(path.relative_to(source_root)).replace("\\", "/"),
            source_sha256=snapshot["sha256"],
            transformation="Replay slots: reward, terminated, truncated, length; states and optimizer omitted",
        )
        del state, arrays, records
    if weights_manifest is None:
        weights_manifest = read(source_root / "weights/manifest.json")
    write_json(output / "weights.json", weights_manifest)
    files = {
        p.relative_to(output).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    write_json(
        output / "manifest.json",
        dict(
            format="snake-report-evidence-v1",
            seed=0,
            historical=historical,
            inputs=inputs,
            files=files,
            note="Episode data, validation histories and diagnostic fields. Verify with python evaluate.py.",
        ),
    )
    return files
