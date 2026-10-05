"""Isolated fixed-budget extension of completed phase-2 scratch runs.

This file intentionally lives outside snake/classic: the running base campaign's
source manifest stays unchanged. Reuse its collector and update implementation.
"""

from copy import deepcopy
from pathlib import Path
import json
import shutil

from .classic.config import CONFIG, ROOT
from .classic.training import TrainingRun, assert_compatible, read_snapshot
from .training_state import atomic_json, atomic_save, digest

BASE_BUDGET = 2048000
TOTAL_BUDGET = 4096000
EXTENSION = "classic-scratch-extension-1"


def implementation_hashes():
    return {name: digest(ROOT / name) for name in ("experiments/classic_extend.py", "snake/classic_extension.py")}


def checked_parent(parent_root, regime):
    folder = Path(parent_root).resolve() / regime / "scratch"
    state = read_snapshot(folder / "last.resume.pt")
    info = state["summary"]
    assert_compatible(info["manifest"])
    if (info["stage"] != "scratch" or info["regime"] != regime
            or info["counter"] != BASE_BUDGET or info["budget"] != BASE_BUDGET
            or not info["complete"] or info.get("extension") is not None):
        raise ValueError(f"{regime}: finish the original 2.048M scratch run first")
    if len(state["envs"]) != CONFIG.slots or state["replay"] is None:
        raise ValueError("Parent must include active games and replay")
    expected = (BASE_BUDGET - CONFIG.warmup) // CONFIG.slots
    if info["stage_updates"] != expected or state["learner"]["updates"] != expected:
        raise ValueError("Parent optimizer counters do not match its budget")
    record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    if not record["complete"] or record["counter"] != BASE_BUDGET:
        raise ValueError("Parent has not finished committing its final outputs")
    if record["history"] != info["history"]:
        raise ValueError("Parent final record and snapshot disagree")
    paths = ("last.resume.pt", "last.pt", "best.resume.pt", "best.pt", "run.json")
    hashes = {name: digest(folder / name) for name in paths}
    return folder, state, hashes


class ExtendedScratchRun(TrainingRun):
    def __init__(self, folder, regime):
        self.extension = None
        super().__init__(folder, regime, "scratch")

    @property
    def budget(self):
        return TOTAL_BUDGET

    def summary(self):
        value = super().summary()
        value["extension"] = deepcopy(self.extension)
        value["additional_transitions"] = max(0, self.counter - BASE_BUDGET)
        value["additional_updates"] = max(0, self.stage_updates -
                                          (BASE_BUDGET - CONFIG.warmup) // CONFIG.slots)
        return value

    def inherit(self, state, parent_folder, hashes):
        # Restore everything, including Adam moments, target lag, games and RNGs.
        super().restore(state)
        self.extension = dict(protocol=EXTENSION, base_budget=BASE_BUDGET,
                              total_budget=TOTAL_BUDGET, source_hashes=implementation_hashes(),
                              parent_folder=str(Path(parent_folder).resolve()), parent_hashes=hashes)
        self.manifest["extension"] = deepcopy(self.extension)

    def restore(self, state):
        extension = state["summary"].get("extension")
        if (extension is None or extension["protocol"] != EXTENSION
                or extension["base_budget"] != BASE_BUDGET
                or extension["total_budget"] != TOTAL_BUDGET
                or extension["source_hashes"] != implementation_hashes()):
            raise ValueError("Extension configuration/source mismatch")
        if not BASE_BUDGET <= state["summary"]["counter"] <= TOTAL_BUDGET:
            raise ValueError("Invalid extension counter")
        super().restore(state)
        self.extension = deepcopy(extension)
        self.manifest["extension"] = deepcopy(extension)


def prepare_extension(parent_root, output, regime):
    """Fork the final state; preserve the original and its selected checkpoint."""
    parent, state, hashes = checked_parent(parent_root, regime)
    folder = Path(output).resolve() / regime / "scratch"
    if folder == parent or parent in folder.parents:
        raise ValueError("Extension output must be separate from the original run")
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Extension already exists; use --resume: {folder}")
    run = ExtendedScratchRun(folder, regime)
    run.inherit(state, parent, hashes)
    folder.mkdir(parents=True, exist_ok=True)
    # An inherited best can remain best through the entire extension.
    for name in ("best.pt", "best.resume.pt"):
        shutil.copyfile(parent / name, folder / name)
    atomic_json(folder / "parent.json", run.extension)
    atomic_save(folder / "resume.pt", run.snapshot())
    atomic_json(folder / "run.json", run.summary())
    return run


def run_extension(parent_root, output, regime, resume=False):
    folder = Path(output).resolve() / regime / "scratch"
    if resume:
        run = ExtendedScratchRun(folder, regime)
        run.restore(read_snapshot(folder / "resume.pt"))
    else:
        run = prepare_extension(parent_root, output, regime)
    try:
        # Existing history suppresses duplicate validation at the old endpoint.
        return run.run()
    except (Exception, KeyboardInterrupt) as error:
        atomic_json(folder / "failure.json", dict(error=repr(error), counter=run.counter,
                    note="Resume from the last committed snapshot; base campaign is unchanged"))
        raise


def stability_summary(info):
    history = info["history"]
    recent = history[-5:]
    fields = dict(wins=lambda h: h["summary"]["wins"],
                  mean_fruits=lambda h: h["summary"]["mean_fruits"],
                  unresolved=lambda h: h["summary"]["outcomes"]["total_limit"])
    window = {}
    for name, getter in fields.items():
        values = [getter(h) for h in recent]
        window[name] = dict(min=min(values), max=max(values), first=values[0],
                            last=values[-1], change=values[-1] - values[0])
    parent_selected = next(h for h in history if h["counter"] == info["selected_counter"])
    return dict(counter=info["counter"], complete=info["complete"],
                selected=parent_selected, final=history[-1],
                window_counters=[h["counter"] for h in recent], window=window,
                interpretation="Descriptive stability on reused validation; no convergence claim")


def extension_report(parent_root, output):
    result = dict(protocol=EXTENSION, base_budget=BASE_BUDGET, total_budget=TOTAL_BUDGET,
                  regimes={}, missing=[], automatic_extension=False, final_test_opened=False)
    for regime in ("full", "partial"):
        paths = dict(original=Path(parent_root) / regime / "scratch/run.json",
                     extended=Path(output) / regime / "scratch/run.json")
        result["regimes"][regime] = {}
        for label, path in paths.items():
            if not path.exists():
                result["missing"].append(str(path))
                continue
            info = json.loads(path.read_text(encoding="utf-8"))
            result["regimes"][regime][label] = stability_summary(info)
    atomic_json(Path(output) / "stability.json", result)
    return result
