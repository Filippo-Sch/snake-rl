"""Fixed full-only Watkins experiment; reuse the completed DQN control."""
from copy import deepcopy
from pathlib import Path
import json
import time

from .classic.config import CONFIG, ROOT
from .classic.data import Replay
from .classic.training import TrainingRun, read_snapshot
from .classic.evaluation import selection_key
from .classic_policy import PolicyRun, checked_parent, SOURCES as POLICY_SOURCES
from .classic_watkins import (HORIZON, LAMBDA, SequenceReplay, WatkinsLearner, value_audit)
from .training_state import atomic_json, atomic_save, digest

PROTOCOL = "classic-watkins-full-v1"
START = 4096000
ADDITIONAL = 1024000
SOURCES = (*POLICY_SOURCES, "experiments/classic_watkins.py", "snake/classic_watkins.py",
           "snake/classic_watkins_run.py")


def settings():
    return dict(protocol=PROTOCOL, regime="full", start=START,
                additional_transitions=ADDITIONAL, trace_lambda=LAMBDA, horizon=HORIZON,
                learning_rate=1e-4, gamma=1, batch_size=64, replay_capacity=50000,
                validation_interval=128000, seed=0, target_policy="target-network-first-argmax",
                trace_cut="next-action-not-greedy", loss="start-transition-only",
                target_interval=500, protected_replay=False, double_dqn=False)


def source_hashes():
    return {name: digest(ROOT / name) for name in SOURCES}


def checked_control(control_root, parent_hashes):
    path = Path(control_root).resolve() / "full/control/run.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    experiment = record.get("experiment", {})
    saved_settings = experiment.get("settings", {})
    if (not record["complete"] or record["counter"] != START + ADDITIONAL
            or saved_settings.get("regime") != "full" or saved_settings.get("arm") != "control"
            or saved_settings.get("learning_rate") != 1e-4
            or experiment.get("parent_hashes") != parent_hashes
            or experiment.get("source_hashes") != {n: digest(ROOT / n) for n in POLICY_SOURCES}):
        raise ValueError("Existing control does not match the agreed parent/protocol")
    return dict(path=str(path), sha256=digest(path))


class WatkinsRun(PolicyRun):
    def __init__(self, folder):
        super().__init__(folder, "full", "control")
        self.arm = "watkins"
        self.learner = WatkinsLearner("full")
        self.latest_value_audit = None
        self.block_diagnostics = None

    def inherit(self, state, parent_folder, hashes, control):
        TrainingRun.restore(self, state)
        if any(g["lr"] != 1e-4 for g in self.learner.optimizer.param_groups):
            raise ValueError("Unexpected parent learning rate")
        self.replay = SequenceReplay.from_state(state["replay"])
        initial = next(h for h in self.history if h["counter"] == START)
        self.history = [deepcopy(initial)]
        self.selected_counter = START
        self.best_key = selection_key(initial["summary"])
        self.trace, self.finished_games = [], []
        self.elapsed = 0.
        self.experiment = dict(settings=settings(), source_hashes=source_hashes(),
                               parent_folder=str(parent_folder), parent_hashes=hashes,
                               parent_checkpoint="last", control=control)
        self.manifest["watkins"] = deepcopy(self.experiment)

    def summary(self):
        result = super().summary()
        result["value_audit"] = deepcopy(self.latest_value_audit)
        result["watkins_diagnostics"] = deepcopy(self.block_diagnostics)
        return result

    def restore(self, state):
        experiment = state["summary"].get("experiment")
        if (not experiment or experiment["settings"] != settings()
                or experiment["source_hashes"] != source_hashes()):
            raise ValueError("Watkins settings/source mismatch")
        if not START <= state["summary"]["counter"] <= START + ADDITIONAL:
            raise ValueError("Invalid Watkins counter")
        TrainingRun.restore(self, state)
        self.replay = SequenceReplay.from_state(state["replay"])
        if self.replay.horizon != HORIZON or self.replay.capacity != 50000:
            raise ValueError("Sequence replay configuration mismatch")
        if any(g["lr"] != 1e-4 for g in self.learner.optimizer.param_groups):
            raise ValueError("Saved optimizer learning rate mismatch")
        self.latest_value_audit = deepcopy(state["summary"]["value_audit"])
        self.block_diagnostics = deepcopy(state["summary"]["watkins_diagnostics"])
        self.experiment = deepcopy(experiment)
        self.manifest["watkins"] = deepcopy(experiment)

    def validate_and_save(self):
        self.latest_value_audit = value_audit(self.learner.network, self.replay)
        if self.trace:
            keys = ("sequence_mean_length", "trace_mean_span", "trace_mean_weighted_depth",
                    "trace_cut_fraction", "trace_first_cut_fraction", "loss", "max_abs_q")
            self.block_diagnostics = {
                k: sum(r["diagnostics"][k] for r in self.trace) / len(self.trace) for k in keys}
        # Parent implementation writes all inference/resume checkpoints and JSON.
        super().validate_and_save()
        atomic_json(self.folder / "validation" / f"{self.counter:09d}" / "learning.json",
                    dict(counter=self.counter, value_audit=self.latest_value_audit,
                         watkins=self.block_diagnostics))
        print("learning: " + json.dumps(dict(value_audit=self.latest_value_audit,
                                            watkins=self.block_diagnostics)), flush=True)

    def run(self):
        # Refresh wall-clock origin on resume; no duplicate boundary evaluation.
        self._clock = time.perf_counter()
        return super().run()


def preflight(parent_root, control_root):
    parent, state, hashes = checked_parent(parent_root, "full")
    control = checked_control(control_root, hashes)
    return parent, state, hashes, control


def prepare(parent_root, control_root, output):
    parent, state, hashes, control = preflight(parent_root, control_root)
    folder = Path(output).resolve() / "full/watkins"
    for root in (Path(parent_root).resolve(), Path(control_root).resolve()):
        if folder == root or root in folder.parents or folder in root.parents:
            raise ValueError("Use an output separate from historical runs")
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Watkins run exists; use --resume: {folder}")
    run = WatkinsRun(folder)
    run.inherit(state, parent, hashes, control)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "parent.json", run.experiment)
    atomic_save(folder / "resume.pt", run.snapshot())
    atomic_json(folder / "run.json", run.summary())
    return run


def train(parent_root, control_root, output, resume=False):
    folder = Path(output).resolve() / "full/watkins"
    if resume:
        run = WatkinsRun(folder)
        run.restore(read_snapshot(folder / "resume.pt"))
        control = run.experiment["control"]
        if digest(control["path"]) != control["sha256"]:
            raise ValueError("Saved historical control changed")
    else:
        run = prepare(parent_root, control_root, output)
    try:
        result = run.run()
        report(output, control_root, save=True)
        return result
    except (Exception, KeyboardInterrupt) as error:
        atomic_json(folder / "failure.json", dict(error=repr(error), counter=run.counter,
                    note="Resume the last committed boundary; no automatic protocol changes"))
        raise


def report(output, control_root, save=False):
    folder = Path(output).resolve() / "full/watkins"
    path = folder / "run.json"
    control_path = Path(control_root).resolve() / "full/control/run.json"
    if not path.exists():
        return dict(protocol=PROTOCOL, status="not-started")
    info = json.loads(path.read_text(encoding="utf-8"))
    control = json.loads(control_path.read_text(encoding="utf-8"))
    expected = info["experiment"]["control"]
    if str(control_path) != expected["path"] or digest(control_path) != expected["sha256"]:
        raise ValueError("Comparison control path/hash mismatch")
    controls = {h["counter"]: h["summary"] for h in control["history"]}
    rows = []
    for h in info["history"]:
        if h["counter"] not in controls:
            raise ValueError("No same-budget control validation")
        row = dict(counter=h["counter"], additional=h["counter"] - START)
        for label, summary in (("watkins", h["summary"]), ("control", controls[h["counter"]])):
            row[label] = {k: summary.get(k) for k in ("wins", "mean_fruits", "exact_cycles")}
        learning = folder / "validation" / f"{h['counter']:09d}" / "learning.json"
        if learning.exists():
            row["learning"] = json.loads(learning.read_text(encoding="utf-8"))
        rows.append(row)
    recent = [r for r in rows if r["counter"] > START][-4:]
    window = {}
    for label in ("control", "watkins"):
        window[label] = {}
        for metric in ("wins", "mean_fruits", "exact_cycles"):
            values = [r[label][metric] for r in recent]
            window[label][metric] = (dict(mean=sum(values) / len(values),
                                         minimum=min(values), maximum=max(values)) if values else None)
    result = dict(protocol=PROTOCOL, complete=info["complete"], selected_counter=info["selected_counter"],
                  rows=rows, last_four=window, last_four_counters=[r["counter"] for r in recent],
                  final_test_opened=False)
    if save:
        atomic_json(folder / "comparison.json", result)
    return result

