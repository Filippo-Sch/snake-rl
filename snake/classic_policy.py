"""Versioned, explicitly launched continuations; historical sources stay untouched."""
from copy import deepcopy
from pathlib import Path
import json
import time

from .classic.config import CONFIG, ROOT
from .classic.environment import ClassicSnake
from .classic.evaluation import selection_key, summarize, write_evaluation
from .classic.learner import choose_actions, save_policy
from .classic.training import TrainingRun, assert_compatible, read_snapshot
from .classic_extension import implementation_hashes as extension_hashes
from .classic_replay import ProtectedReplay
from .training_state import atomic_json, atomic_save, digest

PROTOCOL = "classic-policy-continuation-1"
ADDITIONAL = 1024000
START = dict(full=4096000, partial=2432000)
CHECKPOINT = dict(full="last", partial="best")
SOURCES = ("experiments/classic_policy.py", "snake/classic_policy.py", "snake/classic_replay.py")


def settings(regime, arm):
    if regime not in START or arm not in ("control", "variant"):
        raise ValueError("Choose full/partial and control/variant")
    return dict(protocol=PROTOCOL, regime=regime, arm=arm,
                start=START[regime], additional_transitions=ADDITIONAL,
                learning_rate=5e-5 if (regime, arm) == ("full", "variant") else 1e-4,
                protected_capacity=5000 if (regime, arm) == ("partial", "variant") else 0,
                protected_per_batch=8 if (regime, arm) == ("partial", "variant") else 0,
                replay_capacity=50000, batch_size=64, gamma=1,
                validation_interval=128000, seed=0)


def source_hashes():
    return {name: digest(ROOT / name) for name in SOURCES}


def checked_parent(parent_root, regime):
    folder = Path(parent_root).resolve() / regime / "scratch"
    name = CHECKPOINT[regime]
    path = folder / (name + ".resume.pt")
    state = read_snapshot(path)
    info = state["summary"]
    assert_compatible(info["manifest"])
    record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    if (info["stage"] != "scratch" or info["regime"] != regime
            or info["counter"] != START[regime]
            or info["budget"] != 4096000
            or not record["complete"] or record["counter"] != 4096000
            or (regime == "partial" and record["selected_counter"] != START[regime])):
        raise ValueError("Expected the completed historical extension and agreed checkpoint")
    extension = info.get("extension", {})
    if extension.get("source_hashes") != extension_hashes():
        raise ValueError("Historical extension sources changed")
    if (len(state["envs"]) != CONFIG.slots or state["replay"] is None
            or state["replay"]["capacity"] != CONFIG.replay_capacity):
        raise ValueError("Parent needs all active games and the complete replay")
    expected = (START[regime] - CONFIG.warmup) // CONFIG.slots
    if info["stage_updates"] != expected or state["learner"]["updates"] != expected:
        raise ValueError("Parent update counters disagree")
    rows = [h for h in info["history"] if h["counter"] == START[regime]]
    historical = [h for h in record["history"] if h["counter"] == START[regime]]
    if len(rows) != 1 or historical != rows:
        raise ValueError("Parent checkpoint validation and final record disagree")
    return folder, state, dict(snapshot=digest(path), record=digest(folder / "run.json"))


def evaluate_policy(regime, network, games=CONFIG.validation_games):
    """Same greedy evaluation and seeds; full-state repeats are diagnostics only."""
    records = []
    for game_id in range(games):
        env = ClassicSnake(regime, "validation", game_id,
                           limit=CONFIG.eval_limit, fruit_free_limit=None)
        seen, cycle = set(), False
        while not env.done:
            key = (tuple(env.body), env.heading, env.fruit)
            if key in seen:
                cycle = True
            seen.add(key)
            obs = env.observe()
            action = int(choose_actions(network, obs.encode()[None], obs.legal[None])[0])
            env.step(action)
        records.append(dict(env.record(), exact_cycle=cycle))
    summary = summarize(records)
    summary["exact_cycles"] = sum(r["exact_cycle"] for r in records)
    return dict(regime=regime, role="validation", cap=CONFIG.eval_limit,
                policy="dqn", summary=summary, records=records)


class PolicyRun(TrainingRun):
    def __init__(self, folder, regime, arm):
        self.arm = arm
        self.experiment = None
        settings(regime, arm)
        super().__init__(folder, regime, "scratch")

    @property
    def budget(self):
        return START[self.regime] + ADDITIONAL

    def summary(self):
        value = super().summary()
        value["experiment"] = deepcopy(self.experiment)
        value["additional_transitions"] = self.counter - START[self.regime]
        value["additional_updates"] = self.stage_updates - (
            START[self.regime] - CONFIG.warmup) // CONFIG.slots
        if isinstance(self.replay, ProtectedReplay):
            value["replay_diagnostics"] = self.replay.summary()
        return value

    def inherit(self, state, parent_folder, hashes):
        super().restore(state)
        config = settings(self.regime, self.arm)
        for group in self.learner.optimizer.param_groups:
            group["lr"] = config["learning_rate"]
        if config["protected_capacity"]:
            self.replay = ProtectedReplay(self.replay, self.envs,
                                          config["protected_capacity"],
                                          config["protected_per_batch"])
        initial = next(h for h in self.history if h["counter"] == self.counter)
        self.history = [deepcopy(initial)]
        self.selected_counter = self.counter
        self.best_key = selection_key(initial["summary"])
        self.trace, self.finished_games = [], []
        self.elapsed = 0.
        self.experiment = dict(settings=config, source_hashes=source_hashes(),
                               parent_folder=str(parent_folder), parent_hashes=hashes,
                               parent_checkpoint=CHECKPOINT[self.regime])
        self.manifest["policy_continuation"] = deepcopy(self.experiment)

    def restore(self, state):
        experiment = state["summary"].get("experiment")
        if (not experiment or experiment["settings"] != settings(self.regime, self.arm)
                or experiment["source_hashes"] != source_hashes()):
            raise ValueError("Policy continuation settings/source mismatch")
        if not START[self.regime] <= state["summary"]["counter"] <= self.budget:
            raise ValueError("Invalid continuation counter")
        protected = experiment["settings"]["protected_capacity"] > 0
        if protected:
            replay = ProtectedReplay.from_state(state["replay"])
            if (replay.capacity != CONFIG.replay_capacity
                    or replay.protected_capacity != 5000 or replay.quota != 8):
                raise ValueError("Protected replay configuration mismatch")
            base = dict(state, replay=state["replay"]["ordinary"])
        else:
            base = state
        super().restore(base)
        if protected:
            self.replay = replay
        if any(g["lr"] != settings(self.regime, self.arm)["learning_rate"]
               for g in self.learner.optimizer.param_groups):
            raise ValueError("Saved optimizer learning rate mismatch")
        self.experiment = deepcopy(experiment)
        self.manifest["policy_continuation"] = deepcopy(experiment)

    def validate_and_save(self):
        result = evaluate_policy(self.regime, self.learner.network)
        self.evaluation_actions += result["summary"]["actions"]
        write_evaluation(self.folder / "validation" / f"{self.counter:09d}", result)
        # Replace the inherited baseline when adding cycle diagnostics.
        if self.history and self.history[-1]["counter"] == self.counter:
            self.history.pop()
        row = dict(counter=self.counter, lineage_updates=self.learner.updates,
                   summary=result["summary"], holdout=None)
        if isinstance(self.replay, ProtectedReplay):
            row["replay"] = self.replay.summary()
        self.history.append(row)
        key = selection_key(result["summary"])
        improved = self.counter == START[self.regime] or key > self.best_key
        if improved:
            self.best_key, self.selected_counter = key, self.counter
        self._flush_trace()
        self.elapsed += time.perf_counter() - self._clock
        self._clock = time.perf_counter()
        state = self.snapshot()
        metadata = dict(regime=self.regime, stage=self.stage, counter=self.counter,
                        lineage_updates=self.learner.updates, manifest=self.manifest,
                        demo_hash=None, summary=result["summary"])
        if improved:
            atomic_save(self.folder / "best.resume.pt", state)
            save_policy(self.folder / "best.pt", self.learner, metadata)
        if self.counter == self.budget:
            atomic_save(self.folder / "last.resume.pt", state)
            save_policy(self.folder / "last.pt", self.learner, metadata)
        atomic_save(self.folder / "resume.pt", state)
        atomic_json(self.folder / "run.json", self.summary())
        print(f"{self.regime}/{self.arm} +{self.counter - START[self.regime]}/{ADDITIONAL}: "
              f"wins={key[0]}, fruit={key[1]:.3f}, cycles={result['summary']['exact_cycles']}",
              flush=True)

    def run(self):
        if "exact_cycles" not in self.history[-1]["summary"]:
            self.validate_and_save()
        return super().run()


def prepare(parent_root, output, regime, arm):
    parent, state, hashes = checked_parent(parent_root, regime)
    folder = Path(output).resolve() / regime / arm
    original_root = Path(parent_root).resolve()
    if folder == original_root or original_root in folder.parents or folder in original_root.parents:
        raise ValueError("Use an output separate from historical checkpoints")
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Continuation exists; use --resume: {folder}")
    run = PolicyRun(folder, regime, arm)
    run.inherit(state, parent, hashes)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "parent.json", run.experiment)
    atomic_save(folder / "resume.pt", run.snapshot())
    atomic_json(folder / "run.json", run.summary())
    return run


def run_experiment(parent_root, output, regime, arm, resume=False):
    folder = Path(output).resolve() / regime / arm
    if resume:
        run = PolicyRun(folder, regime, arm)
        run.restore(read_snapshot(folder / "resume.pt"))
    else:
        run = prepare(parent_root, output, regime, arm)
    try:
        return run.run()
    except (Exception, KeyboardInterrupt) as error:
        atomic_json(folder / "failure.json", dict(error=repr(error), counter=run.counter,
                    note="Resume from the last committed snapshot; parents are unchanged"))
        raise


def report(output):
    result = dict(protocol=PROTOCOL, runs={}, missing=[], test_opened=False)
    for regime in START:
        for arm in ("control", "variant"):
            path = Path(output) / regime / arm / "run.json"
            label = regime + "/" + arm
            if not path.exists():
                result["missing"].append(label)
                continue
            info = json.loads(path.read_text(encoding="utf-8"))
            recent = [h for h in info["history"] if h["counter"] > START[regime]][-4:]
            metrics = {}
            for field in ("wins", "mean_fruits", "exact_cycles"):
                values = [h["summary"][field] for h in recent]
                metrics[field] = (dict(mean=sum(values) / len(values),
                                       minimum=min(values), maximum=max(values))
                                  if values else None)
            result["runs"][label] = dict(
                complete=info["complete"], additional_transitions=info["additional_transitions"],
                selected_counter=info["selected_counter"],
                final=info["history"][-1]["summary"],
                window_counters=[h["counter"] for h in recent], last_four=metrics,
                replay=info.get("replay_diagnostics"))
    return result

