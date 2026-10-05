"""Isolated one-step DQN discount experiment; historical implementations are frozen."""
from copy import deepcopy
from pathlib import Path
import json
import time

import torch
from torch import nn

from .classic.config import CONFIG, ROOT
from .classic.evaluation import selection_key
from .classic.learner import Learner
from .classic.training import TrainingRun, read_snapshot
from .classic_policy import PolicyRun, checked_parent
from .classic_watkins import value_audit
from .classic_watkins_run import checked_control, SOURCES as PREVIOUS_SOURCES
from .training_state import atomic_json, atomic_save, digest

GAMMA = 0.995
START = 4096000
ADDITIONAL = 1024000
PROTOCOL = "classic-dqn-discount-full-v1"
SOURCES = (*PREVIOUS_SOURCES, "experiments/classic_discount.py", "snake/classic_discount.py")


@torch.no_grad()
def discounted_targets(target, rewards, next_states, terminated, next_legal, gamma=GAMMA):
    if not 0 <= gamma <= 1:
        raise ValueError("Gamma must be in [0, 1]")
    result = rewards.clone()
    active = ~terminated
    if active.any() and gamma:
        values = target(next_states[active])
        masks = next_legal[active]
        if not masks.any(dim=1).all() or not torch.isfinite(values).all():
            raise FloatingPointError("Invalid discounted DQN target values/masks")
        bootstrap = values.masked_fill(~masks, -torch.inf).max(1).values
        result[active] += gamma * bootstrap
    return result


def discounted_loss(network, target, batch, gamma=GAMMA):
    if batch["eligible"].any() or batch["demo"].any():
        raise ValueError("This experiment supports ordinary self-replay only")
    states = torch.as_tensor(batch["states"], dtype=torch.float32)
    actions = torch.as_tensor(batch["actions"], dtype=torch.long)
    legal = torch.as_tensor(batch["legal"], dtype=torch.bool)
    q = network(states)
    if not legal.gather(1, actions[:, None]).all():
        raise ValueError("Illegal replay action")
    chosen = q.gather(1, actions[:, None]).squeeze(1)
    targets = discounted_targets(target,
        torch.as_tensor(batch["rewards"], dtype=torch.float32),
        torch.as_tensor(batch["next_states"], dtype=torch.float32),
        torch.as_tensor(batch["terminated"], dtype=torch.bool),
        torch.as_tensor(batch["next_legal"], dtype=torch.bool), gamma)
    weights = torch.as_tensor(batch["weights"], dtype=torch.float32).detach()
    loss = (weights * nn.functional.smooth_l1_loss(chosen, targets, reduction="none")).mean()
    if not torch.isfinite(q).all() or not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite discounted DQN loss")
    return loss, dict(loss=float(loss.detach()), td=float(loss.detach()),
                      imitation=0., eligible_fraction=0.,
                      max_abs_q=float(q.detach().abs().max()),
                      mean_target=float(targets.mean()))


class DiscountLearner(Learner):
    def update(self, batch):
        loss, diagnostics = discounted_loss(self.network, self.target, batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.network.parameters(), CONFIG.gradient_norm,
                                       error_if_nonfinite=True)
        self.optimizer.step()
        if not all(torch.isfinite(p).all() for p in self.network.parameters()):
            raise FloatingPointError("Nonfinite parameters after Adam")
        self.updates += 1
        if self.updates % CONFIG.target_interval == 0:
            self.target.load_state_dict(self.network.state_dict())
        return dict(diagnostics, gradient_norm=float(norm), lineage_updates=self.updates)


def settings():
    return dict(protocol=PROTOCOL, regime="full", algorithm="one-step-DQN", gamma=GAMMA,
                start=START, additional_transitions=ADDITIONAL,
                learning_rate=1e-4, batch_size=64, replay_capacity=50000,
                validation_interval=128000, target_interval=500, seed=0,
                double_dqn=False, protected_replay=False, memory=False,
                evaluation="unchanged-undiscounted-validation",
                automatic_followup=False)


def source_hashes():
    return {name: digest(ROOT / name) for name in SOURCES}


def preflight(parent_root, control_root):
    parent, state, hashes = checked_parent(parent_root, "full")
    control = checked_control(control_root, hashes)
    record = json.loads(Path(control["path"]).read_text(encoding="utf-8"))
    baseline = record["experiment"]["settings"]
    if (baseline.get("gamma") != 1 or baseline.get("protected_capacity") != 0
            or baseline.get("additional_transitions") != ADDITIONAL
            or baseline.get("batch_size") != CONFIG.batch_size):
        raise ValueError("Control must be the unchanged one-step DQN gamma=1 continuation")
    return parent, state, hashes, control


class DiscountRun(PolicyRun):
    def __init__(self, folder):
        super().__init__(folder, "full", "control")
        self.arm = "discount"
        self.learner = DiscountLearner("full")
        self.latest_value_audit = None
        self.block_diagnostics = None

    def inherit(self, state, parent_folder, hashes, control):
        TrainingRun.restore(self, state)
        if any(g["lr"] != 1e-4 for g in self.learner.optimizer.param_groups):
            raise ValueError("Unexpected parent learning rate")
        initial = next(h for h in self.history if h["counter"] == START)
        self.history = [deepcopy(initial)]
        self.selected_counter, self.best_key = START, selection_key(initial["summary"])
        self.trace, self.finished_games = [], []
        self.elapsed = 0.
        self.experiment = dict(settings=settings(), source_hashes=source_hashes(),
                               parent_folder=str(parent_folder), parent_hashes=hashes,
                               parent_checkpoint="last", control=control)
        self.manifest["discount_experiment"] = deepcopy(self.experiment)

    def summary(self):
        result = super().summary()
        result["value_audit"] = deepcopy(self.latest_value_audit)
        result["learning_diagnostics"] = deepcopy(self.block_diagnostics)
        return result

    def restore(self, state):
        experiment = state["summary"].get("experiment")
        if (not experiment or experiment["settings"] != settings()
                or experiment["source_hashes"] != source_hashes()):
            raise ValueError("Discount settings/source mismatch")
        counter = state["summary"]["counter"]
        expected_updates = (counter - CONFIG.warmup) // CONFIG.slots
        if (not START <= counter <= START + ADDITIONAL or counter % CONFIG.slots
                or state["learner"]["updates"] != expected_updates
                or state["summary"]["stage_updates"] != expected_updates):
            raise ValueError("Invalid discount counters")
        if (state["replay"].get("sequence_format") is not None
                or state["replay"]["capacity"] != CONFIG.replay_capacity):
            raise ValueError("Expected ordinary one-step replay")
        TrainingRun.restore(self, state)
        if any(g["lr"] != 1e-4 for g in self.learner.optimizer.param_groups):
            raise ValueError("Saved optimizer learning rate mismatch")
        self.latest_value_audit = deepcopy(state["summary"]["value_audit"])
        self.block_diagnostics = deepcopy(state["summary"]["learning_diagnostics"])
        self.experiment = deepcopy(experiment)
        self.manifest["discount_experiment"] = deepcopy(experiment)

    def validate_and_save(self):
        # This undiscounted bound is also a conservative upper bound for gamma<1.
        # Values below it are not thereby proven calibrated under either objective.
        self.latest_value_audit = value_audit(self.learner.network, self.replay)
        self.latest_value_audit["bound"] = "common-undiscounted-optimistic-return-bound"
        self.latest_value_audit["training_gamma"] = GAMMA
        if self.trace:
            keys = ("loss", "max_abs_q", "mean_target", "gradient_norm")
            self.block_diagnostics = {
                key: sum(r["diagnostics"][key] for r in self.trace) / len(self.trace)
                for key in keys}
            self.block_diagnostics["clipped_fraction"] = sum(
                r["diagnostics"]["gradient_norm"] > CONFIG.gradient_norm
                for r in self.trace) / len(self.trace)
        super().validate_and_save()
        atomic_json(self.folder / "validation" / f"{self.counter:09d}" / "learning.json",
                    dict(counter=self.counter, value_audit=self.latest_value_audit,
                         diagnostics=self.block_diagnostics))
        print("learning: " + json.dumps(dict(value_audit=self.latest_value_audit,
                                            diagnostics=self.block_diagnostics)), flush=True)

    def run(self):
        self._clock = time.perf_counter()
        return super().run()


def prepare(parent_root, control_root, output):
    parent, state, hashes, control = preflight(parent_root, control_root)
    folder = Path(output).resolve() / "full/discount"
    for root in (Path(parent_root).resolve(), Path(control_root).resolve()):
        if folder == root or root in folder.parents or folder in root.parents:
            raise ValueError("Use an output separate from historical runs")
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Discount run exists; use --resume: {folder}")
    run = DiscountRun(folder)
    run.inherit(state, parent, hashes, control)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "parent.json", run.experiment)
    atomic_save(folder / "resume.pt", run.snapshot())
    atomic_json(folder / "run.json", run.summary())
    return run


def train(parent_root, control_root, output, resume=False):
    folder = Path(output).resolve() / "full/discount"
    if resume:
        run = DiscountRun(folder)
        run.restore(read_snapshot(folder / "resume.pt"))
        control = run.experiment["control"]
        requested = Path(control_root).resolve() / "full/control/run.json"
        if str(requested) != control["path"] or digest(requested) != control["sha256"]:
            raise ValueError("Saved historical control changed")
    else:
        run = prepare(parent_root, control_root, output)
    try:
        result = run.run()
        report(output, control_root, save=True)
        return result
    except (Exception, KeyboardInterrupt) as error:
        atomic_json(folder / "failure.json", dict(error=repr(error), counter=run.counter,
                    note="Resume the last committed boundary; no automatic follow-up experiments"))
        raise


def report(output, control_root, save=False):
    folder = Path(output).resolve() / "full/discount"
    path = folder / "run.json"
    if not path.exists():
        return dict(protocol=PROTOCOL, status="not-started", automatic_followup=False)
    info = json.loads(path.read_text(encoding="utf-8"))
    control_path = Path(control_root).resolve() / "full/control/run.json"
    expected = info["experiment"]["control"]
    if str(control_path) != expected["path"] or digest(control_path) != expected["sha256"]:
        raise ValueError("Comparison control path/hash mismatch")
    control = json.loads(control_path.read_text(encoding="utf-8"))
    controls = {h["counter"]: h["summary"] for h in control["history"]}
    rows = []
    for h in info["history"]:
        if h["counter"] not in controls:
            raise ValueError("No same-budget control validation")
        row = dict(counter=h["counter"], additional=h["counter"] - START)
        for label, summary in (("discount", h["summary"]), ("control", controls[h["counter"]])):
            row[label] = {key: summary.get(key) for key in ("wins", "mean_fruits", "exact_cycles")}
        learning = folder / "validation" / f"{h['counter']:09d}" / "learning.json"
        if learning.exists():
            row["learning"] = json.loads(learning.read_text(encoding="utf-8"))
        rows.append(row)
    recent = [r for r in rows if r["counter"] > START][-4:]
    window = {}
    for label in ("control", "discount"):
        window[label] = {}
        for metric in ("wins", "mean_fruits", "exact_cycles"):
            values = [r[label][metric] for r in recent]
            window[label][metric] = (dict(mean=sum(values)/len(values),
                                         minimum=min(values), maximum=max(values)) if values else None)
    result = dict(protocol=PROTOCOL, complete=info["complete"], selected_counter=info["selected_counter"],
                  rows=rows, last_four=window, last_four_counters=[r["counter"] for r in recent],
                  final_test_opened=False, automatic_followup=False,
                  next_action="Review results for the report; further training needs a concrete high-value hypothesis",
                  interpretation="Discount changes the training objective; evaluation metrics remain undiscounted")
    if save:
        atomic_json(folder / "comparison.json", result)
    return result

