"""Explicit fixed-budget stages and complete boundary snapshots.

Full resume files are trusted local project artifacts (pickle); inference files
are separate and load with weights_only=True. Importing this module never trains.
"""

from copy import deepcopy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from ..training_state import atomic_json, atomic_save, digest
from .config import CONFIG, PROTOCOL, demo_probability, epsilon, manifest, rng, seed_training
from .data import Demonstrations, Replay, mixed_batch, transition
from .environment import ClassicSnake
from .evaluation import evaluate, selection_key, write_evaluation
from .learner import Learner, choose_actions, loss_terms, save_policy
from .policies import Decision

SNAPSHOT_FORMAT = "classic-dqn-resume-v1"


def read_snapshot(path):
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("format") != SNAPSHOT_FORMAT or value.get("protocol") != PROTOCOL:
        raise ValueError("Not a classic training snapshot")
    if value.get("boundary") != "before_next_round":
        raise ValueError("Snapshot is not at a resumable boundary")
    return value


def assert_compatible(saved):
    current = manifest()
    for key in ("config_hash", "source_hash", "software"):
        if saved[key] != current[key]:
            raise ValueError(f"Classic {key} changed; do not silently alter an existing run")


class TrainingRun:
    def __init__(self, folder, regime, stage, demos=None):
        if stage not in ("scratch", "offline", "online"):
            raise ValueError("Unknown training stage")
        if (stage != "scratch") != (demos is not None):
            raise ValueError("Only assisted stages require demonstrations")
        if demos is not None and demos.regime != regime:
            raise ValueError("Teacher and student observation regimes must match")
        self.folder, self.regime, self.stage = Path(folder), regime, stage
        self.demos = demos
        self.manifest = manifest()
        seed_training()
        self.learner = Learner(regime)
        self.replay = Replay(regime) if stage != "offline" else None
        self.exploration_rng, self.mixing_rng = rng("exploration"), rng("mixing")
        self.envs = []
        self.next_game_id = 0
        self.counter = self.stage_updates = self.inherited_spent_updates = 0
        self.evaluation_actions = self.inherited_evaluation_actions = 0
        self.elapsed = 0.0
        self.history = []
        self.selected_counter = None
        self.best_key = None
        self.collection = dict(completed_games=0, fruits=0, reward=0.0,
                               outcomes={k: 0 for k in ("win", "wall_death", "body_death",
                                                        "total_limit", "fruit_free_limit")},
                               length_counts=[0] * 26)
        self.parent = None
        self.trace, self.finished_games = [], []
        self._clock = time.perf_counter()

    @property
    def budget(self):
        return CONFIG.offline_updates if self.stage == "offline" else CONFIG.online_budget

    def new_game(self):
        env = ClassicSnake(self.regime, "training", self.next_game_id)
        self.next_game_id += 1
        return env

    def initialize_games(self):
        if self.stage != "offline":
            self.envs = [self.new_game() for _ in range(CONFIG.slots)]

    def _rng_state(self):
        return dict(python=random.getstate(), torch=torch.get_rng_state().clone(),
                    exploration=deepcopy(self.exploration_rng.bit_generator.state),
                    mixing=deepcopy(self.mixing_rng.bit_generator.state),
                    demo=deepcopy(self.demos.rng.bit_generator.state) if self.demos else None)

    def _restore_rng(self, value):
        random.setstate(value["python"])
        torch.set_rng_state(value["torch"])
        self.exploration_rng.bit_generator.state = value["exploration"]
        self.mixing_rng.bit_generator.state = value["mixing"]
        if self.demos is not None:
            self.demos.rng.bit_generator.state = value["demo"]

    def summary(self):
        return dict(stage=self.stage, regime=self.regime, counter=self.counter,
                    budget=self.budget, complete=self.counter == self.budget,
                    lineage_updates=self.learner.updates, stage_updates=self.stage_updates,
                    inherited_spent_updates=self.inherited_spent_updates,
                    spent_updates=self.inherited_spent_updates + self.stage_updates,
                    evaluation_actions=self.evaluation_actions,
                    inherited_evaluation_actions=self.inherited_evaluation_actions,
                    collection=deepcopy(self.collection), selected_counter=self.selected_counter,
                    selection="fruit fallback" if self.best_key and self.best_key[0] == 0 else "completion first",
                    history=deepcopy(self.history), elapsed_seconds=self.elapsed,
                    demo_hash=self.demos.file_hash if self.demos else None,
                    demo_summary=self.demos.summary() if self.demos else None,
                    parent=self.parent, manifest=self.manifest)

    def snapshot(self):
        return dict(format=SNAPSHOT_FORMAT, protocol=PROTOCOL, boundary="before_next_round",
                    summary=self.summary(), learner=self.learner.state_dict(),
                    replay=self.replay.state_dict() if self.replay else None,
                    envs=[e.state_dict() for e in self.envs], next_game_id=self.next_game_id,
                    rng=self._rng_state(), best_key=self.best_key,
                    trace=deepcopy(self.trace), finished_games=deepcopy(self.finished_games))

    def restore(self, state):
        summary = state["summary"]
        assert_compatible(summary["manifest"])
        if summary["regime"] != self.regime or summary["stage"] != self.stage:
            raise ValueError("Snapshot stage/regime mismatch")
        if summary["demo_hash"] != (self.demos.file_hash if self.demos else None):
            raise ValueError("Demonstration file changed")
        self.learner.load_state_dict(state["learner"])
        self.replay = Replay.from_state(state["replay"]) if state["replay"] else None
        self.envs = [ClassicSnake.from_state(s) for s in state["envs"]]
        self.next_game_id = state["next_game_id"]
        for name in ("counter", "stage_updates", "inherited_spent_updates", "evaluation_actions",
                     "inherited_evaluation_actions", "history", "selected_counter", "collection", "parent"):
            setattr(self, name, deepcopy(summary[name]))
        self.elapsed = summary["elapsed_seconds"]
        self.best_key = state["best_key"]
        self.trace, self.finished_games = deepcopy(state["trace"]), deepcopy(state["finished_games"])
        self._clock = time.perf_counter()
        self._restore_rng(state["rng"])

    def inherit_offline(self, path, final_summary):
        state = read_snapshot(path)
        summary = state["summary"]
        assert_compatible(summary["manifest"])
        if (self.stage != "online" or summary["stage"] != "offline"
                or summary["regime"] != self.regime or summary["demo_hash"] != self.demos.file_hash):
            raise ValueError("Continuation requires matching selected offline snapshot and demonstrations")
        if not final_summary["complete"] or final_summary["stage"] != "offline":
            raise ValueError("Finish the fixed offline budget before continuation")
        assert_compatible(final_summary["manifest"])
        if final_summary["regime"] != self.regime or final_summary["demo_hash"] != self.demos.file_hash:
            raise ValueError("Final offline accounting does not match the selected lineage")
        if final_summary["selected_counter"] != summary["counter"]:
            raise ValueError("Continuation must use the selected offline checkpoint")
        self.learner.load_state_dict(state["learner"])
        self._restore_rng(state["rng"])
        self.inherited_spent_updates = final_summary["spent_updates"]
        self.inherited_evaluation_actions = final_summary["evaluation_actions"]
        self.parent = dict(snapshot_hash=digest(path), selected_offline_update=summary["counter"],
                           discarded_offline_updates=final_summary["stage_updates"] - summary["counter"])
        self.initialize_games()

    def advance(self):
        """One offline update or one 16-transition round. Explicit caller only."""
        if self.counter >= self.budget:
            raise ValueError("Fixed training budget is already complete")
        if self.stage == "offline":
            batch = self.demos.sample(CONFIG.batch_size)
            self.counter += 1
        else:
            observations = [e.observe() for e in self.envs]
            states = np.stack([o.encode() for o in observations])
            masks = np.stack([o.legal for o in observations])
            actions = choose_actions(self.learner.network, states, masks,
                                     epsilon(self.counter, self.stage == "online"), self.exploration_rng)
            for slot, (env, observation, action) in enumerate(zip(self.envs, observations, actions)):
                decision = Decision(int(action), "student")
                result = env.step(int(action))
                self.replay.add(transition(observation, decision, result, env.game_id, len(env.body)))
                self.collection["fruits"] += int(result.fruit)
                self.collection["reward"] += result.reward
                self.collection["length_counts"][len(env.body)] += 1
                if env.done:
                    self.collection["completed_games"] += 1
                    self.collection["outcomes"][env.outcome] += 1
                    self.finished_games.append(env.record())
                    self.envs[slot] = self.new_game()
            self.counter += CONFIG.slots
            if self.counter <= CONFIG.warmup:
                return None
            rho = demo_probability(self.counter) if self.stage == "online" else 0.0
            batch = mixed_batch(self.replay, self.demos, rho, self.mixing_rng)
        diagnostics = self.learner.update(batch)
        self.stage_updates += 1
        self.trace.append(dict(counter=self.counter, diagnostics=diagnostics,
                               **{k: batch[k].copy() for k in ("game_ids", "probabilities", "weights",
                                                              "eligible", "demo")}))
        return diagnostics

    def holdout_diagnostics(self):
        if self.demos is None or not len(self.demos.holdout):
            return None
        td_sum, eligible, agree = 0.0, 0, 0
        indices = self.demos.holdout
        with torch.no_grad():
            for start in range(0, len(indices), CONFIG.batch_size):
                selected = indices[start:start + CONFIG.batch_size]
                batch = {k: a[selected].copy() for k, a in self.demos.arrays.items()}
                batch["weights"] = np.ones(len(selected), dtype=np.float32)
                _, diagnostics = loss_terms(self.learner.network, self.learner.target, batch)
                td_sum += diagnostics["td"] * len(selected)
                actions = choose_actions(self.learner.network, batch["states"], batch["legal"])
                eligible += int(batch["eligible"].sum())
                agree += int(((actions == batch["actions"]) & batch["eligible"]).sum())
        return dict(transitions=len(indices), mean_td=td_sum / len(indices), eligible=eligible,
                    eligible_action_agreement=agree / eligible if eligible else None)

    def _flush_trace(self):
        """Idempotent blocks at snapshot boundaries avoid duplicate resume logs."""
        folder = self.folder / "trace"
        folder.mkdir(parents=True, exist_ok=True)
        if self.trace:
            values = {key: np.stack([r[key] for r in self.trace])
                      for key in ("game_ids", "probabilities", "weights", "eligible", "demo")}
            values["counters"] = np.asarray([r["counter"] for r in self.trace])
            for key in self.trace[0]["diagnostics"]:
                values[key] = np.asarray([r["diagnostics"][key] for r in self.trace])
            path = folder / f"samples_{self.counter:09d}.npz"
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **values)
            temporary.replace(path)
        atomic_json(folder / f"games_{self.counter:09d}.json", self.finished_games)
        self.trace, self.finished_games = [], []

    def validate_and_save(self):
        result = evaluate(self.regime, network=self.learner.network)
        self.evaluation_actions += result["summary"]["actions"]
        write_evaluation(self.folder / "validation" / f"{self.counter:09d}", result)
        holdout = self.holdout_diagnostics() if self.stage == "offline" else None
        self.history.append(dict(counter=self.counter, lineage_updates=self.learner.updates,
                                 summary=result["summary"], holdout=holdout))
        key = selection_key(result["summary"])
        eligible = self.counter > 0 or self.stage == "online"
        improved = eligible and (self.best_key is None or key > self.best_key)
        if improved:
            self.best_key, self.selected_counter = key, self.counter
        self._flush_trace()
        self.elapsed += time.perf_counter() - self._clock
        self._clock = time.perf_counter()
        state = self.snapshot()
        metadata = dict(regime=self.regime, stage=self.stage, counter=self.counter,
                        lineage_updates=self.learner.updates, manifest=self.manifest,
                        demo_hash=self.demos.file_hash if self.demos else None,
                        summary=result["summary"])
        if improved:
            atomic_save(self.folder / "best.resume.pt", state)
            save_policy(self.folder / "best.pt", self.learner, metadata)
        if self.stage == "online" and self.counter in (0, 128000, 512000):
            atomic_save(self.folder / f"boundary_{self.counter}.resume.pt", state)
            save_policy(self.folder / f"boundary_{self.counter}.pt", self.learner, metadata)
        if self.counter == self.budget:
            atomic_save(self.folder / "last.resume.pt", state)
            save_policy(self.folder / "last.pt", self.learner, metadata)
        atomic_save(self.folder / "resume.pt", state)
        atomic_json(self.folder / "run.json", self.summary())
        print(f"{self.regime}/{self.stage} {self.counter}/{self.budget}: "
              f"wins={key[0]}, fruit={key[1]:.3f}", flush=True)

    def run(self):
        if not self.history:
            self.validate_and_save()
        interval = (CONFIG.offline_validation_interval if self.stage == "offline"
                    else CONFIG.validation_interval)
        while self.counter < self.budget:
            self.advance()
            if self.counter % interval == 0 or self.counter == self.budget:
                self.validate_and_save()
        return self.summary()


def run_stage(root, regime, stage, resume=False):
    root = Path(root)
    folder = root / regime / stage
    demos = (Demonstrations.load(root / regime / "demonstrations.npz", regime)
             if stage != "scratch" else None)
    if demos is not None:
        assert_compatible(demos.metadata["manifest"])
    if not resume and folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Stage already exists; use --resume: {folder}")
    run = TrainingRun(folder, regime, stage, demos)
    if resume:
        run.restore(read_snapshot(folder / "resume.pt"))
    elif stage == "online":
        offline = root / regime / "offline"
        final = json.loads((offline / "run.json").read_text(encoding="utf-8"))
        run.inherit_offline(offline / "best.resume.pt", final)
    else:
        run.initialize_games()
    folder.mkdir(parents=True, exist_ok=True)
    try:
        return run.run()
    except Exception as error:
        atomic_json(folder / "failure.json", dict(error=repr(error), counter=run.counter,
                    completed_updates_this_attempt=run.stage_updates,
                    note="Stopped without changing protocol; last committed resume.pt remains available"))
        raise
