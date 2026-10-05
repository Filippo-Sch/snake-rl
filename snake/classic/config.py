"""Fixed phase-2 protocol and independent seed-0 random streams."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import random

import numpy as np
import torch

PROTOCOL = "classic-snake-1.1"
ROOT = Path(__file__).resolve().parents[2]
ROLES = dict(training=0, demos=1, validation=2, test=3, audit=4,
             exploration=5, replay=6, mixing=7, sampling=8, teacher=9)


@dataclass(frozen=True)
class Config:
    board_size: int = 7
    partial_size: int = 5
    initial_length: int = 3
    fruit_reward: float = 1.0
    win_reward: float = 100.0
    death_reward: float = -1.0
    step_reward: float = -0.001
    gamma: float = 1.0
    train_limit: int = 1000
    fruit_free_limit: int = 250
    eval_limit: int = 5000
    slots: int = 16
    replay_capacity: int = 50000
    batch_size: int = 64
    warmup: int = 4096
    target_interval: int = 500
    online_budget: int = 2048000
    demo_budget: int = 256000
    offline_updates: int = 20000
    validation_interval: int = 128000
    offline_validation_interval: int = 2000
    validation_games: int = 100
    audit_games: int = 500
    test_games: int = 500
    learning_rate: float = 1e-4
    margin: float = 0.8
    gradient_norm: float = 10.0


CONFIG = Config()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def manifest():
    paths = sorted(ROOT.glob("*.py")) + [ROOT / "requirements.txt"]
    paths += sorted((ROOT / "snake").rglob("*.py"))
    paths += sorted((ROOT / "experiments").rglob("*.py"))
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"):
              hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return dict(protocol=PROTOCOL, config=asdict(CONFIG),
                config_hash=canonical_hash(asdict(CONFIG)), source_hashes=hashes,
                source_hash=canonical_hash(hashes), seed=0,
                software=dict(python=platform.python_version(), numpy=np.__version__,
                              torch=str(torch.__version__)), device="cpu", dtype="float32")


def rng(role, game_id=0):
    return np.random.Generator(np.random.PCG64(
        np.random.SeedSequence(0, spawn_key=(2, ROLES[role], int(game_id)))))


def configure_torch():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def seed_training():
    configure_torch()
    random.seed(0)
    torch.manual_seed(0)


def input_size(regime):
    if regime not in ("full", "partial"):
        raise ValueError("Regime must be full or partial")
    return (7 if regime == "full" else 5) ** 2 * 4 + 4


def epsilon(t, assisted=False):
    if assisted:
        return 0.10 - 0.09 * min(t / 256000, 1.0)
    return 1.0 - 0.99 * min(max(t - CONFIG.warmup, 0) / (512000 - CONFIG.warmup), 1.0)


def demo_probability(t):
    return 0.25 * (1.0 - min(max(t - 128000, 0) / 384000, 1.0))
