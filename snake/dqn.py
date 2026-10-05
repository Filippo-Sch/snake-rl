"""Standard DQN components for undiscounted, finite-horizon Snake."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .evaluation import REWARD_PROFILES, EvaluationConfig, PolicySpec


FORMAT = "snake-dqn-v2"
TIME_INPUT = "remaining_actions / horizon"


def input_side(config, regime):
    return config.board_size if regime == "full" else max(5, 2 * config.mask_size + 1)


def encode(observation, step, horizon, canvas_side=None):
    """Append the public clock; step == horizon is a terminal successor."""
    if not 0 <= step <= horizon or horizon < 1:
        raise ValueError("Clock must satisfy 0 <= step <= horizon")
    observation = np.asarray(observation, dtype=np.float32)
    side = observation.shape[1]
    if canvas_side is not None and canvas_side != side:
        if canvas_side < side or (canvas_side - side) % 2:
            raise ValueError("Input canvas must center the observation without cropping")
        border = (canvas_side - side) // 2
        observation = np.pad(observation, ((0, 0), (border, border), (border, border), (0, 0)))
    flat = observation.reshape(len(observation), -1)
    clock = np.full((len(flat), 1), (horizon - step) / horizon, dtype=np.float32)
    return np.concatenate((flat, clock), axis=1)


class QNetwork(nn.Sequential):
    def __init__(self, input_size):
        super().__init__(nn.Linear(input_size, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 4))


class ReplayBuffer:
    """Uniform replay with owned storage, including both clocks and terminal flags."""

    def __init__(self, capacity, input_size, seed=0):
        self.states = np.empty((capacity, input_size), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminals = np.empty(capacity, dtype=bool)
        self.capacity, self.position, self.size = capacity, 0, 0
        self.rng = np.random.default_rng(seed)

    @property
    def nbytes(self):
        return sum(value.nbytes for value in (self.states, self.next_states,
                   self.actions, self.rewards, self.terminals))

    def add(self, states, actions, rewards, next_states, terminal):
        n = len(states)
        if n > self.capacity:
            raise ValueError("A collected batch cannot exceed replay capacity")
        indices = (self.position + np.arange(n)) % self.capacity
        self.states[indices] = states
        self.actions[indices] = np.asarray(actions).reshape(n)
        self.rewards[indices] = np.asarray(rewards).reshape(n)
        self.next_states[indices] = next_states
        self.terminals[indices] = terminal
        self.position = (self.position + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        indices = self.rng.choice(self.size, batch_size, replace=False)
        return tuple(torch.from_numpy(array[indices]) for array in (
            self.states, self.actions, self.rewards, self.next_states, self.terminals))


def epsilon(transitions, budget, warmup=4096, *, decay_end=None):
    """Linear decay; an explicit endpoint is measured in collected transitions."""
    if transitions < warmup:
        return 1.0
    duration = 0.30 * (budget - warmup) if decay_end is None else decay_end - warmup
    if duration <= 0:
        raise ValueError("Exploration decay must end after warmup")
    return 1.0 - 0.95 * min(1.0, (transitions - warmup) / duration)


def choose_actions(network, states, exploration, rng):
    """Independent epsilon-greedy draws; all four actions are always available."""
    with torch.no_grad():
        actions = network(torch.from_numpy(states)).argmax(dim=1).numpy()
    random_mask = rng.random(len(states)) < exploration
    actions[random_mask] = rng.integers(0, 4, size=int(random_mask.sum()))
    return actions[:, None]


@torch.no_grad()
def dqn_targets(target, rewards, next_states, terminals):
    values = rewards.clone()
    active = ~terminals
    # Do not even evaluate terminal successors: their value is exactly zero.
    if active.any():
        values[active] += target(next_states[active]).max(dim=1).values
    return values


def optimize(network, target, optimizer, batch):
    states, actions, rewards, next_states, terminals = batch
    values = network(states).gather(1, actions[:, None]).squeeze(1)
    expected = dqn_targets(target, rewards, next_states, terminals)
    loss = nn.functional.smooth_l1_loss(values, expected)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite DQN loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), 10.0, error_if_nonfinite=True)
    optimizer.step()
    return loss.item(), values.detach().abs().max().item()


def save_checkpoint(path, network, config, regime, transitions):
    side = config.board_size if regime == "full" else 2 * config.mask_size + 1
    metadata = {"format": FORMAT, "regime": regime, "board_size": config.board_size,
                "mask_size": config.mask_size, "horizon": config.steps,
                "observation_shape": [side, side, 4], "hidden_sizes": [128, 128],
                "time_input": TIME_INPUT, "gamma": 1.0, "seed": 0,
                "transitions": transitions, "inference": "argmax_first_index",
                "action_mask": False}
    metadata.update(input_canvas_side=input_side(config, regime),
                    training_reward_profile=config.reward_profile)
    from .training_state import atomic_save
    atomic_save(path, {"metadata": metadata,
                "state_dict": {k: v.detach().cpu().clone()
                               for k, v in network.state_dict().items()}})


class DQNPolicy:
    device = "cpu"

    def __init__(self, network, observation_shape, horizon, canvas_side=None):
        self.network = network.eval().requires_grad_(False)
        self.observation_shape = tuple(observation_shape)
        self.horizon, self.step = horizon, 0
        self.canvas_side = canvas_side

    def act(self, observation):
        if tuple(observation.shape[1:]) != self.observation_shape:
            raise ValueError("Observation shape does not match the DQN checkpoint")
        if self.step >= self.horizon:
            raise ValueError("DQN episode has ended; create a fresh policy")
        states = encode(observation, self.step, self.horizon, self.canvas_side)
        with torch.no_grad():
            values = self.network(torch.from_numpy(states))
            if not torch.isfinite(values).all():
                raise ValueError("Checkpoint produced non-finite Q values")
            actions = values.argmax(dim=1).numpy()[:, None]
        self.step += 1
        return actions


def load_policy(path, config, regime):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    side = config.board_size if regime == "full" else 2 * config.mask_size + 1
    if metadata.get("format") not in ("snake-dqn-v1", FORMAT):
        raise ValueError("Unsupported DQN checkpoint format")
    if metadata["format"] == FORMAT and metadata.get("training_reward_profile") not in REWARD_PROFILES:
        raise ValueError("DQN checkpoint must declare its training reward profile")
    expected = {"regime": regime, "board_size": config.board_size,
                "horizon": config.steps, "observation_shape": [side, side, 4],
                "hidden_sizes": [128, 128], "time_input": TIME_INPUT,
                "gamma": 1.0, "seed": 0, "inference": "argmax_first_index",
                "action_mask": False}
    if regime == "partial":
        expected["mask_size"] = config.mask_size
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"DQN checkpoint {key}: expected {value}, got {metadata.get(key)}")
    canvas = side if metadata["format"] == "snake-dqn-v1" else input_side(config, regime)
    if metadata["format"] == FORMAT and metadata.get("input_canvas_side") != canvas:
        raise ValueError("DQN checkpoint input canvas does not match the configuration")
    metadata.setdefault("training_reward_profile", "R0")
    network = QNetwork(canvas * canvas * 4 + 1)
    network.load_state_dict(checkpoint["state_dict"])
    if not all(torch.isfinite(p).all() for p in network.parameters()):
        raise ValueError("Checkpoint contains non-finite weights")
    return DQNPolicy(network, (side, side, 4), config.steps, canvas), metadata


def make_dqn_spec(config: EvaluationConfig, weights):
    """Bind public task settings while preserving the shared policy factory API."""
    paths = {regime: Path(path) for regime, path in weights.items()}
    metadata = {regime: load_policy(path, config, regime)[1]
                for regime, path in paths.items()}

    def factory(*, seed, regime):
        if seed != 0:
            raise ValueError("The project requires seed=0")
        return load_policy(paths[regime], config, regime)[0]

    return PolicySpec("dqn", factory, parameters=metadata, weights=paths)
