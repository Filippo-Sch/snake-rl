"""Standard DQN with Huber TD loss and eligible BFS large-margin labels."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from ..dqn import QNetwork
from ..training_state import atomic_save
from .config import CONFIG, PROTOCOL, input_size

POLICY_FORMAT = "classic-dqn-policy-v1"


def choose_actions(network, states, legal, exploration=0.0, rng=None):
    with torch.no_grad():
        values = network(torch.as_tensor(states, dtype=torch.float32))
        if not torch.isfinite(values).all():
            raise FloatingPointError("Non-finite classic Q values")
        masks = torch.as_tensor(legal, dtype=torch.bool)
        if not masks.any(dim=1).all():
            raise ValueError("No legal action")
        actions = values.masked_fill(~masks, -torch.inf).argmax(1).numpy()
    if exploration:
        if rng is None:
            raise ValueError("Exploration needs its own RNG")
        for i in np.flatnonzero(rng.random(len(actions)) < exploration):
            actions[i] = rng.choice(np.flatnonzero(legal[i]))
    return actions


@torch.no_grad()
def dqn_targets(target, rewards, next_states, terminated, next_legal):
    result = rewards.clone()
    active = ~terminated
    if active.any():
        values = target(next_states[active])
        masks = next_legal[active]
        if not masks.any(dim=1).all() or not torch.isfinite(values).all():
            raise FloatingPointError("Invalid classic target values/masks")
        result[active] += values.masked_fill(~masks, -torch.inf).max(1).values
    return result


def loss_terms(network, target, batch):
    states = torch.as_tensor(batch["states"], dtype=torch.float32)
    next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32)
    actions = torch.as_tensor(batch["actions"], dtype=torch.long)
    rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
    terminated = torch.as_tensor(batch["terminated"], dtype=torch.bool)
    legal = torch.as_tensor(batch["legal"], dtype=torch.bool)
    next_legal = torch.as_tensor(batch["next_legal"], dtype=torch.bool)
    weights = torch.as_tensor(batch["weights"], dtype=torch.float32).detach()
    eligible = torch.as_tensor(batch["eligible"], dtype=torch.bool)
    q = network(states)
    chosen = q.gather(1, actions[:, None]).squeeze(1)
    if not legal.gather(1, actions[:, None]).all():
        raise ValueError("Illegal action stored in a classic batch")
    targets = dqn_targets(target, rewards, next_states, terminated, next_legal)
    td = nn.functional.smooth_l1_loss(chosen, targets, reduction="none")
    margins = torch.full_like(q, CONFIG.margin)
    margins.scatter_(1, actions[:, None], 0.0)
    imitation = (q + margins).masked_fill(~legal, -torch.inf).max(1).values - chosen
    imitation = torch.where(eligible, imitation, torch.zeros_like(imitation))
    loss = (weights * (td + imitation)).mean()
    if not torch.isfinite(q).all() or not torch.isfinite(loss):
        raise FloatingPointError("Non-finite classic loss")
    diagnostics = dict(loss=float(loss.detach()), td=float((weights * td).mean().detach()),
                       imitation=float((weights * imitation).mean().detach()),
                       eligible_fraction=float(eligible.float().mean()),
                       max_abs_q=float(q.detach().abs().max()))
    return loss, diagnostics


class Learner:
    def __init__(self, regime):
        self.regime = regime
        self.network = QNetwork(input_size(regime)).float()
        self.target = deepcopy(self.network).eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=CONFIG.learning_rate,
                                          betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                                          amsgrad=False, foreach=False, fused=False)
        self.updates = 0

    def update(self, batch):
        loss, diagnostics = loss_terms(self.network, self.target, batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.network.parameters(), CONFIG.gradient_norm,
                                       error_if_nonfinite=True)
        self.optimizer.step()
        if not all(torch.isfinite(p).all() for p in self.network.parameters()):
            raise FloatingPointError("Non-finite parameters after Adam update")
        self.updates += 1
        if self.updates % CONFIG.target_interval == 0:
            self.target.load_state_dict(self.network.state_dict())
        return dict(diagnostics, gradient_norm=float(norm), lineage_updates=self.updates)

    def state_dict(self):
        return dict(regime=self.regime, network=deepcopy(self.network.state_dict()),
                    target=deepcopy(self.target.state_dict()),
                    optimizer=deepcopy(self.optimizer.state_dict()), updates=self.updates)

    def load_state_dict(self, value):
        if value["regime"] != self.regime:
            raise ValueError("Learner regime mismatch")
        self.network.load_state_dict(value["network"])
        self.target.load_state_dict(value["target"])
        self.optimizer.load_state_dict(value["optimizer"])
        self.updates = value["updates"]


def save_policy(path, learner, metadata):
    atomic_save(path, dict(format=POLICY_FORMAT, protocol=PROTOCOL, regime=learner.regime,
                           metadata=metadata, state_dict=deepcopy(learner.network.state_dict())))


def load_policy(path, regime):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (value.get("format") != POLICY_FORMAT or value.get("protocol") != PROTOCOL
            or value.get("regime") != regime):
        raise ValueError("Checkpoint is not a matching classic DQN policy")
    # Loading a policy does not advance any caller's Torch initialization stream.
    with torch.random.fork_rng(devices=[]):
        network = QNetwork(input_size(regime))
        network.load_state_dict(value["state_dict"])
    if not all(torch.isfinite(p).all() for p in network.parameters()):
        raise ValueError("Non-finite classic policy weights")
    return network.eval().requires_grad_(False), value
