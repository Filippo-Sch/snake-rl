"""Watkins lambda targets and chronological sequence replay, without agent memory."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from .classic.config import CONFIG
from .classic.data import Replay
from .classic.learner import Learner, dqn_targets

LAMBDA = 0.8
HORIZON = 20
REPLAY_FORMAT = "classic-sequence-replay-v1"


class SequenceReplay(Replay):
    """Uniform transition starts; follow same-game successors up to H steps.

    Links are derived from FIFO order and game IDs, including inherited replay.
    No row is discarded, reweighted, or sampled from another game.
    """
    def __init__(self, regime, capacity=CONFIG.replay_capacity, horizon=HORIZON):
        super().__init__(regime, capacity)
        if horizon < 1:
            raise ValueError("Sequence horizon must be positive")
        self.horizon = horizon
        self.successor = np.full(capacity, -1, dtype=np.int64)
        self.predecessor = np.full(capacity, -1, dtype=np.int64)
        self.tails = {}

    def _link(self, index):
        gid = int(self.arrays["game_ids"][index])
        previous = self.tails.get(gid)
        if previous is not None:
            if (not np.array_equal(self.arrays["next_states"][previous], self.arrays["states"][index])
                    or not np.array_equal(self.arrays["next_legal"][previous], self.arrays["legal"][index])):
                raise ValueError("Noncontiguous same-game transitions")
            self.successor[previous], self.predecessor[index] = index, previous
        if self.arrays["terminated"][index] or self.arrays["truncated"][index]:
            self.tails.pop(gid, None)
        else:
            self.tails[gid] = index

    def add(self, row):
        index = self.position
        if self.size == self.capacity:
            old_gid = int(self.arrays["game_ids"][index])
            if self.tails.get(old_gid) == index:
                del self.tails[old_gid]
            previous, following = self.predecessor[index], self.successor[index]
            if previous >= 0:
                self.successor[previous] = -1
            if following >= 0:
                self.predecessor[following] = -1
        self.successor[index] = self.predecessor[index] = -1
        super().add(row)
        self._link(index)

    def sample(self, count):
        indices = self.rng.integers(self.size, size=count)
        batch = {k: a[indices].copy() for k, a in self.arrays.items()}
        batch.update(weights=np.ones(count, dtype=np.float32),
                     probabilities=np.full(count, 1 / self.size),
                     demo=np.zeros(count, dtype=bool))
        batch["eligible"][:] = False
        path = np.zeros((count, self.horizon), dtype=np.int64)
        valid = np.zeros((count, self.horizon), dtype=bool)
        path[:, 0], valid[:, 0] = indices, True
        for step in range(1, self.horizon):
            next_indices = self.successor[path[:, step - 1]]
            valid[:, step] = valid[:, step - 1] & (next_indices >= 0)
            path[:, step] = np.where(valid[:, step], next_indices, path[:, step - 1])
        for key in ("actions", "rewards", "terminated", "truncated", "next_states", "next_legal"):
            batch["seq_" + key] = self.arrays[key][path].copy()
        batch["seq_valid"] = valid
        return batch

    def state_dict(self):
        return dict(super().state_dict(), sequence_format=REPLAY_FORMAT, horizon=self.horizon)

    @classmethod
    def from_state(cls, state):
        if state.get("sequence_format", REPLAY_FORMAT) != REPLAY_FORMAT:
            raise ValueError("Unknown sequence replay format")
        ordinary = Replay.from_state(state)
        instance = cls(state["regime"], state["capacity"], state.get("horizon", HORIZON))
        instance.arrays = ordinary.arrays
        instance.size, instance.position, instance.rng = ordinary.size, ordinary.position, ordinary.rng
        order = (np.concatenate((np.arange(instance.position, instance.size),
                                 np.arange(instance.position))) if instance.size == instance.capacity
                 else np.arange(instance.size))
        for index in order:
            instance._link(int(index))
        return instance


@torch.no_grad()
def lambda_returns(rewards, terminated, truncated, valid, next_q, next_legal,
                   actions, trace_lambda=LAMBDA):
    """Backward forward-view returns, using target-network greedy trace cuts.

    G_t = r_t + V(s_next) + c_next * (G_next - V(s_next)),
    c_next = lambda iff the next stored action is the target's first argmax.
    A true terminal removes bootstrap; truncation/window end retains it but
    never continues the trace. The starting action need not be greedy.
    """
    if not 0 <= trace_lambda <= 1:
        raise ValueError("Lambda must be in [0, 1]")
    if rewards.ndim != 2 or not valid[:, 0].all():
        raise ValueError("Every sequence needs a valid starting transition")
    if (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("Sequence padding must be a suffix")
    if (valid[:, 1:] & (terminated[:, :-1] | truncated[:, :-1])).any():
        raise ValueError("Sequence crosses an episode boundary")
    active = valid & ~terminated
    if not torch.isfinite(next_q[active]).all() or not next_legal[active].any(dim=-1).all():
        raise FloatingPointError("Invalid nonterminal bootstrap")
    masked = next_q.masked_fill(~next_legal, -torch.inf)
    # Avoid NaN/inf from unused terminal predictions and padding.
    values = torch.where(active, masked.max(dim=-1).values, 0.)
    greedy = masked.argmax(dim=-1)
    base = rewards + values
    continue_trace = (valid[:, 1:] & ~terminated[:, :-1] & ~truncated[:, :-1]
                      & (actions[:, 1:] == greedy[:, :-1]))
    result = base[:, -1]
    weighted_depth = torch.ones_like(result)
    span = torch.ones_like(result)
    for step in range(rewards.shape[1] - 2, -1, -1):
        propagate = continue_trace[:, step]
        coefficient = trace_lambda * propagate.float()
        result = base[:, step] + coefficient * (result - values[:, step])
        weighted_depth = 1 + coefficient * weighted_depth
        span = 1 + propagate.float() * span
    # Exact one-step identity, including floating-point operation order.
    if trace_lambda == 0:
        result = base[:, 0]
    available = valid[:, 1:] & ~terminated[:, :-1] & ~truncated[:, :-1]
    available_count = int(available.sum())
    diagnostics = dict(sequence_mean_length=float(valid.sum(1).float().mean()),
                       trace_mean_span=float(span.mean()),
                       trace_mean_weighted_depth=float(weighted_depth.mean()),
                       trace_cut_fraction=(float((available & ~continue_trace).sum()) /
                                           available_count if available_count else 0.),
                       trace_first_cut_fraction=(float((available[:, 0] & ~continue_trace[:, 0]).float().mean())
                                                 if rewards.shape[1] > 1 else 0.))
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite lambda targets")
    return result, diagnostics


@torch.no_grad()
def watkins_targets(target, batch, trace_lambda=LAMBDA):
    if trace_lambda == 0:
        result = dqn_targets(target, torch.as_tensor(batch["rewards"], dtype=torch.float32),
                             torch.as_tensor(batch["next_states"], dtype=torch.float32),
                             torch.as_tensor(batch["terminated"], dtype=torch.bool),
                             torch.as_tensor(batch["next_legal"], dtype=torch.bool))
        return result, dict(sequence_mean_length=1., trace_mean_span=1.,
                            trace_mean_weighted_depth=1., trace_cut_fraction=0.,
                            trace_first_cut_fraction=0.)
    rewards = torch.as_tensor(batch["seq_rewards"], dtype=torch.float32)
    valid = torch.as_tensor(batch["seq_valid"], dtype=torch.bool)
    terminated = torch.as_tensor(batch["seq_terminated"], dtype=torch.bool)
    active = valid & ~terminated
    values = torch.zeros((*valid.shape, 4), dtype=torch.float32)
    # Evaluate valid nonterminal successors only; frozen target does both
    # selection and evaluation. There is deliberately no Double DQN here.
    if active.any():
        successors = torch.as_tensor(batch["seq_next_states"], dtype=torch.float32)
        values[active] = target(successors[active])
    return lambda_returns(rewards, terminated,
                          torch.as_tensor(batch["seq_truncated"], dtype=torch.bool),
                          valid, values,
                          torch.as_tensor(batch["seq_next_legal"], dtype=torch.bool),
                          torch.as_tensor(batch["seq_actions"], dtype=torch.long),
                          trace_lambda)


def watkins_loss(network, target, batch, trace_lambda=LAMBDA):
    q = network(torch.as_tensor(batch["states"], dtype=torch.float32))
    actions = torch.as_tensor(batch["actions"], dtype=torch.long)
    legal = torch.as_tensor(batch["legal"], dtype=torch.bool)
    if not legal.gather(1, actions[:, None]).all():
        raise ValueError("Illegal replay action")
    chosen = q.gather(1, actions[:, None]).squeeze(1)
    targets, diagnostics = watkins_targets(target, batch, trace_lambda)
    loss = (torch.as_tensor(batch["weights"], dtype=torch.float32) *
            nn.functional.smooth_l1_loss(chosen, targets, reduction="none")).mean()
    if not torch.isfinite(q).all() or not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite Watkins loss")
    diagnostics.update(loss=float(loss.detach()), td=float(loss.detach()), imitation=0.,
                       eligible_fraction=0., max_abs_q=float(q.detach().abs().max()),
                       mean_target=float(targets.mean()))
    return loss, diagnostics


class WatkinsLearner(Learner):
    def update(self, batch):
        loss, diagnostics = watkins_loss(self.network, self.target, batch)
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


@torch.no_grad()
def value_audit(network, replay):
    """Full-state upper return bound; no learned-action override or Q clipping."""
    a = replay.arrays
    count = replay.size
    chosen_values, violations = [], []
    for start in range(0, count, 2048):
        stop = min(start + 2048, count)
        states = a["states"][start:stop]
        q = network(torch.as_tensor(states, dtype=torch.float32))
        chosen = q.gather(1, torch.as_tensor(a["actions"][start:stop])[:, None]).squeeze(1).numpy()
        length = states[:, :-4].reshape(-1, 7, 7, 4)[..., 2:].sum(axis=(1, 2, 3))
        bound = 100 + (25 - length) * .999
        chosen_values.append(chosen)
        violations.append(chosen > bound + 1e-5)
    chosen = np.concatenate(chosen_values)
    deaths = a["terminated"][:count] & (a["rewards"][:count] < 0)
    return dict(transitions=count, mean_chosen_q=float(chosen.mean()),
                fraction_above_return_bound=float(np.concatenate(violations).mean()),
                death_mean_q=float(chosen[deaths].mean()) if deaths.any() else None,
                winning_terminals=int((a["terminated"][:count] & (a["rewards"][:count] > 100)).sum()))

