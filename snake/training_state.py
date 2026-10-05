"""Atomic, complete CPU training snapshots at episode-batch boundaries.

Only load snapshots produced by this project: full snapshots contain Python
and NumPy RNG objects and therefore require trusted local pickle loading.
Inference checkpoints remain separate and use weights_only=True.
"""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


FORMAT = 'snake-training-v1'
REPLAY_ARRAYS = ('states', 'next_states', 'actions', 'rewards', 'terminals')


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def atomic_save(path, value, rotate=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if rotate and path.exists():
        try:
            load_snapshot(path)
        except Exception:
            pass  # Keep the readable previous version if latest was corrupt.
        else:
            os.replace(path, path.with_name('resume_previous.pt'))
    os.replace(temporary, path)


def capture(network, target, optimizer, replay, action_rng):
    # All unused replay storage is excluded (np.empty is not initialized).
    return dict(network=deepcopy(network.state_dict()), target=deepcopy(target.state_dict()),
        optimizer=deepcopy(optimizer.state_dict()), replay=dict(
            capacity=replay.capacity, size=replay.size, position=replay.position,
            arrays={key: getattr(replay, key)[:replay.size].copy() for key in REPLAY_ARRAYS},
            rng=deepcopy(replay.rng.bit_generator.state)),
        rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                 torch=torch.get_rng_state().clone(), action=deepcopy(action_rng.bit_generator.state)))


def restore(state, network, target, optimizer, replay, action_rng):
    network.load_state_dict(state['network'])
    target.load_state_dict(state['target'])
    optimizer.load_state_dict(state['optimizer'])
    saved = state['replay']
    if saved['capacity'] != replay.capacity or not 0 <= saved['size'] <= replay.capacity:
        raise ValueError('Replay capacity/size mismatch')
    if not 0 <= saved['position'] < replay.capacity:
        raise ValueError('Invalid replay cursor')
    replay.size, replay.position = saved['size'], saved['position']
    for key in REPLAY_ARRAYS:
        destination = getattr(replay, key)[:replay.size]
        if destination.shape != saved['arrays'][key].shape:
            raise ValueError('Replay shape mismatch')
        destination[:] = saved['arrays'][key]
    replay.rng.bit_generator.state = saved['rng']
    # Restore RNGs LAST: model/optimizer construction must not consume restored state.
    random.setstate(state['rng']['python'])
    np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch'])
    action_rng.bit_generator.state = state['rng']['action']


def load_snapshot(path):
    value = torch.load(path, map_location='cpu', weights_only=False)
    if value.get('format') != FORMAT or value.get('boundary') != 'before_next_episode_batch':
        raise ValueError('Not a complete training snapshot at a safe boundary')
    return value


def latest_snapshot(folder):
    """Recover from a torn latest write using the previous committed snapshot."""
    errors = []
    for name in ('resume_latest.pt', 'resume_previous.pt'):
        path = Path(folder) / name
        if path.exists():
            try:
                load_snapshot(path)
                return path
            except Exception as error:
                errors.append(f'{name}: {error}')
    raise ValueError('No readable committed snapshot: ' + '; '.join(errors))
