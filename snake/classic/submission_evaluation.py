"""Passive physical-state revisits; never changes actions or stopping rules."""
from .config import CONFIG
from .environment import ClassicSnake
from .policies import Heuristic
from .learner import choose_actions
from .evaluation import summarize


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run_game(regime, role, game_id, *, network=None, heuristic=None, limit=5000):
    require((network is None) != (heuristic is None), 'Choose one policy')
    env = ClassicSnake(regime, role, game_id, limit=limit, fruit_free_limit=None)
    policy = Heuristic(heuristic, game_id) if heuristic else None
    seen, first_repeat, period = {}, None, None
    initial = dict(body=[list(position) for position in env.body], heading=env.heading,
                   fruit=list(env.fruit) if env.fruit is not None else None)
    while not env.done:
        key = (tuple(env.body), env.heading, env.fruit)
        if key in seen and first_repeat is None:
            first_repeat, period = env.steps, env.steps-seen[key]
        seen.setdefault(key, env.steps)
        obs = env.observe()
        action = (policy.act(obs).action if policy else
                  int(choose_actions(network, obs.encode()[None], obs.legal[None])[0]))
        env.step(action)
    # Randomized baselines may escape repeated physical states. Do not label
    # those repeats as absorbing deterministic-policy cycles.
    return dict(env.record(), initial_state=initial, physical_revisit=first_repeat is not None,
                exact_cycle=(first_repeat is not None) if network is not None else None,
                first_repeat_step=first_repeat, cycle_period=period)


def evaluate_with_revisits(regime, *, role, games, network=None, heuristic=None):
    records = []
    for game_id in range(games):
        records.append(run_game(regime, role, game_id, network=network, heuristic=heuristic))
        if (game_id + 1) % 25 == 0:
            print(f"{regime}/{heuristic or 'dqn'}: {game_id + 1}/{games}", flush=True)
    summary = summarize(records)
    summary['physical_revisits'] = sum(r['physical_revisit'] for r in records)
    summary['exact_cycles'] = sum(r['exact_cycle'] for r in records) if network is not None else None
    return dict(regime=regime, role=role, cap=CONFIG.eval_limit, policy=heuristic or 'dqn',
                summary=summary, records=records)
