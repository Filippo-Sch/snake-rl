"""Native factorial and LR schedules; default prints a plan, never trains."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from train import TrainingConfig, compatible_resume
from snake.evaluation import EvaluationConfig
from snake.training_state import latest_snapshot, load_snapshot
ROOT = Path(__file__).resolve().parents[1]
FIRST, SECOND, FINAL, INTERVAL = 1_024_000, 2_048_000, 4_096_000, 128_000
NAMES = ('A', 'B', 'C')
CASES = ('b7_full', 'b7_partial', 'b11_full', 'b11_partial')


def environment(case, boards=16, test=False):
    if case not in CASES:
        raise ValueError(f'Unknown case: {case}')
    return EvaluationConfig(board_size=int(case.split('_')[0][1:]), mask_size=2,
        n_boards=boards, split='test' if test else 'validation', test_stream=3 if test else 1)


def settings_for(case, variant='A'):
    if variant not in NAMES or (case.startswith('b7_') and variant != 'A'):
        raise ValueError('Only A is trained on board 7')
    budget = SECOND if case.startswith('b7_') else FINAL
    steps = () if variant == 'A' else ((FIRST, 5e-5),)
    if variant == 'C':
        steps += ((SECOND, 2.5e-5),)
    settings = TrainingConfig(budget=budget, evaluation_interval=INTERVAL,
        exploration_budget=FIRST, lr_steps=steps,
        selection_budgets=(FIRST, SECOND) if budget == SECOND else (FIRST, SECOND, FINAL))
    settings.validate(environment(case))
    return settings


def run_folder(output, case, variant):
    return output / 'training' / case / variant


def training_command(case, variant, folder, stop, resume=None, fork=False):
    settings = settings_for(case, variant)
    command = [sys.executable, str(ROOT / 'train.py'), '--stage', 'main',
        '--regimes', case.split('_')[1], '--board-size', str(environment(case).board_size),
        '--boards', '16', '--steps', '1000', '--mask-size', '2', '--reward-profile', 'R0',
        '--budget', str(settings.budget), '--evaluation-interval', str(INTERVAL),
        '--evaluation-boards', '100', '--exploration-budget', str(FIRST),
        '--learning-rate', '0.0001', '--selection-budgets', *map(str, settings.selection_budgets),
        '--run-directory', str(folder), '--stop-at', str(stop)]
    for step, rate in settings.lr_steps:
        command += ['--lr-step', f'{step}:{rate}']
    if resume is not None:
        command += ['--resume', str(resume)]
    if fork:
        command += ['--fork']
    return command


def ensure_segment(case, variant, output, stop, parent=None):
    folder = run_folder(output, case, variant)
    resume, fork = parent, parent is not None
    if folder.exists():
        resume = latest_snapshot(folder)
        saved = load_snapshot(resume)
        compatible_resume(saved, settings_for(case, variant), environment(case), case.split('_')[1], 100, False)
        if saved['transitions'] >= stop:
            # Restore artifacts/metadata as well if a crash occurred after checkpoint commit.
            stop = saved['transitions']
        fork = False
    command = training_command(case, variant, folder, stop, resume, fork)
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    with (logs / f'{case}_{variant}.log').open('a', encoding='utf-8') as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    saved = load_snapshot(folder / 'resume_latest.pt')
    if saved['transitions'] < stop:
        raise ValueError('Training did not reach its declared stop')
    return folder


def train_case(case, output):
    print(f'Case {case}: starting/resuming declared segments', flush=True)
    if case.startswith('b7_'):
        ensure_segment(case, 'A', output, SECOND)
        return
    a = ensure_segment(case, 'A', output, FIRST)
    ensure_segment(case, 'A', output, FINAL)
    b = ensure_segment(case, 'B', output, SECOND, a / f'resume_{FIRST}.pt')
    ensure_segment(case, 'B', output, FINAL)
    ensure_segment(case, 'C', output, FINAL, b / f'resume_{SECOND}.pt')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('plan', 'train'), default='plan')
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--output', type=Path, default=ROOT / 'results/native')
    args = parser.parse_args(argv)
    if len(set(args.cases)) != len(args.cases):
        parser.error('Cases must be unique')
    if args.phase == 'plan':
        print(json.dumps({case: {v: asdict(settings_for(case, v)) for v in
            (('A',) if case.startswith('b7_') else NAMES)} for case in args.cases}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        train_case(case, args.output)


if __name__ == '__main__':
    main()
