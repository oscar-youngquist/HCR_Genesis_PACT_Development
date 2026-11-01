"""Wait for an existing local Linux PACT trainer, then dispatch frozen evaluation.

Standard library only: waiting never imports Torch/Isaac Lab or touches training.
The existing three-seed script remains authoritative for conditions and episodes.
"""
import argparse
import ast
import math
import os
from pathlib import Path
import re
import shlex
import sys
import time
import zipfile

REPO = Path(__file__).resolve().parents[2]


def option(argv, name):
    value = None
    for i, word in enumerate(argv):
        if word.startswith(name + "="):
            value = word.split("=", 1)[1]
        elif word == name and i + 1 < len(argv):
            value = argv[i + 1]
    return value


def process_state(pid, proc=Path('/proc')):
    try:
        # comm may contain spaces/parentheses. Field 22 is ticks since boot.
        fields = (proc / str(pid) / 'stat').read_text().rpartition(')')[2].split()
        return int(fields[19]), fields[0]
    except (FileNotFoundError, ProcessLookupError):
        return None


def inspect_process(pid, repo=REPO, proc=Path('/proc')):
    root = proc / str(pid)
    state = process_state(pid, proc)
    if state is None or state[1] in ('Z', 'X'):
        raise ValueError(f'PID {pid} is not running; start this wrapper while training is active.')
    argv = [v.decode() for v in (root / 'cmdline').read_bytes().split(b'\0') if v]
    cwd = (root / 'cwd').resolve(strict=True)
    if option(argv, '--task') != 'b1z1_pact' or not any(
        Path(word).name == 'train.py' and (cwd / word).resolve() == repo / 'legged_gym/scripts/train.py'
        for word in argv
    ):
        raise ValueError(f'PID {pid} is not this checkout\'s b1z1_pact train.py process.')
    boot = next(int(line.split()[1]) for line in (proc / 'stat').read_text().splitlines() if line.startswith('btime '))
    environment = dict(item.decode().split('=', 1) for item in (root / 'environ').read_bytes().split(b'\0') if b'=' in item)
    # Copy only interpreter/device settings, not the whole training environment.
    keys = ('PATH', 'CONDA_PREFIX', 'CONDA_DEFAULT_ENV', 'VIRTUAL_ENV', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES')
    settings = {k: environment[k] for k in keys if k in environment}
    executable = (root / 'exe').resolve(strict=True)
    current = process_state(pid, proc)
    if current is None or current[0] != state[0]:
        raise ValueError('Training exited during inspection; no evaluation was started.')
    return dict(pid=pid, ticks=state[0], started=boot + state[0] / os.sysconf('SC_CLK_TCK'),
                argv=argv, environment=settings, executable=executable)


def find_process(pid=None, repo=REPO, proc=Path('/proc')):
    if pid is not None:
        return inspect_process(pid, repo, proc)
    matches = []
    for root in proc.glob('[0-9]*'):
        try:
            if root.stat().st_uid == os.getuid():
                matches.append(inspect_process(int(root.name), repo, proc))
        except (OSError, ValueError):
            continue
    if len(matches) != 1:
        raise ValueError(f'Found {len(matches)} active PACT trainers: {[p["pid"] for p in matches]}. Specify --pid; run this on the training host.')
    return matches[0]


def find_run(process, supplied=None, repo=REPO):
    # TensorBoard embeds the writer PID and timestamp. Pin this run now, never
    # choose the globally newest folder after waiting (another job may start).
    matches = {p.parent.resolve() for p in (repo / 'logs/b1z1').glob(f'*/*/events.out.tfevents.*.{process["pid"]}.*')
               if int(p.name.split('.')[3]) >= process['started'] - 1}
    if supplied is not None:
        run = supplied.expanduser().resolve()
        if not run.is_dir() or (matches and run not in matches):
            raise ValueError('--run is missing or does not belong to the selected process.')
        return run
    if len(matches) != 1:
        raise ValueError(f'Cannot uniquely identify the run from its TensorBoard filename: {sorted(map(str, matches))}. Supply --run.')
    return matches.pop()


def expected_checkpoint(run, argv, explicit=None):
    if explicit is not None:
        return explicit
    # Parse the archived config as data; importing it would import training code.
    tree = ast.parse((run / 'b1z1_pact_config.py').read_text())
    policy = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'B1Z1PACTCfgPPO')
    runner = next(n for n in policy.body if isinstance(n, ast.ClassDef) and n.name == 'runner')
    values = {}
    for node in runner.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('max_iterations', 'resume'):
                    values[target.id] = ast.literal_eval(node.value)
    if '--resume' in argv or values.get('resume'):
        raise ValueError('Resumed training needs --expected-checkpoint FINAL_ITERATION (including its starting iteration).')
    return int(option(argv, '--max_iterations') or values['max_iterations'])


def latest_checkpoint(run):
    choices = [(int(m[1]), p) for p in run.glob('model_*.pt') if (m := re.fullmatch(r'model_(\d+)\.pt', p.name))]
    if not choices:
        raise ValueError(f'No model_<iteration>.pt checkpoints in {run}')
    return max(choices)


def wait_for_exit(process, poll_seconds, proc=Path('/proc')):
    while True:
        state = process_state(process['pid'], proc)
        # PID reuse and unreaped zombies must not leave the watcher waiting forever.
        if state is None or state[0] != process['ticks'] or state[1] in ('Z', 'X'):
            return
        time.sleep(poll_seconds)


def validate_checkpoint(path):
    # torch.save uses ZIP here. Check all records once on CPU before 18 evaluations;
    # a partial final save is an error, not permission to use an older checkpoint.
    try:
        with zipfile.ZipFile(path) as archive:
            if not any(name.endswith('/data.pkl') for name in archive.namelist()) or archive.testzip() is not None:
                raise ValueError(f'Incomplete checkpoint: {path}')
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f'Unreadable/incomplete checkpoint: {path}') from exc


def evaluation_command(process, run, number, output, gpu, repo=REPO):
    root = (output or repo / 'evaluation/pilot_v02' / run.name / f'model_{number}').expanduser().resolve()
    env = dict(os.environ)
    for key in ('CONDA_PREFIX', 'CONDA_DEFAULT_ENV', 'VIRTUAL_ENV', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES'):
        env.pop(key, None)
    env.update(process['environment'])
    prefix = env.get('VIRTUAL_ENV') or env.get('CONDA_PREFIX')
    bin_dir = Path(prefix) / 'bin' if prefix else process['executable'].parent
    env['PATH'] = str(bin_dir) + os.pathsep + env.get('PATH', os.defpath)
    env['EVAL_GPU'] = gpu or os.environ.get('EVAL_GPU') or option(process['argv'], '--gpu') or 'cuda:0'
    env['EXP2_OUTPUT_BASE'] = str(root / 'exp2')
    command = ['/bin/bash', str(repo / 'legged_gym/scripts/eval_b1z1_pact_three_seeds.sh'), str(run), str(number), str(root / 'exp1')]
    return command, env


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--pid', type=int, help='Existing trainer PID; auto-detect when exactly one PACT trainer is running.')
    cli.add_argument('--run', type=Path, help='Explicit run directory, otherwise inferred from this PID\'s event file.')
    cli.add_argument('--gpu', help='Evaluation CUDA device; defaults to EVAL_GPU or the training --gpu.')
    cli.add_argument('--output-dir', type=Path, help='Parent for exp1/exp2 outputs; default is a fresh run/checkpoint-specific directory.')
    cli.add_argument('--expected-checkpoint', type=int, help='Final iteration; auto-read for fresh runs, required for resumed runs.')
    cli.add_argument('--allow-early-exit', action='store_true', help='Explicitly permit evaluating the last checkpoint after an early stop/crash.')
    cli.add_argument('--poll-seconds', type=float, default=30., help='Process polling interval (default 30).')
    cli.add_argument('--settle-seconds', type=float, default=5., help='Grace period after exit before checkpoint selection (default 5).')
    cli.add_argument('--dry-run', action='store_true', help='Print detected process/run/settings; do not wait or evaluate.')
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0 or not math.isfinite(args.settle_seconds) or args.settle_seconds < 0:
        raise ValueError('Polling must be positive and settling nonnegative, both finite seconds.')
    process = find_process(args.pid)
    run = find_run(process, args.run)
    try:
        expected = expected_checkpoint(run, process['argv'], args.expected_checkpoint)
    except (OSError, ValueError, KeyError, StopIteration, SyntaxError) as exc:
        raise ValueError(f'Cannot infer completion: {exc}. Set --expected-checkpoint explicitly.') from exc
    if expected <= 0:
        raise ValueError('--expected-checkpoint must be positive.')
    command, env = evaluation_command(process, run, expected, args.output_dir, args.gpu)
    print(f'Training PID: {process["pid"]}\nRun: {run}\nExpected final checkpoint: model_{expected}.pt\n'
          f'Python environment: {process["executable"]}\nEvaluation GPU: {env["EVAL_GPU"]}\n'
          f'Planned evaluation: {shlex.join(command)}\nExp2 output: {env["EXP2_OUTPUT_BASE"]}', flush=True)
    if args.dry_run:
        print('Dry run only; no waiting or evaluation. Checkpoint selection happens after exit.', flush=True)
        return 0
    print('Waiting for the selected training process to exit. Ctrl-C cancels this watcher, not training.', flush=True)
    wait_for_exit(process, args.poll_seconds)
    time.sleep(args.settle_seconds)
    number, checkpoint = latest_checkpoint(run)
    # A non-child process has no waitable exit status. The expected final save is
    # the completion guard; process disappearance alone could also mean a crash.
    if number < expected and not args.allow_early_exit:
        raise ValueError(f'Training stopped with model_{number}.pt, before expected model_{expected}.pt; evaluation NOT started. Use --allow-early-exit deliberately to permit this.')
    if number < expected:
        print('WARNING: evaluating an early-stop checkpoint by explicit request.', flush=True)
    validate_checkpoint(checkpoint)
    command, env = evaluation_command(process, run, number, args.output_dir, args.gpu)
    for destination in (Path(command[-1]), Path(env['EXP2_OUTPUT_BASE'])):
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(f'Evaluation output is not empty: {destination}; choose a fresh --output-dir.')
    print(f'Trainer exited. Launching {shlex.join(command)}', flush=True)
    os.chdir(REPO)
    os.execve(command[0], command, env)  # Preserve suite exit status; launch once.


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print('\nWatcher cancelled; the training process was not signalled.', file=sys.stderr)
        raise SystemExit(130)
