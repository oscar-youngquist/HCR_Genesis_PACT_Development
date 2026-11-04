"""Low-frequency worker snapshots and coordinator episode progress display."""
import json
import subprocess
import sys
import time


def episode_progress(counts, job, phase='collecting'):
    counts = [int(n) for n in counts]
    quota = job.get('episodes_per_env')
    target = quota * job['num_envs'] if quota is not None else job['episodes']
    done = sum(min(n, quota) for n in counts) if quota is not None else min(sum(counts), target)
    return dict(phase=phase, completed=done, target=target, episodes_per_env=counts,
                environments_complete=sum(n >= quota for n in counts) if quota is not None else None)


def format_progress(label, progress):
    done, target = progress['completed'], progress['target']
    fraction = min(done / target, 1.)
    filled = int(24 * fraction)
    line = f'{label} [{"#" * filled}{"-" * (24-filled)}] {done}/{target} episodes ({fraction:.0%})'
    if progress['environments_complete'] is not None:
        line += f" | envs {progress['environments_complete']}/{len(progress['episodes_per_env'])}"
    return line + f" | {progress['phase']}"


def wait_with_progress(proc, attempt, job, label, timeout, stream=None):
    """Keep the original worker deadline; leave termination to the coordinator."""
    stream = sys.stdout if stream is None else stream
    tty = stream.isatty()
    deadline = time.monotonic() + timeout
    previous = None

    def display():
        nonlocal previous
        try:
            progress = json.loads((attempt/'progress.json').read_text())
        except FileNotFoundError:
            progress = episode_progress([0] * job['num_envs'], job, 'initializing')
        line = format_progress(label, progress)
        if line != previous:
            print(('\r\033[2K' if tty else '') + line, end='' if tty else '\n', file=stream, flush=True)
            previous = line

    try:
        while True:
            display()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(proc.args, timeout)
            try:
                code = proc.wait(timeout=min(1., remaining))
            except subprocess.TimeoutExpired:
                continue
            display()  # Include the final snapshot even for very short runs.
            return code
    finally:
        if tty:
            print(file=stream, flush=True)
