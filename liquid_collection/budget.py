"""Episode quota and timeout arithmetic shared by planning and execution."""
import math


def budget_reached(completed, job):
    target = job.get('episodes_per_env')
    if target is not None:
        return bool((completed >= target).all())
    return int(completed.sum()) >= job['episodes']


def episode_steps(seconds, policy_dt):
    if not math.isfinite(seconds) or not math.isfinite(policy_dt) or seconds <= 0 or policy_dt <= 0:
        raise ValueError('Episode duration and policy dt must be positive and finite')
    steps = math.floor(seconds/policy_dt+1e-9)
    if steps < 1:
        raise ValueError(f'Episode time limit must allow at least one policy step ({policy_dt:g} s)')
    return steps
