"""Deterministic round plans with immutable input identities and fresh attempts."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from .payload import sample_payload, supported_bounds
from .storage import atomic_json, digest, verified_complete

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path.home()/'.conda/envs/genesis_lr/bin/python'


def terminate_worker(proc):
    import signal
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def make_plan(manifest, *, seed=1, rounds=1, num_envs=100, episodes=100,
              max_steps=20000, max_seconds=1800., feature_dt=.02, safety_margin=.001,
              chunk_size=128, max_buffer_mb=64, smoke=False, snapshots=0, benchmark=False,
              property_expansion=.10, geometry_expansion=.02, particle_size=.010,
              substeps=8, contact_geometry='boxes', episode_seconds=None, smoke_step_height=-.16,
              save_mode='200hz', episodes_per_env=None):
    if min(rounds, num_envs, episodes, max_steps, chunk_size, max_buffer_mb) < 1:
        raise ValueError('Counts and limits must be positive')
    if benchmark and max_steps < 2:
        raise ValueError('Benchmark needs at least two policy steps to separate first-step cost')
    if not all(math.isfinite(x) and x > 0 for x in (max_seconds, feature_dt)) or safety_margin < 0:
        raise ValueError('Invalid timing or clearance limit')
    if snapshots < 0:
        raise ValueError('Snapshot cadence cannot be negative')
    if episodes_per_env is not None and episodes_per_env < 1:
        raise ValueError('Episodes per environment must be positive')
    if save_mode not in ('200hz', 'native'):
        raise ValueError('save_mode must be 200hz or native')
    if not math.isfinite(particle_size) or particle_size <= 0 or substeps < 1:
        raise ValueError('Particle size/substeps must be positive')
    if contact_geometry not in ('mesh', 'boxes'):
        raise ValueError('Unknown contact geometry')
    if episode_seconds is not None and (not math.isfinite(episode_seconds) or episode_seconds <= 0):
        raise ValueError('Episode duration must be positive')
    if not -.184 <= smoke_step_height <= -.04:
        raise ValueError('Smoke descending steps must lie within existing generator support [-.184,-.04] m')
    manifest = Path(manifest).resolve()
    entries = json.loads(manifest.read_text())['policies']
    if not entries or len({p['id'] for p in entries}) != len(entries):
        raise ValueError('Manifest requires nonempty, uniquely named policies')
    policies = []
    for entry in entries:
        p = dict(entry)
        if p.get('adapter', 'dact') != 'dact' or p['robot'] not in ('go1', 'go2'):
            raise ValueError('Only DACT Go1/Go2 adapters are implemented')
        p['adapter'] = 'dact'
        for name in ('checkpoint', 'config'):
            path = (manifest.parent/p[name]).resolve()
            if not path.is_file():
                raise ValueError(f'Missing {name}: {path}')
            p[name], p[name+'_sha256'] = str(path), digest(path)
        policies.append(p)
    plans = []
    for i in range(rounds):
        round_seed = random.Random(f'{seed}:{i}').randrange(2**31)
        rng = random.Random(round_seed)
        policy = rng.choice(policies)
        plans.append(dict(round=i, seed=round_seed, policy=policy,
                          payload=sample_payload(rng, max(.012, particle_size) if benchmark else particle_size, safety_margin,
                              property_expansion=property_expansion, geometry_expansion=geometry_expansion)))
    plan = dict(schema_version=2, save_mode=save_mode, manifest_sha256=digest(manifest), seed=seed,
                num_envs=num_envs, episodes=episodes,
                max_steps=max_steps, max_seconds=max_seconds,
                feature_dt=feature_dt, safety_margin=safety_margin, particle_size=particle_size,
                substeps=substeps, contact_geometry=contact_geometry,
                episode_seconds=episode_seconds,
                smoke_step_height=smoke_step_height,
                property_expansion=property_expansion, geometry_expansion=geometry_expansion,
                chunk_size=chunk_size, max_buffer_mb=max_buffer_mb, snapshots=snapshots,
                smoke=smoke, benchmark=benchmark,
                payload_bounds=supported_bounds(property_expansion, geometry_expansion), rounds=plans)
    if episodes_per_env is not None:
        plan['episodes_per_env'] = episodes_per_env
    return plan


def run_plan(plan, output, resume=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Advisory lock is held by the coordinator for its entire lifetime.
    with (output/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = output/'plan.json'
        if saved.exists():
            if not resume:
                raise ValueError('Output already has a plan; use --resume')
            if json.loads(saved.read_text()) != plan:
                raise ValueError('Resume plan/input identity differs; choose a new output directory')
        else:
            atomic_json(saved, plan)
        failed = False
        for round_plan in plan['rounds']:
            modes = ['reference', 'baseline', 'collection'] if plan['benchmark'] else ['collection']
            for mode in modes:
                directory = output/f"round-{round_plan['round']:05d}"/mode
                directory.mkdir(parents=True, exist_ok=True)
                attempts = sorted(directory.glob('attempt-*'))
                if any(verified_complete(p) for p in attempts):
                    continue
                attempt = directory/f'attempt-{len(attempts):04d}'
                attempt.mkdir()
                job = dict(plan, selected=round_plan, baseline=mode != 'collection', benchmark_mode=mode)
                if mode == 'reference':
                    # Existing player's geometry and liquid integration settings,
                    # with the same payload/terrain/commands and domain/reset fixes.
                    job.update(contact_geometry='mesh', particle_size=.012, substeps=2)
                atomic_json(attempt/'job.json', job)
                atomic_json(attempt/'status.json', dict(status='running', started=time.time()))
                env = dict(os.environ, SIMULATOR='genesis_pact_water', PYTHONUNBUFFERED='1',
                           PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH', ''))
                env.setdefault('NUMBA_CACHE_DIR', '/tmp/liquid-numba')
                env.setdefault('MPLCONFIGDIR', '/tmp/liquid-mpl')
                command = [str(DEFAULT_PYTHON), '-m', 'liquid_collection.worker', str(attempt)]
                with (attempt/'worker.log').open('w') as log:
                    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = proc.wait(timeout=plan['max_seconds'])
                        status = 'complete' if code == 0 and verified_complete(attempt) else 'incomplete' if code == 2 else 'failed'
                    except subprocess.TimeoutExpired:
                        terminate_worker(proc)
                        status, code = 'incomplete', None
                    except BaseException as exc:
                        terminate_worker(proc)
                        atomic_json(attempt/'status.json', dict(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                                                               error=repr(exc)))
                        raise
                atomic_json(attempt/'status.json', dict(status=status, returncode=code, ended=time.time()))
                print(f'{attempt}: {status}', flush=True)
                failed |= status != 'complete'
            if plan['benchmark']:
                results = {}
                for mode in modes:
                    attempts = sorted((output/f"round-{round_plan['round']:05d}"/mode).glob('attempt-*'))
                    completed = [p for p in attempts if verified_complete(p)]
                    if completed:
                        results[mode] = json.loads((completed[-1]/'result.json').read_text())
                if len(results) == 3:
                    reference = results['reference']['steady_seconds_per_policy_step']
                    baseline = results['baseline']['steady_seconds_per_policy_step']
                    collected = results['collection']['steady_seconds_per_policy_step']
                    summary = dict(reference_seconds_per_step=reference, baseline_seconds_per_step=baseline,
                        collection_seconds_per_step=collected, collection_overhead_fraction=collected/baseline-1,
                        relative_to_original_player=collected/reference, runtime_budget_passed=collected <= reference,
                        num_envs=plan['num_envs'], steps=plan['max_steps'])
                    atomic_json(output/f"round-{round_plan['round']:05d}"/'benchmark.json', summary)
                    failed |= not summary['runtime_budget_passed']
        return 2 if failed else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--num-envs', type=int, default=100)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--episodes-per-env', type=int, help='Require this many completed episodes in every environment; overrides total --episodes')
    p.add_argument('--episode-seconds', type=float, help='Override saved episode duration; smoke defaults to 0.2 s')
    p.add_argument('--max-steps', type=int, default=20000)
    p.add_argument('--max-seconds', type=float, default=1800)
    p.add_argument('--feature-dt', type=float, default=.02)
    p.add_argument('--safety-margin', type=float, default=.001)
    p.add_argument('--chunk-size', type=int, default=128)
    p.add_argument('--max-buffer-mb', type=int, default=64)
    p.add_argument('--snapshots', type=int, default=0, help='Optional native-step particle snapshot cadence')
    p.add_argument('--save-mode', choices=['200hz', 'native'], default='200hz',
                   help='Aggregate to 200 Hz (default); native retains full substeps for debugging')
    p.add_argument('--property-expansion', type=float, default=.10, help='Fractional endpoint expansion; mounts excluded')
    p.add_argument('--geometry-expansion', type=float, default=.02, help='Fractional endpoint expansion; mounts excluded')
    p.add_argument('--particle-size', type=float, default=.010, help='Metres; tuned small-particle starting point')
    p.add_argument('--substeps', type=int, default=8, help='Internal subdivisions only; policy/PD timing stays unchanged')
    p.add_argument('--contact-geometry', choices=['mesh', 'boxes'], default='boxes')
    p.add_argument('--smoke', action='store_true', help='Small negative-terrain and partial-reset audit')
    p.add_argument('--smoke-step-height', type=float, default=-.16)
    p.add_argument('--benchmark', action='store_true', help='Original mesh player, current player-loop, and instrumented subprocesses')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = vars(p.parse_args())
    manifest, output = args.pop('manifest'), args.pop('output')
    dry, resume = args.pop('dry_run'), args.pop('resume')
    plan = make_plan(manifest, **args)
    if dry:
        print(json.dumps(plan, indent=2))
        return 0
    return run_plan(plan, output, resume)


if __name__ == '__main__':
    sys.exit(main())
