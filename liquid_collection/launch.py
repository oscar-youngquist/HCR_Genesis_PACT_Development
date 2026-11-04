"""Top-level collection campaign: every chosen platform runs every payload round."""
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path

from .budget import episode_steps
from .device import selected_cuda_device
from .plan import ROOT, make_plan, run_plan
from .storage import atomic_json

PLATFORMS = {
    'go1-dact': ROOT/'liquid_collection/go1_manifest.json',
    'go2-dact': ROOT/'liquid_collection/example_manifest.json',
    'go2-hardpact': None,
}


def control_dt(config):
    """Read the saved literal policy clock without importing Genesis or running config code."""
    tree = ast.parse(Path(config).read_text())
    cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name.endswith('Cfg'))
    control = next(n for n in cfg.body if isinstance(n, ast.ClassDef) and n.name == 'control')
    for node in control.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'dt' for t in node.targets):
            dt = float(ast.literal_eval(node.value))
            if math.isfinite(dt) and dt > 0:
                return dt
    raise ValueError(f'{config}: an explicit positive control.dt is required for automatic step limits')


def campaign_plan(platforms, num_envs, episodes_per_env, rounds, episode_seconds, seed=1, max_seconds=None, gpu=0):
    if not platforms or len(set(platforms)) != len(platforms):
        raise ValueError('Choose a nonempty list of distinct platforms')
    if min(num_envs, episodes_per_env, rounds) < 1:
        raise ValueError('Environment, episode and round counts must be positive')
    for platform in platforms:
        if platform not in PLATFORMS:
            raise ValueError(f'Unknown platform: {platform}')
        if PLATFORMS[platform] is None:
            raise ValueError('go2-hardpact is reserved: no HardPACT collection adapter/checkpoint is implemented. '
                             'Select go1-dact and/or go2-dact; no DACT fallback will be substituted.')
    selected_device = selected_cuda_device(gpu)
    plans = {}
    for platform in platforms:
        plan = make_plan(PLATFORMS[platform], seed=seed, rounds=rounds, num_envs=num_envs,
                         episodes=num_envs*episodes_per_env,
                         episodes_per_env=episodes_per_env, episode_seconds=episode_seconds)
        plan.update(gpu=gpu, cuda_visible_device=selected_device)
        clocks = [control_dt(r['policy']['config']) for r in plan['rounds']]
        steps = max(episode_steps(episode_seconds, dt) for dt in clocks)
        plan['max_steps'] = (steps+2)*episodes_per_env+1
        # Conservative watchdog only, not a throughput prediction. It scales
        # with the selected work instead of truncating long campaigns at 30 min.
        plan['max_seconds'] = float(max_seconds if max_seconds is not None else
                                    max(1800., 300.+plan['max_steps']*max(1., num_envs/2)))
        if not math.isfinite(plan['max_seconds']) or plan['max_seconds'] <= 0:
            raise ValueError('Wall-clock watchdog must be positive and finite')
        plans[platform] = plan
    # Match payloads across platform comparisons while retaining robot mount Z.
    reference = plans[platforms[0]]['rounds']
    for plan in plans.values():
        for r, shared in zip(plan['rounds'], reference):
            r['payload'] = deepcopy(shared['payload'])
    return dict(schema_version=1, platforms=platforms, num_envs=num_envs,
                gpu=gpu, cuda_visible_device=selected_device,
                episodes_per_env=episodes_per_env, rounds=rounds, episode_seconds=episode_seconds,
                seed=seed, worker_runs=len(platforms)*rounds,
                requested_episodes=len(platforms)*rounds*num_envs*episodes_per_env, plans=plans)


def run_campaign(campaign, output, resume=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = output/'campaign.json'
        if saved.exists():
            if not resume:
                raise ValueError('Campaign already exists; use --resume with the same settings')
            if json.loads(saved.read_text()) != campaign:
                raise ValueError('Campaign settings or input identities differ; choose a new output directory')
        else:
            atomic_json(saved, campaign)
        code = 0
        for round_index in range(campaign['rounds']):
            for platform in campaign['platforms']:
                plan = campaign['plans'][platform]
                print(f'Running round {round_index+1}/{campaign["rounds"]}: {platform}; '
                      f'output {output/platform}', flush=True)
                code |= run_plan(plan, output/platform, resume=resume or round_index > 0,
                                 round_index=round_index)
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platforms', nargs='+', choices=tuple(PLATFORMS), default=['go1-dact', 'go2-dact'])
    parser.add_argument('--gpu', type=int, default=0, help='CUDA index within inherited visibility; each worker uses only this GPU')
    parser.add_argument('--num-envs', type=int, default=2)
    parser.add_argument('--episodes-per-env', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=1, help='Sampled configs per selected platform')
    parser.add_argument('--episode-seconds', type=float, default=5., help='Maximum simulated duration per episode')
    parser.add_argument('--output', type=Path, help='Default: a fresh timestamped directory under runs/liquid_collection')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max-seconds', type=float, help='Optional wall-clock watchdog per worker, including initialization')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume and args.output is None:
        parser.error('--resume requires --output pointing to the existing campaign')
    try:
        campaign = campaign_plan(args.platforms, args.num_envs, args.episodes_per_env,
                                 args.rounds, args.episode_seconds, args.seed, args.max_seconds, args.gpu)
        output = args.output or ROOT/'runs/liquid_collection'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        print(f'Output: {output.resolve()}', flush=True)
        print(f'{campaign["worker_runs"]} sequential worker runs; {campaign["requested_episodes"]} requested episodes; '
              f'200 Hz saving; genesis_lr; GPU {args.gpu} (CUDA_VISIBLE_DEVICES={campaign["cuda_visible_device"]}; worker cuda:0)', flush=True)
        if args.dry_run:
            print(json.dumps(campaign, indent=2))
            return 0
        return run_campaign(campaign, output, args.resume)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
