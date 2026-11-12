"""FACET-only schedule/resume CLI; the ordinary --task b1z1_facet remains supported."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume', type=Path, help='Exact checkpoint; restore saved task/schedule before constructing the simulator')
    p.add_argument('--legacy-contract', type=Path, help='Verified metadata for a Stage 5 checkpoint without its sidecar')
    p.add_argument('--seed', type=int)
    p.add_argument('--num-envs', type=int)
    p.add_argument('--max-iterations', type=int, help='Explicit TOTAL budget override, not additional updates')
    p.add_argument('--student-start-iteration', type=int)
    p.add_argument('--adaptation-iterations', type=int)
    p.add_argument('--save-interval', type=int)
    p.add_argument('--gpu', default='cuda:0')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--log-dir', type=Path)
    a = p.parse_args()
    os.environ['SIMULATOR'] = 'isaaclab_b1z1_pact'
    import json
    import torch
    import legged_gym.envs
    from legged_gym.envs.b1z1.b1z1_facet import B1Z1FACETCfg, B1Z1FACETCfgPPO
    from legged_gym.utils import get_args, task_registry
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.facet_contract import checkpoint_contract, apply_config_snapshot
    from rsl_rl.runners.b1z1_facet_runner import B1Z1FACETRunner
    cfg, training = B1Z1FACETCfg(), B1Z1FACETCfgPPO()
    contract = None
    if a.resume:
        state = torch.load(a.resume, map_location='cpu', weights_only=False)
        contract = checkpoint_contract(state, a.resume, json.loads(a.legacy_contract.read_text()) if a.legacy_contract else None)
        if 'environment' not in contract:
            raise ValueError('Resume CLI requires the original resolved environment settings')
        apply_config_snapshot(cfg, contract['environment'])
        apply_config_snapshot(training, state['config'])
        # Older checkpoints predate the mode field and were FIXED impedance.
        cfg.facet_reference.sampling_mode = contract['reference'].get('sampling_mode', 'fixed')
    overrides = {key: value for key in ('max_iterations', 'student_start_iteration', 'adaptation_iterations', 'save_interval')
                 if (value := getattr(a, key)) is not None}
    for key, value in overrides.items():
        setattr(training.runner if key in ('max_iterations', 'save_interval') else training.algorithm, key, value)
    if a.seed is not None:
        if a.resume and a.seed != training.seed:
            raise ValueError('Resume retains the checkpoint training-seed identity; omit --seed or use the saved seed')
        training.seed = a.seed
    if a.num_envs is not None:
        cfg.env.num_envs = a.num_envs
    sys.argv = [sys.argv[0], '--task=b1z1_facet', '--gpu='+a.gpu, '--num_envs='+str(cfg.env.num_envs), '--seed='+str(training.seed)]
    if a.headless:
        sys.argv.append('--headless')
    cli = get_args()
    log = a.log_dir or ROOT/'logs/b1z1/b1z1_facet'/datetime.now().strftime('%b%d_%H-%M-%S_facet')
    env, _ = task_registry.make_env('b1z1_facet', args=cli, env_cfg=cfg)
    try:
        runner = B1Z1FACETRunner(env, class_to_dict(training), str(log), a.gpu)
        if a.resume:
            runner.load(a.resume, schedule_overrides=overrides, legacy_contract=contract)
        runner.learn(init_at_random_ep_len=not bool(a.resume))
        (log/'budget.json').write_text(json.dumps(dict(phase_updates=runner.phase_updates,
            phase_interactions=runner.phase_interactions, total_interactions=sum(runner.phase_interactions.values()),
            simulated_robot_seconds=runner.simulated_seconds, collection_update_wall_seconds=runner.training_wall_seconds,
            accounting_exact=runner.accounting_exact), indent=2))
    except BaseException:
        import traceback
        error = traceback.format_exc()
        log.mkdir(parents=True, exist_ok=True)
        (log/'ERROR.txt').write_text(error)
        print(error, file=sys.stderr, flush=True)
        raise
    finally:
        env.simulator._app_launcher.app.close()


if __name__ == '__main__':
    main()
