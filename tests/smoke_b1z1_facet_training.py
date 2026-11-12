"""Short real FACET PPO/handoff smoke; overrides are test-local, not production."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='cuda:0')
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, default=Path('/tmp/facet-stage5-smoke'))
    args = parser.parse_args()
    os.environ['SIMULATOR'] = 'isaaclab_b1z1_pact'
    import torch
    import legged_gym.envs
    from legged_gym.utils import get_args, task_registry
    from legged_gym.envs.b1z1.b1z1_facet import B1Z1FACETCfg, B1Z1FACETCfgPPO
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.runners.b1z1_facet_runner import B1Z1FACETRunner
    sys.argv = [sys.argv[0], '--task=b1z1_facet', '--gpu='+args.gpu, '--headless', '--num_envs='+str(args.num_envs)]
    cli = get_args()
    cfg, training = B1Z1FACETCfg(), B1Z1FACETCfgPPO()
    cfg.env.num_envs = args.num_envs
    cfg.env.episode_length_s = .8  # Exercise timeout/reset packets in a short run.
    training.algorithm.student_start_iteration = 3
    training.runner.max_iterations = 6
    training.runner.save_interval = 3
    env, _ = task_registry.make_env('b1z1_facet', args=cli, env_cfg=cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        runner = B1Z1FACETRunner(env, class_to_dict(training), str(args.output_dir), args.gpu)
        def snapshot(module):
            return {k: v.detach().clone() for k, v in module.state_dict().items()}
        def changed(module, state):
            return any(not torch.equal(v, state[k]) for k, v in module.state_dict().items())
        before = {k: snapshot(getattr(runner.model, k)) for k in ('teacher', 'critic', 'estimator', 'priv_encoder', 'force_encoder')}
        start = time.perf_counter()
        runner.learn(3)
        assert runner.alg.phase == 'student'
        for key, state in before.items():
            assert changed(getattr(runner.model, key), state), key
        frozen = {k: snapshot(getattr(runner.model, k)) for k in ('teacher', 'priv_encoder', 'force_encoder', 'obs_norm', 'priv_norm', 'wrench_norm')}
        for key, value in runner.model.teacher.state_dict().items():
            torch.testing.assert_close(value, runner.model.student.state_dict()[key], rtol=0, atol=0)
        student, estimator, critic = (snapshot(getattr(runner.model, key)) for key in ('student', 'estimator', 'critic'))
        runner.learn(6)
        assert changed(runner.model.student, student) and changed(runner.model.estimator, estimator) and changed(runner.model.critic, critic)
        assert all(not changed(getattr(runner.model, key), state) for key, state in frozen.items())
        assert all(torch.isfinite(p).all() for p in runner.model.parameters())
        assert all(torch.isfinite(torch.tensor(v)) for v in runner.last_metrics.values())
        assert not hasattr(runner, 'dynamics')
        assert (args.output_dir/'transition_3.pt').exists()
        for index, name in enumerate(cfg.asset.dof_names):
            torch.testing.assert_close(env.simulator._p_gains[index], torch.tensor(40. if name.startswith('z1_') else 100., device=env.device))
            torch.testing.assert_close(env.simulator._d_gains[index], torch.tensor(1. if name.startswith('z1_') else 2., device=env.device))
        # Strict resume must not recopy teacher over the now-trained student.
        saved = snapshot(runner.model)
        runner.load(args.output_dir/'model_6.pt')
        assert not changed(runner.model, saved) and runner.current_learning_iteration == 6
        assert runner.hidden.count_nonzero() == runner.ema_hidden.count_nonzero() == 0
        result = dict(elapsed_seconds=time.perf_counter()-start, num_envs=args.num_envs,
            updates=6, switch_after=3, last_metrics=runner.last_metrics,
            observations=env.get_facet_observation_schema(), device=args.gpu)
        (args.output_dir/'smoke.json').write_text(json.dumps(result, indent=2))
        print('FACET_STAGE5_SMOKE_OK', json.dumps(result), flush=True)
    finally:
        env.simulator._app_launcher.app.close()


if __name__ == '__main__':
    main()
