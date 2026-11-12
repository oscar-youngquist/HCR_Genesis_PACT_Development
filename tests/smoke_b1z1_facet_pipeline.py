"""Fresh-process training -> adaptation -> fine-tuning -> portable export -> evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def train(args):
    os.environ['SIMULATOR'] = 'isaaclab_b1z1_pact'
    import torch
    import legged_gym.envs
    from legged_gym.envs.b1z1.b1z1_facet import B1Z1FACETCfg, B1Z1FACETCfgPPO
    from legged_gym.utils import get_args, task_registry
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.runners.b1z1_facet_runner import B1Z1FACETRunner
    sys.argv = [sys.argv[0], '--task=b1z1_facet', '--headless', '--gpu='+args.gpu, '--num_envs=6']
    cli = get_args()
    cfg, training = B1Z1FACETCfg(), B1Z1FACETCfgPPO()
    cfg.env.num_envs, cfg.env.episode_length_s = 6, .9
    for key in ('push_gripper_stators', 'push_robot_base', 'apply_ee_external_forces', 'apply_base_external_forces', 'apply_base_external_torques'):
        setattr(cfg.commands, key, True)
    training.algorithm.student_start_iteration = 2
    training.algorithm.adaptation_iterations = 2
    training.runner.max_iterations = 6
    env, _ = task_registry.make_env('b1z1_facet', args=cli, env_cfg=cfg)
    try:
        # Test-only held loads go through native physical application once.
        force = torch.tensor([[0., 12., 0.]], device=env.device).expand(6, -1)
        base = torch.tensor([[-8., 0., 0.]], device=env.device).expand(6, -1)
        moment = torch.tensor([[0., 0., 1.]], device=env.device).expand(6, -1)
        def ee(ids):
            env.ee_force_ext_world.copy_(force)
            env.simulator.apply_ee_force(force)
        def torso(ids):
            env.base_force_ext_world.copy_(base)
            env.base_torque_ext_world.copy_(moment)
            env.simulator.apply_base_force(base)
            env.simulator.apply_base_torque(moment)
        env._push_gripper, env._push_robot_base = ee, torso
        runner = B1Z1FACETRunner(env, class_to_dict(training), str(args.output_dir/'training'), args.gpu)
        runner.learn(2)
        assert runner.alg.phase == 'adaptation'
        actor = {k: v.clone() for k, v in runner.model.student.state_dict().items()}
        runner.learn(4)
        assert runner.alg.phase == 'student'
        assert all(torch.equal(v, actor[k]) for k, v in runner.model.student.state_dict().items())
        runner.learn(6)
        assert any(not torch.equal(v, actor[k]) for k, v in runner.model.student.state_dict().items())
        assert runner.phase_interactions == dict(teacher=288, adaptation=288, student=288)
        assert torch.isfinite(torch.tensor(list(runner.last_metrics.values()))).all()
        result = dict(phase_updates=runner.phase_updates, phase_interactions=runner.phase_interactions,
            total_interactions=sum(runner.phase_interactions.values()), simulated_seconds=runner.simulated_seconds,
            wall_seconds=runner.training_wall_seconds, metrics=runner.last_metrics)
        (args.output_dir/'training.json').write_text(json.dumps(result, indent=2))
    finally:
        env.simulator._app_launcher.app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='cuda:0')
    parser.add_argument('--output-dir', type=Path, default=Path('/tmp/facet67-pipeline'))
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.train:
        return train(args)
    root = Path(__file__).resolve().parents[1]
    commands = [
        [sys.executable, str(Path(__file__).resolve()), '--train', '--gpu', args.gpu, '--output-dir', str(args.output_dir)],
        [sys.executable, '-m', 'rsl_rl.facet_deployment', '--checkpoint', str(args.output_dir/'training/model_6.pt'), '--output', str(args.output_dir/'student.pt')],
        [sys.executable, 'legged_gym/scripts/eval_b1z1_facet.py', '--artifact', str(args.output_dir/'student.pt'),
            '--gpu', args.gpu, '--headless', '--num-envs', '6', '--episodes', '36', '--horizon-seconds', '3',
            '--event-times', '.5', '1', '1.5', '--output-dir', str(args.output_dir/'evaluation')]]
    for index, command in enumerate(commands):
        with (args.output_dir/f'process_{index}.log').open('w') as stream:
            subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=360)
    assert (args.output_dir/'evaluation/COMPLETE').exists(), 'Isaac shutdown is not proof of successful evaluation'
    summary = json.loads((args.output_dir/'evaluation/summary.json').read_text())
    assert summary['trials'] == 36 and summary['evaluation_interactions'] > 0
    trials = json.loads((args.output_dir/'evaluation/trials.json').read_text())
    assert len(trials) == 36
    reached = [r for r in trials if r['survival_seconds'] > .6]
    assert reached and any(r['force_peak_N'] > 0 for r in reached)
    assert any(r['base_force_peak_N'] > 0 for r in reached)
    for row in trials:
        if row['condition'] == 'base-only':
            assert row['force_peak_N'] == 0
        if row['condition'] == 'ee-only':
            assert row['base_force_peak_N'] == 0
    (args.output_dir/'validation.json').write_text(json.dumps(dict(
        pipeline_passed=True, teacher_adaptation_student_updates=[2, 2, 2],
        training_interactions=summary['training_interactions'], evaluated_trials=len(trials),
        physical_failures=summary['failed_trials'], evaluation_interactions=summary['evaluation_interactions'],
        end_to_end_wall_seconds=summary['wall_seconds'], inference_seconds=summary['inference_seconds'],
        rollout_seconds=summary['rollout_seconds'],
        interpretation='Integration only; trained task performance and full-budget baseline comparisons pending'), indent=2))
    print('FACET67_PIPELINE_OK', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
