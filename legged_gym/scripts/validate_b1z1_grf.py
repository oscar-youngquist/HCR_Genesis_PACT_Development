"""Short, reproducible Isaac Lab GRF check; never starts a full training run.

Run once per controller family; --training-update adds exactly one tiny PPO
update. A checkpoint enables meaningful standing/load response; without one the
test checks interfaces and conservation, not learned standing performance.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import eval_b1z1_exp1 as evaluation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', default='b1z1_pact')
    p.add_argument('--run')
    p.add_argument('--ckpt', default='-1')
    p.add_argument('--gpu', default='cuda:0')
    p.add_argument('--num-envs', type=int, default=2)
    p.add_argument('--steps-per-stage', type=int, default=50)
    p.add_argument('--frictionless', action='store_true')
    p.add_argument('--training-update', action='store_true')
    p.add_argument('--actor-trunk-film', action='store_true', help='Explicit V01 architecture for older FiLM checkpoints')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.num_envs <= 8 or not 2 <= args.steps_per_stage <= 200:
        p.error('Validation is bounded to 1-8 environments and 2-200 steps per stage')
    args.seed, args.headless, args.horizon_seconds = 7, True, 30.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'ERROR.txt').unlink(missing_ok=True)
    os.environ['SIMULATOR'] = evaluation.simulator_family(args.task)
    import numpy as np
    import torch
    import legged_gym.envs
    from legged_gym.utils import task_registry
    from legged_gym.utils.helpers import class_to_dict
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = copy.deepcopy(evaluation.EVAL_CONFIG)
    config['overrides']['noise']['add_noise'] = False
    config['overrides']['domain_rand']['randomize_friction'] = False
    config['overrides']['terrain'].update(static_friction=0. if args.frictionless else 1.,
                                         dynamic_friction=0. if args.frictionless else 1.)
    config['initial_joint_jitter_rad'] = 0.
    config['initial_yaw_range_rad'] = [0., 0.]
    config['ee_amplitude_sphere'] = [0., 0., 0.]
    cfg = evaluation.configure(task_registry.env_cfgs[args.task], args, config)
    training = evaluation.util.materialize(task_registry.train_cfgs[args.task])
    if args.actor_trunk_film:
        training.policy.film_v02_enabled = False
        training.policy.conditioning_mode = 'film'
    env = None
    report = {'task': args.task, 'frictionless': args.frictionless, 'gpu': args.gpu,
              'actor_trunk_film_override': args.actor_trunk_film}
    try:
        env = task_registry.get_task_class(args.task)(cfg, class_to_dict(cfg.sim), args.gpu, True)
        s = env.simulator
        from legged_gym.simulator.b1z1_contact_forces import momentum_residual
        report['grf_semantics'] = s.grf_semantics
        if args.training_update:
            from rsl_rl.utils.runner_registry import runner_registry
            training.runner.num_steps_per_env = 4
            training.algorithm.num_learning_epochs = training.algorithm.num_mini_batches = 1
            log = args.output_dir / 'training'
            log.mkdir(exist_ok=True)
            runner = runner_registry.get_runner_class(training.runner_class_name)(
                env, class_to_dict(training), str(log), device=args.gpu)
            runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
            assert torch.stack([p.isfinite().all() for p in runner.alg.actor_critic.parameters()]).all()
            report['training_updates'] = 1
            report['training_parameters_finite'] = True
            del runner
        state = None
        if args.run:
            checkpoint = evaluation.resolve_checkpoint(args.run, args.ckpt, training.runner.experiment_name)
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            report['checkpoint'] = str(checkpoint)
        actor = evaluation.build_actor(env, training, state)
        before = evaluation.model_hash(actor)
        adapter = evaluation.EvaluationAdapter(env, config, args.seed)
        with torch.inference_mode():
            adapter.reset([evaluation.util.Scenario(args.seed, i, 'nominal', 'in', 'stationary', config)
                           for i in range(args.num_envs)])
        adapter.body_masses = s._robot.root_physx_view.get_masses().to(env.device)
        if args.frictionless:
            # Both colliders must be frictionless; terrain=0 alone can combine
            # with nonzero robot material friction and is not a valid control.
            view = s._robot.root_physx_view
            material = view.get_material_properties()
            material[..., :2] = 0.
            view.set_material_properties(material, torch.arange(args.num_envs, device='cpu'))
        mass = adapter.body_masses.sum(1)
        gravity = torch.tensor(cfg.sim.gravity, device=env.device)
        report.update(mass_kg=mass.tolist(), gravity_world_m_s2=gravity.tolist(),
            substep_force_columns=[f'{component}_{axis}_n' for component in ('normal', 'tangential', 'total', 'external') for axis in 'xyz'])
        original_refresh = adapter.sensor_update
        timings, normal_timings, samples, stages = [], [], [], []

        def momentum():
            return (s._robot.data.body_com_vel_w[..., :3] * adapter.body_masses[..., None]).sum(1)

        def refresh():
            normal_start, normal_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            normal_start.record()
            s._contact_sensors.contact_physx_view.get_net_contact_forces(dt=s._sim_params['dt'])
            normal_end.record()
            normal_timings.append((normal_start, normal_end))
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            original_refresh()
            end.record()
            timings.append((start, end))
            normal, tangent, total = s._body_ground_normal_w, s._body_ground_tangential_w, s._body_ground_total_w
            torch.testing.assert_close(total, normal+tangent)
            filtered_normal = s._contact_sensors.data.force_matrix_w.sum(2)[:, s._contact_indices]
            torch.testing.assert_close(normal, filtered_normal, atol=.05, rtol=.002)
            if args.frictionless:
                torch.testing.assert_close(tangent, torch.zeros_like(tangent), atol=1e-4, rtol=0.)
            samples.append(torch.cat((normal.sum(1), tangent.sum(1), total.sum(1),
                                      s._external_force_world.sum(1)), -1).clone())

        adapter.sensor_update = refresh
        loads = [('standing', (0., 0., 0.)), ('downward_ee', (0., 0., -25.)), ('horizontal_ee', (-25., 0., 0.))]
        with torch.inference_mode():
            for name, force in loads:
                start_sample, p0 = len(samples), momentum().clone()
                start_time = time.perf_counter()
                for step in range(args.steps_per_stage):
                    adapter.step_index += 1
                    s.apply_ee_force(torch.tensor(force, device=env.device).expand(args.num_envs, -1))
                    s.apply_base_force(torch.zeros(args.num_envs, 3, device=env.device))
                    s.apply_base_torque(torch.zeros(args.num_envs, 3, device=env.device))
                    obs, history, *_ = env.get_observations()
                    action = evaluation.inference(actor, obs, history)
                    adapter.substeps.clear()
                    env.step(action)
                torch.cuda.synchronize()
                values = torch.stack(samples[start_sample:])
                dt = s._sim_params['dt']
                residual = momentum_residual(p0, momentum(), mass, gravity, values[..., 6:9], values[..., 9:12], dt)
                stages.append(dict(stage=name, mean_normal_n=values[..., :3].mean((0, 1)).tolist(),
                    mean_tangential_n=values[..., 3:6].mean((0, 1)).tolist(),
                    mean_total_n=values[..., 6:9].mean((0, 1)).tolist(),
                    momentum_residual_ns=residual.tolist(), duration_s=len(values)*dt,
                    momentum_start_ns=p0.tolist(), momentum_end_ns=momentum().tolist(),
                    momentum_residual_fraction_weight_impulse=(residual.norm(dim=1)/(mass*gravity.norm()*len(values)*dt)).tolist(),
                    wall_seconds=time.perf_counter()-start_time, failures=adapter.failures()))
        assert before == evaluation.model_hash(actor), 'Evaluation changed model weights'
        # Partial reset clears only that environment; no old contact label may survive.
        other = s._body_ground_total_w[1:].clone()
        with torch.inference_mode():
            s.reset_idx(torch.tensor([0], device=env.device))
        assert not s._body_ground_total_w[0].any()
        torch.testing.assert_close(s._body_ground_total_w[1:], other)
        milliseconds = [a.elapsed_time(b) for a, b in timings]
        report.update(stages=stages, contact_query_reduce_ms_mean=float(np.mean(milliseconds[1:])),
                      contact_query_reduce_ms_p95=float(np.percentile(milliseconds[1:], 95)),
                      normal_only_query_ms_mean=float(np.mean([a.elapsed_time(b) for a,b in normal_timings[1:]])),
                      filtered_normal_matrix_parity=True,
                      frozen_policy=True, partial_reset=True,
                      torch_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        np.savez_compressed(args.output_dir / 'substeps.npz',
                            forces=torch.stack(samples).cpu().numpy(), physics_dt=s._sim_params['dt'])
        (args.output_dir / 'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps(report, indent=2), flush=True)
    except Exception:
        # Kit close may terminate Python before its normal exception printer runs.
        import traceback
        error = traceback.format_exc()
        (args.output_dir / 'ERROR.txt').write_text(error)
        print(error, file=sys.stderr, flush=True)
        raise
    finally:
        if env is not None:
            env.simulator._app_launcher.app.close()


if __name__ == '__main__':
    if os.environ.get('B1Z1_GRF_VALIDATION_WORKER') == '1':
        main()
    else:
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument('--output-dir', type=Path)
        args, _ = p.parse_known_args()
        if args.output_dir and (args.output_dir / 'validation.json').exists():
            raise SystemExit('Choose a new output directory; validation never overwrites a completed report')
        result = subprocess.run([sys.executable, __file__, *sys.argv[1:]],
            env={**os.environ, 'B1Z1_GRF_VALIDATION_WORKER': '1', 'PYTHONUNBUFFERED': '1'})
        complete = args.output_dir and (args.output_dir / 'validation.json').exists()
        raise SystemExit(result.returncode or (0 if complete or '--help' in sys.argv else 1))
