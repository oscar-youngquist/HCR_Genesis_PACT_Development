"""GPU integration check: rough terrain, curriculum application, training, resume.

CUDA_VISIBLE_DEVICES=1 SIMULATOR=isaaclab python -m \
    legged_gym.scripts.smoke_go2_dreamwaq --headless --gpu cuda:0
"""
import os
import tempfile

import torch
from legged_gym import SIMULATOR
from legged_gym.envs import task_registry
from legged_gym.utils import get_args, init_genesis


def check_contact_indexing(env):
    if SIMULATOR != 'isaaclab':
        return
    sim = env.simulator
    body_names, sensor_names = sim._robot.body_names, sim._contact_sensors.body_names
    feet_names = env.cfg.asset.feet_names
    expected_body = [body_names.index(name) for name in feet_names]
    expected_sensor = [sensor_names.index(name) for name in feet_names]
    assert sim.feet_indices == expected_body
    assert sim.feet_contact_indices == expected_sensor
    raw = sim._contact_sensors.data.net_forces_w
    torch.testing.assert_close(sim.link_contact_forces, raw)
    contacts = (raw[:, expected_sensor].norm(dim=-1) > env.cfg.rewards.contact_force_threshold).float()
    torch.testing.assert_close(env.explicit_labels_buf[:, 3:7], contacts)
    state_ids = [sensor_names.index(name) for name in env.cfg.asset.contact_state_link_names]
    torch.testing.assert_close(sim.link_contact_states, (raw[:, state_ids].norm(dim=-1) > 1.).float())
    for patterns, indices in ((env.cfg.asset.penalize_contacts_on, sim.penalized_contact_indices),
                              (env.cfg.asset.terminate_after_contacts_on, sim.termination_contact_indices)):
        assert indices == [i for i, name in enumerate(sensor_names) if any(part in name for part in patterns)]
    print(f'DREAMWAQ_CONTACT_INDICES_PASSED: bodies={expected_body}, sensors={expected_sensor}', flush=True)


@torch.inference_mode()
def check_torque_clipping(env):
    sim = env.simulator
    old_clip = env.cfg.normalization.clip_actions
    env.cfg.normalization.clip_actions = 1.
    try:
        raw = torch.full_like(env.actions, 1000.)
        raw[:, ::2] *= -1
        # Fill both delay queues before checking a deliberately saturated command.
        for _ in range(env._raw_action_queue.shape[1]):
            actions = env._pre_sim_step(raw)
        sim.step(actions)
        assert (sim.torques.abs() <= sim.torque_limits + 1.e-5).all()
        assert (sim.requested_torques.abs() > sim.torque_limits).any()
        expected = (sim.requested_torques.abs() - sim.torque_limits *
                    env.cfg.rewards.soft_torque_limit).clamp_min(0).sum(-1)
        torch.testing.assert_close(env._reward_torque_limits(), expected)
        if SIMULATOR == 'isaaclab':
            torch.testing.assert_close(
                sim._robot.data.joint_effort_target[:, sim._dof_indices], sim.torques)
            assert (sim._robot.data.applied_torque[:, sim._dof_indices].abs()
                    <= sim.torque_limits + 1.e-5).all()
        print('DREAMWAQ_TORQUE_CLIPPING_PASSED: bounded execution, raw delayed reward', flush=True)
    finally:
        env.cfg.normalization.clip_actions = old_clip
        env.reset_idx(torch.arange(env.num_envs, device=env.device))
    assert not env._raw_action_queue.any()
    assert not sim.requested_torques.any()
    env.reset()


def main():
    args = get_args()
    args.task = 'go2_dreamwaq'
    args.num_envs = args.num_envs or 16
    args.headless = True
    args.resume = False
    if args.cpu:
        raise ValueError('This smoke test requires CUDA')
    if SIMULATOR == 'genesis':
        from legged_gym import gs
        init_genesis(args, gs)
    cfg, train_cfg = task_registry.get_cfgs(args.task, args)
    cfg.domain_rand.reset_resample_episodes = 3  # Exercise cadence in a short test.
    cfg.env.episode_length_s = .2  # Exercise resets and episode-based gating.
    cfg.terrain.num_rows = 2
    cfg.terrain.num_cols = 5
    cfg.terrain.border_size = 5.
    train_cfg.algorithm.num_learning_epochs = 1
    train_cfg.algorithm.num_mini_batches = 2
    train_cfg.runner.save_interval = 1
    train_cfg.runner.resume = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    try:
        runner, _ = task_registry.make_alg_runner(env, args.task, args=args,
            train_cfg=train_cfg, log_root=tempfile.mkdtemp(prefix='dreamwaq_smoke_'))
        assert env.obs_buf.is_cuda and env.privileged_obs_buf.is_cuda
        assert env.explicit_labels_buf.shape == (args.num_envs, 11)
        assert env.privileged_obs_buf.shape[1] == cfg.env.num_privileged_obs
        check_contact_indexing(env)
        fields = ('_friction_values', '_added_base_mass', '_base_com_bias',
                  '_joint_armature', '_joint_friction', '_joint_damping',
                  '_kp_scale', '_kd_scale', '_motor_strength', '_joint_stiffness')
        before = {name: getattr(env.simulator, name).clone() for name in fields}
        for _ in range(2):
            env.reset()
            for name in fields:
                torch.testing.assert_close(getattr(env.simulator, name), before[name], rtol=0, atol=0)
        env.reset()  # Third completed episode: every physical parameter is due.
        for name in fields:
            assert not torch.equal(getattr(env.simulator, name), before[name]), name
        print('DREAMWAQ_CADENCE_PASSED: physical samples held for 3 episodes', flush=True)
        runner.learn(2)
        check_contact_indexing(env)
        # Accelerate schedules only in this test so all endpoints are exercised.
        cfg.domain_rand.push_warmup = 0
        cfg.domain_rand.step_interval = 1
        cfg.domain_rand.min_reward_to_step = 0.
        cfg.domain_rand.recovery_ratio = 0.
        curriculum = env.simulator.domain_rand_curriculum
        curriculum.deltas = {phase: 1. for phase in curriculum.phases}
        cfg.rewards.reward_curriculum.warmup_steps = 2
        cfg.rewards.reward_curriculum.curr_steps = 2
        # Force one simultaneous XYZ/angular disturbance event.
        env.simulator._push_timers.zero_()
        runner.learn(3)
        check_contact_indexing(env)
        assert curriculum.phase == 'complete', curriculum.state_dict()
        env.reset()  # Install final physical randomization ranges.
        assert cfg.domain_rand.added_mass_range == (cfg.domain_rand.added_mass_min, cfg.domain_rand.max_added_mass_max)
        assert torch.isfinite(env.simulator.dof_pos).all()
        assert torch.isfinite(env.simulator.torques).all()
        assert all(torch.isfinite(p).all() for p in runner.alg.actor_critic.parameters())
        expected = [p.detach().clone() for p in runner.alg.actor_critic.parameters()]
        checkpoint = os.path.join(runner.log_dir, 'model_5.pt')
        runner.load(checkpoint)
        assert runner.current_learning_iteration == 5
        assert curriculum.phase == 'complete'
        for actual, saved in zip(runner.alg.actor_critic.parameters(), expected):
            torch.testing.assert_close(actual, saved)
        runner.learn(1)
        check_contact_indexing(env)
        assert runner.current_learning_iteration == 6
        check_torque_clipping(env)
        print('DREAMWAQ_SMOKE_PASSED: 6 iterations, all curriculum phases, checkpoint resume', flush=True)
        print('Checkpoint: ' + os.path.join(runner.log_dir, 'model_6.pt'), flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        print("DREAMWAQ_SMOKE_FAILED", flush=True)
        raise
    finally:
        if SIMULATOR == 'isaaclab':
            env.simulator._app_launcher.app.close()


if __name__ == '__main__':
    main()
