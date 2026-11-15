"""Play-only QP setup; training configuration and checkpoints are not mutated."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path


def configure_play_qp(env, runner, train_cfg, enabled=None):
    enabled = (getattr(getattr(env.cfg, 'play', None), 'qp_enabled', False)
               if enabled is None else enabled)
    if not enabled:
        if hasattr(env, 'set_hard_pact_qp_enabled'):
            env.set_hard_pact_qp_enabled(False)
        return None
    actor = runner.alg.actor_critic
    if (getattr(env.cfg, 'task_backend', None) != 'isaaclab'
            or not getattr(runner, 'is_hard_pact', False)
            or getattr(actor, 'num_policy_actions', 24) != 24):
        raise ValueError('Play QP requires a coupled HardPACT Isaac Lab policy, not HardPACTPos/position-only')
    from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig, HardPACTDifferentiableQP
    from legged_gym.dynamics import create_go2_dynamics
    from legged_gym import LEGGED_GYM_ROOT_DIR

    config = HardPACTQPConfig.from_dict(deepcopy(train_cfg.algorithm.hard_pact_qp))
    config = replace(config, enabled=True, warmup_iterations=0,
        qp_update_mode='every_substep', correction_ramp_enabled=False,
        objective_curriculum_enabled=False, torso_stability_curriculum_enabled=False,
        torque_rate_limit_nm_s=env.cfg.control.torque_rate_limit_nm_s,
        height_target=(env.cfg.rewards.base_height_target if config.height_target is None else config.height_target),
        height_velocity_obs_scale=env.cfg.normalization.obs_scales.lin_vel)
    simulator = env.simulator
    qp = HardPACTDifferentiableQP(config, simulator.torque_limits,
        simulator.dof_pos_limits_hard[:, 0], simulator.dof_pos_limits_hard[:, 1], simulator.dof_vel_limits)
    dynamics = runner.alg.physics_dynamics
    if dynamics is None:
        path = Path(getattr(train_cfg.algorithm, 'bard_urdf_path', 'resources/robots/go2/urdf/go2.urdf'))
        if not path.is_absolute(): path = Path(LEGGED_GYM_ROOT_DIR) / path
        dynamics = create_go2_dynamics(train_cfg.algorithm.dynamics_backend, str(path),
            device=env.device, batch_capacity=env.num_envs,
            default_joint_position=simulator.default_dof_pos.reshape(-1, 12)[0])
    # An explicit deployment opt-in may enable execution for a coupled no-QP
    # checkpoint. Never change the actor/algorithm's training ablation flags.
    env.hard_pact_features = replace(env.hard_pact_features, execution_qp=True)
    env.configure_hard_pact_substep_qp(actor, dynamics, qp)
    env._cache_qp_replay_mechanics = False
    env._qp_execution_alpha = 1.0
    qp.clear_warm_start()
    env.set_hard_pact_qp_enabled(True)
    if not env._hard_pact_rollout_qp_enabled:
        raise RuntimeError('Deployment QP failed to activate')
    print(f'Play QP: every_substep, alpha=1, warmup=0, '
          f'solver={config.rollout_qp_solver or config.qp_solver}; configured objective weights/limits')
    return qp
