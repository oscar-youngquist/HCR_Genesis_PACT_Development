#!/usr/bin/env python3
"""Run the B1 environment's existing UniFP reference gait open-loop.

The environment already owns the gait clock and reference calculation:

    _step_contact_targets() -> _get_phase() -> compute_ref_state()

This script only converts the resulting absolute reference joint positions to
the normalized residual actions expected by the existing simulator:

    q_target = q_default + action_scale * action

Therefore:

    action = (q_ref - q_default) / action_scale

The normal ``env.step(actions)`` path then calls the simulator's existing
``_compute_torques(actions)`` implementation.  No second gait generator or
controller is implemented here.
"""

import torch

from legged_gym import SIMULATOR, gs
import legged_gym.envs  # noqa: F401 -- registers environments
from legged_gym.utils import get_args, init_genesis, task_registry, quat_rotate_inverse


# Test settings. These can be edited directly for a particular run.
TASK_NAME = "b1_unifp"
NUM_ENVS = 64
TEST_DURATION_S = 10.0
WALK_COMMAND = (0.5, 0.0, 0.0)

# This is a diagnostic safety clamp on the normalized residual action. Set it
# high enough to reproduce the configured reference, but inspect the printed
# required-action maximum before increasing it further.
MAX_ABS_ACTION = 2.0
PRINT_EVERY_STEPS = 25


def disable_training_randomization(env_cfg) -> None:
    """Keep this control-path test deterministic and on flat ground."""
    env_cfg.env.num_envs = NUM_ENVS
    # env_cfg.terrain.mesh_type = "plane"
    # env_cfg.terrain.curriculum = False
    # env_cfg.terrain.measure_heights = False
    # env_cfg.terrain.obtain_terrain_info_around_feet = False
    # env_cfg.commands.curriculum = False
    # env_cfg.commands.push_robot_base = False
    # env_cfg.commands.apply_base_external_forces = False
    # env_cfg.noise.add_noise = False

    for name in (
        "randomize_friction",
        "randomize_base_mass",
        "randomize_com_displacement",
        "randomize_ctrl_delay",
        "randomize_pd_gain",
        "randomize_motor_strength",
        "randomize_joint_armature",
        "randomize_joint_friction",
        "randomize_joint_stiffness",
        "randomize_joint_damping",
        "push_robots",
    ):
        if hasattr(env_cfg.domain_rand, name):
            setattr(env_cfg.domain_rand, name, False)


def reference_position_to_action(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the existing reference and convert it to policy actions."""
    env.compute_ref_state()

    phase = env._get_phase()
    idx = env.leg_dof_indices

    vx = env.commands[:, 0]

    direction = torch.sign(vx)
    command_scale = torch.clamp(
        torch.abs(vx) / 0.5,
        min=0.0,
        max=1.0,
    )

    sweep_amplitude = 0.10  # radians; begin conservatively

    sweep = (
        sweep_amplitude
        * command_scale
        * direction
        * torch.cos(2.0 * torch.pi * phase)
    )

    # FR/RL diagonal
    env.ref_dof_pos[:, idx["FR_thigh_joint"]] += sweep
    env.ref_dof_pos[:, idx["RL_thigh_joint"]] += sweep

    # FL/RR diagonal: half-cycle offset
    env.ref_dof_pos[:, idx["FL_thigh_joint"]] -= sweep
    env.ref_dof_pos[:, idx["RR_thigh_joint"]] -= sweep


    q_ref = env.ref_dof_pos[:, :12]
    q_default = env.simulator.default_dof_pos[:, :12]
    action_scale = float(env.cfg.control.action_scale)

    if action_scale <= 0.0:
        raise ValueError(f"control.action_scale must be positive: {action_scale}")

    required_actions = (q_ref - q_default) / action_scale
    applied_actions = required_actions.clamp(
        min=-MAX_ABS_ACTION,
        max=MAX_ABS_ACTION,
    )
    return applied_actions, required_actions


def main() -> None:
    args = get_args()
    args.task = TASK_NAME

    if "genesis" in SIMULATOR:
        init_genesis(args, gs)

    env_cfg, _ = task_registry.get_cfgs(name=args.task, args=args)
    disable_training_randomization(env_cfg)
    env, _ = task_registry.make_env(
        name=args.task,
        args=args,
        env_cfg=env_cfg,
    )
    env.reset()

    if env.num_actions != 12:
        raise RuntimeError(f"Expected 12 B1 leg actions, got {env.num_actions}")

    command = torch.tensor(
        WALK_COMMAND,
        device=env.device,
        dtype=env.commands.dtype,
    )
    num_steps = int(round(TEST_DURATION_S / env.dt))
    reset_count = 0

    for step in range(num_steps):
        # A nonzero command lets the environment's existing
        # _step_contact_targets() advance gait_indices during env.step().
        # Reapply it every step so periodic command resampling cannot stop the
        # gait clock during this deterministic test.
        env.commands[:, :3] = command

        actions, required_actions = reference_position_to_action(env)

        # This follows the normal task path. B1UniFP.step() calls
        # simulator.step(actions), which calls simulator._compute_torques().
        env.step(actions)

        resets = env.reset_buf.bool()
        reset_count += int(resets.sum().item())

        feet_from_base_world = (
            env.simulator.feet_pos
            - env.simulator.base_pos.unsqueeze(1)
        )

        base_quat_per_foot = env.simulator.base_quat.unsqueeze(1).expand(
            -1, 4, -1
        )

        feet_pos_base = quat_rotate_inverse(
            base_quat_per_foot.reshape(-1, 4),
            feet_from_base_world.reshape(-1, 3),
        ).view(env.num_envs, 4, 3)

        if step % PRINT_EVERY_STEPS == 0 or step == num_steps - 1:
            phase = env._get_phase()
            contacts = (
                env.simulator.link_contact_forces[
                    :, env.simulator.feet_indices, 2
                ] > 5.0
            ).float().sum(dim=1)

            print(
                f"step={step:4d} "
                f"time={step * env.dt:6.2f}s "
                f"phase={phase[0].item():.3f} "
                f"contacts={contacts.float().mean().item():.2f} "
                f"body_vx: {env.simulator.base_lin_vel[:, 0].mean().item():.3f} "
                f"foot_x: {feet_pos_base[:, :, 0].mean(dim=0).tolist()} "
                f"height={env.simulator.base_pos[:, 2].mean().item():.3f} "
                f"|roll|={env.simulator.base_euler[:, 0].abs().mean().item():.3f} "
                f"|pitch|={env.simulator.base_euler[:, 1].abs().mean().item():.3f} "
                f"required|max_action|={required_actions.abs().max().item():.3f} "
                f"applied|max_action|={actions.abs().max().item():.3f} "
                f"resets={reset_count}"
            )


if __name__ == "__main__":
    main()
