"""Optional RL version of the detached arm compensation teacher."""

import math

import torch

from legged_gym.torque_action_scaling import simulator_torque_action_scale


NAME = "tracking_arm_feedforward"


def enabled(env):
    return (env.cfg.env.num_policy_actions == 2 * env.num_actions
            and env.reward_scales.get(NAME, 0.) != 0.)


def configure(env, backend):
    """Borrow the runner's BARD instance; no duplicate dynamics allocation."""
    if not enabled(env):
        return
    sigma = env.cfg.rewards.arm_feedforward_tracking_sigma
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("arm_feedforward_tracking_sigma must be finite and positive")
    ids = [int(i) for i in env.simulator._arm_dof_cfg_ids if int(i) < env.num_actions]
    if not ids or len(set(ids)) != len(ids) or min(ids) < 0:
        raise ValueError("Arm feedforward reward needs policy-controlled arm indices")
    env._arm_ff_reward_backend = backend
    env._arm_ff_reward_indices = torch.tensor(ids, device=env.device, dtype=torch.long)
    env._arm_ff_reward_scale = simulator_torque_action_scale(env.simulator).index_select(
        0, env._arm_ff_reward_indices).detach().clone()


@torch.no_grad()
def capture(env, actions):
    """Capture after force sampling and action delay, before physical integration."""
    if not enabled(env):
        return
    if getattr(env, "_arm_ff_reward_backend", None) is None:
        raise RuntimeError("Arm feedforward reward requires the B1Z1 PACT runner with dynamics_backend='bard'")
    state = env.get_pact_dynamics_state().detach().clone()
    ids = env._arm_ff_reward_indices
    # Same effective FF branch as the PINN, before clamping the summed PD+FF torque.
    # Explicit scaling avoids backend differences in cached pre/post-motor torques.
    gain = state[:, 97:97 + len(env.cfg.asset.dof_names)].index_select(1, ids) * state[:, 155:156]
    ff = actions[:, env.num_actions:].index_select(1, ids) * env._arm_ff_reward_scale * gain
    env._arm_ff_reward_snapshot = (state, ff.detach().clone(), (gain.abs() > 1e-8).all(-1))


@torch.inference_mode(False)
@torch.no_grad()
def reward(env):
    """exp(-mean(((tau_FF - tau_FF*) / S_tau)^2) / sigma), in [0, 1].

    Reuse h_a + M_au a_u - J_EE,a^T F_applied exactly. For RL, a_u is measured
    over the completed simulator interval, rather than the actor-PINN's model
    prediction. Rewards never backpropagate through this teacher. The backend
    runs outside inference_mode so its shared buffers remain usable by PINNs.
    """
    result = env.rew_buf.new_zeros(env.num_envs)
    if not enabled(env):
        return result
    from rsl_rl.algorithms.b1z1_actor_feedforward import compensation_target
    from rsl_rl.algorithms.b1z1_bard_pinn import mechanics
    state, command, authority = env._arm_ff_reward_snapshot
    simulator = env.simulator
    velocity = torch.cat((simulator._base_world_lin_vel, simulator._base_world_ang_vel,
                          simulator.dof_vel), -1)
    acceleration = (velocity - state[:, 26:51]) / env.dt
    # Skip resets, nonphysical teleports, unavailable FF authority, and bad states.
    displacement = simulator.base_pos - state[:, :3] - env.dt * state[:, 26:29]
    valid = ~env.reset_buf.bool().flatten() & authority & (displacement.norm(dim=-1) <= .5)
    for value in (state, command, acceleration):
        valid &= torch.isfinite(value).all(-1)
    valid &= state[:, 3:7].norm(dim=-1) > 1e-6
    backend = env._arm_ff_reward_backend
    device = getattr(backend, "device", state.device)
    ids = env._arm_ff_reward_indices.to(device)
    for rows in valid.nonzero().flatten().split(backend.batch_capacity):
        if not len(rows):
            continue
        # Clone under inference_mode(False) before touching shared BARD buffers.
        initial = state[rows].to(device).clone()
        fixed = mechanics(backend, {"rollout_initial_state": initial[:, :51], "dynamics_state": initial})
        target = compensation_target(fixed, acceleration[rows].to(device), initial[:, 88:91],
                                     ids, len(env.cfg.asset.dof_names))[0].to(result.device)
        good = torch.isfinite(target).all(-1)
        error = (command[rows][good] - target[good]) / env._arm_ff_reward_scale
        result[rows[good]] = torch.exp(-error.square().mean(-1) / env.cfg.rewards.arm_feedforward_tracking_sigma)
    return result
