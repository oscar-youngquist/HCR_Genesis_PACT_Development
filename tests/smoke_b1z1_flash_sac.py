"""Bounded real B1Z1/BARD smoke: two environments, five training steps.

Run with SIMULATOR=genesis_b1z1_pact or isaaclab_b1z1_pact in the matching environment.
Requires CUDA and BARD. Never starts a full training run or writes checkpoints.
"""
import os
import torch
import legged_gym
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact import B1Z1PACT
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg, B1Z1PACTCfgPPO
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners.b1z1_pact_runner import B1Z1PACTRunner


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("The real BARD simulator smoke requires CUDA")
    torch.set_num_threads(1)
    is_lab = legged_gym.SIMULATOR.startswith("isaaclab")
    device = os.environ.get("PACT_SMOKE_DEVICE", "cuda:1" if is_lab else "cuda:0")
    if not is_lab:
        legged_gym.gs.init(backend=legged_gym.gs.gpu, logging_level="warning")
    env_cfg, train = B1Z1PACTCfg(), class_to_dict(B1Z1PACTCfgPPO())
    env_cfg.env.num_envs = 2
    # Genesis supports heightfields; the shared config also serves Isaac Gym.
    env_cfg.terrain.mesh_type = "heightfield"
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.max_init_terrain_level = 0
    train["runner"]["num_steps_per_env"] = 5
    train["runner"]["curriculum_metrics_interval_env_steps"] = 1
    env_cfg.commands.force_curriculum_gate_start_env_step = 2
    env_cfg.commands.force_curriculum_latest_start_env_step = 2
    env_cfg.commands.force_curriculum_external_ramp_env_steps = 2
    env_cfg.rewards.reward_curriculum.warmup_env_steps = 2
    env_cfg.rewards.reward_curriculum.curr_env_steps = 2
    env_cfg.rewards.gait_guidance_decay_enabled = True
    env_cfg.rewards.gait_guidance_decay_env_steps = 4
    train["algorithm"].update(sac_batch_size=4, sac_critic_width=16, sac_critic_blocks=1,
        replay_capacity=16, replay_warmup=4, bard_batch_capacity=4)
    for channel in ("position", "leg_torque", "arm_torque"):
        train["algorithm"][f"sac_{channel}_action_range"] = float(
            os.environ.get(f"PACT_SMOKE_{channel.upper()}_ACTION_RANGE", "1.0"))
    train["policy"]["pretrained_path"] = None
    env = B1Z1PACT(env_cfg, class_to_dict(env_cfg.sim), device, True)
    try:
        runner = B1Z1PACTRunner(env, train, log_dir=None, device=device)
        runner.alg.cfg["pinn_warmup_env_steps"] = 1
        # Exercise timeout bootstrapping before auto-reset on the first step.
        runner.env.episode_length_buf[0] = runner.env.max_episode_length
        runner.learn(1)
        assert runner.completed_env_steps == runner.env.completed_env_steps == 5
        assert runner.env.external_force_scale == 1.
        for key in runner.env.reward_curr_keys:
            if key in runner.env.reward_scales:
                expected = runner.env.reward_curr_bounds[key][1] * runner.env.dt
                assert abs(runner.env.reward_scales[key] - expected) < 1e-8
        assert runner.total_timesteps == 10 and runner.alg.schedule_env_steps == 5
        assert runner.alg.env_steps == 5 and runner.alg.update_step == 8
        assert runner.alg.replay.data["truncated"][:runner.alg.replay.size].any()
        assert runner.alg.replay.data["actions"][:runner.alg.replay.size].abs().max() <= 1.
        print(f"Environment action-range multiplier: {runner.alg.action_ranges}")
        per_transition = runner.alg.replay.estimated_bytes // runner.alg.replay.capacity
        print(f"PASS: two environments, five vector steps, eight updates; "
              f"default 32768-transition replay estimate: {per_transition * 32768 / 2**20:.1f} MiB")
    finally:
        launcher = getattr(env.simulator, "_app_launcher", None)
        if launcher is not None:
            launcher.app.close()



if __name__ == "__main__":
    main()
