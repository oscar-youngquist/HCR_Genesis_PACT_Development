"""Visually inspect a Go2 HardPACTPos checkpoint with random commands."""

import os

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR, SIMULATOR
from legged_gym.envs import *  # noqa: F403 - imports task registrations
from legged_gym.utils import get_args, init_genesis, task_registry
from legged_gym.utils.helpers import get_load_path


# ---------------------------------------------------------------------------
# MANUAL TERRAIN SETTINGS
# ---------------------------------------------------------------------------
# Edit TERRAIN_PRESET to choose what is shown in every terrain cell.  "mixed"
# keeps the task's normal randomized terrain mixture.  For any other preset,
# edit the physical values (metres unless stated otherwise) in the matching
# dictionary below.  Negative slope/step height gives downhill terrain.
TERRAIN_PRESET = "mixed"
TERRAIN_PRESETS = {
    "flat": {
        "type": "terrain_utils.pyramid_sloped_terrain",
        "slope": 0.0,
        "platform_size": 3.0,
    },
    "rough": {
        "type": "terrain_utils.random_uniform_terrain",
        "min_height": -0.05,
        "max_height": 0.05,
        "step": 0.005,
        "downsampled_scale": 0.2,
    },
    "slope": {
        "type": "terrain_utils.pyramid_sloped_terrain",
        "slope": 0.25,
        "platform_size": 3.0,
    },
    "stairs": {
        "type": "terrain_utils.pyramid_stairs_terrain",
        "step_width": 0.31,
        "step_height": 0.10,
        "platform_size": 3.0,
    },
    "obstacles": {
        "type": "terrain_utils.discrete_obstacles_terrain",
        "max_height": 0.10,
        "min_size": 1.0,
        "max_size": 2.0,
        "num_rects": 20,
        "platform_size": 3.0,
    },
    "waves": {
        "type": "terrain_utils.wave_terrain",
        "amplitude": 0.10,
        "num_waves": 2,
    },
    "stepping_stones": {
        "type": "terrain_utils.stepping_stones_terrain",
        "stone_size": 1.0,
        "stone_distance": 0.20,
        "max_height": 0.05,
        "platform_size": 3.0,
    },
    "gap": {
        "type": "terrain_utils.gap_terrain",
        "gap_size": 0.20,
        "platform_size": 3.0,
    },
    "pit": {
        "type": "terrain_utils.pit_terrain",
        "depth": 0.20,
        "platform_size": 3.0,
    },
}


def _configure_play_terrain(terrain_cfg):
    """Apply the manually editable terrain selection without changing training."""
    if TERRAIN_PRESET == "mixed":
        terrain_cfg.selected = False
        terrain_cfg.curriculum = False
        return
    if TERRAIN_PRESET not in TERRAIN_PRESETS:
        choices = ", ".join(("mixed", *TERRAIN_PRESETS))
        raise ValueError(
            f"Unknown TERRAIN_PRESET={TERRAIN_PRESET!r}; choose one of: {choices}"
        )

    terrain_cfg.selected = True
    terrain_cfg.curriculum = False
    # Terrain.selected_terrain() pops "type", so give it a fresh dictionary.
    terrain_cfg.terrain_kwargs = dict(TERRAIN_PRESETS[TERRAIN_PRESET])


def play(args):
    if args.load_run is None:
        raise SystemExit("--load_run RUN is required (use --ckpt N, or -1 for latest)")
    if SIMULATOR in ("genesis", "genesis_pact", "genesis_pact_pos"):
        import genesis as gs

        init_genesis(args, gs)

    env_cfg, train_cfg = task_registry.get_cfgs(args.task, args)
    env_cfg.env.num_envs = 20
    env_cfg.viewer.rendered_envs_idx = list(range(10))
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 2.0
    env_cfg.commands.ranges.lin_vel_x = [-1.0, 1.0]
    env_cfg.commands.ranges.lin_vel_y = [-0.5, 0.5]
    env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]
    # Ten terrain cells keep visible startup light while retaining the task's
    # normal terrain generator and reward/observation behavior.
    if env_cfg.terrain.mesh_type in ("heightfield", "trimesh"):
        env_cfg.terrain.num_rows = 2
        env_cfg.terrain.num_cols = 5
        env_cfg.terrain.num_subterrains = 10
        _configure_play_terrain(env_cfg.terrain)

    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env, name=args.task, args=args, log_root=None
    )
    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", "hardpact_iclr",
        train_cfg.runner.experiment_name,
    )
    checkpoint = get_load_path(
        log_root, load_run=args.load_run, checkpoint=args.ckpt
    )
    runner.load(checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    obs, obs_history, _, _ = env.get_observations()
    print(f"Playing {checkpoint} with 10 robots; velocity commands resample every 2 s.")

    # Roughly ten episodes; Ctrl-C can be used to switch checkpoints sooner.
    # The environment override is useful for automated smoke checks.
    play_steps = int(os.getenv(
        "HARD_PACT_PLAY_STEPS", 10 * env.max_episode_length
    ))
    with torch.inference_mode():
        for _ in range(play_steps):
            actions = policy(obs, obs_history)
            obs, _, obs_history, _, _, _, _, _ = env.step(actions)


if __name__ == "__main__":
    play(get_args())
