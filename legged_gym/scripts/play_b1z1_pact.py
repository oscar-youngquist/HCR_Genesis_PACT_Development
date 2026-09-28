"""IsaacLab PACT disturbance-rejection playback; no force-shifted commands."""
import argparse
import sys
import numpy as np
import torch


def configure_play(cfg, options):
    cfg.env.num_envs = 1
    cfg.use_force_shifted_target = False
    cfg.use_force_compensation = False
    cfg.commands.use_external_impedance_compensation = False
    cfg.commands.curriculum = False  # Never expand the trained command bounds in play.
    if options.use_joystick:
        cfg.commands.heading_command = False
    cfg.commands.apply_ee_external_forces = options.ee_forces
    cfg.commands.apply_base_external_forces = options.base_forces
    cfg.commands.apply_base_external_torques = options.base_forces and cfg.commands.apply_base_external_torques
    cfg.commands.push_gripper_stators = options.ee_forces
    cfg.commands.push_robot_base = options.base_forces
    cfg.env.render_ee_goal_debug = False  # Dedicated nominal-only visualization below.
    cfg.env.render_ee_frame_debug = False


def joystick_command(axis, bounds):
    """Map each joystick half to the corresponding trained bound, including asymmetry."""
    axis = float(np.clip(axis, -1., 1.))
    return float(np.clip(axis * (bounds[1] if axis >= 0 else -bounds[0]), *bounds))


def force_arrow(origin, force, scale):
    """World-frame shaft and arrowhead; length in metres per applied newton."""
    origin, force = np.asarray(origin), np.asarray(force)
    length = np.linalg.norm(force) * scale
    if not np.isfinite(length) or length < 1e-8:
        return [], []
    direction = force / np.linalg.norm(force)
    tip = origin + direction * length
    axis = np.eye(3)[np.argmin(np.abs(direction))]
    side = np.cross(direction, axis)
    side /= np.linalg.norm(side)
    head = min(.06, length * .25)
    return [origin.tolist(), tip.tolist(), tip.tolist()], [
        tip.tolist(), (tip-head*direction+head*.5*side).tolist(),
        (tip-head*direction-head*.5*side).tolist()]


class ForceVisualizer:
    def __init__(self, scale):
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.util.debug_draw")
        from isaacsim.util.debug_draw import _debug_draw
        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.scale = scale

    def update(self, env):
        self.draw.clear_points()
        self.draw.clear_lines()
        ee = env.simulator.ee_pos[0].detach().cpu().numpy()
        target = env.curr_ee_goal_cart_world[0].detach().cpu().numpy()
        self.draw.draw_points([target.tolist(), ee.tolist()],
                              [(1.,1.,0.,1.), (0.,.4,1.,1.)], [14.,12.])
        # Applied disturbance, not sampled peak, estimated force, or target offset.
        force = env.ee_force_ext_world[0].detach().cpu().numpy()
        starts, ends = force_arrow(ee, force, self.scale)
        if starts:
            self.draw.draw_lines(starts, ends, [(1.,.2,.1,1.)]*len(starts), [3.]*len(starts))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ee-forces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-forces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-arrows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-arrow-scale", type=float, default=.005, help="Metres per newton")
    parser.add_argument("--play-steps", type=int, default=10000)
    options, remaining = parser.parse_known_args()
    if options.force_arrow_scale <= 0 or options.play_steps < 1:
        parser.error("Arrow scale and play steps must be positive")
    sys.argv = [sys.argv[0], *remaining]
    from legged_gym import SIMULATOR
    import legged_gym.envs
    from legged_gym.utils import get_args, task_registry
    args = get_args()
    if SIMULATOR != "isaaclab_b1z1_pact" or args.task != "b1z1_pact":
        raise ValueError("Use SIMULATOR=isaaclab_b1z1_pact and --task=b1z1_pact")
    options.use_joystick = args.use_joystick
    cfg, train = task_registry.get_cfgs(args.task, args)
    if args.ckpt < -1:
        parser.error("--ckpt must be -1 (latest) or a nonnegative checkpoint number")
    # The shared config updater ignores --ckpt; apply it locally for playback.
    train.runner.checkpoint = args.ckpt
    configure_play(cfg, options)
    args.num_envs = 1
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    train.runner.resume = True
    runner, _ = task_registry.make_alg_runner(env, args.task, args=args, train_cfg=train)
    policy = runner.get_inference_policy(device=env.device)
    # Evaluate enabled disturbances at their trained final bounds, not warmup zero.
    curriculum = env._staged_force_curriculum
    curriculum.gate_latched = True
    curriculum.trigger_iteration = env.training_iteration - curriculum.external_ramp
    joystick = None
    if args.use_joystick:
        from legged_gym.scripts.joystick import Joystick
        joystick = Joystick(joystick_type=args.joystick_type)
    visualizer = ForceVisualizer(options.force_arrow_scale) if options.force_arrows and not args.headless else None
    print("PACT rejection: yellow=nominal EE target, blue=EE, red arrow=applied EE force.")
    print("Base command bounds:", env.command_ranges)
    print("External disturbances use final configured bounds; EE target is never force-shifted.")
    try:
        with torch.inference_mode():
            env.reset()
            obs, history, _, _ = env.get_observations()
            for _ in range(options.play_steps):
                if joystick is not None:
                    joystick.update()
                    for i, (axis, name) in enumerate(zip((-joystick.ly,-joystick.lx,-joystick.rx),
                            ("lin_vel_x", "lin_vel_y", "ang_vel_yaw"))):
                        env.commands[:,i] = joystick_command(axis, env.command_ranges[name])
                    # Refresh command channels without advancing the observation history twice.
                    cmd = env.commands * env.commands_scale
                    obs[:,-cmd.shape[-1]:] = cmd
                    history[:,-cmd.shape[-1]:] = cmd
                actions = policy(obs, history)
                obs, _, history, _, _, _, _, _ = env.step(actions)
                if args.follow_robot and not args.headless:
                    base = env.simulator.base_pos[0].detach().cpu().numpy()
                    env.set_viewer_camera(base + np.asarray(cfg.viewer.pos),
                                          base + np.asarray(cfg.viewer.lookat))
                if visualizer is not None:
                    visualizer.update(env)
    finally:
        runner.dynamics.close()


if __name__ == "__main__":
    main()
