"""Single-robot visual Exp2 playback; frozen policies, no training or recordings."""
import argparse
import copy
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("exp2_visual_protocol", HERE / "eval_b1z1_exp2.py")
exp2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp2)
exp1 = exp2.exp1
ALIASES = {"pact": "b1z1_pact", "pilot": "b1z1_pact", "pos": "b1z1_ppo_pos",
           "unifp": "b1z1_unifp_original", "unifp-reject": "b1z1_unifp_reject"}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", "--approach", default="b1z1_pact",
                   help="Registered B1Z1 task/ablation, or pact/pilot/pos/unifp/unifp-reject")
    p.add_argument("--profile", choices=("pushing", "pulling", "carrying"), default="pushing")
    p.add_argument("--run", required=True, help="Existing run directory/resolver selection")
    p.add_argument("--ckpt", default="-1", help="Checkpoint selection; -1 selects latest")
    p.add_argument("--loading", choices=("loaded", "unloaded"), default="loaded")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--trial-start", type=int, default=0)
    p.add_argument("--horizon-seconds", type=float, default=exp2.EVAL_CONFIG["finish_s"])
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--headless", action="store_true", help="For tiny smoke checks only; disables visualization")
    p.add_argument("--max-steps", type=int, help="Optional total control-step cap for smoke checks")
    p.add_argument("--speed", type=float, default=1., help="Playback speed relative to simulated time")
    p.add_argument("--follow-robot", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force-arrow-scale", type=float, default=.005, help="Arrow length in metres per applied newton")
    return p


def validate(args):
    args.task = ALIASES.get(args.task.lower(), args.task)
    if not args.task.startswith("b1z1_"):
        raise ValueError("Select a registered B1Z1 task or an approach alias")
    if args.episodes < 1 or args.seed < 0 or args.trial_start < 0:
        raise ValueError("episodes must be positive; seed/trial-start must be nonnegative")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if any(not math.isfinite(v) or v <= 0 for v in (args.speed, args.force_arrow_scale, args.horizon_seconds)):
        raise ValueError("speed, arrow scale and horizon must be finite and positive")
    args.num_envs = 1  # Never allocate hidden evaluation robots for visual playback.
    exp2.tasks.validate(exp2.EVAL_CONFIG, args.horizon_seconds)


class VisualAdapter(exp2.Adapter):
    # Keep the same control/targets/reset/noise hooks, without recorder substep copies.
    def torque(self, *args, **kwargs):
        return self.compute_torques(*args, **kwargs)

    def contact_update(self, *args, **kwargs):
        return self.sensor_update(*args, **kwargs)


def main(argv=None):
    args = parser().parse_args(argv)
    validate(args)
    config = copy.deepcopy(exp2.EVAL_CONFIG)  # Includes current edited task/load timings.
    os.environ["SIMULATOR"] = exp1.simulator_family(args.task)
    # Select the backend before imports that register task/simulator families.
    import numpy as np
    import torch
    import legged_gym.envs
    from legged_gym.utils import task_registry
    from legged_gym.utils.helpers import class_to_dict

    if args.task not in task_registry.task_classes:
        raise ValueError(f"Unregistered B1Z1 task: {args.task}")
    training = exp1.util.materialize(task_registry.train_cfgs[args.task])
    cfg = exp1.configure(task_registry.env_cfgs[args.task], args, config)
    cfg.viewer.rendered_envs_idx = [0]
    cfg.env.render_ee_goal_debug = cfg.env.render_ee_frame_debug = False
    cfg.env.debug_viz = False
    env_class = task_registry.get_task_class(args.task)
    reject = args.task in ("b1z1_unifp_original", "b1z1_unifp_reject")
    if reject:
        # Original UniFP checkpoint, with the same prediction-only rejection interface as Exp2.
        env_class = task_registry.get_task_class("b1z1_unifp_reject")
        commands = exp1.util.materialize(task_registry.env_cfgs["b1z1_unifp_reject"].commands)
        for key, value in vars(commands).items():
            if not hasattr(cfg.commands, key):
                setattr(cfg.commands, key, value)
        cfg.commands.use_external_impedance_compensation = True
        cfg.commands.compensate_ee_external_force = cfg.commands.compensate_base_external_force = True
    checkpoint = exp1.resolve_checkpoint(args.run, args.ckpt, training.runner.experiment_name)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    env = None
    try:
        with exp1.isolated_rng(args.seed, args.gpu):
            env = env_class(cfg, class_to_dict(cfg.sim), args.gpu, args.headless)
            actor = exp1.build_actor(env, training, state)
        before = exp1.model_hash(actor)
        if reject:
            env._staged_force_curriculum.beta = config["reject_beta"]
        adapter = VisualAdapter(env, config, args.seed)
        app = env.simulator._app_launcher.app
        visualizer = None
        if not args.headless:
            # Reuse play's yellow nominal target, blue measured EE, and red applied-force arrow.
            play = exp2.load_module("exp2_pact_visualizer", HERE / "play_b1z1_pact.py")
            visualizer = play.ForceVisualizer(args.force_arrow_scale)
        print(f"{args.task}: {args.profile}, {args.loading}; checkpoint {checkpoint}", flush=True)
        print("One robot. Yellow=nominal target; blue=EE; red=applied EE force. No data recorded.", flush=True)
        total_steps = 0
        with torch.inference_mode():
            for trial in range(args.trial_start, args.trial_start + args.episodes):
                if not app.is_running():
                    break
                scenario = exp2.tasks.Scenario(args.seed, trial, args.profile, args.loading, config)
                adapter.reset([scenario])
                last_stage = None
                for step in range(math.ceil(args.horizon_seconds / env.dt)):
                    if not app.is_running() or (args.max_steps is not None and total_steps >= args.max_steps):
                        break
                    started = time.perf_counter()
                    adapter.step_index = step
                    obs, history, _, _ = env.get_observations()
                    action = exp1.inference(actor, obs, history)
                    if reject:
                        env.set_impedance_force_estimates(actor.last_prediction.detach())
                    # Interval-start load follows inference; successor target is refreshed by post_control.
                    adapter.loads(step * env.dt)
                    env.step(action)
                    total_steps += 1
                    if not args.headless and (args.follow_robot or step == 0):
                        base = exp1.numpy_value(env.simulator.base_pos[0])
                        env.set_viewer_camera(base + np.asarray(cfg.viewer.pos), base + np.asarray(cfg.viewer.lookat))
                    if visualizer is not None:
                        visualizer.update(env)
                    stage = scenario.stage((step + 1) * env.dt)
                    if stage != last_stage:
                        print(f"Trial {trial}: t={(step+1)*env.dt:.2f}s, {exp2.tasks.STAGES[stage]}", flush=True)
                        last_stage = stage
                    failure = adapter.failures()[0]
                    if failure:
                        print(f"Trial {trial} ended: {failure}", flush=True)
                        break
                    if not args.headless:
                        time.sleep(max(0., env.dt / args.speed - (time.perf_counter() - started)))
                if args.max_steps is not None and total_steps >= args.max_steps:
                    break
        if exp1.model_hash(actor) != before:
            raise RuntimeError("Playback changed policy weights")
        print(f"Playback complete: {total_steps} control steps; frozen weights verified.", flush=True)
        marker = os.environ.get("B1Z1_EXP2_PLAY_SUCCESS")
        if marker:
            Path(marker).write_text("complete\n")
    except Exception:
        # Kit may terminate during close; print the original error before shutdown.
        import traceback
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.simulator._app_launcher.app.close()


if __name__ == "__main__":
    if os.environ.get("B1Z1_EXP2_PLAY_WORKER") == "1":
        main()
    else:
        validate(parser().parse_args())
        # Kit shutdown may exit(0) during an exception; require an explicit success marker.
        with tempfile.TemporaryDirectory(prefix="b1z1-exp2-play-") as temporary:
            marker = Path(temporary) / "success"
            result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:]],
                env={**os.environ, "B1Z1_EXP2_PLAY_WORKER": "1", "B1Z1_EXP2_PLAY_SUCCESS": str(marker)})
            raise SystemExit(result.returncode or (0 if marker.exists() else 1))
