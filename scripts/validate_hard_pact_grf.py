"""Small real-PhysX GRF validation, optionally followed by exactly one PPO update.

Uses nominal PD standing, not an untrained policy, for force-balance checks.
Run each task in a fresh process; Kit owns process-global simulator state.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(args):
    os.environ["SIMULATOR"] = "isaaclab"
    from legged_gym.scripts.train_hard_pact import prepare_solver_runtime
    prepare_solver_runtime(["--qp_solver", "cupiqp"])
    import torch
    import legged_gym.envs
    from legged_gym.utils import task_registry
    from legged_gym.utils.helpers import class_to_dict
    cfg = copy.deepcopy(task_registry.env_cfgs[args.task])
    cfg.env.num_envs = 2
    cfg.terrain.mesh_type = "plane"
    cfg.terrain.curriculum = cfg.commands.curriculum = False
    cfg.noise.add_noise = False
    # Isolate force accounting from impulses, resets and randomized mechanics.
    for key in dir(cfg.domain_rand):
        if key.startswith(("randomize_", "use_", "push_", "wrench_", "vert_")) and isinstance(getattr(cfg.domain_rand, key), bool):
            setattr(cfg.domain_rand, key, False)
    cfg.terrain.static_friction = cfg.terrain.dynamic_friction = 0. if args.frictionless else 1.
    env = task_registry.get_task_class(args.task)(cfg, class_to_dict(cfg.sim), args.gpu, True)
    sim = env.simulator
    app = sim._app_launcher.app
    try:
        from legged_gym.simulator.ground_contact_forces import momentum_residual
        # Legacy terrain diagnostics index a height grid even on a plane.
        # A zero grid represents this validator's flat plane exactly.
        sim._height_samples = torch.zeros(2, 2, device=env.device)
        robot = sim._robot
        ids = torch.arange(env.num_envs, device="cpu")
        materials = robot.root_physx_view.get_material_properties().clone()
        materials[..., :2] = cfg.terrain.static_friction
        robot.root_physx_view.set_material_properties(materials, ids)
        mass = robot.root_physx_view.get_masses().to(env.device)
        gravity = torch.tensor([0., 0., -9.81], device=env.device)
        wrench = torch.zeros(env.num_envs, 6, device=env.device)
        actions = torch.zeros_like(env.actions)
        actions = env._pre_sim_step(actions)  # Initialize the real clip/delay/gain contract before substeps.
        # Keep nominal PD and no feedforward; no policy inference needed for this physical check.
        pre, post = sim._hard_pact_pre_physics_substep, sim._hard_pact_grf_post_physics_substep
        def before():
            pre()
            sim.hard_pact_apply_base_wrench_world(wrench)
        normal, tangent, total, timings = [], [], [], []
        update = sim._update_ground_contacts
        def timed_update():
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            update()
            end.record()
            timings.append((start, end))
        def after():
            post()
            normal.append(sim.ground_normal_forces_world.sum(1).clone())
            tangent.append(sim.ground_tangential_forces_world.sum(1).clone())
            total.append(sim.ground_total_forces_world.sum(1).clone())
        sim._hard_pact_pre_physics_substep = before
        sim._hard_pact_grf_post_physics_substep = after
        sim._update_ground_contacts = timed_update
        def momentum():
            return (robot.data.body_com_vel_w[..., :3] * mass[..., None]).sum(1).clone()
        # Settle before measuring; direct simulator steps avoid episode-reset momentum jumps.
        for _ in range(50):
            sim.step(actions)
        rows = []
        for name, load in (("standing", [0., 0., 0.]), ("downward", [0., 0., -25.]),
                           ("horizontal", [-25., 0., 0.])):
            wrench[:, :3] = torch.tensor(load, device=env.device)
            normal.clear(); tangent.clear(); total.clear(); timings.clear()
            start = momentum()
            for _ in range(args.steps):
                sim.step(actions)
            end = momentum()
            n, t, f = torch.stack(normal), torch.stack(tangent), torch.stack(total)
            torch.testing.assert_close(f, n+t, atol=1e-3, rtol=1e-5)
            torch._assert_async(torch.isfinite(f).all(), "Nonfinite GRF")
            if args.frictionless:
                torch.testing.assert_close(t, torch.zeros_like(t), atol=1e-4, rtol=0)
            residual = momentum_residual(start, end, mass.sum(1), gravity, f,
                                         wrench[None, :, :3].expand_as(f), sim._sim_params["dt"])
            relative = residual.norm(dim=-1) / (mass.sum(1)*9.81*len(f)*sim._sim_params["dt"])
            if relative.max() > .02:
                raise AssertionError(f"Momentum residual exceeds 2% gravitational impulse: {relative}")
            torch.cuda.synchronize()
            rows.append(dict(stage=name, applied_force_world_n=load,
                normal_mean_n=n.mean((0,1)).tolist(), tangential_mean_n=t.mean((0,1)).tolist(),
                total_mean_n=f.mean((0,1)).tolist(), momentum_residual_ns=residual.tolist(),
                relative_momentum_residual=relative.tolist(),
                contact_query_reduce_ms=sum(a.elapsed_time(b) for a,b in timings)/len(timings)))
        other = sim.ground_total_forces_world[1].clone()
        env.reset_idx(torch.tensor([0], device=env.device))
        assert not sim.ground_total_forces_world[0].any()
        torch.testing.assert_close(sim.ground_total_forces_world[1], other)
        sim._hard_pact_pre_physics_substep, sim._hard_pact_grf_post_physics_substep = pre, post
        sim._update_ground_contacts = update
        env.reset()
        updates = 0
        if args.training_update:
            from rsl_rl.utils.runner_registry import runner_registry
            train = copy.deepcopy(task_registry.train_cfgs[args.task])
            train.runner.num_steps_per_env = 4
            train.runner.resume = False
            train.policy.pretrained_path = ""
            train.algorithm.num_learning_epochs = train.algorithm.num_mini_batches = 1
            if hasattr(train.algorithm, "ppo_qp_shard_percentage"):
                train.algorithm.ppo_qp_shard_percentage = 100.
                train.algorithm.ppo_qp_passes_per_iteration = 1
            if hasattr(train.algorithm, "bard_batch_capacity"):
                train.algorithm.bard_batch_capacity = 8
            runner = runner_registry.get_runner_class(train.runner_class_name)(
                env, class_to_dict(train), str(args.output_dir / "training"), device=args.gpu)
            runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
            assert all(torch.isfinite(p).all() for p in runner.alg.actor_critic.parameters())
            updates = 1
        report = dict(task=args.task, frictionless=args.frictionless, stages=rows,
            grf_semantics=sim.grf_semantics, partial_reset=True, training_updates=updates,
            note="Nominal PD physical validation, not trained-policy performance; includes all ground contacts.")
        (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2)+"\n")
        (args.output_dir / "failure.txt").unlink(missing_ok=True)
    except BaseException:
        (args.output_dir / "failure.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="go2_hard_pact_full_isaaclab")
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--frictionless", action="store_true")
    p.add_argument("--training-update", action="store_true")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if not 2 <= args.steps <= 100:
        p.error("steps must be 2..100; this is a bounded smoke test")
    if args.worker:
        try:
            main(args)
        except BaseException:
            (args.output_dir / "failure.txt").write_text(traceback.format_exc())
            traceback.print_exc()
            raise
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if (args.output_dir / "validation.json").exists():
            p.error("Choose a new output directory; completed reports are never overwritten")
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"])
        # Kit may swallow an exception during shutdown; require the success artifact too.
        sys.exit(0 if child.returncode == 0 and (args.output_dir / "validation.json").exists() else 1)
