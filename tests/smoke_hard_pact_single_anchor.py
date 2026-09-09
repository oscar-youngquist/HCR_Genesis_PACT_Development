"""Opt-in real simulator smoke (not collected by pytest); no benchmark.

Run in the simulator environment with the normal training CLI arguments.
Only this process overrides warmup, rollout length, and terrain grid size.
"""
import json
import tempfile
from pathlib import Path
import traceback

from legged_gym.scripts.train_hard_pact import prepare_solver_runtime
import sys
prepare_solver_runtime(sys.argv[1:])

import torch
from legged_gym import SIMULATOR
from legged_gym.envs import *  # noqa: F403 -- register real tasks
from legged_gym.utils import get_args, task_registry


def main():
    args = get_args()
    assert SIMULATOR == "isaaclab" and torch.cuda.is_available()
    cfg, train = task_registry.get_cfgs(args.task, args)
    cfg.terrain.num_rows = 2
    cfg.terrain.num_cols = 2
    cfg.terrain.max_init_terrain_level = 1
    train.algorithm.hard_pact_qp["warmup_iterations"] = 0
    train.algorithm.hard_pact_qp["qp_update_mode"] = "single_anchor_held_correction"
    train.runner.num_steps_per_env = 4
    train.algorithm.num_mini_batches = 1
    # Keep the usual five epochs, 20% disjoint QP shards, and all objectives.
    train.policy.pinn_init_steps = -1
    train.policy.pinn_loss_weight = -1.0
    train.runner.resume = False
    if train.policy.pretrained_path:
        train.policy.pretrained_path = str((
            Path(__file__).resolve().parents[1] / "legged_gym/scripts"
            / train.policy.pretrained_path
        ).resolve())
    env, _ = task_registry.make_env(args.task, args, env_cfg=cfg)
    app = env.simulator._app_launcher.app
    try:
        runner, _ = task_registry.make_alg_runner(
            env, args.task, args, train_cfg=train,
            log_root=tempfile.mkdtemp(prefix="hard_pact_single_anchor_"),
        )
        qp = runner.alg.hard_pact_qp
        counts = {"anchor": 0, "held": 0, "ppo": 0}
        finite_checks = []
        bound_checks = []
        gradient_checks = []
        solve = qp.solve

        def checked_solve(**kwargs):
            result = solve(**kwargs)
            phase = kwargs.get("diagnostics_phase")
            if phase == "ppo":
                counts["ppo"] += 1
                if result.tau_safe.requires_grad:
                    result.tau_safe.register_hook(
                        lambda grad: gradient_checks.append((torch.isfinite(grad).all(), grad.abs().sum()))
                    )
            else:
                counts["anchor"] += 1
            finite_checks.append(torch.isfinite(result.tau_safe).all())
            return result

        qp.solve = checked_solve
        callback = env._solve_hard_pact_rollout_qp_substep

        def checked_callback(*values, **kwargs):
            k = env._qp_substep
            previous = env._hard_pact_previous_substep_torque.clone()
            callback(*values, **kwargs)
            counts["held"] += int(k != 0)
            executed = env._hard_pact_previous_substep_torque
            rate = qp.cfg.torque_rate_limit_nm_s * float(env.cfg.sim.dt)
            bound_checks.append(((executed.abs() <= qp.torque_limits + 1e-6).all()
                                 & ((executed - previous).abs() <= rate + 1e-5).all()))
            assert torch.count_nonzero(env._qp_sampled_substep_index) == 0

        env._solve_hard_pact_rollout_qp_substep = checked_callback
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        assert counts["anchor"] == 4, counts
        assert counts["held"] == 12, counts
        assert counts["ppo"] == 5, counts
        assert torch.stack(finite_checks + bound_checks).all()
        assert gradient_checks, "No differentiable QP backward reached the solver output"
        assert all(bool(finite) for finite, _ in gradient_checks)
        assert sum(float(norm) for _, norm in gradient_checks) > 0
        for value in env.get_observations():
            assert torch.isfinite(value).all()
        print("SINGLE_ANCHOR_SMOKE_PASS " + json.dumps({
            **counts, "device": str(env.device), "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__, "backward_calls": len(gradient_checks),
            "log_dir": runner.log_dir,
        }), flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
