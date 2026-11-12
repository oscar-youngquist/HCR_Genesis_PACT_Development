"""Independent Isaac Lab replay parity; no learning, checkpoint, or FACET teacher.

Run: python tests/smoke_b1z1_facet.py --gpu cuda:1 --output-dir /tmp/facet-smoke
Each process owns one simulator. Compare actual inputs/properties, not just seeds.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def record(args):
    os.environ["SIMULATOR"] = "isaaclab_b1z1_pact"
    import torch
    import legged_gym.envs
    from legged_gym.utils import get_args, task_registry
    from legged_gym.envs.b1z1.b1z1_unifp_original.b1z1_unifp_original_config import copy_sections
    from legged_gym.utils.helpers import class_to_dict

    sys.argv = [sys.argv[0], "--task=" + args.record_task, "--gpu=" + args.gpu,
                "--headless", "--num_envs=2"]
    cli = get_args()
    cfg = copy_sections(task_registry.env_cfgs[args.record_task])
    if args.record_task == "b1z1_facet":
        cfg.facet_reference.enabled = not args.references_disabled
        cfg.facet_tracking.enabled = args.tracking_enabled
        cfg.facet_observations.enabled = not args.observations_disabled
    train = copy_sections(task_registry.train_cfgs[args.record_task])
    cfg.terrain.mesh_type = "plane"
    cfg.terrain.curriculum = False
    cfg.noise.add_noise = False
    # Test-only explicit matched mechanics. The production configs retain all
    # baseline randomization. Non-unit gain/motor samples are installed below.
    for key in dir(cfg.domain_rand):
        value = getattr(cfg.domain_rand, key)
        if isinstance(value, bool) and (key.startswith("randomize_") or
                                        "curriculum" in key or key == "push_robots"):
            setattr(cfg.domain_rand, key, False)
    cfg.domain_rand.randomize_ctrl_delay = True
    cfg.domain_rand.ctrl_delay_step_range = [0, 2]
    cfg.control.randomize_pact_weights = False
    cfg.compliance.compile_geometry = False  # No compiler warmup in this tiny test.
    cfg.env.num_envs = 2
    env, _ = task_registry.make_env(args.record_task, args=cli, env_cfg=cfg)
    sim = env.simulator
    output = {}
    def save(name, value):
        output[name] = value.detach().cpu().numpy().copy()
    try:
        if args.record_task == "b1z1_facet":
            # Exercise the real registered runner/model, without optimizer steps.
            train.runner.num_steps_per_env = 2
            runner, _ = task_registry.make_alg_runner(env, name=args.record_task,
                args=cli, train_cfg=train, log_root=None)
            assert runner.dynamics is None
            obs, history, _, _ = env.get_observations()
            with torch.no_grad():
                action = runner.get_inference_policy()(obs, history)
            assert action.shape == (2, 17) and torch.isfinite(action).all()
            facet_model = None
            if cfg.facet_observations.enabled:
                from rsl_rl.facet_model import FACETModel
                packet = env.get_facet_observations()
                # Test model initialization must not perturb task/reset random draws.
                with torch.random.fork_rng(devices=[torch.device(env.device).index]):
                    facet_model = FACETModel(packet['observation'].shape[-1], packet['privileged'].shape[-1]).to(env.device).eval()
                facet_hidden = torch.zeros(2, 128, device=env.device)
        env.reset_idx(env.all_env_ids)
        ids = env.all_env_ids
        q = sim.default_dof_pos.expand(2, -1).clone()
        qd = torch.zeros_like(q)
        root = sim.base_init_pos[None].repeat(2, 1) + sim._env_origins
        quat = root.new_tensor([0., 0., 0., 1.]).expand(2, -1)
        sim.reset_dofs(ids, q, qd)
        sim.reset_root_states(ids, root, quat, torch.zeros_like(root), torch.zeros_like(root))
        sim._robot.update(0.)
        sim.post_physics_step()
        if getattr(env, "facet_reference", None) is not None:
            env._reset_facet_reference(env.all_env_ids)  # Fixture teleported the measured state.
        sim._kp_scale[:] = root.new_tensor([.9, 1.1])[:, None]
        sim._kd_scale[:] = root.new_tensor([1.1, .95])[:, None]
        sim._motor_strength[:] = root.new_tensor([.8, 1.2])[:, None]
        sim.feedback_tau_weight.fill_(1.)
        env.action_queue.zero_()
        env.action_delay[:] = torch.tensor([0, 2], device=env.device)
        env.commands[:, :3] = root.new_tensor([.1, 0., 0.])
        sphere = root.new_tensor([.5, .1, 0.])
        env.ee_start_sphere[:] = sphere
        env.ee_goal_sphere[:] = sphere
        env.goal_timer.zero_()
        env.traj_timesteps.fill_(100.)
        env.update_curr_ee_goal()
        # Record effective initial states and articulation mechanics explicitly.
        for name, value in dict(initial_q=sim.dof_pos, initial_qd=sim.dof_vel,
                initial_root=sim._robot.data.root_link_state_w,
                masses=sim._robot.root_physx_view.get_masses(),
                inertias=sim._robot.root_physx_view.get_inertias(),
                coms=sim._robot.root_physx_view.get_coms(),
                materials=sim._robot.root_physx_view.get_material_properties(),
                kp=sim._kp_scale, kd=sim._kd_scale, motor=sim._motor_strength,
                p_gains=sim._p_gains, d_gains=sim._d_gains, limits=sim.torque_limits,
                armature=sim._joint_armature, friction=sim._joint_friction,
                damping=sim._joint_damping, stiffness=sim._joint_stiffness,
                delay=env.action_delay, branch_pd=sim.feedback_tau_weight,
                branch_ff=sim.feedforward_tau_weight).items():
            save(name, value)
        load = torch.zeros(2, 3, device=env.device)
        torso, moment = torch.zeros_like(load), torch.zeros_like(load)
        def ee_event(ids):
            env.ee_force_ext_world.copy_(load)
            sim.apply_ee_force(load)
        def torso_event(ids):
            env.base_force_ext_world.copy_(torso)
            env.base_torque_ext_world.copy_(moment)
            sim.apply_base_force(torso)
            sim.apply_base_torque(moment)
        # Bypass stochastic event draws only in this fixture; forces still enter
        # the production simulator application path once at every physics substep.
        env._push_gripper, env._push_robot_base = ee_event, torso_event
        traces = {key: [] for key in ("actions", "commands", "ee_reference", "q", "qd",
            "root", "torques", "pd", "ff", "external_force", "external_moment", "position_target",
            "obs", "privileged", "history", "explicit", "reward")}
        reference_traces = []
        tracking_traces = []
        nominal_traces = []
        for step in range(args.steps):
            load[:] = load.new_tensor([5., -3., -10.]) if 1 <= step < args.steps - 1 else 0.
            torso[:] = torso.new_tensor([-2., 4., 0.]) if step >= 2 else 0.
            moment[:] = moment.new_tensor([0., 0., .5]) if step >= 2 else 0.
            actions = torch.sin(torch.arange(34, device=env.device).reshape(2, 17) + step) * .04
            commands, reference = env.commands.clone(), env.curr_ee_goal_cart_world.clone()
            if getattr(env, "facet_reference", None) is not None:
                start_origin = env._facet_measured_state()[0]
                start_point = sim.applied_torso_wrench_source()[2]
                # PhysX link/root buffers can differ at float32 roundoff. Test
                # the actual lever convention, not a presumed exactly-zero lever.
                expected_yaw_moment = (moment + torch.linalg.cross(start_point-start_origin, torso))[:, 2].clone()
            result = env.step(actions)
            assert not result[5].any(), "Unexpected reset in short parity interval"
            assert all(torch.isfinite(x).all() for x in result[:6])
            values = dict(actions=actions, commands=commands, ee_reference=reference,
                q=sim.dof_pos, qd=sim.dof_vel, root=sim._robot.data.root_link_state_w,
                torques=sim.interval_executed_torque, pd=sim.combined_feedback_torques,
                ff=sim.combined_feedforward_torques, external_force=sim._external_force_world,
                external_moment=sim._external_torque_world, position_target=env.dof_tracking_target,
                obs=result[0], privileged=result[1], history=result[2], explicit=result[3], reward=result[4])
            if getattr(env, "facet_reference", None) is not None:
                virtual = env.get_facet_references()
                assert all(torch.isfinite(v).all() and not v.requires_grad for v in virtual.values())
                torch.testing.assert_close(virtual["elapsed_seconds"], torch.full_like(virtual["elapsed_seconds"], (step+1)*env.dt))
                torch.testing.assert_close(virtual["applied_ee_force_world"], load)
                torch.testing.assert_close(virtual["applied_base_force_world"], torso)
                torch.testing.assert_close(virtual["applied_yaw_moment"], expected_yaw_moment)
                reference_traces.append({key: value.cpu().numpy() for key, value in virtual.items()})
            if args.record_task == "b1z1_facet":
                nominal_traces.append(torch.stack([env.episode_sums[key] for key in
                    ("tracking_lin_vel_force_world", "tracking_ang_vel", "tracking_ee_force_world")], -1).cpu().numpy().copy())
                if facet_model is not None:
                    packet = env.get_facet_observations()
                    assert all(torch.isfinite(v).all() for v in packet.values())
                    with torch.no_grad():
                        student, facet_hidden, reconstructed = facet_model.student_action(packet['observation'], facet_hidden, packet['is_init'])
                        teacher = facet_model.teacher_action(packet['observation'], packet['privileged'], packet['wrench'])
                        value = facet_model.value(packet['observation'], packet['privileged'], packet['wrench'])
                    assert student.shape == teacher.shape == (2, 17)
                    assert reconstructed.shape == (2, 9) and value.shape == (2, 1)
                    assert all(torch.isfinite(v).all() for v in (student, teacher, reconstructed, value))
                    assert env._facet_observations.actions.shape == (2, 3, 17)
                    torch.testing.assert_close(env._facet_observations.actions[:, 0], actions)
            if args.tracking_enabled:
                snapshot = env.get_facet_tracking()
                expected_valid = torch.tensor([step+1 >= h for h in cfg.facet_tracking.horizons], device=env.device).expand(2, -1)
                assert torch.equal(snapshot["valid"], expected_valid)
                if step < min(cfg.facet_tracking.horizons)-1:
                    assert snapshot["rewards_unweighted"].count_nonzero() == 0
                tracking_traces.append({key: value.cpu().numpy() for key, value in snapshot.items()})
            assert values["ff"].count_nonzero() == 0
            for key, value in values.items():
                traces[key].append(value.detach().cpu().numpy().copy())
        if getattr(env, "facet_reference", None) is not None:
            if args.tracking_enabled:
                # Exercise the parent's real timeout/reward/reset order after a
                # mature bank. Terminal reward must retain valid pre-reset targets.
                env.episode_length_buf[0] = env.max_episode_length
                terminal_result = env.step(torch.zeros(2, 17, device=env.device))
                assert terminal_result[5][0] and not terminal_result[5][1]
                terminal = env.get_facet_tracking()
                assert terminal["valid"].all()
                assert (env.facet_history._ages[0] == -1).all()
                if facet_model is not None:
                    packet = env.get_facet_observations()
                    assert packet['is_init'][0] and not packet['is_init'][1]
                    assert packet['privileged'][0, 13:].count_nonzero() == 0
                    assert env._facet_observations.actions[0].count_nonzero() == 0
                for name in ("tracking_lin_vel_force_world", "tracking_ang_vel", "tracking_ee_force_world"):
                    assert "rew_"+name in env.extras["episode"]
                assert "facet/h32/ee_position_mse" in env.extras["episode"]
                env.step(torch.zeros(2, 17, device=env.device))
                assert not env.get_facet_tracking()["valid"][0].any()
                assert env.get_facet_tracking()["valid"][1].all()
                # Test-only pose injection exercises physical roll termination,
                # without relaxing or replacing the production failure criteria.
                roll = cfg.termination.roll_threshold + .3
                tilted = root.new_tensor([np.sin(roll/2), 0., 0., np.cos(roll/2)])[None]
                sim.reset_root_states(ids[1:], sim.base_pos[1:].clone(), tilted,
                                      torch.zeros_like(root[1:]), torch.zeros_like(root[1:]))
                physical_result = env.step(torch.zeros(2, 17, device=env.device))
                assert physical_result[5][1] and not env.time_out_buf[1]
                assert env.extras["episode"]["roll_termination_rate"] > 0
                assert env.get_facet_tracking()["valid"][1].all()
                assert (env.facet_history._ages[1] == -1).all()
            before = env.get_facet_references()
            env.reset_idx(ids[:1])
            after = env.get_facet_references()
            for key in before:
                torch.testing.assert_close(before[key][1:], after[key][1:], rtol=0, atol=0)
            measured = env._facet_measured_state()
            for key, value in zip(("base_position_xy", "base_velocity_xy", "yaw", "yaw_rate",
                                   "ee_position_world", "ee_velocity_world"), measured):
                expected = value[:1, :2] if key.endswith("_xy") else value[:1]
                torch.testing.assert_close(after[key][:1], expected)
            assert after["elapsed_seconds"][0] == 0
        output.update({key: np.stack(values) for key, values in traces.items()})
        name = args.record_task + ("_disabled" if args.references_disabled else "") + ("_tracking" if args.tracking_enabled else "") + ("_no_packets" if args.observations_disabled else "")
        if reference_traces:
            np.savez_compressed(args.output_dir / "facet_references.npz",
                **{key: np.stack([row[key] for row in reference_traces]) for key in reference_traces[0]})
        if tracking_traces:
            np.savez_compressed(args.output_dir / "facet_tracking.npz",
                **{key: np.stack([row[key] for row in tracking_traces]) for key in tracking_traces[0]})
        if nominal_traces:
            np.savez_compressed(args.output_dir / (name+"_nominal.npz"), sums=np.stack(nominal_traces))
        np.savez_compressed(args.output_dir / (name + ".npz"), **output)
        (args.output_dir / (name + "_config.json")).write_text(
            json.dumps(class_to_dict(cfg), default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x), indent=2))
        print("FACET_RECORD_OK", args.record_task, flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        sim._app_launcher.app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/facet-smoke"))
    parser.add_argument("--record-task", choices=("b1z1_ppo_pos", "b1z1_facet"))
    parser.add_argument("--references-disabled", action="store_true")
    parser.add_argument("--tracking-enabled", action="store_true")
    parser.add_argument("--observations-disabled", action="store_true")
    args = parser.parse_args()
    if (args.tracking_enabled or args.record_task is None) and args.steps < 33:
        parser.error("Stage 3 smoke requires at least 33 uninterrupted intervals")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.record_task:
        record(args)
        return
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, SIMULATOR="isaaclab_b1z1_pact", PYTHONPATH=str(root))
    for task, disabled, tracking, no_packets in (("b1z1_ppo_pos", False, False, False), ("b1z1_facet", True, False, False),
                                     ("b1z1_facet", False, False, False), ("b1z1_facet", False, True, False),
                                     ("b1z1_facet", False, True, True)):
        command = [sys.executable, str(Path(__file__).resolve()), "--record-task", task,
                   "--gpu", args.gpu, "--steps", str(args.steps),
                   "--output-dir", str(args.output_dir.resolve())]
        if disabled:
            command.append("--references-disabled")
        if tracking:
            command.append("--tracking-enabled")
        if no_packets:
            command.append("--observations-disabled")
        name = task + ("_disabled" if disabled else "") + ("_tracking" if tracking else "") + ("_no_packets" if no_packets else "")
        with (args.output_dir / (name + ".log")).open("w") as log:
            subprocess.run(command, cwd=root, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=300)
        if not (args.output_dir / (name + ".npz")).exists():
            raise RuntimeError(f"Simulator did not finish recording; inspect {args.output_dir / (name + '.log')}")
    errors = {}
    with np.load(args.output_dir / "b1z1_ppo_pos.npz") as baseline, np.load(args.output_dir / "b1z1_facet.npz") as facet:
        assert set(baseline.files) == set(facet.files)
        for key in baseline.files:
            if key in {"obs", "privileged", "history", "explicit", "reward"}:
                continue  # Stage 1 intentionally removed baseline compliance observations/rewards.
            # Independent GPU scene solves may differ slightly; inputs and
            # sampled effective properties must match exactly, not merely seed.
            dynamic = key in {"q", "qd", "root", "torques", "pd", "ee_reference"}
            np.testing.assert_allclose(facet[key], baseline[key], rtol=1e-4 if dynamic else 0,
                                       atol=1e-4 if dynamic else 0, err_msg=key)
            errors[key] = float(np.max(np.abs(facet[key] - baseline[key])))
        assert np.max(np.abs(facet["external_force"])) > 0
        assert np.max(np.abs(facet["external_moment"])) > 0
    with np.load(args.output_dir / "b1z1_facet_disabled.npz") as disabled, np.load(args.output_dir / "b1z1_facet.npz") as enabled:
        for key in disabled.files:
            np.testing.assert_array_equal(disabled[key], enabled[key], err_msg=key)
            errors["reference_toggle/" + key] = 0.
    with np.load(args.output_dir / "b1z1_facet.npz") as stage2, np.load(args.output_dir / "b1z1_facet_tracking.npz") as stage3:
        for key in stage2.files:
            if key == "reward":
                assert not np.array_equal(stage2[key], stage3[key])
            else:
                np.testing.assert_array_equal(stage2[key], stage3[key], err_msg=key)
                errors["tracking_toggle/"+key] = 0.
    with np.load(args.output_dir / "b1z1_facet_nominal.npz") as stage2, np.load(args.output_dir / "b1z1_facet_tracking_nominal.npz") as stage3:
        np.testing.assert_array_equal(stage2["sums"], stage3["sums"])
        errors["nominal_metric_sums"] = 0.
    with np.load(args.output_dir / "b1z1_facet_tracking.npz") as enabled, np.load(args.output_dir / "b1z1_facet_tracking_no_packets.npz") as disabled:
        for key in enabled.files:
            np.testing.assert_array_equal(enabled[key], disabled[key], err_msg=key)
            errors['observation_api_toggle/'+key] = 0.
    (args.output_dir / "parity.json").write_text(json.dumps(errors, indent=2))
    print("FACET Stage 1/2/3/4 physics/observation/nominal-metric parity passed; Stage 3 rewards differ:", args.output_dir)


if __name__ == "__main__":
    main()
