"""Focused tests for Isaac Gym's sensor-backed foot GRF interface."""

import os
from types import SimpleNamespace
import unittest

# Importing the simulator initializes Isaac Gym before PyTorch, as required by
# Isaac Gym's Python bindings.
from legged_gym.simulator.isaacgym_simulator import (
    reshape_foot_force_sensor_tensor,
)
import torch


class IsaacGymFootForceSensorTests(unittest.TestCase):
    def test_force_sensor_tensor_reshape_preserves_actor_and_foot_order(self):
        num_envs, num_feet = 3, 4
        raw = torch.arange(num_envs * num_feet * 6, dtype=torch.float32).reshape(-1, 6)

        wrenches = reshape_foot_force_sensor_tensor(raw, num_envs, num_feet)

        self.assertEqual(wrenches.shape, (num_envs, num_feet, 6))
        for env_id in range(num_envs):
            for foot_id in range(num_feet):
                sensor_id = env_id * num_feet + foot_id
                self.assertTrue(torch.equal(wrenches[env_id, foot_id], raw[sensor_id]))

    def test_force_sensor_tensor_reshape_rejects_wrong_sensor_count(self):
        with self.assertRaisesRegex(AssertionError, "Expected force-sensor tensor"):
            reshape_foot_force_sensor_tensor(
                torch.zeros(7, 6), num_envs=2, num_feet=4
            )

    @unittest.skipUnless(
        os.environ.get("RUN_ISAACGYM_SENSOR_TESTS") == "1",
        "set RUN_ISAACGYM_SENSOR_TESTS=1 to launch the PhysX integration test",
    )
    def test_b1z1_force_sensors_report_physical_grfs(self):
        """Validate ordering, aerial zero, stance support, and weight consistency."""
        self.assertEqual(os.environ.get("SIMULATOR"), "isaacgym_b1z1_unifp")

        # Import after SIMULATOR is selected, matching the training entrypoint.
        from legged_gym.envs import task_registry

        args = SimpleNamespace(
            task="b1z1_unifp",
            headless=True,
            cpu=False,
            gpu="cuda:0",
            num_envs=1,
            max_iterations=None,
            resume=False,
            sync_wandb=False,
            export_onnx=False,
            debug=False,
            load_run=None,
            ckpt=-1,
            use_joystick=False,
            joystick_type="xbox",
            follow_robot=False,
            record_frames=False,
            seed=1,
            pinn_loss_weight=0.0,
        )
        env_cfg, _ = task_registry.get_cfgs(name=args.task, args=args)
        env_cfg.env.num_envs = 1
        env_cfg.terrain.mesh_type = "plane"
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.measure_heights = False
        env_cfg.terrain.obtain_terrain_info_around_feet = False
        env_cfg.commands.curriculum = False
        env_cfg.commands.push_gripper_stators = False
        env_cfg.commands.push_robot_base = False
        env_cfg.commands.apply_ee_external_forces = False
        env_cfg.commands.apply_base_external_forces = False
        env_cfg.noise.add_noise = False
        env_cfg.asset.fix_base_link = False

        for name in (
            "randomize_friction",
            "randomize_base_mass",
            "randomize_com_displacement",
            "randomize_pd_gain",
            "randomize_motor_strength",
            "randomize_joint_armature",
            "randomize_joint_friction",
            "randomize_joint_stiffness",
            "randomize_joint_damping",
            "randomize_control_delay",
            "push_robots",
        ):
            if hasattr(env_cfg.domain_rand, name):
                setattr(env_cfg.domain_rand, name, False)

        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        simulator = env.simulator
        actions = torch.zeros(1, env.num_actions, device=env.device)
        env_ids = torch.arange(1, dtype=torch.long, device=env.device)
        num_feet = len(env_cfg.asset.foot_name)

        self.assertEqual(
            simulator._foot_force_sensor_names, tuple(env_cfg.asset.foot_name)
        )
        self.assertEqual(simulator.foot_force_wrenches.shape, (1, num_feet, 6))
        self.assertEqual(simulator._grfs_buf.shape, (1, num_feet * 3))
        body_props = simulator._gym.get_actor_rigid_body_properties(
            simulator._envs[0], simulator._actor_handles[0]
        )
        expected_weight = sum(prop.mass for prop in body_props) * 9.81

        # Raising the robot makes every foot aerial. Since forward-dynamics
        # terms are disabled, contact-only sensor forces should approach zero.
        raised_pos = simulator.base_pos.clone()
        raised_pos[:, 2] += 0.5
        simulator.reset_root_states(
            env_ids,
            raised_pos,
            simulator.base_quat.clone(),
            torch.zeros_like(simulator.base_lin_vel),
            torch.zeros_like(simulator.base_ang_vel),
        )
        simulator.step(actions)
        simulator.post_physics_step()
        self.assertLess(
            torch.linalg.vector_norm(simulator.foot_contact_forces, dim=-1).max().item(),
            0.05 * expected_weight,
        )

        # Restore the nominal free-base pose and average settled support forces.
        nominal_pos = simulator.base_init_pos.unsqueeze(0) + simulator._env_origins[:1]
        simulator.reset_root_states(
            env_ids,
            nominal_pos,
            simulator.base_init_quat.unsqueeze(0),
            torch.zeros(1, 3, device=env.device),
            torch.zeros(1, 3, device=env.device),
        )
        samples = []
        for step in range(250):
            simulator.step(actions)
            simulator.post_physics_step()
            if step >= 150:
                samples.append(simulator.foot_contact_forces.clone())
        mean_forces = torch.stack(samples).mean(dim=0)[0]

        # Settled world-Z support is positive and should sum to robot weight.
        self.assertGreaterEqual((mean_forces[:, 2] > 1.0).sum().item(), 3)
        measured_weight = mean_forces[:, 2].sum().item()
        self.assertAlmostEqual(
            measured_weight,
            expected_weight,
            delta=0.35 * expected_weight,
            msg=(
                f"mean foot forces={mean_forces.tolist()}, "
                f"base_pos={simulator.base_pos[0].tolist()}, "
                f"base_euler={simulator.base_euler[0].tolist()}"
            ),
        )

        self.assertTrue(
            torch.allclose(
                simulator._grfs_buf.view(1, num_feet, 3),
                simulator.foot_contact_forces,
            )
        )
        self.assertTrue(
            torch.allclose(
                simulator.link_contact_forces[:, simulator.feet_indices],
                simulator.foot_contact_forces,
            )
        )


if __name__ == "__main__":
    unittest.main()
