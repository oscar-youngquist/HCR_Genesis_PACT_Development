import io
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

os.environ.setdefault("SIMULATOR", "genesis_pact")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_hard_pact_action_replay")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_hard_pact_action_replay")

import torch

from rsl_rl.algorithms.ppo_hard_pact import PPO_HardPACT
from rsl_rl.modules.actor_critic_hard_pact import (
    ActorCritic_HardPACT,
    ContextDecoder,
)
from rsl_rl.storage.rollout_storage_pact import RolloutStoragePACT
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import GO2HardPACTCfg
from legged_gym.envs.go2.go2_hard_pact.deployment import calculate_physics_head_gains


def make_algorithm(action_clip=1.0):
    torch.manual_seed(31)
    gains = calculate_physics_head_gains(GO2HardPACTCfg())
    actor = ActorCritic_HardPACT(
        num_actor_obs=57, num_critic_obs=95, num_actions=12,
        actor_layers=[32, 16], critic_layers=[32, 16],
        cenet_in_dim=57 * 20, cenet_enc_layers=[32, 16],
        cenet_explicit_layers=[16, 16],
        grf_decoder_layers=[16, 16], wrench_decoder_layers=[16, 16],
        grf_scale_n=gains.grf_scale_n,
        wrench_scale=gains.wrench_scale_n_nm,
        wrench_qp_clip=gains.wrench_qp_clip_n_nm,
    )
    decoder = ContextDecoder(input_dim=27, layers=[32, 24, 16], decode_dim=133)
    with redirect_stdout(io.StringIO()):
        algorithm = PPO_HardPACT(
            actor, decoder, 181, bard_enabled=False,
            use_adaptive_entropy=False, action_clip=action_clip,
        )
    algorithm.use_boot = True
    return algorithm


def action_transform(actions):
    return actions[:, :12] * 0.5, actions[:, 12:] * 2.0


def feedback(desired, position, velocity):
    return 3.0 * (desired - position) - 0.2 * velocity


class CompactStorageReplayTests(unittest.TestCase):
    def make_storage(self):
        storage = RolloutStoragePACT(
            2, 4, [1], [1], [1], [2], [24], [1], [1], [1], "cpu"
        )
        storage.configure_action_replay(2)
        for timestep in range(4):
            for environment in range(2):
                value = 10.0 * timestep + environment
                storage.observations[timestep, environment] = value
                storage.observation_history[timestep, environment] = value + 100.0
                storage.action_noise[timestep, environment] = value + 200.0
        storage._action_replay_boundary_observations[:, :, 0] = torch.tensor(
            [[-20.0, -19.0], [-10.0, -9.0]]
        )
        storage._action_replay_boundary_history[:, :, :] = (
            storage._action_replay_boundary_observations + 100.0
        )
        storage._action_replay_boundary_noise[:, :, :] = (
            storage._action_replay_boundary_observations + 200.0
        )
        return storage

    def test_delay_zero_maximum_and_rollout_boundary_sources(self):
        storage = self.make_storage()
        # flat indices are (t=0,e=0), (t=1,e=1), and (t=3,e=0).
        indices = torch.tensor([0, 3, 6])
        delay = torch.tensor([0, 2, 2])
        observation, history, noise = storage._action_replay_sources(
            indices, delay
        )
        torch.testing.assert_close(
            observation[:, 0], torch.tensor([0.0, -9.0, 10.0])
        )
        torch.testing.assert_close(history[:, 0], observation[:, 0] + 100.0)
        torch.testing.assert_close(noise[:, 0], observation[:, 0] + 200.0)

    def test_only_noise_and_small_boundary_cache_add_persistent_vram(self):
        storage = self.make_storage()
        # Source observations/history are indexed from existing rollout
        # tensors. Persistent additions are one noise rollout plus D boundary
        # rows; there is no [T,E,history] replay duplicate.
        self.assertEqual(storage.action_noise.shape, storage.actions.shape)
        self.assertEqual(storage._action_replay_boundary_history.shape, (2, 2, 2))
        replay_tensors = {
            name for name, value in vars(storage).items()
            if "action_replay" in name and torch.is_tensor(value)
        }
        self.assertEqual(replay_tensors, {
            "_action_replay_boundary_observations",
            "_action_replay_boundary_history",
            "_action_replay_boundary_noise",
        })
        self.assertFalse(hasattr(storage, "context_latent_noise"))

    def test_clear_preserves_only_required_cross_rollout_sources(self):
        storage = self.make_storage()
        expected_obs = storage.observations[-2:].clone()
        expected_history = storage.observation_history[-2:].clone()
        expected_noise = storage.action_noise[-2:].clone()
        storage.clear()
        torch.testing.assert_close(
            storage._action_replay_boundary_observations, expected_obs
        )
        torch.testing.assert_close(
            storage._action_replay_boundary_history, expected_history
        )
        torch.testing.assert_close(
            storage._action_replay_boundary_noise, expected_noise
        )


class StochasticActionReplayTests(unittest.TestCase):
    def replay_inputs(self, algorithm, batch=3):
        observation = torch.randn(batch, 57)
        history = torch.randn(batch, 57 * 20)
        algorithm.actor_critic.act(observation, history)
        mean = algorithm.actor_critic.action_mean
        noise = torch.randn(batch, 24)
        raw = mean.detach() + algorithm.actor_critic.std.detach() * noise
        transition = {
            "standardized_action_noise": noise.clone(),
            "delayed_source_observation": observation.clone(),
            "delayed_source_history": history.clone(),
            "delayed_source_noise": noise.clone(),
            "delayed_action_source_valid": torch.tensor(
                [[True], [True], [False]]
            )[:batch],
            "sampled_action_delay": torch.tensor([[0], [2], [2]])[:batch],
            "raw_sampled_action": raw.clone(),
        }
        return observation, history, mean, raw, transition

    def test_frozen_policy_exact_raw_delayed_action_and_torque(self):
        algorithm = make_algorithm(action_clip=1.0)
        observation, _, mean, raw, transition = self.replay_inputs(algorithm)
        result = algorithm._replay_action_path(
            mean, observation, transition, action_transform, feedback,
            torch.zeros(12), 1.0,
        )
        expected_delayed = torch.clamp(raw, -1.0, 1.0)
        expected_delayed[2] = 0.0  # reset/boundary queue entry
        expected_torque = algorithm._nominal_torque(
            expected_delayed, observation, action_transform, feedback,
            torch.zeros(12), 1.0,
        )
        transition["delayed_action"] = expected_delayed
        transition["nominal_torque"] = expected_torque
        torch.testing.assert_close(
            result["raw_action"], transition["raw_sampled_action"]
        )
        torch.testing.assert_close(
            result["delayed_action"], transition["delayed_action"]
        )
        torch.testing.assert_close(
            result["nominal_torque"], transition["nominal_torque"]
        )

    def test_replayed_stochastic_torque_routes_actor_encoder_and_noise_scale_gradients(self):
        algorithm = make_algorithm(action_clip=10.0)
        observation, _, mean, _, transition = self.replay_inputs(algorithm, batch=2)
        result = algorithm._replay_action_path(
            mean, observation, transition, action_transform, feedback,
            torch.zeros(12), 1.0,
        )
        result["nominal_torque"].square().mean().backward()
        modules = (
            algorithm.actor_critic.act_trunk,
            algorithm.actor_critic.context_encoder,
        )
        for module in modules:
            gradient = sum(
                parameter.grad.abs().sum().item()
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0)
        self.assertIsNotNone(algorithm.actor_critic.std.grad)
        self.assertGreater(algorithm.actor_critic.std.grad.abs().sum().item(), 0.0)

    def test_frozen_policy_sampled_substep_nominal_torque_matches_rollout_formula(self):
        algorithm = make_algorithm(action_clip=1.0)
        observation, _, mean, _, transition = self.replay_inputs(algorithm)
        replay = algorithm._replay_action_path(
            mean, observation, transition, action_transform, feedback,
            torch.zeros(12), 1.0,
        )
        sampled_q = torch.randn(3, 12)
        sampled_qdot = torch.randn(3, 12)
        replayed = replay["feedforward_torque"] + feedback(
            replay["desired_position"], sampled_q, sampled_qdot
        )
        rollout = replay["feedforward_torque"].detach() + feedback(
            replay["desired_position"].detach(), sampled_q, sampled_qdot
        )
        torch.testing.assert_close(replayed, rollout)


class EnvironmentActionCaptureTests(unittest.TestCase):
    def test_legacy_queue_delay_boundaries_and_nominal_torque_are_captured(self):
        task = Go2HardPACT.__new__(Go2HardPACT)
        task.cfg = GO2HardPACTCfg()
        task.device = "cpu"
        task.num_envs = 2
        task.num_actions = 12
        task.actions = torch.zeros(2, 24)
        task.last_actions = torch.zeros_like(task.actions)
        task.llast_actions = torch.zeros_like(task.actions)
        task.action_queue = torch.zeros(2, 3, 24)
        task.action_delay = torch.tensor([0, 2])
        task._action_replay_valid_queue = torch.zeros(2, 3, dtype=torch.bool)
        task.simulator = SimpleNamespace(
            default_dof_pos=torch.linspace(-0.2, 0.2, 12),
            _dof_pos=torch.linspace(-0.1, 0.1, 12).repeat(2, 1),
            _dof_vel=torch.linspace(-0.3, 0.3, 12).repeat(2, 1),
            _torques=torch.linspace(-2.0, 2.0, 12).repeat(2, 1),
        )
        task._get_pinn_feedback = feedback

        first = torch.linspace(-0.8, 0.8, 48).reshape(2, 24)
        first_delayed = task._pre_sim_step(first)
        self.assertTrue(task._pending_action_replay_transition[
            "delayed_action_source_valid"
        ][0])
        self.assertFalse(task._pending_action_replay_transition[
            "delayed_action_source_valid"
        ][1])
        torch.testing.assert_close(first_delayed[0], first[0])
        torch.testing.assert_close(first_delayed[1], torch.zeros(24))
        torch.testing.assert_close(
            task._pending_action_replay_transition[
                "previous_executed_torque"
            ],
            task.simulator._torques,
        )

        task._pre_sim_step(first + 0.1)
        third_delayed = task._pre_sim_step(first + 0.2)
        self.assertTrue(task._pending_action_replay_transition[
            "delayed_action_source_valid"
        ].all())
        torch.testing.assert_close(third_delayed[1], first[1])
        expected_torque = task._pending_action_replay_transition[
            "nominal_torque"
        ]
        desired, feedforward_torque = task._get_pinn_actions(third_delayed)
        torch.testing.assert_close(
            expected_torque,
            feedforward_torque + feedback(
                desired, task.simulator._dof_pos, task.simulator._dof_vel
            ),
        )


if __name__ == "__main__":
    unittest.main()
