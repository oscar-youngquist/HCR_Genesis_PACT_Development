import io
import os
import unittest
from unittest.mock import Mock, patch
from contextlib import redirect_stdout
from types import SimpleNamespace

os.environ.setdefault("SIMULATOR", "genesis_pact")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_hard_pact_action_replay")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_hard_pact_action_replay")

import torch

from rsl_rl.algorithms.ppo_hard_pact import PPO_HardPACT
from rsl_rl.modules.hard_pact_control import bounded_nominal_torque
from rsl_rl.modules.hard_pact_physics import GRFSwingConfig
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
        observation, history, noise, latent_noise, boot_mask = storage._action_replay_sources(
            indices, delay
        )
        self.assertIsNone(latent_noise)
        self.assertIsNone(boot_mask)
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

    def test_latent_noise_and_boot_mask_follow_shuffled_sources_and_boundary(self):
        storage = RolloutStoragePACT(
            2, 4, [1], [1], [1], [2], [24], [1], [1], [1], "cpu",
            latent_noise_dim=3,
        )
        storage.configure_action_replay(2)
        row_id = torch.arange(8).reshape(4, 2, 1).float()
        storage.observations.copy_(row_id)
        storage.latent_noise.copy_(row_id + 100)
        storage.latent_boot_mask.copy_(row_id.long().remainder(2).bool())
        storage.clear()
        torch.testing.assert_close(storage._action_replay_boundary_latent_noise, storage.latent_noise[-2:])
        torch.testing.assert_close(storage._action_replay_boundary_boot_mask, storage.latent_boot_mask[-2:])
        # Overwrite current rollout, as training does after update()/clear().
        storage.observations.add_(8)
        storage.latent_noise.add_(8)
        storage.latent_boot_mask.logical_not_()
        storage.hard_pact_fields = {"sampled_action_delay": torch.tensor(
            [[[0], [2]], [[1], [2]], [[2], [0]], [[1], [2]]]
        )}
        for _ in storage.mini_batch_generator(2, 3):
            fields = storage.current_hard_pact_batch
            source = fields["delayed_source_observation"]
            torch.testing.assert_close(fields["delayed_source_latent_noise"], (source + 100).expand(-1, 3))
            expected_boot = source.long().remainder(2).bool() ^ source.ge(8)
            torch.testing.assert_close(fields["delayed_source_boot_mask"], expected_boot)
        self.assertEqual(storage._action_replay_boundary_latent_noise.shape, (2, 2, 3))


class StochasticActionReplayTests(unittest.TestCase):
    def replay_inputs(self, algorithm, batch=3):
        observation = torch.randn(batch, 57)
        history = torch.randn(batch, 57 * 20)
        latent_noise = torch.randn(batch, algorithm.actor_critic.context_encoder.ce_out_mean.out_features)
        boot_mask = torch.ones(batch, 1, dtype=torch.bool)
        algorithm.actor_critic.act(
            observation, history, latent_noise=latent_noise, latent_boot_mask=boot_mask,
        )
        mean = algorithm.actor_critic.action_mean
        noise = torch.randn(batch, 24)
        raw = mean.detach() + algorithm.actor_critic.std.detach() * noise
        transition = {
            "standardized_action_noise": noise.clone(),
            "delayed_source_observation": observation.clone(),
            "delayed_source_history": history.clone(),
            "delayed_source_noise": noise.clone(),
            "delayed_source_latent_noise": latent_noise.clone(),
            "delayed_source_boot_mask": boot_mask.clone(),
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
        algorithm.use_boot = False  # Source boot state, not this flag, wins.
        # Actual actuator parameters and physical state deliberately differ
        # from the noisy observation and the old feedback helper's gains.
        transition.update({
            "control_kp": torch.full((3, 12), 4.),
            "control_kd": torch.full((3, 12), .3),
            "control_motor_strength": torch.tensor([[.8], [1.2], [1.1]]),
            "control_feedback_weight": torch.full((3, 1), .9),
            "control_feedforward_weight": torch.full((3, 1), .7),
            "control_torque_limits": torch.full((3, 12), 2.),
            "pre_q": torch.randn(3, 19), "pre_v": torch.randn(3, 18),
        })
        rng_state = torch.get_rng_state()
        result = algorithm._replay_action_path(
            mean, observation, transition, action_transform, feedback,
            torch.zeros(12), 1.0,
        )
        torch.testing.assert_close(torch.get_rng_state(), rng_state)
        expected_delayed = torch.clamp(raw, -1.0, 1.0)
        expected_delayed[2] = 0.0  # reset/boundary queue entry
        desired, ff = action_transform(expected_delayed)
        expected_torque = (transition["control_motor_strength"] * (
            .9 * (4. * (desired-transition["pre_q"][:, 7:]) - .3*transition["pre_v"][:, 6:])
            + .7 * ff)).clamp(-2., 2.)
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

    def test_missing_source_latent_noise_or_boot_mask_fails_clearly(self):
        algorithm = make_algorithm()
        observation, _, mean, _, transition = self.replay_inputs(algorithm)
        for key in ("delayed_source_latent_noise", "delayed_source_boot_mask"):
            missing = dict(transition)
            missing.pop(key)
            with self.assertRaisesRegex(RuntimeError, "stored latent noise and boot mask"):
                algorithm._replay_action_path(mean, observation, missing,
                    action_transform, feedback, torch.zeros(12), 1.)

    def test_invalid_reset_source_is_zero_without_evaluating_nan_history(self):
        algorithm = make_algorithm()
        observation, _, mean, _, transition = self.replay_inputs(algorithm)
        transition["delayed_source_history"][2] = float("nan")
        result = algorithm._replay_action_path(mean, observation, transition,
            action_transform, feedback, torch.zeros(12), 1.)
        assert result["delayed_action"][2].eq(0).all()
        result["nominal_torque"].square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in algorithm.actor_critic.parameters() if p.grad is not None)

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
        self._check_sampled_substep_nominal_and_grf("every_substep")

    def test_random_substep_replay_holds_initial_grf_conditioning_but_refreshes_sampled_pd(self):
        self._check_sampled_substep_nominal_and_grf("random_one_substep")

    def _check_sampled_substep_nominal_and_grf(self, mode):
        algorithm = make_algorithm(action_clip=1.0)
        observation, _, mean, _, transition = self.replay_inputs(algorithm)
        replay = algorithm._replay_action_path(
            mean, observation, transition, action_transform, feedback,
            torch.zeros(12), 1.0,
        )
        sampled_q = torch.randn(3, 12)
        sampled_qdot = torch.randn(3, 12)
        transition.update({
            "control_kp": torch.full((3, 12), 4.),
            "control_kd": torch.full((3, 12), .3),
            "control_motor_strength": torch.full((3, 1), 1.2),
            "control_feedback_weight": torch.full((3, 1), .9),
            "control_feedforward_weight": torch.full((3, 1), .7),
            "control_torque_limits": torch.full((3, 12), 2.),
            "control_dt": torch.full((3, 1), .02),
            "equivalent_mass_com_wrench_world": torch.zeros(3, 6),
            "interval_executed_torque": torch.zeros(3, 12),
            "pre_q": torch.zeros(3, 19), "pre_v": torch.zeros(3, 18),
        })
        transition["pre_q"][:, 6] = 1.
        transition["sampled_qp_q"] = transition["pre_q"].clone()
        transition["sampled_qp_q"][:, 7:] = sampled_q
        transition["sampled_qp_v"] = transition["pre_v"].clone()
        transition["sampled_qp_v"][:, 6:] = sampled_qdot
        transition["sampled_qp_grf_conditioning_q"] = transition["pre_q"][:,7:].clone()
        transition["sampled_qp_grf_conditioning_v"] = transition["pre_v"][:,6:].clone()
        rollout = bounded_nominal_torque(
            replay["desired_position"].detach(), replay["feedforward_torque"].detach(),
            sampled_q, sampled_qdot, transition)
        algorithm.bard_enabled = True
        algorithm.qp_enabled_at_iteration = lambda: True
        algorithm.storage = SimpleNamespace(
            current_hard_pact_batch=transition, current_batch_indices=torch.arange(3))
        algorithm._materialize_mechanics_cache = lambda **_: SimpleNamespace(
            mass_matrix=torch.eye(18).expand(3, -1, -1), bias=torch.zeros(3, 18),
            foot_jacobians=torch.zeros(3, 4, 3, 18), base_jacobian=torch.zeros(3, 6, 18),
            foot_acceleration_bias=torch.zeros(3, 4, 3))
        # Exercise the actual sampled-QP assembly caller; stop at the solver
        # boundary, whose certified solve/backward is covered by QP tests.
        algorithm.hard_pact_qp = SimpleNamespace(
            cfg=SimpleNamespace(qp_update_mode=mode),
            solve=Mock(side_effect=RuntimeError("captured QP input")))
        heads = algorithm.actor_critic.physics_estimator
        heads.grf_swing = GRFSwingConfig(enabled=True)
        _, _, latent, explicit = algorithm.actor_critic.cenet_enc_forward(torch.randn(3, 1140))
        explicit = explicit.clone()
        explicit[:, 3:7] = torch.tensor([.25, .5, .75, .1])
        with self.assertRaisesRegex(RuntimeError, "captured QP input"):
            algorithm._compute_bard_loss(
                replay["nominal_torque"], observation, torch.randn(3, 1140),
                torch.zeros(3, 12), torch.zeros(12),
                desired_position=replay["desired_position"],
                feedforward_torque=replay["feedforward_torque"], fb_func=feedback,
                qp_rows=torch.arange(3), compute_pinn=False,
                policy_features=(latent, explicit))
        replayed = algorithm.hard_pact_qp.solve.call_args.kwargs["tau_nom"]
        torch.testing.assert_close(replayed, rollout, rtol=0, atol=0)
        qp_forces = algorithm.hard_pact_qp.solve.call_args.kwargs["force_pred_world"]
        grf_torque = (bounded_nominal_torque(
            replay["desired_position"], replay["feedforward_torque"],
            transition["pre_q"][:, 7:], transition["pre_v"][:, 6:], transition,
        ))
        raw = heads.predict_grf(latent, explicit, grf_torque)
        deployment = heads.grf_to_physical(raw).reshape(3, 4, 3)
        torch.testing.assert_close(qp_forces, deployment, rtol=0, atol=0)
        # The shared builder gates the reference; its input remains raw physical GRF.
        torch.testing.assert_close(qp_forces[:, [1, 2]], heads.grf_to_physical(raw).reshape(3, 4, 3)[:, [1, 2]])
        replayed.square().mean().backward()
        self.assertGreater(algorithm.actor_critic.act_tau_out.weight.grad.abs().sum(), 0.)


class EnvironmentActionCaptureTests(unittest.TestCase):
    def test_legacy_queue_delay_boundaries_and_nominal_torque_are_captured(self):
        task = Go2HardPACT.__new__(Go2HardPACT)
        task.cfg = GO2HardPACTCfg()
        task.device = "cpu"
        task.num_envs = 2
        task.num_actions = 12  # Legacy joint count; the action vector is 24-D.
        task.cfg.normalization.clip_actions = .5
        task.actions = torch.zeros(2, 24)
        task.last_actions = torch.zeros_like(task.actions)
        task.llast_actions = torch.zeros_like(task.actions)
        task.action_queue = torch.zeros(2, 3, 24)
        task.action_delay = torch.tensor([0, 2])
        task._action_replay_valid_queue = torch.zeros(2, 3, dtype=torch.bool)
        task._hard_pact_raw_action_queue = torch.zeros(2, 3, 24)
        task.simulator = SimpleNamespace(
            default_dof_pos=torch.linspace(-0.2, 0.2, 12),
            _dof_pos=torch.linspace(-0.1, 0.1, 12).repeat(2, 1),
            _dof_vel=torch.linspace(-0.3, 0.3, 12).repeat(2, 1),
            _torques=torch.linspace(-2.0, 2.0, 12).repeat(2, 1),
            _kp_scale=torch.ones(2, 12), _kd_scale=torch.ones(2, 12),
            _p_gains=torch.full((12,), 3.), _d_gains=torch.full((12,), .2),
            _motor_strength=torch.tensor([[.8], [1.2]]),
            feedforward_tau_weight=.9, feedback_tau_weight=1.1,
            torque_limits=torch.full((12,), 2.),
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
        torch.testing.assert_close(first_delayed[0], first[0].clamp(-.5, .5))
        torch.testing.assert_close(task._hard_pact_raw_delayed_action[0], first[0])
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
        torch.testing.assert_close(third_delayed[1], first[1].clamp(-.5, .5))
        torch.testing.assert_close(task._hard_pact_raw_delayed_action[1], first[1])
        expected_torque = task._pending_action_replay_transition[
            "nominal_torque"
        ]
        desired, feedforward_torque = task._get_pinn_actions(third_delayed)
        torch.testing.assert_close(
            expected_torque,
            bounded_nominal_torque(
                desired, feedforward_torque, task.simulator._dof_pos,
                task.simulator._dof_vel, task._hard_pact_control_parameters,
            ),
        )
        # Reset clears raw reward/delay and executed-rate state for just the
        # affected environments, without retaining a previous episode request.
        untouched = task._hard_pact_raw_action_queue[1].clone()
        task._hard_pact_executed_torque = torch.ones(2, 12)
        with patch.object(task._legacy_task_class, "reset_idx"):
            task.reset_idx(torch.tensor([0]))
        self.assertTrue(task._hard_pact_raw_action_queue[0].eq(0).all())
        self.assertTrue(task._hard_pact_raw_delayed_action[0].eq(0).all())
        self.assertTrue(task._hard_pact_executed_torque[0].eq(0).all())
        torch.testing.assert_close(task._hard_pact_raw_action_queue[1], untouched)


if __name__ == "__main__":
    unittest.main()
