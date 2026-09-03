"""Fast deterministic coverage for HardPACT ablation selection and logging."""

import dataclasses
import os
import pathlib
import subprocess
import tempfile
import unittest
from types import SimpleNamespace

os.environ.setdefault("SIMULATOR", "genesis_pact")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_hard_pact_ablation_tests")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_hard_pact_ablation_tests")

import torch

import legged_gym.envs  # noqa: F401
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.utils.task_registry import task_registry
from rsl_rl.algorithms.hard_pact_qp import HardPACTDifferentiableQP, HardPACTQPConfig
from rsl_rl.algorithms.ppo_hard_pact import PPO_HardPACT
from rsl_rl.hard_pact_ablations import (
    HARD_PACT_ABLATIONS, HARD_PACT_BACKENDS, HARD_PACT_VARIANTS,
    hard_pact_task_names, resolve_hard_pact_features,
)
from rsl_rl.hard_pact_logging import (
    STABLE_HARD_PACT_SCALARS, collect_hard_pact_scalars,
)
from rsl_rl.modules.actor_critic_hard_pact import ActorCritic_HardPACT
from rsl_rl.runners.pact_runner import OnPolicyRunnerPACT


EXPECTED = {
    "baseline": (0, 0, 0, 0, 0, 0),
    "soft": (1, 1, 0, 0, 0, 0),
    "hard": (0, 0, 1, 1, 1, 0),
    "full": (1, 1, 1, 1, 1, 0),
    "stopgrad": (1, 1, 1, 0, 1, 0),
    "soft_penalty": (0, 0, 0, 0, 0, 1),
    "inverse": (1, 0, 0, 0, 0, 0),
    "rollout": (0, 1, 0, 0, 0, 0),
}


def qp_data(dtype=torch.float64):
    z = lambda *shape: torch.zeros(*shape, dtype=dtype)
    return dict(
        mass_matrix=torch.eye(18, dtype=dtype).unsqueeze(0), bias=z(1, 18),
        foot_jacobians=z(1, 4, 3, 18), base_jacobian=z(1, 6, 18),
        foot_acceleration_bias=z(1, 4, 3),
        tau_nom=torch.full((1, 12), 0.2, dtype=dtype, requires_grad=True),
        force_pred_world=z(1, 4, 3).requires_grad_(),
        wrench_pred_world=z(1, 6).requires_grad_(),
        contact_probability=torch.full((1, 4), 0.5, dtype=dtype, requires_grad=True),
        previous_torque=z(1, 12), joint_position=z(1, 12),
        joint_velocity=z(1, 12), dt=torch.full((1, 1), 0.02, dtype=dtype),
    )


def make_qp():
    return HardPACTDifferentiableQP(
        HardPACTQPConfig(max_iter=50, not_improved_limit=10),
        torch.full((12,), 23.5), torch.full((12,), -2.0),
        torch.full((12,), 2.0), torch.full((12,), 30.0),
    )


class HardPACTAblationTests(unittest.TestCase):
    def test_exact_immutable_matrix(self):
        self.assertEqual(tuple(EXPECTED), HARD_PACT_VARIANTS)
        for name, expected in EXPECTED.items():
            spec = HARD_PACT_ABLATIONS[name]
            actual = tuple(int(getattr(spec, field)) for field in (
                "inverse_loss", "rollout_loss", "execution_qp",
                "qp_gradient", "projection_metric", "soft_constraint_penalty",
            ))
            self.assertEqual(actual, expected)
            self.assertIs(resolve_hard_pact_features(name), spec)
            with self.assertRaises(dataclasses.FrozenInstanceError):
                spec.inverse_loss = False
        with self.assertRaises(TypeError):
            HARD_PACT_ABLATIONS["new"] = HARD_PACT_ABLATIONS["full"]

    def test_all_registrations_share_class_and_exact_profiles(self):
        self.assertEqual(len(hard_pact_task_names()), 27)
        for variant in HARD_PACT_VARIANTS:
            for backend in HARD_PACT_BACKENDS:
                name = f"go2_hard_pact_{variant}_{backend}"
                self.assertIs(task_registry.task_classes[name], Go2HardPACT)
                self.assertEqual(task_registry.env_cfgs[name].ablation_variant, variant)
                self.assertEqual(task_registry.train_cfgs[name].algorithm.ablation_variant, variant)
                self.assertEqual(task_registry.train_cfgs[name].runner.task_backend, backend)
        for backend in HARD_PACT_BACKENDS:
            alias = f"go2_hard_pact_{backend}"
            self.assertEqual(task_registry.env_cfgs[alias].ablation_variant, "full")

    def test_dimensions_architecture_and_state_dict_keys_match(self):
        reference_cfg = task_registry.env_cfgs["go2_hard_pact_full_genesis"]
        reference_train = task_registry.train_cfgs["go2_hard_pact_full_genesis"]
        dims = (
            reference_cfg.env.num_observations, reference_cfg.env.num_privileged_obs,
            reference_cfg.env.num_actions, reference_cfg.env.num_obs_hist,
        )
        shape_signature = None
        for variant in HARD_PACT_VARIANTS:
            env_cfg = task_registry.env_cfgs[f"go2_hard_pact_{variant}_genesis"]
            train_cfg = task_registry.train_cfgs[f"go2_hard_pact_{variant}_genesis"]
            self.assertEqual((env_cfg.env.num_observations, env_cfg.env.num_privileged_obs,
                              env_cfg.env.num_actions, env_cfg.env.num_obs_hist), dims)
            policy = train_cfg.policy
            torch.manual_seed(7)
            actor = ActorCritic_HardPACT(
                num_actor_obs=env_cfg.env.num_observations,
                num_critic_obs=env_cfg.env.num_privileged_obs * env_cfg.env.num_priv_stack,
                num_actions=env_cfg.env.num_actions, actor_layers=policy.actor_layers,
                critic_layers=policy.critic_layers,
                cenet_in_dim=env_cfg.env.num_observations * env_cfg.env.num_obs_hist,
                cenet_latent_dim=policy.cenet_enc_latent_dim,
                cenet_velo_dim=policy.cenet_velo_dim,
                cenet_enc_layers=policy.cenet_enc_layers,
                activation=policy.activation, init_noise_std=policy.init_noise_std,
                cenet_explicit_layers=policy.cenet_explicit_layers,
                grf_decoder_layers=policy.grf_decoder_layers,
                wrench_decoder_layers=policy.wrench_decoder_layers,
                ablation_features=variant,
            )
            signature = tuple((key, tuple(value.shape)) for key, value in actor.state_dict().items())
            shape_signature = signature if shape_signature is None else shape_signature
            self.assertEqual(signature, shape_signature)
            self.assertIs(actor.hard_pact_features, HARD_PACT_ABLATIONS[variant])

    def test_disabled_profile_elides_every_expensive_path(self):
        algorithm = PPO_HardPACT.__new__(PPO_HardPACT)
        algorithm.hard_pact_features = HARD_PACT_ABLATIONS["baseline"]
        algorithm.bard_enabled = False
        nominal = torch.ones(2, 12, requires_grad=True)
        loss = algorithm._compute_bard_loss(
            nominal, None, None, None, None
        )
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(nominal.grad, torch.zeros_like(nominal))

    def test_stopgrad_and_full_qp_forward_match_but_gradients_differ(self):
        full_data, stop_data = qp_data(), qp_data()
        full = make_qp().solve(differentiable=True, **full_data)
        stop = make_qp().solve(differentiable=False, **stop_data)
        for a, b in ((full.qdd, stop.qdd), (full.force_world, stop.force_world),
                     (full.tau_safe, stop.tau_safe),
                     (full.contact_slack, stop.contact_slack)):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
        full.tau_safe.square().sum().backward()
        self.assertIsNotNone(full_data["tau_nom"].grad)
        self.assertGreater(full_data["tau_nom"].grad.abs().sum().item(), 0.0)
        self.assertFalse(stop.tau_safe.requires_grad)
        self.assertIsNone(stop_data["tau_nom"].grad)

    def test_stable_logging_keys_and_disabled_semantics(self):
        algorithm = SimpleNamespace(
            last_physics_loss_metrics={}, last_inverse_dynamics_metrics={},
            last_rollout_dynamics_metrics={}, last_auxiliary_metrics={},
            last_qp_metrics={}, last_physics_gradient_metrics={},
        )
        key_sets = []
        for variant in HARD_PACT_VARIANTS:
            values = collect_hard_pact_scalars(algorithm, variant)
            key_sets.append(set(values))
            self.assertEqual(values["qp/enabled"], float(EXPECTED[variant][2]))
            self.assertEqual(values["physics/inverse/enabled"], float(EXPECTED[variant][0]))
        self.assertTrue(all(keys == key_sets[0] for keys in key_sets))
        self.assertTrue(set(STABLE_HARD_PACT_SCALARS) <= key_sets[0])

    def test_rollout_qp_logging_is_scalar_aggregated_on_device(self):
        runner = OnPolicyRunnerPACT.__new__(OnPolicyRunnerPACT)
        runner._rollout_qp_metric_sums = {}
        runner._rollout_qp_metric_count = 0
        runner._rollout_disturbance_active_sum = torch.zeros(())
        runner._rollout_disturbance_metric_count = 0
        interval = {
            "interval_qp_stage_fractions": torch.tensor([[1., 0., 0.], [0., 1., 0.]]),
            "interval_qp_correction": torch.tensor([[1., -2.], [3., -4.]]),
            "interval_qp_contact_slack": torch.tensor([[0., 2.], [1., 3.]]),
            "interval_qp_residuals": torch.tensor([[0.1, 0.2, 0., 0.], [0.3, 0.4, 0., 0.]]),
            "interval_qp_timing_ms": torch.tensor([[2.], [4.]]),
        }
        runner._accumulate_rollout_qp_metrics({
            "hard_pact_qp_interval": interval,
            "hard_pact_transition": {
                "sustained_wrench_active_mask": torch.tensor([[True], [False]])
            },
        })
        self.assertEqual(runner._rollout_qp_metric_count, 1)
        self.assertTrue(all(value.ndim == 0 for value in runner._rollout_qp_metric_sums.values()))
        torch.testing.assert_close(
            runner._rollout_qp_metric_sums["qp/minimal/full_fraction"],
            torch.tensor(0.5),
        )
        torch.testing.assert_close(
            runner._rollout_qp_metric_sums[
                "qp/minimal/normalized_inequality_violation_max"
            ], torch.tensor(0.4),
        )
        torch.testing.assert_close(
            runner._rollout_disturbance_active_sum, torch.tensor(0.5)
        )

    def test_launcher_validates_and_forwards_without_injected_defaults(self):
        launcher = (pathlib.Path(__file__).parents[1] / "legged_gym" /
                    "scripts" / "train_go2_hard_pact.sh")
        with tempfile.TemporaryDirectory() as directory:
            fake = pathlib.Path(directory) / "python"
            output = pathlib.Path(directory) / "args"
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$SIMULATOR" "$@" > "$HARDPACT_TEST_OUTPUT"\n')
            fake.chmod(0o755)
            env = dict(os.environ, PATH=directory + os.pathsep + os.environ["PATH"],
                       HARDPACT_TEST_OUTPUT=str(output),
                       HARD_PACT_SKIP_CONDA_ACTIVATE="1")
            result = subprocess.run(
                [str(launcher), "--task", "go2_hard_pact_full_genesis",
                 "--seed", "19", "--num_envs=7"], env=env,
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = output.read_text().splitlines()
            self.assertEqual(lines[0], "genesis_pact")
            self.assertEqual(lines[-5:], ["--task", "go2_hard_pact_full_genesis",
                                          "--seed", "19", "--num_envs=7"])
            bad = subprocess.run([str(launcher), "--task", "go2"], env=env,
                                 text=True, capture_output=True)
            self.assertEqual(bad.returncode, 2)
            self.assertIn("Available HardPACT tasks:", bad.stderr)


if __name__ == "__main__":
    unittest.main()
