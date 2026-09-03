"""Deterministic compatibility tests for the legacy HardPACT task aliases."""

from __future__ import annotations

import contextlib
import copy
import io
import os
from types import SimpleNamespace
import unittest

os.environ.setdefault("SIMULATOR", "genesis_pact")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_hard_pact_tests")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_hard_pact_tests")

import torch

import legged_gym.envs  # noqa: F401 - populates the task registry
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.envs.go2.go2_hard_pact.transition import DISTURBANCE_CRITIC_DIM
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import (
    GO2HardPACTCfg,
    GO2HardPACTCfgPPO,
)
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos import Go2HardPACTPos
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos_config import (
    GO2HardPACTPosCfg,
    GO2HardPACTPosCfgPPO,
)
from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT
from legged_gym.envs.go2.go2_pact.go2_pact_config import GO2PACTCfg, GO2PACTCfgPPO
from legged_gym.envs.go2.go2_pact_pos.go2_pact_pos import Go2PACTPos
from legged_gym.envs.go2.go2_pact_pos.go2_pact_pos_config import (
    GO2PACTPosCfg,
    GO2PACTPosCfgPPO,
)
from legged_gym.simulator.genesis_simulator_pact import GenesisSimulator_PACT
from legged_gym.simulator.genesis_simulator_pact_pos import GenesisSimulator_PACT_Pos
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.task_registry import task_registry
from rsl_rl.modules import (
    ActorCritic_HardPACT,
    ActorCritic_HardPACT_Pos,
    ActorCritic_PACT,
    ActorCritic_PACT_Pos,
)
from rsl_rl.modules.actor_critic_pact import ContextDecoder as PACTContextDecoder
from rsl_rl.modules.actor_critic_pact_pos import ContextDecoder as PACTPosContextDecoder
from rsl_rl.runners.pact_runner import OnPolicyRunnerPACT
from rsl_rl.runners.pact_pos_runner import (
    OnPolicyRunnerPACTPos,
    build_hard_pact_start_checkpoint,
)


CASES = (
    (
        "go2_hard_pact",
        Go2PACT,
        Go2HardPACT,
        GO2PACTCfg,
        GO2HardPACTCfg,
        GO2PACTCfgPPO,
        GO2HardPACTCfgPPO,
        ActorCritic_PACT,
        24,
    ),
    (
        "go2_hard_pact_pos",
        Go2PACTPos,
        Go2HardPACTPos,
        GO2PACTPosCfg,
        GO2HardPACTPosCfg,
        GO2PACTPosCfgPPO,
        GO2HardPACTPosCfgPPO,
        ActorCritic_PACT_Pos,
        12,
    ),
)


def _actor_kwargs(env_cfg, train_cfg):
    policy = train_cfg.policy
    return dict(
        num_actor_obs=env_cfg.env.num_observations,
        num_critic_obs=env_cfg.env.num_privileged_obs * env_cfg.env.num_priv_stack,
        num_actions=env_cfg.env.num_actions,
        actor_layers=policy.actor_layers,
        critic_layers=policy.critic_layers,
        cenet_in_dim=env_cfg.env.num_observations * env_cfg.env.num_obs_hist,
        cenet_latent_dim=policy.cenet_enc_latent_dim,
        cenet_velo_dim=policy.cenet_velo_dim,
        cenet_enc_layers=policy.cenet_enc_layers,
        activation=policy.activation,
        init_noise_std=policy.init_noise_std,
    )


def _synthetic_task(task_cls, cfg, action_width):
    task = task_cls.__new__(task_cls)
    task.cfg = cfg
    task.device = "cpu"
    task.num_envs = 2
    task.num_actions = cfg.env.num_actions
    task.actions = torch.zeros(2, action_width)
    task.last_actions = torch.full((2, action_width), -1.0)
    task.llast_actions = torch.full((2, action_width), -2.0)
    task.action_queue = torch.zeros(2, 3, action_width)
    task.action_delay = torch.tensor([0, 2])
    task.simulator = SimpleNamespace(default_dof_pos=torch.linspace(-0.2, 0.2, 12))
    return task


def _synthetic_torque_sim(sim_cls, cfg):
    sim = sim_cls.__new__(sim_cls)
    sim._cfg = cfg
    sim._num_envs = 2
    sim._p_gains = torch.linspace(10.0, 21.0, 12)
    sim._d_gains = torch.linspace(0.5, 1.6, 12)
    sim._kp_scale = torch.tensor([[1.0], [0.9]])
    sim._kd_scale = torch.tensor([[1.0], [1.1]])
    sim._default_dof_pos = torch.linspace(-0.25, 0.25, 12)
    sim._dof_pos = torch.stack((torch.zeros(12), torch.linspace(-0.1, 0.1, 12)))
    sim._dof_vel = torch.stack((torch.linspace(-1.0, 1.0, 12), torch.ones(12) * 0.2))
    sim._motor_strength = torch.tensor([[1.0], [0.95]])
    sim.first_loop = True
    sim.feedforward_tau_weight = torch.tensor([[0.4], [0.7]])
    sim.feedback_tau_weight = torch.tensor([[1.6], [1.2]])
    return sim


def _synthetic_domain_sim(sim_cls, cfg):
    sim = sim_cls.__new__(sim_cls)
    sim._cfg = cfg
    sim._print_domain_rand_values = lambda *_: None
    sim.push_warmup_step = 0
    sim.max_mass_bounds, sim.mass_bounds_diff = [1.0, 4.0], 3.0
    sim.com_delta_x_bounds, sim.com_delta_x_diff = [0.01, 0.11], 0.10
    sim.com_delta_y_bounds, sim.com_delta_y_diff = [0.02, 0.12], 0.10
    sim.com_delta_z_bounds, sim.com_delta_z_diff = [0.03, 0.13], 0.10
    sim.push_bounds, sim.push_diff = [0.1, 1.1], 1.0
    sim.angular_push_bounds, sim.angular_push_diff = [0.2, 1.2], 1.0
    sim.vert_bounds, sim.vert_diff = [0.3, 1.3], 1.0
    sim.joint_stiffness_bounds_start = torch.tensor([0.0, 0.01])
    sim.joint_stiffness_range = torch.tensor([0.1, 0.1])
    sim.joint_damping_bounds_start = torch.tensor([0.2, 0.3])
    sim.joint_damping_range = torch.tensor([0.2, 0.2])
    sim.joint_friction_bounds_start = torch.tensor([0.0, 0.02])
    sim.joint_friction_range = torch.tensor([0.04, 0.04])
    sim.com_rand_z_positive = cfg.domain_rand.com_rand_z_positive
    sim._init_domain_rand_curriculum_state()
    return sim


class TestHardPACTAliases(unittest.TestCase):
    def test_hard_pact_and_pos_observation_contracts_align(self):
        hard = GO2HardPACTCfg()
        pos = GO2HardPACTPosCfg()
        for field in (
            "num_observations",
            "num_privileged_obs",
            "num_priv_stack",
            "num_obs_hist",
            "num_explicit_recon_obs",
        ):
            self.assertEqual(getattr(hard.env, field), getattr(pos.env, field))

        task = Go2HardPACTPos.__new__(Go2HardPACTPos)
        task.cfg = pos
        task.obs_buf = torch.zeros(2, pos.env.num_observations)
        task.obs_history = torch.zeros(
            2, pos.env.num_observations * pos.env.num_obs_hist
        )
        task.explicit_labels_buf = torch.zeros(
            2, pos.env.num_explicit_recon_obs
        )
        task.privileged_obs_buf = torch.zeros(
            2, pos.env.num_privileged_obs * pos.env.num_priv_stack
        )
        task._validate_hard_pact_observation_dimensions()
        self.assertTrue(task._hard_pact_observation_schema_validated)

        task._hard_pact_observation_schema_validated = False
        task.obs_buf = torch.zeros(2, pos.env.num_observations - 1)
        with self.assertRaisesRegex(RuntimeError, "actor observation"):
            task._validate_hard_pact_observation_dimensions()

    def test_hard_pact_and_pos_console_policies_align(self):
        hard = GO2HardPACTCfgPPO.runner
        pos = GO2HardPACTPosCfgPPO.runner
        for field in (
            "console_iteration",
            "console_model_summary",
            "console_reward_terms",
            "console_detailed_losses",
            "console_pinn_timing",
            "console_qp_timing",
        ):
            self.assertEqual(getattr(hard, field), getattr(pos, field))
        self.assertFalse(GO2HardPACTCfg.sim.console_debug)
        self.assertFalse(GO2HardPACTPosCfg.sim.console_debug)
        self.assertTrue(GO2HardPACTCfg.sim.suppress_backend_warnings)
        self.assertTrue(GO2HardPACTPosCfg.sim.suppress_backend_warnings)

    def test_hard_pact_pos_exports_strict_hard_pact_start_weights(self):
        torch.manual_seed(2027)
        pos_actor = ActorCritic_HardPACT_Pos(
            **_actor_kwargs(GO2HardPACTPosCfg(), GO2HardPACTPosCfgPPO())
        )
        hard_actor = ActorCritic_HardPACT(
            **_actor_kwargs(GO2HardPACTCfg(), GO2HardPACTCfgPPO())
        )
        pos_state = pos_actor.state_dict()
        decoder_state = {"probe": torch.arange(3.0)}
        converted = build_hard_pact_start_checkpoint(
            pos_state, decoder_state, iteration=123
        )

        incompatible = hard_actor.load_state_dict(
            converted["model_state_dict"], strict=True
        )
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        for key, value in pos_state.items():
            if key != "std":
                torch.testing.assert_close(
                    converted["model_state_dict"][key], value
                )
        expected_std = torch.ones_like(converted["model_state_dict"]["std"])
        torch.testing.assert_close(
            converted["model_state_dict"]["std"], expected_std
        )
        self.assertEqual(converted["iter"], 123)
        self.assertTrue(converted["hard_pact_start"])
        self.assertEqual(converted["source_task"], "go2_hard_pact_pos")
        torch.testing.assert_close(converted["decoder_state_dict"]["probe"],
                                   decoder_state["probe"])

    def test_configs_are_standalone_legged_robot_configs(self):
        """HardPACT config modules must not depend on either legacy config."""
        self.assertIs(GO2HardPACTCfg.__bases__[0], GO2PACTCfg.__bases__[0])
        self.assertIs(GO2HardPACTCfgPPO.__bases__[0], GO2PACTCfgPPO.__bases__[0])
        self.assertIs(GO2HardPACTPosCfg.__bases__[0], GO2PACTPosCfg.__bases__[0])
        self.assertIs(
            GO2HardPACTPosCfgPPO.__bases__[0], GO2PACTPosCfgPPO.__bases__[0]
        )
        self.assertNotIn(GO2PACTCfg, GO2HardPACTCfg.__mro__)
        self.assertNotIn(GO2PACTCfgPPO, GO2HardPACTCfgPPO.__mro__)
        self.assertNotIn(GO2PACTPosCfg, GO2HardPACTPosCfg.__mro__)
        self.assertNotIn(GO2PACTPosCfgPPO, GO2HardPACTPosCfgPPO.__mro__)

    def test_pact_sim_dt_is_derived_from_control_interval(self):
        config_classes = (
            GO2PACTCfg, GO2PACTPosCfg, GO2HardPACTCfg, GO2HardPACTPosCfg,
        )
        for config_class in config_classes:
            with self.subTest(config=config_class.__name__):
                self.assertEqual(
                    config_class.sim.dt,
                    config_class.control.dt / config_class.control.decimation,
                )

        for task_name, registered in task_registry.env_cfgs.items():
            if not task_name.startswith("go2_hard_pact"):
                continue
            with self.subTest(task=task_name):
                self.assertEqual(
                    registered.sim.dt,
                    registered.control.dt / registered.control.decimation,
                )

    def test_task_registration_and_resolved_configuration_parity(self):
        for name, legacy_task, alias_task, legacy_cfg_cls, alias_cfg_cls, legacy_train_cls, alias_train_cls, _, _ in CASES:
            with self.subTest(name=name):
                self.assertIs(task_registry.get_task_class(name), alias_task)
                self.assertTrue(issubclass(alias_task, legacy_task))
                # HardPACT wraps this one legacy callback solely to record the
                # already-selected clip/delay/torque path for exact replay.

                registered_env = task_registry.env_cfgs[name]
                registered_train = task_registry.train_cfgs[name]
                self.assertIsInstance(registered_env, alias_cfg_cls)
                self.assertIsInstance(registered_train, alias_train_cls)
                alias_env_dict = class_to_dict(alias_cfg_cls())
                legacy_env_dict = class_to_dict(legacy_cfg_cls())
                grf_cfg = alias_env_dict["sim"].pop("grf")
                if name == "go2_hard_pact":
                    self.assertFalse(alias_env_dict["sim"].pop("console_debug"))
                    self.assertTrue(
                        alias_env_dict["sim"].pop("suppress_backend_warnings")
                    )
                self.assertEqual(
                    set(grf_cfg),
                    {
                        "vertical_deadband_n",
                        "clip_min_n",
                        "clip_max_n",
                        "ema_alpha",
                        "contact_threshold_n",
                        "use_ema_grfs_buf",
                        "prediction_scale_n",
                    },
                )
                self.assertEqual(
                    grf_cfg["use_ema_grfs_buf"],
                    alias_cfg_cls.sim.grf.use_ema_grfs_buf,
                )
                alias_env_dict.pop("deployment_physics")
                alias_env_dict["env"]["num_explicit_recon_obs"] = legacy_env_dict["env"]["num_explicit_recon_obs"]
                alias_env_dict["env"]["num_privileged_obs"] = legacy_env_dict["env"]["num_privileged_obs"]
                alias_env_dict["normalization"]["obs_scales"].pop("base_wrench")
                persistent_fields = {
                    key for key in alias_env_dict["domain_rand"]
                    if key.startswith("persistent_")
                }
                self.assertTrue(persistent_fields)
                for field in persistent_fields:
                    alias_env_dict["domain_rand"].pop(field)
                if name == "go2_hard_pact":
                    # These are intentional HardPACT-only B1Z1 reward ports.
                    for field in (
                        "torque_cancellation_deadband",
                        "foot_clearance_excess_margin",
                        "foot_clearance_excess_weight",
                    ):
                        alias_env_dict["rewards"].pop(field)
                    alias_env_dict["rewards"]["scales"][
                        "torque_conflict_symmetric"
                    ] = legacy_env_dict["rewards"]["scales"][
                        "torque_conflict_symmetric"
                    ]
                    alias_env_dict["rewards"]["scales"].pop(
                        "torque_cancellation"
                    )
                self.assertEqual(
                    alias_env_dict["domain_rand"]["push_robots"],
                    legacy_env_dict["domain_rand"]["push_robots"],
                )
                self.assertEqual(alias_env_dict, legacy_env_dict)

                alias_train_dict = class_to_dict(alias_train_cls())
                legacy_train_dict = class_to_dict(legacy_train_cls())
                for field in ("cenet_velo_dim", "cenet_explicit_layers", "grf_decoder_layers", "wrench_decoder_layers", "cenet_dec_input_dim", "cenet_dec_out_dim", "pretrained_path"):
                    if field in legacy_train_dict["policy"]:
                        alias_train_dict["policy"][field] = legacy_train_dict["policy"][field]
                    else:
                        alias_train_dict["policy"].pop(field, None)
                alias_train_dict["runner"]["policy_class_name"] = legacy_train_dict["runner"]["policy_class_name"]
                if name == "go2_hard_pact_pos":
                    alias_train_dict["runner"].pop("export_hard_pact_start")
                    alias_train_dict["runner"].pop(
                        "hard_pact_start_filename"
                    )
                    for field in (
                        "console_iteration",
                        "console_model_summary",
                        "console_reward_terms",
                        "console_detailed_losses",
                        "console_pinn_timing",
                        "console_qp_timing",
                    ):
                        alias_train_dict["runner"].pop(field)
                if name == "go2_hard_pact":
                    alias_train_dict["runner"]["algorithm_class_name"] = legacy_train_dict["runner"]["algorithm_class_name"]
                    for field in (
                        "console_iteration", "console_model_summary",
                        "console_reward_terms", "console_detailed_losses",
                        "console_pinn_timing", "console_qp_timing",
                    ):
                        alias_train_dict["runner"].pop(field)
                    for field in ("num_steps_per_env", "max_iterations"):
                        alias_train_dict["runner"][field] = (
                            legacy_train_dict["runner"][field]
                        )
                    for field in (
                        "bard_enabled", "bard_randomize_base_inertia",
                        "bard_scale_rotational_inertia", "bard_batch_capacity",
                        "bard_inverse_enabled", "bard_rollout_enabled",
                        "lambda_inverse", "lambda_rollout",
                        "lambda_projection", "hard_pact_qp",
                        "action_clip",
                        "grf_observation_scale",
                        "base_wrench_observation_scale",
                        "auxiliary_learning_rate", "privileged_loss_weight",
                        "explicit_loss_weight", "grf_loss_weight",
                        "active_wrench_loss_weight", "neutral_wrench_loss_weight",
                        "profile_bard_timing", "console_debug",
                        "dynamics_backend", "pinocchio_num_workers",
                        "pcgrad_diagnostics_enabled",
                        "pcgrad_diagnostics_start_iteration",
                        "pcgrad_diagnostics_interval",
                        "cache_rollout_mechanics",
                        "ppo_qp_sampling",
                        "ppo_qp_passes_per_iteration",
                        "ppo_qp_shard_percentage",
                        "ppo_qp_stratify_by_anchor",
                        "ppo_qp_sampling_seed",
                        "ppo_qp_sampling_logging_enabled",
                    ):
                        alias_train_dict["algorithm"].pop(field)
                self.assertEqual(alias_train_dict, legacy_train_dict)

                legacy_train = legacy_train_cls()
                alias_train = alias_train_cls()
                self.assertEqual(alias_train.runner.runner_class_name if hasattr(alias_train.runner, "runner_class_name") else alias_train.runner_class_name,
                                 legacy_train.runner.runner_class_name if hasattr(legacy_train.runner, "runner_class_name") else legacy_train.runner_class_name)
                self.assertEqual(alias_train.runner.policy_class_name,
                                 "ActorCritic_HardPACT" if name == "go2_hard_pact" else "ActorCritic_HardPACT_Pos")
                self.assertEqual(
                    alias_train.runner.algorithm_class_name,
                    "PPO_HardPACT" if name == "go2_hard_pact"
                    else legacy_train.runner.algorithm_class_name,
                )

    def test_observation_critic_and_action_dimensions(self):
        for name, _, _, legacy_cfg_cls, alias_cfg_cls, _, _, _, expected_action_dim in CASES:
            with self.subTest(name=name):
                legacy, alias = legacy_cfg_cls(), alias_cfg_cls()
                self.assertEqual(alias.env.num_observations, legacy.env.num_observations)
                self.assertEqual(
                    alias.env.num_privileged_obs,
                    legacy.env.num_privileged_obs + DISTURBANCE_CRITIC_DIM,
                )
                self.assertEqual(alias.env.num_priv_stack, legacy.env.num_priv_stack)
                self.assertEqual(alias.env.num_obs_hist, legacy.env.num_obs_hist)
                self.assertEqual(alias.env.num_actions, legacy.env.num_actions)
                self.assertEqual(
                    alias.env.num_privileged_obs * alias.env.num_priv_stack,
                    legacy.env.num_privileged_obs * legacy.env.num_priv_stack
                    + DISTURBANCE_CRITIC_DIM * alias.env.num_priv_stack,
                )
                multiplier = 2 if name == "go2_hard_pact" else 1
                self.assertEqual(multiplier * alias.env.num_actions, expected_action_dim)

    def test_actor_outputs_distributions_state_shapes_and_strict_checkpoints(self):
        for name, _, _, legacy_cfg_cls, alias_cfg_cls, legacy_train_cls, alias_train_cls, actor_cls, expected_action_dim in CASES:
            with self.subTest(name=name):
                legacy_cfg, alias_cfg = legacy_cfg_cls(), alias_cfg_cls()
                legacy_train, alias_train = legacy_train_cls(), alias_train_cls()
                torch.manual_seed(1729)
                legacy_actor = actor_cls(**_actor_kwargs(legacy_cfg, legacy_train))
                torch.manual_seed(1729)
                reload_actor = actor_cls(**_actor_kwargs(legacy_cfg, legacy_train))

                legacy_shapes = {key: tuple(value.shape) for key, value in legacy_actor.state_dict().items()}
                reload_shapes = {key: tuple(value.shape) for key, value in reload_actor.state_dict().items()}
                self.assertEqual(reload_shapes, legacy_shapes)

                obs = torch.linspace(-0.5, 0.5, 2 * legacy_cfg.env.num_observations).reshape(2, -1)
                hist_dim = legacy_cfg.env.num_observations * legacy_cfg.env.num_obs_hist
                history = torch.linspace(-1.0, 1.0, 2 * hist_dim).reshape(2, -1)
                torch.manual_seed(99)
                legacy_actions = legacy_actor.act(obs, history)
                torch.manual_seed(99)
                reload_actions = reload_actor.act(obs, history)
                torch.testing.assert_close(reload_actions, legacy_actions, rtol=0, atol=0)
                torch.testing.assert_close(reload_actor.distribution.mean, legacy_actor.distribution.mean, rtol=0, atol=0)
                torch.testing.assert_close(reload_actor.distribution.stddev, legacy_actor.distribution.stddev, rtol=0, atol=0)
                self.assertEqual(tuple(reload_actions.shape), (2, expected_action_dim))

                decoder_cls = PACTContextDecoder if name == "go2_hard_pact" else PACTPosContextDecoder
                legacy_decoder = decoder_cls(
                    legacy_train.policy.cenet_dec_input_dim,
                    legacy_train.policy.cenet_dec_layers,
                    legacy_train.policy.cenet_dec_out_dim,
                )
                reload_decoder = decoder_cls(
                    legacy_train.policy.cenet_dec_input_dim,
                    legacy_train.policy.cenet_dec_layers,
                    legacy_train.policy.cenet_dec_out_dim,
                )
                checkpoint = {
                    "model_state_dict": legacy_actor.state_dict(),
                    "act_optimizer_state_dict": {},
                    "enc_optimizer_state_dict": {},
                    "decoder_state_dict": legacy_decoder.state_dict(),
                    "decoder_opt_state_dict": {},
                    "iter": 17,
                    "infos": {"legacy": True},
                }
                self.assertEqual(
                    set(checkpoint),
                    {"model_state_dict", "act_optimizer_state_dict", "enc_optimizer_state_dict",
                     "decoder_state_dict", "decoder_opt_state_dict", "iter", "infos"},
                )
                checkpoint_buffer = io.BytesIO()
                torch.save(checkpoint, checkpoint_buffer)
                checkpoint_buffer.seek(0)
                runner_cls = OnPolicyRunnerPACT if name == "go2_hard_pact" else OnPolicyRunnerPACTPos
                runner = runner_cls.__new__(runner_cls)
                runner.alg = SimpleNamespace(actor_critic=reload_actor, decoder=reload_decoder)
                loaded_infos = runner.load(checkpoint_buffer, load_optimizer=False)
                self.assertEqual(loaded_infos, {"legacy": True})
                self.assertEqual(reload_actor.load_state_dict(checkpoint["model_state_dict"], strict=True).missing_keys, [])
                self.assertEqual(reload_decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True).missing_keys, [])

    def test_action_history_delay_and_torque_paths(self):
        for name, legacy_task, alias_task, legacy_cfg_cls, alias_cfg_cls, _, _, _, action_width in CASES:
            with self.subTest(name=name):
                legacy = _synthetic_task(legacy_task, legacy_cfg_cls(), action_width)
                alias = _synthetic_task(alias_task, alias_cfg_cls(), action_width)
                incoming = torch.linspace(-2.0, 2.0, 2 * action_width).reshape(2, -1)
                legacy_executed = legacy._pre_sim_step(incoming)
                alias_executed = alias._pre_sim_step(incoming)
                torch.testing.assert_close(alias_executed, legacy_executed)
                torch.testing.assert_close(alias.actions, legacy.actions)
                torch.testing.assert_close(alias.last_actions, legacy.last_actions)
                torch.testing.assert_close(alias.llast_actions, legacy.llast_actions)
                torch.testing.assert_close(alias.action_queue, legacy.action_queue)

                sim_cls = GenesisSimulator_PACT if action_width == 24 else GenesisSimulator_PACT_Pos
                legacy_sim = _synthetic_torque_sim(sim_cls, legacy_cfg_cls())
                alias_sim = _synthetic_torque_sim(sim_cls, alias_cfg_cls())
                executed = torch.linspace(-0.8, 0.8, 2 * action_width).reshape(2, -1)
                legacy_torque = legacy_sim._compute_torques(executed)
                alias_torque = alias_sim._compute_torques(executed)
                torch.testing.assert_close(alias_sim.feedback_torques, legacy_sim.feedback_torques)
                if action_width == 24:
                    torch.testing.assert_close(alias_sim.feedforward_torques, legacy_sim.feedforward_torques)
                    torch.testing.assert_close(alias_sim._unweighted_torques, legacy_sim._unweighted_torques)
                torch.testing.assert_close(alias_torque, legacy_torque)

    def test_representative_rewards(self):
        for name, legacy_task, alias_task, legacy_cfg_cls, alias_cfg_cls, _, _, _, action_width in CASES:
            with self.subTest(name=name):
                legacy = _synthetic_task(legacy_task, legacy_cfg_cls(), action_width)
                alias = _synthetic_task(alias_task, alias_cfg_cls(), action_width)
                sim_values = dict(
                    torques=torch.linspace(-2.0, 2.0, 24).reshape(2, 12),
                    feedforward_torques=torch.linspace(-1.0, 1.0, 24).reshape(2, 12),
                    first_loop_feedback=torch.linspace(0.5, -0.5, 24).reshape(2, 12),
                    dof_vel=torch.linspace(-3.0, 3.0, 24).reshape(2, 12),
                )
                legacy.simulator = SimpleNamespace(**copy.deepcopy(sim_values))
                alias.simulator = SimpleNamespace(**copy.deepcopy(sim_values))
                legacy.actions.copy_(torch.linspace(-0.5, 0.5, 2 * action_width).reshape(2, -1))
                alias.actions.copy_(legacy.actions)
                for reward_name in ("_reward_torques", "_reward_feedback_torques",
                                    "_reward_feedforward_torques", "_reward_dof_power",
                                    "_reward_action_rate", "_reward_action_smoothness"):
                    torch.testing.assert_close(getattr(alias, reward_name)(), getattr(legacy, reward_name)())

    def test_reward_and_domain_randomization_curricula(self):
        for name, legacy_task, alias_task, legacy_cfg_cls, alias_cfg_cls, _, _, _, _ in CASES:
            with self.subTest(name=name):
                legacy = legacy_task.__new__(legacy_task)
                alias = alias_task.__new__(alias_task)
                for task, cfg in ((legacy, legacy_cfg_cls()), (alias, alias_cfg_cls())):
                    task.cfg = cfg
                    task.use_reward_curriculum = True
                    task.reward_curr_keys = list(cfg.rewards.reward_curriculum.curr_reward_keys)
                    task.reward_curr_bounds = copy.deepcopy(cfg.rewards.reward_curriculum.curr_reward_bounds)
                    task.reward_curr_steps = cfg.rewards.reward_curriculum.curr_steps
                    task.reward_warmup_steps = cfg.rewards.reward_curriculum.warmup_steps
                    task.dt = cfg.control.dt
                    task.reward_scales = {key: 123.0 for key in task.reward_curr_keys}
                iteration = legacy.reward_warmup_steps + max(1, legacy.reward_curr_steps // 2)
                with contextlib.redirect_stdout(io.StringIO()):
                    legacy.step_reward_curriculum(iteration)
                    alias.step_reward_curriculum(iteration)
                self.assertEqual(alias.reward_scales, legacy.reward_scales)

                sim_cls = GenesisSimulator_PACT if name == "go2_hard_pact" else GenesisSimulator_PACT_Pos
                legacy_sim = _synthetic_domain_sim(sim_cls, legacy_cfg_cls())
                alias_sim = _synthetic_domain_sim(sim_cls, alias_cfg_cls())
                legacy_sim._step_domian_rand(100, mean_reward=10.0)
                alias_sim._step_domian_rand(100, mean_reward=10.0)
                for attr in (
                    "domain_rand_phase", "domain_rand_joint_dynamics_progress",
                    "domain_rand_mass_com_progress", "domain_rand_disturbance_progress",
                    "domain_rand_reward_ema", "domain_rand_best_reward_ema",
                    "mass_max_value", "push_value", "angular_push_value", "vert_value",
                ):
                    self.assertEqual(getattr(alias_sim, attr), getattr(legacy_sim, attr))
                torch.testing.assert_close(alias_sim.joint_stiffness_bound_current,
                                           legacy_sim.joint_stiffness_bound_current)
                torch.testing.assert_close(alias_sim.joint_damping_bound_current,
                                           legacy_sim.joint_damping_bound_current)
                torch.testing.assert_close(alias_sim.joint_friction_bound_current,
                                           legacy_sim.joint_friction_bound_current)


if __name__ == "__main__":
    unittest.main()
