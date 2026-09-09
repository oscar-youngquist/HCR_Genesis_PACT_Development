"""Iteration-scoped QP warmup without touching PPO/auxiliary or ablation flags."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import GO2HardPACTCfgPPO
from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig
from rsl_rl.algorithms.ppo_hard_pact import PPO_HardPACT
from rsl_rl.hard_pact_ablations import HARD_PACT_ABLATIONS, resolve_hard_pact_features
from rsl_rl.runners.pact_runner import OnPolicyRunnerPACT
from test_hard_pact_auxiliary import make_algorithm
from test_hard_pact_speed_optimizations import _FakeDynamics, _flat_transition_fields


def algorithm_shell(variant="full", warmup=2):
    algorithm = PPO_HardPACT.__new__(PPO_HardPACT)
    algorithm.hard_pact_features = resolve_hard_pact_features(variant)
    algorithm.qp_config = HardPACTQPConfig(warmup_iterations=warmup)
    algorithm.hard_pact_qp = SimpleNamespace(cfg=algorithm.qp_config)
    return algorithm


@pytest.mark.parametrize("variant", HARD_PACT_ABLATIONS)
def test_exact_boundary_default_and_ablation_invariance(variant):
    algorithm = algorithm_shell(variant)
    flags = algorithm.hard_pact_features
    assert not algorithm.qp_enabled_at_iteration(0)
    assert not algorithm.qp_enabled_at_iteration(1)
    assert algorithm.qp_enabled_at_iteration(2) == flags.execution_qp
    assert algorithm.qp_enabled_at_iteration(2000) == flags.execution_qp
    algorithm.qp_config = replace(algorithm.qp_config, warmup_iterations=0)
    assert algorithm.qp_enabled_at_iteration(0) == flags.execution_qp
    assert algorithm.hard_pact_features is flags
    assert GO2HardPACTCfgPPO.algorithm.hard_pact_qp["warmup_iterations"] == 0


@pytest.mark.parametrize("iterations", [-1, 0.5])
def test_invalid_warmup_rejected(iterations):
    with pytest.raises(ValueError, match="nonnegative integer"):
        HardPACTQPConfig(warmup_iterations=iterations)


def test_runner_gate_skips_rollout_head_then_enables_at_boundary():
    algorithm = algorithm_shell()
    env = Go2HardPACT.__new__(Go2HardPACT)
    env.hard_pact_features = algorithm.hard_pact_features
    env.extras = {"hard_pact_qp_interval": {"stale": 1}}
    head = Mock(return_value=torch.zeros(2, 6))
    actor = SimpleNamespace(physics_estimator=SimpleNamespace(predict_wrench=head))
    env.configure_hard_pact_substep_qp(actor, object(), algorithm.hard_pact_qp)
    assert not env._hard_pact_rollout_qp_enabled
    assert "hard_pact_qp_interval" not in env.extras
    runner = OnPolicyRunnerPACT.__new__(OnPolicyRunnerPACT)
    runner.alg, runner.env = algorithm, env
    previous = torch.ones(2, 12) * 0.3
    env._hard_pact_previous_substep_torque = previous.clone()
    for iteration in (0, 1, 2):
        runner._set_hard_pact_qp_iteration(iteration)
        env.set_hard_pact_policy_context(torch.zeros(2, 16), torch.zeros(2, 11))
        assert env._hard_pact_rollout_qp_enabled == (iteration >= 2)
        assert head.call_count == int(iteration >= 2)
    # Enabling projection must not substitute zero for the last applied torque.
    # The existing _pre_sim_step then refreshes it from the simulator normally.
    torch.testing.assert_close(env._hard_pact_previous_substep_torque, previous)


def test_warmup_skips_qp_shards_and_deployment_mechanics_not_actual_pinn():
    algorithm = algorithm_shell()
    algorithm._qp_training_iteration = 1
    algorithm.physics_dynamics = _FakeDynamics()
    algorithm.bard_inverse_enabled = algorithm.bard_rollout_enabled = True
    algorithm.storage = SimpleNamespace(
        hard_pact_fields=_flat_transition_fields(),
        # Deliberately no QP replay keys during warmup.
        current_hard_pact_batch={"pre_q": torch.zeros(4, 19)},
    )
    algorithm._prepare_rollout_mechanics_cache(torch.zeros(12))
    assert algorithm._rollout_actual_mechanics is not None
    assert algorithm._rollout_deployment_qp_mechanics is None
    assert all(not need_qp for _, need_qp, _ in algorithm.physics_dynamics.calls)
    assert algorithm._qp_rows_for_epoch(0, 1) is None


@pytest.mark.parametrize("completed,expected_enabled", [(0, False), (1, True), (20, True)])
def test_periodic_checkpoint_resumes_absolute_warmup(tmp_path, completed, expected_enabled):
    algorithm = algorithm_shell()
    algorithm.actor_critic = torch.nn.Linear(2, 2)
    algorithm.decoder = torch.nn.Linear(2, 2)
    algorithm.act_optimizer = SimpleNamespace(optimizer=Mock(state_dict=lambda: {}))
    algorithm.enc_optimizer = algorithm.decoder_optimizer = Mock(state_dict=lambda: {})
    algorithm._last_completed_iteration = completed
    env = Go2HardPACT.__new__(Go2HardPACT)
    env.hard_pact_features = algorithm.hard_pact_features
    env.configure_hard_pact_substep_qp(object(), object(), algorithm.hard_pact_qp)
    env.domain_rand_curriculum_state_dict = lambda: {"iteration": completed}
    env.load_domain_rand_curriculum_state_dict = Mock()
    runner = OnPolicyRunnerPACT.__new__(OnPolicyRunnerPACT)
    runner.alg, runner.env, runner.is_hard_pact = algorithm, env, True
    runner.current_learning_iteration = 0  # unchanged legacy loop counter
    path = tmp_path / "checkpoint.pt"
    runner.save(path)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["iter"] == completed + 1
    algorithm._last_completed_iteration = 999  # loading replaces old progress
    runner.load(path, load_optimizer=False)
    assert runner.current_learning_iteration == completed + 1
    assert algorithm._last_completed_iteration == completed
    assert env._hard_pact_rollout_qp_enabled == expected_enabled


def test_real_ppo_auxiliary_updates_continue_and_storage_adds_qp_fields_once():
    # Real policy likelihood, shuffled minibatches, auxiliary losses, PCGrad,
    # and optimizer steps. Only numerical dynamics/QP work is a counting mock.
    algorithm = make_algorithm(
        ablation_variant="hard", num_learning_epochs=1, num_mini_batches=2,
        hard_pact_qp={"warmup_iterations": 2}, ppo_qp_sampling="all",
    )
    algorithm.init_storage(4, 2, [57], [95], [133], [1140], [24], [11], [12], [18])
    algorithm.storage.max_action_delay = 0
    solver = Mock(_last_gradient_metrics={})
    solver.solve.side_effect = lambda torque: 1.1 * torque
    algorithm.hard_pact_qp = solver
    algorithm._prepare_rollout_mechanics_cache = Mock()

    def qp_loss(nominal, *_args, qp_rows=None, compute_pinn=True, **_kwargs):
        assert not compute_pinn  # hard ablation; PINN warmup cannot gate QP
        selected = nominal[qp_rows]
        safe = solver.solve(selected)
        return (safe - selected).square().mean()

    previous_parameters = [p.detach().clone() for p in algorithm.actor_critic.parameters()]
    old_storage = None
    for iteration in range(3):
        for _step in range(2):
            obs, history, critic = torch.randn(4, 57), torch.randn(4, 1140), torch.randn(4, 95)
            with torch.no_grad():
                algorithm.act(obs, critic, history, obs, history, obs, history)
            fields = {
                "standardized_action_noise": algorithm.transition.action_noise,
                "delayed_action_source_valid": torch.ones(4, 1, dtype=torch.bool),
                "total_external_wrench_label_yaw_normalized": torch.zeros(4, 6),
                "sustained_wrench_active_mask": torch.zeros(4, 1, dtype=torch.bool),
            }
            if iteration == 2:
                fields.update({
                    "sampled_qp_substep_index": torch.tensor([[0], [2], [0], [2]]),
                    **{key: torch.zeros(4, 1, dtype=torch.bool) for key in (
                        "push_event_mask", "reset_mask", "timeout_mask", "teleport_mask")},
                })
            algorithm.process_env_step(
                torch.ones(4), torch.zeros(4), {"hard_pact_transition": fields},
                torch.zeros(4, 12), torch.randn(4, 133), torch.zeros(4, 11),
                torch.zeros(4, 18), None, None, None,
            )
        storage = algorithm.storage.hard_pact_fields
        assert ("sampled_qp_substep_index" in storage) == (iteration == 2)
        if old_storage is not None:
            assert storage["standardized_action_noise"].data_ptr() == old_storage
        old_storage = storage["standardized_action_noise"].data_ptr()
        algorithm.storage.advantages.normal_()
        with patch.object(algorithm, "_compute_bard_loss", side_effect=qp_loss), \
             patch.object(algorithm, "spectral_normalization"), patch("torch.cuda.synchronize"):
            losses = algorithm.update(
                lambda a: (a[:, :12], a[:, 12:]), lambda q, p, v: q - p - v,
                .02, iteration, torch.zeros(12), 1.,
            )
        assert torch.isfinite(torch.tensor(losses)).all()
        assert algorithm.pinn_weight == 0
        assert solver.solve.call_count == (2 if iteration == 2 else 0)
        assert algorithm._prepare_rollout_mechanics_cache.call_count == int(iteration == 2)
        assert algorithm.last_qp_metrics["qp/minimal/warmup/qp_enabled"] == int(iteration == 2)
        assert any(not torch.equal(old, new) for old, new in
                   zip(previous_parameters, algorithm.actor_critic.parameters()))
        previous_parameters = [p.detach().clone() for p in algorithm.actor_critic.parameters()]
