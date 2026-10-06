"""Small rejection curriculum, target, and actor-physics graph checks."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch

from test_b1z1_actor_physics import setup
from test_b1z1_force_target_projection import ForceTargetProjectionTests
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg, B1Z1PACTCfgPPO
from legged_gym.envs.b1z1.b1z1_pact.ablation_configs import make_b1z1_pact_ablation_configs
from legged_gym.envs.b1z1.b1z1_unifp_original.b1z1_unifp_original_config import B1Z1UniFPOriginalCfg
from legged_gym.envs.b1z1.b1z1_unifp_reject.b1z1_unifp_reject_config import B1Z1UniFPRejectCfg
from legged_gym.envs.b1z1.b1z1_unifp_reject.b1z1_unifp_reject import B1Z1UniFPReject
from legged_gym.envs.b1z1.rejection_curriculum import RejectionCurriculum
from legged_gym.envs.b1z1.force_task_utils import get_force_adjusted_ee_target, init_staged_force_curriculum
from rsl_rl.algorithms import b1z1_actor_rejection as rejection, b1z1_actor_physics as physics


def curriculum_trace(cfg):
    cfg = deepcopy(cfg)
    cfg.reject_warmup_iterations = 2
    cfg.force_curriculum_gate_start_iteration = 2
    cfg.reject_compensation_ramp_iterations = 2
    cfg.reject_external_ramp_iterations = 4
    cfg.reject_min_active_samples = 1
    cfg.force_curriculum_gate_patience = 2
    env = SimpleNamespace(cfg=SimpleNamespace(commands=cfg), device="cpu")
    init_staged_force_curriculum(env)
    c = env._staged_force_curriculum
    assert isinstance(c, RejectionCurriculum)
    force = torch.full((2, 3), 10.)
    result = []
    for iteration in range(10):
        c.observe_forces(force, force, force, force)
        c.update(iteration, 0., 0., 1000.)
        result.append((c.beta, c.external_scale(iteration)))
    return result, c


def test_matched_schedule_all_ablations_and_checkpoint():
    expected, c = curriculum_trace(B1Z1UniFPRejectCfg().commands)
    assert expected[:4] == [(0., .25)] * 4
    assert expected[4] == (.5, .25)
    assert expected[5] == (1., .25)
    assert expected[7] == (1., .625)
    assert expected[9] == (1., 1.)
    for number in range(4, 12):
        env_cfg, train_cfg = make_b1z1_pact_ablation_configs(number)
        assert not env_cfg.use_force_shifted_target
        assert train_cfg.algorithm.actor_phys_arm_rejection_weight == B1Z1PACTCfgPPO.algorithm.actor_phys_arm_rejection_weight
        actual, _ = curriculum_trace(env_cfg().commands)
        assert actual == expected
    restored = RejectionCurriculum(c.cfg, "cpu")
    restored.load_state_dict(c.state_dict())
    assert restored.state_dict() == c.state_dict()
    assert restored.external_scale(9) == 1.


def test_plain_ppo_uses_performance_gate_without_force_samples():
    from legged_gym.envs.b1z1.b1z1_ppo_pos.b1z1_ppo_pos_config import B1Z1PPOPosCfg
    cfg = B1Z1PPOPosCfg().commands
    assert not cfg.reject_require_force_quality
    cfg.reject_warmup_iterations = 0
    cfg.force_curriculum_gate_start_iteration = 0
    cfg.force_curriculum_gate_patience = 1
    cfg.reject_compensation_ramp_iterations = 2
    c = RejectionCurriculum(cfg, "cpu")
    c.update(0, 0., 0., 1000.)
    c.update(1, 0., 0., 1000.)
    assert c.beta == .5 and c.external_scale(1) == .25


def test_shared_rejection_defaults_and_fallback_without_force_gate():
    from legged_gym.envs.b1z1.b1z1_pact_pos.b1z1_pact_pos_config import B1Z1PACTPosCfg
    from legged_gym.envs.b1z1.rejection_curriculum import RejectionCurriculumDefaults
    configurations = [B1Z1UniFPRejectCfg(), B1Z1PACTCfg(), B1Z1PACTPosCfg()]
    configurations += [make_b1z1_pact_ablation_configs(i)[0]() for i in range(4, 12)]
    fields = ("reject_initial_external_scale", "reject_compensation_ramp_iterations",
              "reject_external_ramp_iterations", "force_curriculum_gate_start_iteration",
              "force_curriculum_gate_patience", "force_curriculum_use_latest_start_fallback",
              "force_curriculum_latest_start_iteration")
    for config in configurations:
        cfg = config.commands
        for field in fields:
            assert getattr(cfg, field) == getattr(RejectionCurriculumDefaults, field)
        c = RejectionCurriculum(cfg, "cpu")
        start = cfg.force_curriculum_latest_start_iteration
        c.update(start - 1)  # Neither performance metrics nor force predictions are available.
        assert not c.gate_latched and c.external_scale(start - 1) == .25
        c.update(start)
        assert c.gate_latched and c.beta == 0.
        full = start + cfg.reject_compensation_ramp_iterations
        c.update(full)
        assert c.beta == 1. and c.external_scale(full) == .25
        end = full + cfg.reject_external_ramp_iterations
        c.update(end)
        assert c.external_scale(end) == 1.


@pytest.mark.parametrize("fallback", [False, True])
def test_rejection_gate_matches_original_with_missing_force_predictions(fallback):
    from legged_gym.envs.b1z1.force_task_utils import B1Z1StagedForceCurriculum
    cfg = deepcopy(B1Z1PACTCfg().commands)
    cfg.force_curriculum_gate_start_iteration = 2
    cfg.force_curriculum_gate_patience = 2
    cfg.force_curriculum_use_latest_start_fallback = fallback
    cfg.force_curriculum_latest_start_iteration = 5
    cfg.reject_warmup_iterations = 100  # Legacy rejection warmup must not override the original gate.
    actual = RejectionCurriculum(cfg, "cpu")
    original = B1Z1StagedForceCurriculum(cfg)
    for iteration in range(8):
        # Failed/missing metrics exercise fallback; good metrics exercise patience.
        metrics = (None, None, None) if fallback else (0., 0., 1000.)
        actual.update(iteration, *metrics)
        original.update(iteration, *metrics)
        assert actual.gate_latched == original.gate_latched
        assert actual.gate_patience == original.gate_patience
        assert actual.trigger_iteration == original.trigger_iteration
    assert actual.trigger_iteration == (5 if fallback else 3)


def test_rejection_no_fallback_waits_for_performance():
    cfg = deepcopy(B1Z1PACTCfg().commands)
    cfg.force_curriculum_gate_start_iteration = 0
    cfg.force_curriculum_latest_start_iteration = 1
    cfg.force_curriculum_use_latest_start_fallback = False
    c = RejectionCurriculum(cfg, "cpu")
    for iteration in range(4):
        c.update(iteration)
    assert not c.gate_latched and c.beta == 0.
    assert c.external_scale(3) == .25


def test_nominal_targets_and_original_unifp_unchanged():
    for config in (B1Z1PACTCfg(), B1Z1UniFPRejectCfg(), B1Z1UniFPOriginalCfg()):
        env = ForceTargetProjectionTests._environment([.5, 0., 0.], [.1, .05, 0.],
                                                      commanded_force=[.05, 0., 0.])
        env.cfg.use_force_shifted_target = getattr(config, "use_force_shifted_target", True)
        target = get_force_adjusted_ee_target(env, use_cache=False).effective_target
        if type(config) is B1Z1UniFPOriginalCfg:
            assert not torch.equal(target, env.curr_ee_goal_cart_world)
            assert not getattr(config.commands, "use_shared_rejection_curriculum", False)
        else:
            torch.testing.assert_close(target, env.curr_ee_goal_cart_world, rtol=0, atol=0)
    env.commands = torch.tensor([[.5, 0., 0.]])
    env.simulator.base_lin_vel = torch.tensor([[.5, 0., 0.]])
    env.cfg.rewards = SimpleNamespace(tracking_sigma=.2)
    assert B1Z1UniFPReject._reward_tracking_lin_vel_force_world(env) == 1


def test_projection_recovers_force_and_opposes_disturbance():
    torch.manual_seed(5)
    j = torch.randn(3, 3, 6, dtype=torch.float64)
    a = torch.randn(3, 6, 6, dtype=torch.float64)
    mass = a @ a.transpose(-1, -2) + torch.eye(6)
    external = torch.randn(3, 3, dtype=torch.float64)
    torque = (j.transpose(-1, -2) @ (-external).unsqueeze(-1)).squeeze(-1).requires_grad_()
    generated, valid, _ = rejection.project_arm_force(mass.requires_grad_(), j.requires_grad_(), torque, 1e-10)
    assert valid.all()
    torch.testing.assert_close(generated, -external, atol=1e-8, rtol=1e-8)
    loss, _ = rejection.force_residual(generated, external, valid, 1., [1.]*3, 1.)
    assert loss < 1e-15
    generated.square().mean().backward()
    assert torque.grad.abs().sum() > 0 and mass.grad is None and j.grad is None


def test_frozen_grf_decoder_has_only_torque_gradient():
    a, batch, _, context = setup()
    torque = batch["nominal_torque"].clone().requires_grad_()
    z = context["z"].detach().requires_grad_()
    explicit = context["explicit_condition"].detach().requires_grad_()
    decoder = a.actor_critic.physics_decoder
    output = decoder.predict_grf_actor(z, explicit, torque)
    output.square().mean().backward()
    assert torch.isfinite(torque.grad).all() and torque.grad.abs().sum() > 0
    assert z.grad is None and explicit.grad is None
    assert all(p.grad is None for p in decoder.parameters())
    torch.testing.assert_close(output, decoder.predict_grf(z, explicit, torque))


def configure_actor(a, batch):
    a.cfg.update({k: getattr(B1Z1PACTCfgPPO.algorithm, k) for k in dir(B1Z1PACTCfgPPO.algorithm)
                  if k.startswith(("actor_phys_arm_rejection", "actor_phys_base_rejection",
                                   "actor_phys_rejection", "actor_phys_live_grf",
                                   "actor_phys_grf_torque", "actor_phys_arm_force", "actor_phys_base_force"))})
    a.cfg["actor_phys_arm_dof_indices"] = list(range(12, 18))
    a.force_gate_active = True
    a._force_prediction_blend_alpha = lambda: 1.
    a.rejection_progress = 1.
    batch["observations"] = torch.zeros(len(batch["dones"]), B1Z1PACTCfg.env.num_observations)
    batch["actor_phys_stance"] = torch.ones(len(batch["dones"]), 4, dtype=torch.bool)


@pytest.mark.parametrize("component", ["vel", "ee"])
def test_existing_tracking_receives_grf_mediated_gradient(component):
    a, batch, actions, context = setup()
    configure_actor(a, batch)
    a.rejection_progress = 0.  # No new force loss can explain these gradients.
    a.enable_additional_diagnostics = True
    a.actor_physics_cache.mass_matrix[:] = torch.eye(25)
    a.actor_physics_cache.foot_jacobians[:, :, 0, 0] = 1.
    a.dynamics_backend.ee_position = lambda state: state[:, :3]
    for name in ("vel", "ee", "q", "qd"):
        a.cfg["actor_phys_" + name + "_weight"] = float(name == component)
    loss, _ = physics.objective(a, batch, actions, context)
    grad, = torch.autograd.grad(loss, actions, retain_graph=True)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    loss.backward()
    assert all(p.grad is None for p in a.actor_critic.physics_decoder.parameters())
    assert all(p.grad is None for p in a.actor_critic.context_encoder.parameters())
    assert all(p.grad is None for p in a.actor_critic.explicit_decoder.parameters())
    # With detached GRFs, joint torque has no direct path to this synthetic base.
    a.cfg["actor_phys_live_grf_enabled"] = False
    candidate = actions.detach().requires_grad_()
    detached, _ = physics.objective(a, batch, candidate, context)
    detached_gradient, = torch.autograd.grad(detached, candidate)
    assert detached.isfinite() and detached_gradient.abs().sum() == 0


def test_base_reaction_sign():
    external = torch.tensor([[-20., 0.]])  # Opposes positive commanded vx.
    positive = torch.tensor([[20., 0.]], requires_grad=True)
    negative = -positive
    good, _ = rejection.force_residual(positive, external, torch.ones(1), 20., [1., 1.], 1.)
    bad, _ = rejection.force_residual(negative, external, torch.ones(1), 20., [1., 1.], 1.)
    assert good == 0 and bad > 0


@pytest.mark.parametrize("position_only", [False, True])
def test_counterfactual_actor_only_and_position_control(position_only):
    a, batch, actions, context = setup()
    configure_actor(a, batch)
    if position_only:
        a.actor_critic.action_mode = "position"
        actions = actions[:, :a.actor_critic.num_actions]
    a.actor_physics_cache.ee_jacobian[:, :3, 18:21] = torch.eye(3)
    loss, metrics = physics.objective(a, batch, actions, context)
    assert torch.isfinite(loss) and metrics["rejection_arm_weight"] > 0
    grad, = torch.autograd.grad(loss, actions, retain_graph=True)
    assert grad.abs().sum() > 0 and torch.isfinite(grad).all()
    loss.backward()
    assert all(p.grad is None for p in a.actor_critic.physics_decoder.parameters())
    if position_only:
        zeros = torch.zeros_like(actions)
        state = batch["actor_phys_state"]
        torch.testing.assert_close(a._coupled_torque(actions, state),
                                   a._coupled_torque(torch.cat((actions, zeros), -1), state))


@pytest.mark.parametrize("reason", ["disabled", "zero_weight", "zero_progress"])
def test_counterfactual_skip(monkeypatch, reason):
    a, batch, actions, context = setup()
    configure_actor(a, batch)
    a.cfg["actor_phys_live_grf_enabled"] = False
    if reason == "disabled":
        a.cfg.update(actor_phys_arm_rejection_enabled=False, actor_phys_base_rejection_enabled=False)
    elif reason == "zero_weight":
        a.cfg.update(actor_phys_arm_rejection_weight=0., actor_phys_base_rejection_weight=0.)
    elif reason == "zero_progress":
        a.rejection_progress = 0.
    def forbidden(*args, **kwargs):
        raise AssertionError("inactive rejection computed counterfactual or projection")
    monkeypatch.setattr(rejection, "neutral_torque", forbidden)
    monkeypatch.setattr(rejection, "project_arm_force", forbidden)
    physics.objective(a, batch, actions, context)


def test_actor_rejection_ignores_force_reliability_gate(monkeypatch):
    a, batch, actions, context = setup()
    configure_actor(a, batch)
    a.actor_physics_cache.ee_jacobian[:, :3, 18:21] = torch.eye(3)
    # Old saved configs must not re-enable the removed checks.
    a.cfg.update(actor_phys_rejection_require_force_gate=True,
                 actor_phys_live_grf_require_gate=True)
    def forbidden():
        raise AssertionError("actor rejection consulted legacy force blending")
    monkeypatch.setattr(a, "_force_prediction_blend_alpha", forbidden)
    a.force_gate_active = True
    expected, expected_metrics = physics.objective(a, batch, actions, context)
    a.force_gate_active = False
    actual, metrics = physics.objective(a, batch, actions, context)
    torch.testing.assert_close(actual, expected)
    assert metrics["rejection_grf_blend"] == 1.
    for name in ("rejection_arm_raw", "rejection_base_raw"):
        torch.testing.assert_close(metrics[name], expected_metrics[name])
    grad, = torch.autograd.grad(actual, actions)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
