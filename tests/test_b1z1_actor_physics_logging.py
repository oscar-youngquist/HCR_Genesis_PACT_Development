"""Logging-only filtering must not change losses, gradients, or gate visibility."""
from types import SimpleNamespace

import pytest
import torch

from test_b1z1_actor_physics import setup
from rsl_rl.algorithms import b1z1_actor_physics as physics


def fixture(**settings):
    cfg = dict(actor_phys_enabled=True, actor_phys_coef=.1, pinn_loss_weight=.1,
               actor_phys_live_grf_enabled=True)
    cfg.update(settings)
    return SimpleNamespace(cfg=cfg, actor_critic=SimpleNamespace(action_mode="coupled"),
                           pinn_weight=.1, rejection_progress=0., actor_physics_metrics={})


@pytest.mark.parametrize("settings,names", [
    ({"actor_phys_vel_weight": 1.}, ("vel", "velocity_error")),
    ({"actor_phys_ee_weight": 1.}, ("ee", "ee_error_m")),
    ({"actor_phys_q_weight": 1.}, ("q", "position_violation_rad")),
    ({"actor_phys_qd_weight": 1.}, ("qd", "velocity_violation_radps")),
    ({"actor_phys_pos_fk_enabled": True, "actor_phys_pos_fk_weight": 1.},
     ("pos_fk_raw", "pos_fk_weighted", "pos_fk_error_x_m", "pos_fk_finite_count")),
    ({"actor_phys_arm_manipulability_enabled": True, "actor_phys_arm_manipulability_weight": 1.},
     ("arm_manipulability_raw", "arm_manipulability_sigma_min")),
    ({"actor_phys_ee_stability_weight": 1.}, ("ee_stability_raw", "ee_stability_gate")),
    ({"actor_phys_arm_feedforward_enabled": True, "actor_phys_arm_feedforward_weight": 1.},
     ("arm_feedforward_raw", "arm_feedforward_effective_weight", "arm_feedforward_error_mae_nm")),
    ({"actor_phys_base_balance_enabled": True, "actor_phys_base_balance_weight": 1.},
     ("base_balance_raw", "base_balance_active_fraction")),
    ({"actor_phys_arm_rejection_enabled": True, "actor_phys_arm_rejection_weight": 1.},
     ("rejection_arm_raw", "rejection_arm_weight", "rejection_projection_condition", "rejection_generated_arm_force_norm")),
    ({"actor_phys_base_rejection_enabled": True, "actor_phys_base_rejection_weight": 1.},
     ("rejection_base_raw", "rejection_base_weight", "rejection_live_stance_force_norm", "rejection_incremental_stance_force_norm")),
])
def test_disabled_and_zero_weight_components_are_not_logged(settings, names):
    for key in (None, *settings):
        a = fixture(**settings)
        if key is not None:
            a.cfg[key] = False if isinstance(settings[key], bool) else 0.
        # A disabled tag must be removed before .item(), not merely zero-filled.
        a.actor_physics_metrics = {name: torch.tensor(0.) if key is None else object() for name in names}
        a.actor_physics_metrics["loss_scaled"] = torch.tensor(4.)
        logged = {}
        physics.finish(a, logged, 2)
        assert logged["ActorPhysics/loss_scaled"] == 2.
        assert logged["ActorPhysics/scheduled_coefficient"] == .1
        for name in names:
            assert ("ActorPhysics/" + name in logged) == (key is None)


def test_allocation_orientation_and_live_grf_dependencies():
    a = fixture(actor_phys_arm_rejection_enabled=True, actor_phys_arm_rejection_weight=.5,
                actor_phys_force_allocation_weight=.2, actor_phys_ee_stability_weight=.3)
    assert physics._logged_component(a, "rejection_allocation_raw")
    a.actor_critic.action_mode = "position"
    assert not physics._logged_component(a, "rejection_allocation_raw")
    a.actor_critic.action_mode = "coupled"
    for key in ("actor_phys_force_allocation_weight", "actor_phys_arm_rejection_weight"):
        saved, a.cfg[key] = a.cfg[key], 0.
        assert not physics._logged_component(a, "rejection_allocation_raw")
        a.cfg[key] = saved
    for key, tag in (("use_orientation", "angular_speed"), ("twist_weight", "twist"),
                     ("energy_weight", "energy_violation")):
        assert not physics._logged_component(a, "ee_stability_" + tag)
        a.cfg["actor_phys_ee_stability_" + key] = 1
        assert physics._logged_component(a, "ee_stability_" + tag)
    a.cfg["actor_phys_arm_rejection_enabled"] = False
    assert not physics._logged_component(a, "rejection_external_force_norm")
    assert physics._logged_component(a, "rejection_grf_blend")
    a.cfg["actor_phys_live_grf_enabled"] = False
    assert not physics._logged_component(a, "rejection_trust_clipped_fraction")


def test_actor_disabled_keeps_only_sampling_diagnostics():
    a = fixture(actor_phys_enabled=False)
    a.actor_physics_metrics = {"arm_feedforward_raw": object()}
    a.actor_sampling_metrics = {"configured_fraction": 0.}
    logged = {}
    physics.finish(a, logged, 1)
    assert logged == {"ActorPhysics/configured_fraction": 0.}


def test_finish_preserves_objective_and_gradients():
    a, batch, actions, context = setup()
    loss, metrics = physics.objective(a, batch, actions, context)
    before, = torch.autograd.grad(loss, actions, retain_graph=True)
    a.pinn_weight = .1
    a.cfg["pinn_loss_weight"] = .1
    a.actor_physics_metrics = metrics
    logged = {}
    physics.finish(a, logged, 1)
    after, = torch.autograd.grad(loss, actions)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert logged["ActorPhysics/loss"] == loss.item()
    assert not any("ee_stability_" in tag or "arm_manipulability_" in tag for tag in logged)
