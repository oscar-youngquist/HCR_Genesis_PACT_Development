"""Shared HardPACT contracts without a simulator application."""
import torch
from types import SimpleNamespace
from scripts.eval_hard_pact_frozen import Observer, Moments, PREFIX

from legged_gym.envs.go2.go2_hard_pact.grf import GRFProcessingConfig, IntervalGRFProcessor
from legged_gym.envs.go2.go2_hard_pact.ablation_configs import (
    make_hard_pact_variant_configs, make_hard_pact_pos_backend_configs,
)
from rsl_rl.hard_pact_ablations import HARD_PACT_VARIANTS
from legged_gym.simulator.isaaclab_simulator_pact import IsaacLabSimulator_PACT


def test_all_isaaclab_variants_inherit_total_labels():
    for variant in HARD_PACT_VARIANTS:
        cfg, _ = make_hard_pact_variant_configs(variant, "isaaclab")
        assert cfg.sim.grf.force_mode == "total"
        assert cfg.sim.grf.contact_capacity_per_body > 0
    cfg, _ = make_hard_pact_pos_backend_configs("isaaclab")
    assert cfg.sim.grf.force_mode == "total"


def test_interval_total_labels_and_normal_only_contact_gates():
    p = IntervalGRFProcessor(2, 2, "cpu", torch.float32, GRFProcessingConfig())
    normal = torch.tensor([[[10., 0., 0.], [0., 0., 0.]]]).expand(2, -1, -1)
    total = normal + torch.tensor([[[0., 8., 0.], [0., 20., 0.]]])
    p.begin_interval()
    p.update_substep(total, normal)
    p.update_substep(total * 2, normal * 2)
    expected = total * 1.5
    expected[:, 1] = 0  # Friction alone never enables contact.
    torch.testing.assert_close(p.end_interval(), expected)
    assert p.contacts[:, 0].all() and not p.contacts[:, 1].any()
    before = p.ema[1].clone()
    p.reset(torch.tensor([0]))
    assert not p.interval_average[0].any()
    torch.testing.assert_close(p.ema[1], before)


def test_canonical_foot_order_and_moment_reference():
    s = IsaacLabSimulator_PACT.__new__(IsaacLabSimulator_PACT)
    s._feet_contact_indices = [2, 0]
    s.ground_normal_forces_world = torch.tensor([[[0., 0., 1.], [0., 0., 2.], [0., 0., 3.]]])
    s.ground_total_forces_world = s.ground_normal_forces_world + 1
    points = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
    s.ground_moments_origin_world = torch.cross(points, s.ground_total_forces_world, dim=-1)
    torch.testing.assert_close(s.hard_pact_foot_forces_world(), s.ground_total_forces_world[:, [2, 0]])
    torch.testing.assert_close(s.hard_pact_foot_normal_forces_world(), s.ground_normal_forces_world[:, [2, 0]])
    reference = torch.tensor([[2., 3., 4.]])
    expected = torch.cross(points - reference[:, None], s.ground_total_forces_world, dim=-1)
    torch.testing.assert_close(s.ground_contact_moment_world(reference), expected.sum(1))
    torch.testing.assert_close(s.ground_contact_moment_world(reference, True), expected[:, [2, 0]].sum(1))


def test_frozen_recorder_keeps_raw_components_separate_from_clipped_targets():
    o = Observer.__new__(Observer)
    normal = torch.zeros(2, 4, 3)
    normal[..., 2] = 30.
    tangent = torch.zeros_like(normal)
    tangent[..., 0] = 12.
    total = normal + tangent
    o.sim = SimpleNamespace(grf_semantics={"mode": "total"}, _feet_contact_indices=[2, 0, 3, 1],
        ground_normal_forces_world=normal, ground_tangential_forces_world=tangent,
        ground_total_forces_world=total)
    o.env = SimpleNamespace(grf_processor=SimpleNamespace(
        raw=total, clipped=total.clamp(-10, 10), contacts=torch.ones(2, 4, dtype=torch.bool)))
    o.actor = SimpleNamespace(cenet_torso_velo=torch.zeros(2, 11))
    o._post = lambda: None
    o.pred_force = o.solved_force = torch.zeros_like(normal)
    o.previous_contact = torch.zeros(2, 4, dtype=torch.bool)
    o.step, o.k = PREFIX, 0
    o.metrics, o.latest, o.substeps = Moments(), {}, []
    o.post()
    row = o.substeps[0]
    torch.testing.assert_close(row["measured_total_grf_world_n"],
        row["measured_normal_grf_world_n"] + row["measured_tangential_grf_world_n"])
    assert row["measured_total_grf_world_n"].max() == 30.
    assert row["measured_grf_world_n"].max() == 10.
