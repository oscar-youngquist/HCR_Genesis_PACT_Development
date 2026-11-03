"""CPU checks for Lab curricula and independently ordered contact sensors."""

from types import SimpleNamespace
import copy
from unittest.mock import Mock, patch

import torch
import legged_gym.envs
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg
from legged_gym.simulator.isaaclab_simulator_b1z1 import _IsaacLabSimulatorB1Z1
from legged_gym.simulator.isaaclab_simulator import IsaacLabSimulator
from legged_gym.simulator.isaacgym_simulator_b1z1 import _IsaacGymSimulatorB1Z1
from legged_gym.envs.go2.go2_hard_pact.grf import GRFProcessingConfig, IntervalGRFProcessor


def make_sim():
    sim = _IsaacLabSimulatorB1Z1.__new__(_IsaacLabSimulatorB1Z1)
    sim._cfg = B1Z1PACTCfg()
    sim._cfg.domain_rand.use_domainrand_curriculum = True
    sim._cfg.domain_rand.push_warmup = 0
    sim._cfg.domain_rand.step_interval = 1
    sim._cfg.domain_rand.joint_dynamics_progress_delta = 1.0
    sim._cfg.domain_rand.mass_com_progress_delta = 0.5
    _IsaacGymSimulatorB1Z1._parse_b1z1_cfg(sim, use_final_ranges=False)
    sim._domain_rand_last_iteration = -1
    return sim


def test_curriculum_once_per_iteration_and_checkpoint_bounds():
    sim = make_sim()
    assert sim.mass_max_value == sim._cfg.domain_rand.min_added_mass_max
    sim._step_domian_rand(1, 100.)
    assert sim.domain_rand_phase == "mass_com"
    sim._step_domian_rand(2, 100.)
    assert sim.domain_rand_mass_com_progress == 0.5
    sim._step_domian_rand(2, 100.)
    assert sim.domain_rand_mass_com_progress == 0.5
    assert sim.mass_max_value == sum(sim.max_mass_bounds) / 2
    assert sim.grip_mass_max_value == sum(sim.grip_max_mass_bounds) / 2
    restored = make_sim()
    restored.load_domain_rand_curriculum_state_dict(copy.deepcopy(sim.domain_rand_curriculum_state_dict()))
    assert restored.mass_max_value == sim.mass_max_value
    assert restored.grip_mass_max_value == sim.grip_mass_max_value
    assert restored._domain_rand_last_iteration == 2
    assert list(restored.domain_rand_reward_ema_hist) == list(sim.domain_rand_reward_ema_hist)


def test_disabled_curriculum_uses_final_ranges_and_skips_reset_property_writes():
    sim = make_sim()
    sim._cfg.domain_rand.use_domainrand_curriculum = False
    # Lab's disabled-curriculum mode must not inherit Gym's optional start bounds.
    sim._cfg.domain_rand.isaacgym_use_final_domain_rand_ranges = False
    with patch.object(IsaacLabSimulator, "_parse_cfg"):
        sim._parse_cfg()
    assert sim.mass_max_value == sim.max_mass_bounds[1]
    assert sim.grip_mass_max_value == sim.grip_max_mass_bounds[1]
    assert sim.com_delta_x_value == sim.com_delta_x_bounds[1]
    assert list(sim.joint_friction_bound_current) == list(sim._cfg.domain_rand.joint_friction_range_end)
    sim._robot, sim._contact_sensors = Mock(), Mock()
    sim._randomize_physical_properties = Mock()
    sim._randomize_pd_gain, sim._randomize_motor_strength = Mock(), Mock()
    sim._reset_grf_buffer = Mock()
    sim._grf_processor = None
    sim._ground_contacts = None
    for name in ("_body_ground_normal_w", "_body_ground_total_w", "_body_ground_moment_origin_w"):
        setattr(sim, name, torch.ones(2, 3, 3))
    sim._cfg.domain_rand.randomize_pd_gain = True
    sim._cfg.domain_rand.randomize_motor_strength = True
    for name in ("_last_dof_vel", "_last_base_lin_vel", "_last_base_ang_vel",
                 "_last_base_world_lin_vel", "_last_base_world_ang_vel",
                 "_last_feet_vel", "_dof_tau"):
        setattr(sim, name, torch.ones(2, 3))
    ids = torch.tensor([0])
    sim.reset_idx(ids)
    sim.reset_idx(ids)
    sim._randomize_physical_properties.assert_not_called()
    assert sim._randomize_pd_gain.call_count == 2
    assert sim._randomize_motor_strength.call_count == 2
    # Curriculum-enabled resets still install the current physical ranges.
    sim.use_domainrand_curriculum = True
    sim.reset_idx(ids)
    sim._randomize_physical_properties.assert_called_once_with(ids)


def test_sensor_order_substep_filter_and_indexed_reset():
    sim = _IsaacLabSimulatorB1Z1.__new__(_IsaacLabSimulatorB1Z1)
    # Articulation feet [0,1,2,3] are intentionally not sensor feet [3,1,4,0].
    raw = torch.zeros(2, 5, 3)
    raw[:, :, 2] = torch.tensor([40., 20., 999., 10., 30.])
    sim._contact_sensors = SimpleNamespace(
        body_names=["RL", "FL", "base", "FR", "RR"],
        data=SimpleNamespace(net_forces_w=raw))
    sim._body_names = ["FR", "FL", "RR", "RL", "base"]
    sim._feet_indices = torch.arange(4)
    sim._device = "cpu"
    sim._resolve_contact_indices()
    sim._body_ground_normal_w = raw[:, sim._contact_indices].clone()
    sim._body_ground_total_w = sim._body_ground_normal_w.clone()
    assert sim._feet_contact_indices.tolist() == [3, 1, 4, 0]
    forces = sim._sensor_foot_forces_world()
    torch.testing.assert_close(forces[0, :, 2], torch.tensor([10., 20., 30., 40.]))
    processor = IntervalGRFProcessor(2, 4, "cpu", torch.float32,
        GRFProcessingConfig(15., -25., 25., 0.5, 5.))
    processor.begin_interval()
    processor.update_substep(forces)
    processor.update_substep(forces)
    expected = torch.tensor([0., 20., 25., 25.])
    torch.testing.assert_close(processor.end_interval()[0, :, 2], expected)
    torch.testing.assert_close(processor.ema[0, :, 2], expected * 0.75)
    processor.reset(torch.tensor([0]))
    assert not processor.ema[0].any()
    torch.testing.assert_close(processor.ema[1, :, 2], expected * 0.75)
    sim._grf_processor, sim._num_envs = processor, 2
    sim._feet_names = ["FR", "FL", "RR", "RL"]
    sim._grfs_buf = torch.zeros(2, 12)
    sim._cfg = SimpleNamespace(sim=SimpleNamespace(grf=SimpleNamespace(
        deadband=15., clip_min=-25., clip_max=25., ema_alpha=0.5,
        contact_threshold=5.)))
    sim._configure_grf_processing()
    sim._use_substep_grf_filtering = False
    count_before = processor.interval_count.clone()
    sim._refresh_grf_buffer()
    # One EMA update on the final deadbanded/clipped sample, not raw forces.
    torch.testing.assert_close(sim._grfs_buf[0, 2::3], expected * 0.5)
    assert sim._foot_contact_force_threshold == 5.
    torch.testing.assert_close(processor.interval_count, count_before)
    sim._refresh_grf_buffer()
    torch.testing.assert_close(sim._grfs_buf[0, 2::3], expected * 0.75)
    sim._use_substep_grf_filtering = True
    sim._refresh_grf_buffer()
    torch.testing.assert_close(sim._grfs_buf, processor.ema.flatten(1))


def test_contact_logic_uses_normal_not_tangential_force():
    sim = _IsaacLabSimulatorB1Z1.__new__(_IsaacLabSimulatorB1Z1)
    sim._feet_indices = torch.tensor([1])
    sim._foot_contact_force_threshold = 5.
    sim._body_ground_normal_w = torch.zeros(2, 2, 3)
    sim._body_ground_total_w = torch.zeros_like(sim._body_ground_normal_w)
    sim._body_ground_total_w[:, 1, 0] = 100.
    sim._body_ground_normal_w[1, 1, 0] = 10.  # Sloped/vertical normal, not z support.
    assert sim.foot_contacts.tolist() == [[False], [True]]
    torch.testing.assert_close(sim.foot_contact_forces[..., 0], torch.full((2, 1), 100.))
    torch.testing.assert_close(sim.foot_normal_contact_forces[..., 0], torch.tensor([[0.], [10.]]))
    processor = IntervalGRFProcessor(2, 1, 'cpu', torch.float32,
        GRFProcessingConfig(5., -200., 200., 1., 5.))
    processor.update_substep(sim.foot_contact_forces, sim.foot_normal_contact_forces)
    assert processor.contacts.tolist() == [[False], [True]]
    assert processor.clipped[0].count_nonzero() == 0
    assert processor.clipped[1, 0, 0] == 100.


def test_world_wrench_composer_pose_cache_is_refreshed():
    sim = _IsaacLabSimulatorB1Z1.__new__(_IsaacLabSimulatorB1Z1)
    sim._cfg = SimpleNamespace(commands=SimpleNamespace(push_gripper_stators=True,
        apply_ee_external_forces=True, push_robot_base=True, apply_base_external_forces=True,
        apply_base_external_torques=True))
    sim._external_force_world, sim._external_torque_world = torch.zeros(2, 2, 3), torch.zeros(2, 2, 3)
    sim._gripper_index, sim._base_link_index = 1, 0
    sim._ee_force_world, sim._base_force_world, sim._base_torque_world = [torch.ones(2, 3)*k for k in (2, 3, 4)]
    composer = Mock()
    sim._robot = SimpleNamespace(permanent_wrench_composer=composer)
    sim._apply_external_forces()
    sim._apply_external_forces()
    assert [call[0] for call in composer.mock_calls] == ['reset', 'set_forces_and_torques']*2
    assert composer.set_forces_and_torques.call_args.kwargs['is_global'] is True
    torch.testing.assert_close(sim._external_force_world[:, 1], sim._ee_force_world)


def test_all_b1z1_families_default_to_total_labels():
    from legged_gym.utils import task_registry
    names = [name for name in task_registry.env_cfgs if name.startswith('b1z1_pact')]
    names += ['b1z1_ppo_pos', 'b1z1_unifp', 'b1z1_unifp_original', 'b1z1_unifp_reject']
    for name in names:
        grf = task_registry.env_cfgs[name].sim.grf
        assert grf.force_mode == 'total', name
        assert grf.contact_capacity_per_body > 0
