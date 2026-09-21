"""CPU checks for Lab curricula and independently ordered contact sensors."""

from types import SimpleNamespace
import copy

import torch
import legged_gym.envs
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg
from legged_gym.simulator.isaaclab_simulator_b1z1 import _IsaacLabSimulatorB1Z1
from legged_gym.simulator.isaacgym_simulator_b1z1 import _IsaacGymSimulatorB1Z1
from legged_gym.envs.go2.go2_hard_pact.grf import GRFProcessingConfig, IntervalGRFProcessor
from rsl_rl.utils.simulator_diagnostics import log_grf_metrics


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
    logged = {}
    writer = SimpleNamespace(add_scalar=lambda name, value, iteration: logged.update({name: value}))
    log_grf_metrics(writer, sim, 4)
    assert "GRF/raw_FR_fz" in logged and "GRF/interval_average_RL_fz" in logged
    assert all(torch.isfinite(v) for v in logged.values())
