"""Curriculum clocks follow completed control steps, not optimizer/rollout counts."""
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact import B1Z1PACT
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg
from legged_gym.envs.b1z1.force_task_utils import B1Z1StagedForceCurriculum
from rsl_rl.algorithms.b1z1_training_clock import prepare_step_schedules
from rsl_rl.runners.b1z1_pact_runner import B1Z1PACTRunner
from test_b1z1_flash_sac import make_algorithm
from test_b1z1_lab_curriculum_grf import make_sim


def force_config(**overrides):
    names = vars(B1Z1PACTCfg.commands)
    cfg = {k: v for k, v in names.items() if k.startswith('force_curriculum_')}
    cfg.update(force_curriculum_gate_start_env_step=24,
               force_curriculum_external_ramp_env_steps=48,
               force_curriculum_gate_patience_env_steps=48,
               force_curriculum_use_latest_start_fallback=False,
               force_curriculum_metric_ema_alpha=1.)
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def test_reward_and_gait_boundaries_restore_without_new_collection():
    env = B1Z1PACT.__new__(B1Z1PACT)
    env.cfg = SimpleNamespace(rewards=SimpleNamespace(
        gait_guidance_decay_enabled=True, gait_guidance_decay_env_steps=96))
    env.use_reward_curriculum = True
    env.reward_warmup_steps, env.reward_curr_steps = 24, 48
    env.reward_curr_keys, env.reward_curr_bounds = ['test'], {'test': (1., 3.)}
    env.reward_scales, env.dt = {'test': 99.}, .02
    for step, expected in [(0, 1.), (24, 1.), (48, 2.), (72, 3.), (96, 3.)]:
        env.set_completed_env_steps(step)
        assert env.reward_scales['test'] == pytest.approx(expected * .02)
    env.set_completed_env_steps(48)  # Restore checkpoint progress before any env.step.
    assert env.reward_scales['test'] == pytest.approx(.04)
    assert env._get_gait_guidance_multiplier(1., .01) == pytest.approx(.1)
    with pytest.raises(ValueError):
        env.set_completed_env_steps(-1)


def test_physics_and_kl_do_not_advance_on_repeated_optimizer_work():
    a = make_algorithm(pinn_start_env_step=24, pinn_warmup_env_steps=48,
                       pinn_loss_weight=-.2, kl_warmup_env_steps=96)
    prepare_step_schedules(a, 24)
    assert a.pinn_weight == 0.
    prepare_step_schedules(a, 48)
    assert a.pinn_weight == pytest.approx(.1) and a.schedule_step_delta == 24
    beta = a.kl_controller.warmup_beta(a.schedule_env_steps)
    for _ in range(8):
        prepare_step_schedules(a, 48)
        assert a.pinn_weight == pytest.approx(.1) and a.schedule_step_delta == 0
        assert a.kl_controller.warmup_beta(a.schedule_env_steps) == beta
    prepare_step_schedules(a, 72)
    assert a.pinn_weight == pytest.approx(.2)
    with pytest.raises(ValueError):
        prepare_step_schedules(a, 71)


def test_force_patience_steps_fallback_and_checkpoint():
    c = B1Z1StagedForceCurriculum(force_config())
    c.update(24, .1, .01, 1000.)
    assert c.gate_patience == 0  # No qualifying time before the gate start.
    c.update(48, .1, .01, 1000.)
    c.update(48, .1, .01, 1000.)
    assert c.gate_patience == 24 and not c.gate_latched
    restored = B1Z1StagedForceCurriculum(force_config())
    restored.load_state_dict(c.state_dict())
    restored.update(72, .1, .01, 1000.)
    assert restored.gate_latched and restored.trigger_iteration == 72
    assert restored.external_scale(96) == .5
    fallback = B1Z1StagedForceCurriculum(force_config(
        force_curriculum_use_latest_start_fallback=True,
        force_curriculum_latest_start_env_step=48))
    fallback.update(24)
    assert not fallback.gate_latched
    fallback.update(48)
    assert fallback.gate_latched


def test_environment_curricula_ignore_rollout_partitioning():
    def collect(partitions):
        r = B1Z1PACTRunner.__new__(B1Z1PACTRunner)
        c = B1Z1StagedForceCurriculum(force_config())
        r.env = SimpleNamespace(_staged_force_curriculum=c)
        r.curriculum_metrics_interval, r.completed_env_steps = 24, 0
        r.curriculum_ep_infos, r.curriculum_metrics = [], {}
        r.curriculum_episode_lengths = deque([1000.], maxlen=100)
        calls = []
        r._step_domain_randomization_curriculum = lambda step, infos: calls.append((step, len(infos)))
        for count in partitions:
            for _ in range(count):
                r.completed_env_steps += 1
                r._step_environment_curricula({'episode': {
                    'EE/tracking_l1_mean': .1, 'roll_termination_rate': .01}})
        return calls, c.state_dict()
    one = collect([96])
    many = collect([5] * 19 + [1])
    assert one == many
    assert one[0] == [(24, 24), (48, 24), (72, 24), (96, 24)]


def test_domain_rand_step_warmup_and_spacing():
    sim = make_sim()
    sim.push_warmup_step, sim.domain_rand_step_interval = 24, 24
    sim._step_domian_rand(24, 100.)
    assert sim.domain_rand_phase == 'joint_dynamics'
    sim._step_domian_rand(25, 100.)
    assert sim.domain_rand_phase == 'mass_com'
    sim._step_domian_rand(48, 100.)
    assert sim.domain_rand_phase == 'mass_com'
    sim._step_domian_rand(49, 100.)
    assert sim.domain_rand_mass_com_progress > 0.
    assert sim.domain_rand_last_step_iter == 49


def test_reliability_patience_uses_collection_delta():
    from test_b1z1_force_and_kl_curricula import PACTEventConditionedForceGateTests
    from rsl_rl.algorithms.ppo_b1z1_pact import FORCE_GATE_METRIC_NAMES
    a = PACTEventConditionedForceGateTests._ppo()
    a.cfg.update(force_gate_patience_env_steps=48, pinn_start_env_step=0,
                 pinn_warmup_env_steps=48, pinn_loss_weight=.1)
    errors = {name: .01 for name in FORCE_GATE_METRIC_NAMES}
    samples = {name: 4 for name in FORCE_GATE_METRIC_NAMES}
    prepare_step_schedules(a, 24)
    a._update_event_conditioned_force_gate(errors, samples, .01)
    assert a.force_gate_count == 24 and not a.force_gate_active
    prepare_step_schedules(a, 24)
    a._update_event_conditioned_force_gate(errors, samples, .01)
    assert a.force_gate_count == 24 and not a.force_gate_active
    prepare_step_schedules(a, 48)
    a._update_event_conditioned_force_gate(errors, samples, .01)
    assert a.force_gate_count == 48 and a.force_gate_active


def test_genesis_domain_rand_checkpoint_reconstructs_bounds():
    from legged_gym.simulator.genesis_simulator_b1z1_pact import GenesisSimulatorB1Z1PACT
    def simulator():
        sim = GenesisSimulatorB1Z1PACT.__new__(GenesisSimulatorB1Z1PACT)
        sim._cfg, sim._device, sim._has_gripper = B1Z1PACTCfg(), 'cpu', True
        sim._cfg.env.num_envs = 2
        sim._cfg.domain_rand.use_domainrand_curriculum = True
        sim._cfg.domain_rand.push_warmup_env_steps = 0
        sim._cfg.domain_rand.step_interval_env_steps = 24
        sim._cfg.domain_rand.joint_dynamics_progress_delta = .5
        sim._print_domain_rand_values = lambda *args: None
        sim._parse_cfg()
        return sim
    original = simulator()
    original._step_domian_rand(24, 100.)
    restored = simulator()
    restored.load_domain_rand_curriculum_state_dict(original.domain_rand_curriculum_state_dict())
    assert restored.domain_rand_last_step_iter == 24
    torch.testing.assert_close(torch.as_tensor(restored.joint_damping_bound_current),
                               torch.as_tensor(original.joint_damping_bound_current))
    restored._step_domian_rand(47, 100.)
    assert restored.domain_rand_joint_dynamics_progress == .5
    restored._step_domian_rand(48, 100.)
    assert restored.domain_rand_joint_dynamics_progress == 1.
