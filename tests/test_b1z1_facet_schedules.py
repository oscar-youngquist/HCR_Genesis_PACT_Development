"""FACET gates consume temporal rewards, never nominal tracking diagnostics."""
from types import SimpleNamespace
from collections import deque
import pytest
import torch

from legged_gym.envs.b1z1.b1z1_facet import B1Z1FACETCfg, B1Z1FACETCfgPPO
from legged_gym.envs.b1z1.b1z1_facet.schedules import FACETForceCurriculum, update_force_schedule
from rsl_rl.runners.b1z1_facet_runner import B1Z1FACETRunner


def environment():
    cfg = B1Z1FACETCfg()
    cfg.commands.force_curriculum_gate_start_iteration = 0
    cfg.commands.force_curriculum_gate_patience = 1
    cfg.commands.force_curriculum_use_latest_start_fallback = False
    cfg.commands.reject_nominal_tracking_iterations = 0
    env = SimpleNamespace(cfg=cfg, device='cpu')
    env._staged_force_curriculum = FACETForceCurriculum(cfg, 'cpu')
    return env


def test_threshold_units_and_config_isolation():
    from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfgPPO
    cfg = B1Z1FACETCfgPPO()
    assert cfg.algorithm.adaptive_ent_lin_threshold == pytest.approx(1.125)
    assert cfg.algorithm.adaptive_ent_ang_threshold == pytest.approx(.70)
    assert B1Z1PACTCfgPPO.algorithm.adaptive_ent_lin_threshold == 1.5
    env = environment()
    original = env.cfg.commands.force_curriculum_ee_l1_threshold
    env.cfg.facet_tracking.force_curriculum_reward_fraction = .8
    env._staged_force_curriculum = FACETForceCurriculum(env.cfg, 'cpu')
    assert env.cfg.commands.force_curriculum_ee_l1_threshold == original
    assert env.cfg.commands.force_curriculum_ee_l1_threshold != env._staged_force_curriculum.ee_l1_threshold


@pytest.mark.parametrize('reward,nominal,expected', [(.9,100.,True), (.1,0.,False), (None,0.,False)])
def test_force_gate_uses_reward_not_nominal_distance(reward, nominal, expected):
    env = environment()
    info = {'EE/tracking_l1_mean': nominal, 'roll_termination_rate': 0.}
    if reward is not None:
        info['rew_facet_ee_position'] = reward * env.cfg.facet_tracking.weights['ee_position']
    metrics = update_force_schedule(env, 1, [info], 100000.)
    assert env._staged_force_curriculum.gate_latched == expected
    assert 'ForceCurriculum/ee_l1_ema' not in metrics
    if reward is not None:
        assert metrics['ForceCurriculum/facet_ee_reward_fraction_ema'] == pytest.approx(reward)


def test_force_gate_weight_normalization_and_resume():
    env = environment()
    env.cfg.facet_tracking.weights['ee_position'] = 2.
    update_force_schedule(env, 1, [{'rew_facet_ee_position': 1.8, 'roll_termination_rate': 0.}], 100000.)
    state = env._staged_force_curriculum.state_dict()
    restored = FACETForceCurriculum(env.cfg, 'cpu')
    restored.load_state_dict(state)
    assert restored.ee_l1_ema == pytest.approx(.1)
    assert restored.gate_latched
    del state['facet_reward_gate_version']
    with pytest.warns(UserWarning, match='legacy nominal-error'):
        restored.load_state_dict(state)
    assert restored.ee_l1_ema is None
    assert restored.gate_latched  # Already-exposed loads must not restart.


def test_runner_entropy_and_domain_inputs():
    env = environment()
    calls, entropy = [], []
    env.simulator = SimpleNamespace(use_domainrand_curriculum=True,
        _step_domian_rand=lambda *args: calls.append(args))
    runner = object.__new__(B1Z1FACETRunner)
    runner.env, runner.lengths = env, deque([100000.])
    runner.alg = SimpleNamespace(cfg=vars(B1Z1FACETCfgPPO().algorithm),
        update_entropy=lambda *args: entropy.append(args))
    info = dict(rew_facet_base_velocity=1.2, rew_facet_yaw_rate=.8,
        rew_tracking_lin_vel_force_world=999., rew_tracking_ang_vel=999.)
    runner._schedules(1, [info])
    assert entropy[0][:2] == pytest.approx((1.2,.8))
    assert calls[0][1] == pytest.approx(1.6)  # .8 success in shared 2.0 units.


def test_command_curriculum_ignores_nominal_score():
    from test_b1z1_facet_history import reward_env
    env = reward_env()
    env.cfg.commands.curriculum_threshold = .5
    env.command_ranges = {k: [-.1,.1] for k in ('lin_vel_x','lin_vel_y','ang_vel_yaw')}
    env.episode_sums['tracking_lin_vel_force_world'].fill_(1000.)
    env._update_command_curriculum(torch.arange(2))
    assert env.command_ranges['lin_vel_x'][1] == .1
    env.episode_sums['facet_base_velocity'].fill_(env.max_episode_length * env.reward_scales['facet_base_velocity'])
    env.episode_sums['tracking_lin_vel_force_world'].zero_()
    env._update_command_curriculum(torch.arange(2))
    assert env.command_ranges['lin_vel_x'][1] == pytest.approx(.3)


def test_legacy_entropy_thresholds_migrate_once(tmp_path):
    from test_b1z1_facet_training import make_runner
    runner = make_runner(2)
    runner.alg.cfg.pop('facet_reward_schedule_version')
    runner.alg.cfg['adaptive_ent_lin_threshold'] = 1.5
    path = tmp_path/'legacy.pt'
    runner.save(path)
    restored = make_runner(2)
    with pytest.warns(UserWarning, match='legacy nominal-error'):
        restored.load(path)
    assert restored.alg.cfg['adaptive_ent_lin_threshold'] == pytest.approx(1.125)
    restored.save(tmp_path/'migrated.pt')
    again = make_runner(2)
    again.load(tmp_path/'migrated.pt')
    assert again.alg.cfg['adaptive_ent_lin_threshold'] == pytest.approx(1.125)
