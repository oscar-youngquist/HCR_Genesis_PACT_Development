"""Stage 3 same-time starts, reward reductions, and nominal-metric isolation."""
from types import SimpleNamespace

import pytest
import torch

from legged_gym.envs.b1z1.b1z1_facet import B1Z1FACET, B1Z1FACETCfg
from legged_gym.envs.b1z1.b1z1_facet.reference import FACETReference, FACETReferenceBank
from legged_gym.envs.b1z1.b1z1_facet.tracking import (
    COMPONENTS, LEGACY_TRACKING, CONFLICTING_SHAPING, FACETTrackingCfg, tracking_values,
)
from legged_gym.envs.b1z1.b1z1_ppo_pos.b1z1_ppo_pos import B1Z1PPOPos
from legged_gym.utils.helpers import class_to_dict
from test_b1z1_facet_reference import setup


def packet(step, args):
    # Independently changing measured states, nominal commands and held loads;
    # replaying only the latest packet cannot reproduce the older starts.
    seed = torch.arange(24, dtype=torch.float64).reshape(2, 12)*.001 + step*.003
    seed[:, 4] = step*.02
    inputs = (seed[:, :2], seed[:, 4], args["command_world"]+step*.01,
        args["yaw_command"]+.1, args["ee_local"]+.3,
        args["ee_local_velocity"]+.01*step, args["height"]+.5)
    loads = (args["ee_force"]+(.8 if step < 20 else -1.2),
             args["base_force"]+step*.02, args["yaw_moment"]+step*.01)
    return seed, inputs, loads


def seed_reference(ref, seed):
    ref.reset(torch.arange(len(seed)), seed[:, :2], seed[:, 2:4], seed[:, 4], seed[:, 5], seed[:, 6:9], seed[:, 9:12])


def test_exact_ages_against_independent_offline_replay():
    ref, args = setup()
    bank = FACETReferenceBank(ref)
    packets = []
    for step in range(40):
        seed, inputs, loads = packet(step, args)
        packets.append((seed, inputs, loads))
        bank.advance(seed, .02, inputs, loads)
        snapshot = bank.read()
        expected = torch.tensor([step+1 >= h for h in bank.horizons]).expand(2, -1)
        assert torch.equal(snapshot["valid"], expected)
        if step+1 not in (8, 16, 32, 40):
            continue
        for col, h in enumerate(bank.horizons):
            if step+1 < h:
                continue
            start = step+1-h
            independent = FACETReference(2, "cpu", ref.cfg, torch.float64)
            seed_reference(independent, packets[start][0])
            for _, command, load in packets[start:step+1]:
                independent.advance(.02, *command, *load)
            assert (snapshot["ages"][:, col] == h).all()
            torch.testing.assert_close(snapshot["targets"][:, col], independent._state, rtol=1e-12, atol=1e-12)


def test_partial_reset_rng_parameter_sharing_and_terminal_snapshot():
    ref, args = setup(sample_ranges={"ee_mass": [1, 3], "alpha": [.2, .8]})
    bank = FACETReferenceBank(ref)
    rng = ref._rng.get_state().clone()
    for step in range(33):
        seed, inputs, loads = packet(step, args)
        bank.advance(seed, .02, inputs, loads)
    assert torch.equal(ref._rng.get_state(), rng)
    terminal = bank.read()
    other = bank._state[1].clone()
    ref.reset(torch.tensor([0]), seed[:1, :2], seed[:1, 2:4], seed[:1, 4], seed[:1, 5], seed[:1, 6:9], seed[:1, 9:12])
    bank.clear(torch.tensor([0]))
    torch.testing.assert_close(bank._state[1], other)
    for key in terminal:
        torch.testing.assert_close(terminal[key], bank.read()[key])
    for name, parameter in ref._parameters.items():
        torch.testing.assert_close(parameter, bank._parameters[name][:, 0])
    bank.advance(seed, .02, inputs, loads)
    assert not bank.read()["valid"][0].any()
    assert bank.read()["valid"][1].all()
    result = bank.read(); result["targets"].fill_(99)
    assert not (bank.read()["targets"] == 99).any()


def test_reward_formulas_masks_wrap_and_nonlinear_average():
    settings = FACETTrackingCfg()
    widths = torch.tensor(list(settings.bandwidths.values()), dtype=torch.float64)
    targets = torch.zeros(2, 3, 12, dtype=torch.float64)
    measured = torch.zeros(2, 12, dtype=torch.float64)
    targets[0, 0] = torch.tensor([.2, -.1, .3, .4, 2*torch.pi+.1, .3, .1, .2, .3, .4, .5, .6])
    targets[0, 1] = targets[0, 0]*.5
    valid = torch.tensor([[True, True, False], [False, False, False]])
    values, e2 = tracking_values(targets, measured, valid, widths)
    torch.testing.assert_close(e2[0, 0], torch.tensor([.05, .25, .01, .09, .14, .77], dtype=torch.float64), atol=1e-7, rtol=1e-5)
    expected = torch.exp(-e2[0, :2]/widths)-e2.new_tensor([0, .25, 0, .25, 0, 0])*e2[0, :2]
    torch.testing.assert_close(values[0], expected.mean(0))
    assert values[1].count_nonzero() == 0
    targets.zero_(); targets[0, 0, 6] = .3; targets[0, 1, 6] = -.3
    values, _ = tracking_values(targets, measured, valid, widths)
    assert values[0, 4] == pytest.approx(torch.exp(torch.tensor(-.9)).item())
    assert values[0, 4] < 1  # Averaging these targets first would incorrectly yield reward 1.


def reward_env(enabled=True):
    env = B1Z1FACET.__new__(B1Z1FACET)
    env.cfg, env.device, env.num_envs, env.num_actions, env.dt = B1Z1FACETCfg(), "cpu", 2, 17, .02
    env.cfg.facet_tracking.enabled = enabled
    env.reward_scales = class_to_dict(env.cfg.rewards.scales)
    env._prepare_reward_function()
    env.rew_buf = torch.zeros(2)
    env._facet_snapshot = None
    env.compliance = None
    env.commands = torch.tensor([[.4, -.1, .2], [.1, .2, -.2]])
    env.simulator = SimpleNamespace(base_lin_vel=torch.tensor([[.3, -.1, .1], [0., .2, .1]]),
        base_ang_vel=torch.tensor([[.1, .1, .3], [.1, .1, 0.]]), ee_pos=torch.zeros(2, 3))
    # A nominal target cache is sufficient for the unchanged parent's EE formula.
    env._force_adjusted_ee_target_cache = SimpleNamespace(effective_target=torch.tensor([[.1, -.2, .3], [.2, .1, .2]]))
    env._facet_measured_state = lambda: (torch.zeros(2, 3), torch.zeros(2, 3),
        torch.zeros(2), torch.zeros(2), torch.zeros(2, 3), torch.zeros(2, 3))
    env.max_episode_length = 100
    env.max_episode_length_s = 2.
    return env


def test_optimization_exclusions_legacy_values_and_dt_exactly_once():
    env = reward_env()
    assert not set(env.reward_names) & set(LEGACY_TRACKING+CONFLICTING_SHAPING)
    for name, weight in env.cfg.facet_tracking.weights.items():
        assert env.reward_scales["facet_"+name] == pytest.approx(weight*.02)
    expected_legacy = {key: getattr(B1Z1PPOPos, "_reward_"+key)(env)*env.reward_scales[key] for key in LEGACY_TRACKING}
    # Isolate the six FACET terms to hand-check the aggregator; production keeps
    # all other nonconflicting regularizers and their schedules.
    env.reward_names = ["facet_"+key for key in COMPONENTS]
    env.reward_functions = [getattr(env, "_reward_"+key) for key in env.reward_names]
    env._facet_snapshot = dict(targets=torch.zeros(2, 3, 12), valid=torch.tensor([[True, False, False], [False, False, False]]),
                               ages=torch.tensor([[8, -1, -1], [-1, -1, -1]]))
    env.compute_reward()
    torch.testing.assert_close(env.rew_buf, torch.tensor([sum(env.cfg.facet_tracking.weights.values())*.02, 0.]))
    for key, expected in expected_legacy.items():
        torch.testing.assert_close(env.episode_sums[key], expected)
    before = env.get_facet_tracking()
    for _ in range(3):
        for fn in env.reward_functions:
            fn()
        env.get_facet_tracking()
    for key in before:
        torch.testing.assert_close(before[key], env.get_facet_tracking()[key])
    # Existing command-threshold code still sees the baseline scale and sum.
    env.cfg.commands.curriculum_threshold = 0.
    env.command_ranges = {k: [-.1, .1] for k in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")}
    env._update_command_curriculum(torch.arange(2))
    assert env.command_ranges["lin_vel_x"][1] > .1


def test_stage2_switch_and_schedule_preservation():
    env = reward_env(False)
    baseline = reward_env(False)
    baseline.reward_scales = class_to_dict(baseline.cfg.rewards.scales)
    B1Z1PPOPos._prepare_reward_function(baseline)
    assert env.reward_names == baseline.reward_names
    assert env.reward_scales == baseline.reward_scales
    active = reward_env()
    for name in set(env.reward_names)-set(LEGACY_TRACKING+CONFLICTING_SHAPING):
        assert active.reward_scales[name] == env.reward_scales[name]
    active.use_reward_curriculum = True
    active.reward_warmup_steps, active.reward_curr_steps = 0, 10
    active.reward_curr_keys = ["stand_still", "lin_vel_z"]
    active.reward_curr_bounds = {"stand_still": (-1, -2), "lin_vel_z": (-1, -2)}
    active.step_reward_curriculum(10)
    assert "stand_still" not in active.reward_names and "stand_still" not in active.reward_scales
    assert active.reward_scales["lin_vel_z"] == -.04
    assert not active.cfg.terrain.curriculum


def test_resampled_commands_do_not_retarget_completed_transition():
    env = reward_env()
    env._facet_snapshot = dict(targets=torch.zeros(2, 3, 12), valid=torch.ones(2, 3, dtype=torch.bool),
                               ages=torch.tensor([[8, 16, 32], [8, 16, 32]]))
    env.reward_names = ["facet_"+key for key in COMPONENTS]
    env.reward_functions = [getattr(env, "_reward_"+key) for key in env.reward_names]
    # Mimic the parent's post-step command/goal resampling before rewards.
    env.commands.fill_(100)
    env._force_adjusted_ee_target_cache.effective_target.fill_(100)
    env.compute_reward()
    torch.testing.assert_close(env._facet_values, torch.ones(2, 6))
    for key in LEGACY_TRACKING:
        assert env.episode_sums[key].count_nonzero() == 0


def test_terminal_snapshot_and_legacy_episode_logging(monkeypatch):
    env = reward_env()
    ref = FACETReference(2, "cpu", env.cfg.facet_reference)
    seed_reference(ref, torch.zeros(2, 12))
    env.facet_reference = ref
    env.facet_history = FACETReferenceBank(ref)
    env._facet_interval = None
    env._facet_snapshot = dict(targets=torch.zeros(2, 3, 12), valid=torch.ones(2, 3, dtype=torch.bool),
                               ages=torch.tensor([[8, 16, 32], [8, 16, 32]]))
    env.reward_names = ["facet_"+key for key in COMPONENTS]
    env.reward_functions = [getattr(env, "_reward_"+key) for key in env.reward_names]
    env.compute_reward()
    nominal = {k: v.clone() for k, v in env.episode_sums.items() if k in LEGACY_TRACKING}
    terminal = env.get_facet_tracking()
    def parent_reset(self, ids):
        self.extras = {"episode": {"rew_"+k: v[ids].mean()/self.max_episode_length_s for k, v in self.episode_sums.items()}}
        for value in self.episode_sums.values():
            value[ids] = 0
    monkeypatch.setattr(B1Z1PPOPos, "reset_idx", parent_reset)
    env.reset_idx(torch.tensor([0]))
    for key in terminal:
        torch.testing.assert_close(terminal[key], env.get_facet_tracking()[key])
    for key, value in nominal.items():
        torch.testing.assert_close(env.extras["episode"]["rew_"+key], value[0]/2.)
    assert env.extras["episode"]["facet/h32/valid_fraction"] == 1
    assert (env.facet_history._ages[0] == -1).all()
    assert env._facet_valid_steps[0].count_nonzero() == 0
    assert env._facet_valid_steps[1].count_nonzero() == 3


@pytest.mark.parametrize("horizons", [(), (0,), (8, 8), (True,), (1.5,)])
def test_invalid_horizons(horizons):
    ref, _ = setup()
    with pytest.raises(ValueError):
        FACETReferenceBank(ref, horizons)
