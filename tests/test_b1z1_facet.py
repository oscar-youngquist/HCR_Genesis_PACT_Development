"""FACET configuration isolation and unchanged position-action plumbing."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from legged_gym.envs.b1z1.b1z1_facet import B1Z1FACET, B1Z1FACETCfg, B1Z1FACETCfgPPO
from legged_gym.envs.b1z1.b1z1_ppo_pos.b1z1_ppo_pos import B1Z1PPOPos
from legged_gym.envs.b1z1.b1z1_ppo_pos.b1z1_ppo_pos_config import B1Z1PPOPosCfg, B1Z1PPOPosCfgPPO
from legged_gym.envs.b1z1.b1z1_unifp_original.b1z1_unifp_original_config import copy_sections
from legged_gym.envs.b1z1.compliance_task import configure, configure_algorithm
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict, get_args


def differences(a, b, prefix=""):
    result = set()
    for key in a.keys() | b.keys():
        x, y = a.get(key), b.get(key)
        path = prefix + key
        if isinstance(x, dict) and isinstance(y, dict):
            result |= differences(x, y, path + ".")
        elif not np.array_equal(x, y):
            result.add(path)
    return result


def test_config_allowlist_and_resolved_layout():
    baseline = copy_sections(B1Z1PPOPosCfg)
    facet = B1Z1FACETCfg()
    gain_paths = {f'control.{kind}.{name}' for kind in ('stiffness', 'damping')
                  for name in set(getattr(facet.control, kind)) | set(getattr(baseline.control, kind))}
    assert differences(class_to_dict(facet), class_to_dict(baseline)) == {"compliance.enabled", "facet_reference", "facet_tracking", "facet_observations"} | gain_paths
    train, base_train = B1Z1FACETCfgPPO(), copy_sections(B1Z1PPOPosCfgPPO)
    from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfgPPO
    from legged_gym.envs.b1z1.b1z1_facet.b1z1_facet_config import COMMON_PPO
    pilot = copy_sections(B1Z1PACTCfgPPO)
    for key in COMMON_PPO:
        assert getattr(train.algorithm, key) == getattr(pilot.algorithm, key)
    assert train.runner.max_iterations == 70000 and train.runner.save_interval == 1000
    assert train.runner.num_steps_per_env == 24
    for name in facet.asset.dof_names:
        assert facet.control.stiffness[name] == (40 if name.startswith('z1_') else 100)
        assert facet.control.damping[name] == (1 if name.startswith('z1_') else 2)
    configure(facet)
    configure(baseline)
    # Turning off compliance intentionally avoids its profile slots and its
    # rigid-tracking reward suppression. Stage 3 registration is tested separately;
    # raw nominal coefficients remain available for diagnostic-only bookkeeping.
    assert differences(class_to_dict(facet), class_to_dict(baseline)) == {
        "compliance.enabled", "facet_reference", "facet_tracking", "facet_observations", "env.num_observations", "env.num_critic_state_obs",
        "env.num_privileged_obs", "env.num_privileged_recon_obs",
        "rewards.scales.arm_progress_before_torso", "rewards.scales.ref_dof_leg",
        "rewards.scales.early_torso_tilt", "rewards.scales.feet_contact_number",
        "rewards.scales.stand_still", "rewards.scales.stand_still_contact",
        "rewards.scales.tracking_arm_feedforward",
    } | gain_paths
    for name in ("num_observations", "num_critic_state_obs", "num_privileged_obs", "num_privileged_recon_obs"):
        assert getattr(baseline.env, name) - getattr(facet.env, name) == 12
    assert facet.env.num_observations == 81
    assert facet.env.num_actions == facet.env.num_policy_actions == 17
    assert train.runner_class_name == "B1Z1FACETRunner"
    assert train.policy.action_mode == "position"
    assert not any('pinn' in key or 'actor_phys' in key for key in vars(train.algorithm))
    assert not facet.use_force_shifted_target


def test_config_isolation():
    from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg, B1Z1PACTCfgPPO
    from legged_gym.envs.b1z1.b1z1_unifp.b1z1_unifp_config import B1Z1UniFPCfg, B1Z1UniFPCfgPPO
    sources = (B1Z1PACTCfg, B1Z1PACTCfgPPO, B1Z1PPOPosCfg, B1Z1PPOPosCfgPPO,
               B1Z1UniFPCfg, B1Z1UniFPCfgPPO)
    source_snapshots = [class_to_dict(copy_sections(source)) for source in sources]
    before = {name: (class_to_dict(env), class_to_dict(task_registry.train_cfgs[name]))
              for name, env in task_registry.env_cfgs.items() if name.startswith("b1z1")}
    pristine = class_to_dict(B1Z1FACETCfg())
    cfg, train = B1Z1FACETCfg(), B1Z1FACETCfgPPO()
    cfg.control.stiffness[next(iter(cfg.control.stiffness))] = -123
    cfg.commands.ranges.lin_vel_x[0] = -99
    cfg.compliance.family_weights[0] = 0
    cfg.facet_reference.sample_ranges["ee_mass"] = [1., 3.]
    cfg.facet_tracking.weights["ee_position"] = 9.
    cfg.facet_observations.noise_seed = 999
    cfg.env.num_observations += 100
    train.algorithm.adaptive_ent_bounds[0] = 7
    configure(cfg)
    assert not differences(pristine, class_to_dict(B1Z1FACETCfg()))
    for name, (env, ppo) in before.items():
        assert not differences(env, class_to_dict(task_registry.env_cfgs[name]))
        assert not differences(ppo, class_to_dict(task_registry.train_cfgs[name]))
    for source, snapshot in zip(sources, source_snapshots):
        assert not differences(snapshot, class_to_dict(copy_sections(source)))


def test_cli_registration(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--task=b1z1_facet", "--headless"])
    args = get_args()
    assert task_registry.get_task_class(args.task) is B1Z1FACET
    assert isinstance(task_registry.env_cfgs[args.task], B1Z1FACETCfg)
    assert issubclass(B1Z1FACET, B1Z1PPOPos)
    for method in ("step", "_init_buffers",
                   "check_termination", "_resample_ee_goal", "_push_gripper", "_push_robot_base"):
        assert getattr(B1Z1FACET, method) is getattr(B1Z1PPOPos, method)


@pytest.mark.parametrize("delay", [0, 2])
def test_position_action_replay(delay):
    envs = []
    for cls, cfg in ((B1Z1PPOPos, copy_sections(B1Z1PPOPosCfg)), (B1Z1FACET, B1Z1FACETCfg())):
        cfg.domain_rand.randomize_ctrl_delay = bool(delay)
        env = cls.__new__(cls)
        env.cfg, env.device, env.num_envs, env.num_actions = cfg, "cpu", 2, 17
        env.all_env_ids = torch.arange(2)
        env.actions = torch.zeros(2, 34)
        env.last_actions, env.llast_actions = env.actions.clone(), env.actions.clone()
        env.action_queue = torch.zeros(2, delay + 1, 34)
        env.action_delay = torch.full((2,), delay, dtype=torch.long)
        default = torch.linspace(-.3, .3, 19)[None]
        env.simulator = SimpleNamespace(default_dof_pos=default)
        env.dof_tracking_target = default.repeat(2, 1)
        envs.append(env)
    for scale in (0., 100., -.1, .2):
        action = torch.linspace(-1., 1., 34).reshape(2, 17) * scale
        outputs = [env._pre_sim_step(action) for env in envs]
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        torch.testing.assert_close(envs[0].dof_tracking_target, envs[1].dof_tracking_target)
        assert outputs[1][:, 17:].count_nonzero() == 0
        torch.testing.assert_close(envs[1].dof_tracking_target[:, 17:], default[:, 17:].expand(2, -1))


def test_registered_runner_initializes_without_physics(monkeypatch):
    from rsl_rl.runners.b1z1_pact_runner import B1Z1PACTRunner
    import rsl_rl.runners.b1z1_pact_runner as runner_module
    def forbidden(*args, **kwargs):
        pytest.fail("Stage 1 must not construct a PINN dynamics backend")
    monkeypatch.setattr(runner_module, "BardB1Z1DynamicsBackend", forbidden)
    monkeypatch.setattr(runner_module, "PinocchioWholeBodyDynamics", forbidden)
    cfg, train = B1Z1FACETCfg(), B1Z1FACETCfgPPO()
    settings = class_to_dict(train)
    settings["runner"]["num_steps_per_env"] = 2
    obs = torch.zeros(2, cfg.env.num_observations)
    history = obs.repeat(1, cfg.env.num_obs_hist)
    critic = torch.zeros(2, cfg.env.num_privileged_obs * cfg.env.num_priv_stack)
    env = SimpleNamespace(cfg=cfg, num_envs=2, num_obs=obs.shape[1],
        num_privileged_obs=cfg.env.num_privileged_obs, num_crit_obs_stack=cfg.env.num_priv_stack,
        num_actions=17, num_obs_hist=cfg.env.num_obs_hist, num_exp_labels=cfg.env.num_explicit_recon_obs,
        reward_scales={}, dt=cfg.control.dt, device="cpu",
        obs_scales=cfg.normalization.obs_scales,
        base_velocity_scale=torch.ones(3), base_wrench_scale=torch.ones(6),
        simulator=SimpleNamespace(_cfg=cfg, dof_pos=torch.zeros(2, 19)),
        reset=lambda: (obs, critic))
    from rsl_rl.runners import B1Z1FACETRunner
    env.get_facet_observations = lambda: dict(observation=torch.zeros(2, 237),
        privileged=torch.zeros(2, 52), wrench=torch.zeros(2, 9), is_init=torch.ones(2, 1, dtype=torch.bool))
    monkeypatch.setattr('sys.argv', ['train.py', '--task=b1z1_facet', '--cpu', '--headless'])
    runner, _ = task_registry.make_alg_runner(env, name='b1z1_facet', args=get_args(), train_cfg=train, log_root=None)
    assert isinstance(runner, B1Z1FACETRunner)
    assert not hasattr(runner, 'dynamics')
    with torch.no_grad():
        actions, _ = runner.get_inference_policy()(torch.zeros(2, 237), torch.zeros(2, 128), torch.ones(2, 1, dtype=torch.bool))
    assert actions.shape == (2, 17) and torch.isfinite(actions).all()
