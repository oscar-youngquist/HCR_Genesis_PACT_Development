"""Real observation builders, with synthetic simulator tensors (no simulator)."""
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import GO2HardPACTCfg
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos import Go2HardPACTPos
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos_config import GO2HardPACTPosCfg
from legged_gym.envs.go2.go2_hard_pact.transition import DISTURBANCE_FIELD_DIMS, DISTURBANCE_CRITIC_DIM


def test_all_registered_ablations_share_clean_critic_hook():
    from legged_gym.envs.go2.go2_hard_pact.ablations import HARD_PACT_VARIANTS, HARD_PACT_BACKENDS
    from legged_gym.utils.task_registry import task_registry
    expected = {f"go2_hard_pact_{variant}_{backend}"
                for variant in HARD_PACT_VARIANTS for backend in HARD_PACT_BACKENDS}
    registered = {name: cls for name, cls in task_registry.task_classes.items()
                  if name.startswith("go2_hard_pact")}
    assert expected <= registered.keys()
    for name, cls in registered.items():
        assert cls._get_critic_policy_observation is Go2HardPACT._get_critic_policy_observation, name
        assert cls.compute_observations is Go2HardPACT.compute_observations, name


def fixture(cls, cfg_type, noise):
    task = cls.__new__(cls)
    cfg = task.cfg = cfg_type()
    n, j, width = 2, cfg.env.num_actions, cfg.env.num_observations
    z = lambda *shape: torch.zeros(n, *shape)
    task.num_envs, task.device = n, "cpu"
    task.add_noise = noise
    task.noise_scale_vec = torch.linspace(.01, .2, width)
    task.obs_scales = cfg.normalization.obs_scales
    task.commands = z(3)+.2
    task.commands_scale = torch.tensor([.5, 1., 2.])
    task.actions = torch.arange(2*j).float().expand(n, -1)/100
    task.obs_buf = z(width)
    task.last_obs_buf = z(width)
    task.obs_history = z(width*cfg.env.num_obs_hist)
    task.last_obs_hist = task.obs_history.clone()
    task.obs_history_deque = deque([z(width) for _ in range(cfg.env.num_obs_hist)], maxlen=cfg.env.num_obs_hist)
    legacy_width = cfg.env.num_privileged_obs-DISTURBANCE_CRITIC_DIM
    task.critic_obs_deque = deque([z(legacy_width) for _ in range(cfg.env.num_priv_stack)], maxlen=cfg.env.num_priv_stack)
    task.disturbance_critic_deque = deque([z(DISTURBANCE_CRITIC_DIM) for _ in range(cfg.env.num_priv_stack)], maxlen=cfg.env.num_priv_stack)
    fields = {name: z(size)+.03 for name, size in DISTURBANCE_FIELD_DIMS}
    task._pending_disturbance_transition = fields
    task._build_disturbance_transition = lambda: fields
    task.friction_value_offset = task.kp_scale_offset = task.kd_scale_offset = 0.
    height_count = len(cfg.terrain.measured_points_x)*len(cfg.terrain.measured_points_y)
    task.simulator = SimpleNamespace(
        projected_gravity=z(3)+.1, base_ang_vel=z(3)+.2, base_lin_vel=z(3)+.3,
        dof_pos=z(j)+.4, default_dof_pos=z(j), dof_vel=z(j)+.5,
        feedback_torques=z(j)+.6, link_contact_states=z(4)+1,
        feet_contact_indices=torch.arange(4), feet_pos=z(4,3)+.2,
        height_around_feet=z(4,3), base_pos=z(3)+.5, measured_heights=z(height_count),
        _added_base_mass=z(1)+2, _base_com_bias=z(3)+.01,
        _friction_values=z(1)+.7, _rand_push_vels=z(3), _rand_wrench_vels=z(3),
        _kp_scale=z(j)+1, _kd_scale=z(j)+1, _motor_strength=z(j)+1,
        _joint_armature=z(1), _joint_friction=z(1), _joint_damping=z(1),
        _grfs_buf=z(12)+1, normal_vector_around_feet=z(4,3),
        feedforward_tau_weight=z(1)+.4, feedback_tau_weight=z(1)+.6,
    )
    task.grf_processor = SimpleNamespace(contacts=torch.ones(n,4,dtype=torch.bool), ema=z(4,3)+1)
    return task


@pytest.mark.parametrize("cls,cfg_type", [(Go2HardPACT, GO2HardPACTCfg), (Go2HardPACTPos, GO2HardPACTPosCfg)])
def test_clean_stacked_critic_with_unchanged_noisy_actor_history(cls, cfg_type):
    noisy, clean = fixture(cls,cfg_type,True), fixture(cls,cfg_type,False)
    # Run long enough to exercise both filled histories and multiple critic frames.
    for step in range(noisy.cfg.env.num_obs_hist+1):
        for task in (noisy,clean):
            task.simulator.dof_pos.add_(.01)
        torch.manual_seed(step)
        expected_noise = (2*torch.rand_like(noisy.obs_buf)-1)*noisy.noise_scale_vec
        torch.manual_seed(step)
        noisy.compute_observations()
        clean.compute_observations()
        torch.testing.assert_close(noisy.obs_buf, clean.obs_buf+expected_noise, rtol=0, atol=0)
        assert not torch.equal(noisy.obs_buf,clean.obs_buf)
        torch.testing.assert_close(noisy.privileged_obs_buf,clean.privileged_obs_buf,rtol=0,atol=0)
        torch.testing.assert_close(noisy.explicit_labels_buf,clean.explicit_labels_buf,rtol=0,atol=0)
        torch.testing.assert_close(noisy.obs_history,torch.cat(list(noisy.obs_history_deque),-1))
        torch.testing.assert_close(noisy.obs_history_deque[-1],noisy.obs_buf)
        width=noisy.cfg.env.num_observations
        torch.testing.assert_close(noisy.critic_obs_deque[-1][:,:width],clean.obs_buf)
    assert noisy._hard_pact_observation_schema_validated


@pytest.mark.parametrize("cls,cfg_type", [(Go2HardPACT, GO2HardPACTCfg), (Go2HardPACTPos, GO2HardPACTPosCfg)])
@pytest.mark.parametrize("noise", [False, True])
def test_policy_rng_and_legacy_behavior_are_preserved(cls,cfg_type,noise):
    hard = fixture(cls,cfg_type,noise)
    legacy = fixture(cls._legacy_task_class,cfg_type,noise)
    for step in range(3):
        torch.manual_seed(step)
        hard.compute_observations()
        state = torch.random.get_rng_state()
        torch.manual_seed(step)
        legacy.compute_observations()
        assert torch.equal(state,torch.random.get_rng_state())
        for name in ("obs_buf","obs_history","last_obs_buf","llast_obs_buf","last_obs_hist","llast_obs_hist"):
            torch.testing.assert_close(getattr(hard,name),getattr(legacy,name),rtol=0,atol=0)
        # The hook is HardPACT-only: unrelated legacy critics remain unchanged.
        width=hard.cfg.env.num_observations
        torch.testing.assert_close(legacy.critic_obs_deque[-1][:,:width],legacy.obs_buf)
        if not noise:
            torch.testing.assert_close(hard.critic_obs_deque[-1],legacy.critic_obs_deque[-1])
