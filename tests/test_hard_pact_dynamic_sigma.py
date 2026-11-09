"""Tracking reward relaxation without simulator startup."""
from types import SimpleNamespace as NS
import pytest
import torch
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import GO2HardPACTCfg
from legged_gym.envs.go2.go2_hard_pact.ablation_configs import make_hard_pact_variant_configs
from legged_gym.envs.go2.go2_hard_pact.ablations import HARD_PACT_VARIANTS


def env(enabled=True):
    task=Go2HardPACT.__new__(Go2HardPACT)
    task.cfg=NS(rewards=NS(tracking_sigma=.25,dynamic_tracking_sigma_enabled=enabled,
        dynamic_sigma=dict(min_lin_vel=.5,max_lin_vel=1.5,min_ang_vel=1.,max_ang_vel=2.,
                           max_sigma=[.25,.25,.5,.5,.75,5/12,1.,1.,1.,1.,1.])),
        terrain=NS(curriculum=True,num_cols=4,terrain_proportions=[.5,0,0,0,0,0,0,.5,0]))
    task.commands=torch.tensor([[.5,0.,1.],[1.,-1.,1.5],[1.5,-1.5,2.],[2.,2.,3.]])
    task.simulator=NS(terrain_types=torch.tensor([0,1,2,3]),terrain_levels=torch.tensor([9,9,9,0]),
        base_lin_vel=torch.zeros(4,3),base_ang_vel=torch.zeros(4,3))
    return task


def test_column_mapping_command_and_level_relaxation():
    task=env()
    sigma=task._dynamic_tracking_sigma(task.commands[:,:2])
    # First two columns are slopes; last two gaps, not stairs by column ID.
    torch.testing.assert_close(sigma[:2],torch.full((2,2),.25))
    torch.testing.assert_close(sigma[2],torch.ones(2))
    torch.testing.assert_close(sigma[3],(.25+.75*(torch.exp(torch.tensor(.1))-1)).expand(2))
    task.simulator.terrain_kind_ids=torch.full((4,),7)
    torch.testing.assert_close(task._dynamic_tracking_sigma(task.commands[:,2],True)[:3],
                               torch.tensor([.25,.625,1.]))
    task.cfg.terrain.curriculum=False
    assert task._dynamic_tracking_sigma(task.commands[:,2],True).eq(.25).all()


def test_disabled_exact_rewards_and_unchanged_curriculum_evidence():
    task=env(False)
    for name in ('_reward_tracking_lin_vel','_reward_tracking_ang_vel'):
        torch.testing.assert_close(getattr(task,name)(),getattr(task._legacy_task_class,name)(task),rtol=0,atol=0)
    assert not hasattr(task,'_dynamic_sigma_tables')
    task.cfg.rewards.dynamic_tracking_sigma_enabled=True
    task._command_tracking_collect=True;task._command_tracking_sum=torch.zeros(());task._command_tracking_count=0
    legacy=task._legacy_task_class._reward_tracking_lin_vel(task)
    relaxed=task._reward_tracking_lin_vel()
    assert (relaxed>=legacy).all() and (relaxed>legacy).any()
    torch.testing.assert_close(task._command_tracking_sum,legacy.sum())
    assert task._command_tracking_count==4
    expected=torch.exp(-task.commands[:,2].square()/task._dynamic_tracking_sigma(task.commands[:,2],True))
    torch.testing.assert_close(task._reward_tracking_ang_vel(),expected)


@pytest.mark.parametrize('variant',list(HARD_PACT_VARIANTS))
def test_ablations_inherit_same_master_setting(variant):
    cfg,_=make_hard_pact_variant_configs(variant,'isaaclab')
    assert cfg.rewards is GO2HardPACTCfg.rewards
    assert cfg.rewards.dynamic_tracking_sigma_enabled==GO2HardPACTCfg.rewards.dynamic_tracking_sigma_enabled
    assert cfg.rewards.dynamic_sigma==GO2HardPACTCfg.rewards.dynamic_sigma


def wrench_env():
    task=env()
    task.cfg.rewards.wrench_tracking_sigma_enabled=True
    task.cfg.rewards.wrench_tracking_sigma_gain=.25
    task.domain_rand_curriculum=NS(progress={'disturbance':0.},schema={
        'persistent_force':NS(initial=(-4.,4.),final=(-10.,10.)),
        'persistent_vertical_force':NS(initial=(-15.,0.),final=(-70.,0.)),
        'persistent_torque':NS(initial=(-2.,2.),final=(-15.,15.)),
        'persistent_yaw_torque':NS(initial=(-1.,1.),final=(-4.,4.))})
    task._pending_disturbance_transition={'applied_sustained_wrench_world':torch.tensor([
        [0.,0.,0.,0.,0.,0.],[4.,0.,-15.,2.,2.,1.],
        [10.,0.,-70.,15.,15.,4.],[20.,0.,-140.,30.,30.,8.]])}
    return task


def test_initial_envelope_unchanged_final_capped_and_curriculum_independent():
    task=wrench_env();base=torch.full((4,),.25)
    expected=torch.tensor([.25,.25,.3125,.3125])
    torch.testing.assert_close(task._relax_tracking_sigma_for_wrench(base),expected)
    task.domain_rand_curriculum.progress['disturbance']=1.
    torch.testing.assert_close(task._relax_tracking_sigma_for_wrench(base),expected)
    # No terrain or command-speed gating of disturbance relaxation.
    task.cfg.terrain.curriculum=False
    torch.testing.assert_close(task._dynamic_tracking_sigma(torch.zeros(4,2)),expected[:,None].expand(-1,2))
    torch.testing.assert_close(task._dynamic_tracking_sigma(torch.zeros(4),True),expected)


def test_applied_wrench_smoothstep_and_nonfinite_handling():
    task=wrench_env();base=torch.ones(4)
    task._relax_tracking_sigma_for_wrench(base)
    _,_,dmin,dmax=task._tracking_wrench_envelope
    wrench=task._pending_disturbance_transition['applied_sustained_wrench_world']
    wrench[1]=wrench[2]*((dmin+dmax)/2/dmax)
    wrench[3,0]=float('nan')
    torch.testing.assert_close(task._relax_tracking_sigma_for_wrench(base),torch.tensor([1.,1.125,1.25,1.]))
    # Sampled peak / decoder / mass labels cannot affect reward relaxation.
    task._persistent_wrench_target_world=torch.full_like(wrench,1e6)
    wrench.zero_()
    assert task._relax_tracking_sigma_for_wrench(base).eq(1.).all()


def test_wrench_disabled_master_zero_gain_and_no_expanding_bounds():
    task=wrench_env();base=torch.ones(4)
    task.cfg.rewards.wrench_tracking_sigma_gain=0.
    assert task._relax_tracking_sigma_for_wrench(base) is base
    assert not hasattr(task,'_tracking_wrench_envelope')
    task.cfg.rewards.wrench_tracking_sigma_gain=.25
    task.cfg.rewards.dynamic_tracking_sigma_enabled=False
    for name in ('_reward_tracking_lin_vel','_reward_tracking_ang_vel'):
        torch.testing.assert_close(getattr(task,name)(),getattr(task._legacy_task_class,name)(task),rtol=0,atol=0)
    task.cfg.rewards.wrench_tracking_sigma_enabled=False
    assert task._relax_tracking_sigma_for_wrench(base) is base
    task.cfg.rewards.wrench_tracking_sigma_enabled=True
    for feature in task.domain_rand_curriculum.schema.values():feature.final=feature.initial
    assert task._relax_tracking_sigma_for_wrench(base) is base
