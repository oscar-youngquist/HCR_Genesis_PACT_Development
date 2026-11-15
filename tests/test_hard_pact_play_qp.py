from copy import deepcopy
from types import SimpleNamespace as NS
import pytest
import torch
from rsl_rl.hard_pact_ablations import RSS_HARD_PACT_ABLATIONS
from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig, qp_substep_mask
from legged_gym.scripts.hard_pact_play_qp import configure_play_qp


def fixture(monkeypatch, enabled=True):
    import rsl_rl.algorithms.hard_pact_qp as module
    made=[]
    class QP:
        def __init__(self,cfg,*limits): self.cfg=cfg; self.clears=0; made.append(self)
        def clear_warm_start(self): self.clears+=1
    monkeypatch.setattr(module,'HardPACTDifferentiableQP',QP)
    cfg=NS(play=NS(qp_enabled=enabled),task_backend='isaaclab',
        control=NS(torque_rate_limit_nm_s=900.),rewards=NS(base_height_target=.3),
        normalization=NS(obs_scales=NS(lin_vel=2.)))
    env=NS(cfg=cfg,hard_pact_features=RSS_HARD_PACT_ABLATIONS['guided_no_qp'],
        simulator=NS(torque_limits=torch.ones(12)*24,dof_pos_limits_hard=torch.tensor([[-1.,1.]]*12),dof_vel_limits=torch.ones(12)*30))
    def gate(on): env._hard_pact_rollout_qp_enabled=on
    def bind(actor,dynamics,qp): env.bound=(actor,dynamics,qp)
    env.set_hard_pact_qp_enabled=gate; env.configure_hard_pact_substep_qp=bind
    runner=NS(is_hard_pact=True,alg=NS(actor_critic=NS(num_policy_actions=24),physics_dynamics=object()))
    train=NS(algorithm=NS(hard_pact_qp=dict(qp_update_mode='first_plus_random_substep',
        warmup_iterations=12345,correction_ramp_enabled=True,objective_curriculum_enabled=True,
        contact_acceleration_weight=7.,attitude_weight=9.,planar_velocity_weight=11.,qp_solver='cupiqp')))
    return env,runner,train,made


def test_every_substep_uses_config_and_ignores_training_gates(monkeypatch):
    env,runner,train,made=fixture(monkeypatch)
    original=deepcopy(train.algorithm.hard_pact_qp); features=env.hard_pact_features
    qp=configure_play_qp(env,runner,train)
    assert qp.cfg.qp_update_mode=='every_substep' and qp.cfg.warmup_iterations==0
    assert not qp.cfg.correction_ramp_enabled and not qp.cfg.objective_curriculum_enabled
    assert env._qp_execution_alpha==1 and env._hard_pact_rollout_qp_enabled
    assert env.hard_pact_features.execution_qp and not features.execution_qp
    assert env.bound[:2]==(runner.alg.actor_critic,runner.alg.physics_dynamics)
    assert qp.clears==1 and len(made)==1 and train.algorithm.hard_pact_qp==original
    for key in ('contact_acceleration_weight','attitude_weight','planar_velocity_weight','qp_solver'):
        assert getattr(qp.cfg,key)==original[key]
    selected=torch.tensor([0,1,2,3])
    assert torch.stack([qp_substep_mask(qp.cfg.qp_update_mode,k,selected) for k in range(4)]).sum(0).eq(4).all()
    assert qp.cfg.height_target==.3 and qp.cfg.height_velocity_obs_scale==2.


def test_opt_out_does_not_construct_deployment_qp(monkeypatch):
    env,runner,train,made=fixture(monkeypatch)
    assert configure_play_qp(env,runner,train,False) is None
    assert not made and not env._hard_pact_rollout_qp_enabled
    env.cfg.play.qp_enabled=False
    assert configure_play_qp(env,runner,train) is None


@pytest.mark.parametrize('invalid',['position','backend','pos_runner'])
def test_reject_unsupported_policy_before_constructing_qp(monkeypatch,invalid):
    env,runner,train,made=fixture(monkeypatch)
    if invalid=='position': runner.alg.actor_critic.num_policy_actions=12
    elif invalid=='backend': env.cfg.task_backend='genesis'
    else: runner.is_hard_pact=False
    with pytest.raises(ValueError,match='coupled HardPACT Isaac Lab'):
        configure_play_qp(env,runner,train)
    assert not made


def test_play_wrapper_forwards_deployment_switch_without_training():
    import os
    import subprocess
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run(['bash',str(root/'legged_gym/scripts/play_go2_hard_pact.sh'),
        '--task','go2_hard_pact_isaaclab','--gpu','cuda:1','--load_run','/tmp/example run',
        '--ckpt','3500','--play-qp'],env={**os.environ,'HARD_PACT_SKIP_CONDA_ACTIVATE':'1',
        'HARD_PACT_PYTHON':'/bin/echo'},text=True,capture_output=True,check=True)
    assert 'play_go2_hard_pact_pos.py' in result.stdout
    assert '--play-qp' in result.stdout and '--gpu cuda:1' in result.stdout
    assert '--load_run /tmp/example run' in result.stdout
    assert 'simulator: isaaclab' in result.stderr
