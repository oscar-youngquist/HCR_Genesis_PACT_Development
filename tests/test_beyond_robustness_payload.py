"""Stage-3 tensor contracts and iteration integration; physical checks are separate."""
from dataclasses import replace
from types import SimpleNamespace
from copy import deepcopy
import math
import torch
import pytest
from rsl_rl.beyond_robustness_payload import *
from rsl_rl.beyond_robustness_env import BRObservation
from rsl_rl.beyond_robustness_schema import BeyondRobustnessInputSchema, PARAMETER_FIELDS, InputField
from test_beyond_robustness_runner import SyntheticBR, make


def test_rotating_frame_and_spawn_origin_velocity():
    q=torch.tensor([[math.cos(.4),0.,math.sin(.4),0.]])
    base=torch.tensor([[1.,2.,3.]]); vel=torch.tensor([[.1,.2,.3]])
    omega=torch.tensor([[.7,-.5,1.2]])
    r=torch.tensor([[.1,-.2,.25]]); dr=torch.tensor([[.3,.1,-.2]])
    state=spawn_state(q,base,vel,omega,r,dr)
    labels=load_labels(q,base,vel,omega,state[:,:3],state[:,7:10],torch.tensor([2.]))
    torch.testing.assert_close(labels,torch.cat((r,dr,torch.tensor([[2.]])),-1),atol=2e-7,rtol=1e-6)
    stationary=spawn_state(q,base,vel,omega,r,torch.zeros_like(dr))
    result=load_labels(q,base,vel,omega,stationary[:,:3],stationary[:,7:10],torch.tensor([2.]))
    assert result[:,3:6].abs().max()<2e-7  # Co-moving load, NOT world stationary


def test_payload_geometry_and_mass():
    cfg=SlidingPayloadConfig()
    assert cfg.ranges(0)['mass']==cfg.mass_initial and cfg.ranges(1)['mass']==cfg.mass_final
    r=torch.tensor([[0.,0.,.5],[0.,0.,-.1],[1.,0.,.2]])
    assert lost_payload(r,cfg).tolist()==[False,True,True]
    inertia=cuboid_inertia(torch.tensor([1.,2.]),(.2,.3,.4))
    torch.testing.assert_close(inertia[1],2*inertia[0])
    assert (torch.linalg.eigvalsh(inertia)>0).all()
    with pytest.raises(ValueError): SlidingPayloadConfig(box_size_max=(2.,2.,2.))


def test_owned_terminal_and_failure_precedence():
    tensors=[torch.ones(3,n) for n in (45,84,88,7)]+[torch.ones(3,dtype=torch.bool)]
    packet=BRObservation(*tensors); snapshot=packet.clone()
    for x in tensors: x[1]=0
    assert snapshot.load_labels[1].eq(1).all() and snapshot.supervision_valid.all()
    done,timeout=termination_masks(torch.tensor([True,False,False]),torch.tensor([10,10,2]),10)
    assert done.tolist()==[True,True,False] and timeout.tolist()==[False,True,False]


def test_named_parameter_contract():
    schema=BeyondRobustnessInputSchema(terrain_points_x=(),terrain_points_y=(),
        parameter_fields=PARAMETER_FIELDS+(InputField('tray_mass',1,'kg'),))
    fields={f.name:torch.full((2,f.width),float(i)) for i,f in enumerate(schema.parameter_fields)}
    result=schema.pack('parameters',fields)
    assert result.shape==(2,89) and result[:,-1].eq(7).all()
    assert named_indices(['FL','RL','FR','RR'],['FR','FL','RR','RL'])==[2,0,3,1]
    with pytest.raises(ValueError): named_indices(['FR'],['FL'])
    with pytest.raises(ValueError): schema.pack('parameters',{k:v for k,v in fields.items() if k!='tray_mass'})


def test_runner_iteration_evidence_is_not_entropy_summary():
    runner=make(); env=runner.env
    calls=[]; original=env.step
    def step(actions):
        out=original(actions)
        out.curriculum_tracking=torch.tensor([.2,.4,float('nan')])
        return out
    env.step=step
    env.begin_iteration=lambda iteration:calls.append(('begin',iteration))
    env.end_iteration=lambda iteration,score,count:calls.append(('end',iteration,score,count))
    runner.learn(2)
    assert [c[:2] for c in calls]==[('begin',0),('end',0),('begin',1),('end',1)]
    assert calls[1][2]==pytest.approx(.3) and calls[1][3]==6
    env.step=original
    runner.learn(1)
    assert calls[-1]==('end',2,None,0)


def test_physical_curriculum_missing_evidence_resume_and_reset_cadence():
    from legged_gym.envs.go2.go2_beyond_robustness import IsaacLabBeyondRobustnessEnv as Env
    from legged_gym.envs.go2.go2_hard_pact.ablation_configs import make_hard_pact_backend_configs
    from legged_gym.envs.go2.go2_hard_pact.domain_rand_curriculum import HardPACTDomainRandCurriculum
    from legged_gym.simulator.reset_randomization_cadence import ResetRandomizationCadence
    cfg=make_hard_pact_backend_configs('isaaclab')[0]()
    cfg.domain_rand.use_joint_dynamics_curriculum=False; cfg.domain_rand.use_mass_com_curriculum=False
    cfg.domain_rand.use_disturbance_curriculum=True; cfg.domain_rand.push_warmup=0; cfg.domain_rand.step_interval=1
    cfg.domain_rand.min_reward_to_step=0.; cfg.commands.curriculum=False
    env=Env.__new__(Env); env.cfg=cfg; env.last_completed_iteration=-1; env.command_streak=0
    from rsl_rl.beyond_robustness_stages import BRStageState,BREnvironmentStages
    env.stage_state=BRStageState(BREnvironmentStages())
    env.domain_rand_curriculum=HardPACTDomainRandCurriculum(cfg)
    updates=[]
    env.simulator=SimpleNamespace(use_domainrand_curriculum=True,set_payload_progress=updates.append)
    env._apply_domain_rand_curriculum_state=lambda:None
    env.end_iteration(1,None,0); assert not updates
    env.end_iteration(2,.9,10); p=updates[-1]
    env.end_iteration(2,.9,10); assert updates==[p]
    saved=deepcopy(env.domain_rand_curriculum.state_dict())
    other=HardPACTDomainRandCurriculum(cfg); other.load_state_dict(saved)
    assert other.advance(3,.9)==env.domain_rand_curriculum.advance(3,.9)
    assert other.state_dict()['progress']==env.domain_rand_curriculum.state_dict()['progress']
    cadence=ResetRandomizationCadence(3,'cpu',100)
    cadence.update_ranges({'mass':(.1,.2)})
    ids=torch.tensor([0,1,2]); cadence.begin_reset(ids)
    assert cadence.select('mass',ids).tolist()==[0,1,2]
    for _ in range(99): cadence.begin_reset(torch.tensor([1]))
    assert cadence.select('mass',ids).numel()==0
    cadence.begin_reset(torch.tensor([1])); assert cadence.select('mass',ids).tolist()==[1]
    cadence.update_ranges({'mass':(.2,.4)})
    assert cadence.select('mass',torch.tensor([2])).tolist()==[2]
    assert cadence.episodes.tolist()==[0,100,0]
