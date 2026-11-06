"""Detached local arm teacher and mixed-coordinate timing regressions."""
from types import SimpleNamespace
import pytest
import torch

from test_b1z1_actor_feedforward import fixture
from rsl_rl.algorithms import b1z1_local_feedforward as local
from rsl_rl.algorithms import b1z1_actor_feedforward as ff
from rsl_rl.algorithms.b1z1_actor_physics import integrate_pose
from legged_gym.envs.b1z1.directional_compliance import ComplianceCfg, heading, task_position, task_velocity
from legged_gym.envs.b1z1.compliance_task import configure_algorithm


def local_fixture():
    a,b,actions,context,acc,force,ids,dofs = fixture(dtype=torch.float64)
    a.cfg.update(actor_phys_arm_feedforward_compliant_local_enabled=True, actor_phys_compliance_enabled=True,
        actor_phys_compliance_omega=4., actor_phys_compliance_zeta=1.,
        actor_phys_compliance_dls_damping=.025, actor_phys_compliance_joint_margin=.03)
    state=b['actor_phys_state'].clone()
    state[:,97:116]=1.; state[:,154:156]=1.;state[:,7:26]=0;state[:,32:51]=0
    state[:,156:175]=0
    fixed=a.actor_physics_cache
    fixed.local_h=state.new_zeros((len(state),3,25))
    fixed.local_h[:,0,6+ids[0]]=1;fixed.local_h[:,1,6+ids[1]]=1
    fixed.local_s=state.new_zeros((len(state),3))
    fixed.local_bias=torch.zeros_like(fixed.local_s)
    fixed.local_kinematic_valid=torch.ones((len(state),1),dtype=torch.bool)
    data={k[11:]:v for k,v in b.items() if k.startswith('actor_phys_')}
    data.update(local_ref_position_start=torch.zeros_like(fixed.local_s),
        local_ref_velocity_start=torch.zeros_like(fixed.local_s),local_ref_acceleration=torch.zeros_like(fixed.local_s),
        local_ref_valid=torch.ones((len(state),1),dtype=torch.bool),feedforward_applied_force=force,
        q_min=state.new_full((len(state),dofs),-10),q_max=state.new_full((len(state),dofs),10),
        qd_max=state.new_full((len(state),dofs),100),torque_max=state.new_full((len(state),dofs),100))
    return a,torch.zeros_like(actions),state,fixed,acc,ids,dofs,data


@pytest.mark.parametrize('compliance',[False,True])
@pytest.mark.parametrize('main',[False,True])
@pytest.mark.parametrize('flag',[False,True])
def test_selection_and_override(compliance,main,flag):
    a,*_=local_fixture()
    a.cfg.update(actor_phys_arm_feedforward_enabled=main,
                 actor_phys_arm_feedforward_compliant_local_enabled=flag)
    a.cfg.pop('actor_phys_compliance_enabled')
    c=ComplianceCfg();c.enabled=compliance
    cfg=configure_algorithm(a.cfg,SimpleNamespace(compliance=c))
    assert cfg['actor_phys_arm_feedforward_enabled'] == main
    assert local.enabled(cfg)==(compliance and main and flag)
    assert not local.enabled(cfg,'position')
    assert ff.enabled(cfg)==main


def test_original_parity_and_zero_work(monkeypatch):
    a,act,state,fixed,acc,ids,dofs,data=local_fixture()
    a.cfg['actor_phys_arm_feedforward_compliant_local_enabled']=False
    def forbidden(*args,**kwargs):raise AssertionError('local path called')
    monkeypatch.setattr(local,'target',forbidden)
    before=ff.objective(a,act,state,fixed,acc,dofs,data)[0]
    a.cfg['actor_phys_compliance_enabled']=False
    a.cfg['actor_phys_arm_feedforward_compliant_local_enabled']=True
    torch.testing.assert_close(ff.objective(a,act,state,fixed,acc,dofs,data)[0],before,atol=0,rtol=0)


def test_torque_balance_and_reduction_to_compensation():
    a,act,s,m,acc,ids,n,d=local_fixture()
    act[:,:a.actor_critic.num_actions]=.01
    pd=a._coupled_torque(act[:,:a.actor_critic.num_actions],s)[:,ids]
    maa=m.mass_matrix[:,6+ids][:,:,6+ids]
    a0=torch.linalg.solve(maa,pd[...,None]).squeeze(-1)
    ha=m.local_h[:,:,6+ids]
    au=acc.index_fill(1,6+ids,0)
    d['local_ref_acceleration']=(ha@a0[...,None]).squeeze(-1)+(m.local_h@au[...,None]).squeeze(-1)+m.local_bias
    target,*_=local.target(a,act,s,m,acc,ids,n,d)
    original=ff.compensation_target(m,acc,d['feedforward_applied_force'],ids,n)[0]
    torch.testing.assert_close(target,original)
    d['local_ref_acceleration'][:,0]+=.3
    target,bias,coupling,external,valid,_=local.target(a,act,s,m,acc,ids,n,d)
    qa=torch.linalg.solve(maa,(target+pd-bias-coupling+external)[...,None]).squeeze(-1)
    rhs=d['local_ref_acceleration']-m.local_bias-(m.local_h@au[...,None]).squeeze(-1)-(ha@a0[...,None]).squeeze(-1)
    expected=a0+(ha.transpose(1,2)@torch.linalg.solve(ha@ha.transpose(1,2)+.025**2*torch.eye(3),(rhs)[...,None])).squeeze(-1)
    torch.testing.assert_close(qa,expected)
    assert valid.all()
    # Changing predicted arm acceleration does not copy it into the teacher.
    acc[:,6+ids]*=100
    torch.testing.assert_close(local.target(a,act,s,m,acc,ids,n,d)[0],target)


@pytest.mark.parametrize('position,velocity,accel',[(0,0,0),(.01,.02,.1),(.01,.02,-.2),(-.01,-.02,-.1),(0,-.02,0)])
def test_reference_changes_detached_teacher_and_ff_gradients(position,velocity,accel):
    a,act,s,m,acc,ids,n,d=local_fixture()
    initial=local.target(a,act,s,m,acc,ids,n,d)[0]
    d['local_ref_position_start'][:,0]=position
    d['local_ref_velocity_start'][:,0]=velocity
    d['local_ref_acceleration'][:,0]=accel
    changed=local.target(a,act,s,m,acc,ids,n,d)[0]
    if position or velocity or accel:assert not torch.equal(initial,changed)
    act.requires_grad_();s.requires_grad_();acc.requires_grad_()
    for value in vars(m).values():
        if value.is_floating_point():value.requires_grad_()
    for value in d.values():
        if value.is_floating_point():value.requires_grad_()
    loss,_=ff.objective(a,act,s,m,acc,n,d);loss.backward()
    assert torch.isfinite(loss) and act.grad[:,a.actor_critic.num_actions:].abs().sum()>0
    assert not act.grad[:,:a.actor_critic.num_actions].any()
    assert s.grad is None and acc.grad is None
    assert all(v.grad is None for v in vars(m).values())
    assert all(v.grad is None for v in d.values())


def test_singular_infeasible_saturated_and_reinitialization(monkeypatch):
    a,act,s,m,acc,ids,n,d=local_fixture()
    m.local_h.zero_()
    assert torch.isfinite(local.target(a,act,s,m,acc,ids,n,d)[0]).all()
    d['q_max'].fill_(-1)
    assert not local.target(a,act,s,m,acc,ids,n,d)[4].any()
    d['q_max'].fill_(10);d['torque_max'].zero_()
    assert local.target(a,act,s,m,acc,ids,n,d)[5]['local_saturated_fraction']==1
    d['local_ref_valid'].zero_()
    monkeypatch.setattr(local,'target',lambda *args:pytest.fail('reset row reached solve'))
    assert ff.objective(a,act,s,m,acc,n,d)[0]==0


class RigidPoint:
    """Independent rigidly attached EE, used to check all yaw transport terms."""
    @staticmethod
    def ee_pose_twist(state,use_orientation=False):
        from legged_gym.utils.math_utils import quat_apply
        r=quat_apply(state[:,3:7],state.new_tensor([.5,.2,.1]).expand(len(state),-1))
        return state[:,:3]+r,state[:,26:29]+torch.cross(state[:,29:32],r,dim=-1)
    def ee_position(self,state):return self.ee_pose_twist(state)[0]
    def ee_linear_kinematics(self,state):
        p=self.ee_position(state)
        return p,local.world_jacobian(state,p,state.new_zeros(len(state),3,25))


def test_heading_translation_and_bias_finite_difference():
    torch.manual_seed(4)
    s=torch.randn(8,51,dtype=torch.float64)*.2
    s[:,3:7]=torch.nn.functional.normalize(s[:,3:7]+s.new_tensor([0,0,0,1]),dim=-1)
    backend=RigidPoint();p,v=backend.ee_pose_twist(s)
    jac=s.new_zeros(8,6,25);jac[:,:3,:3]=torch.eye(3)
    r=p-s[:,:3]
    for i in range(3):
        axis=torch.eye(3,dtype=s.dtype)[i].expand_as(r)
        jac[:,:3,3+i]=torch.cross(axis,r,dim=-1)
    terms=local.kinematics(backend,s,jac)
    yaw,rate=heading(s[:,3:7],s[:,29:32])
    torch.testing.assert_close((terms['local_h']@s[:,26:51,None]).squeeze(-1),task_velocity(p,v,s[:,:3],s[:,26:29],yaw,rate))
    eps=1e-4;positions=[]
    for sign in (-1,1):
        shifted=integrate_pose(s,s[:,26:51],sign*eps)
        positions.append(task_position(backend.ee_position(shifted),shifted[:,:3],heading(shifted[:,3:7])))
    torch.testing.assert_close((positions[1]-positions[0])/(2*eps),(terms['local_h']@s[:,26:51,None]).squeeze(-1),atol=1e-7,rtol=1e-5)
    torch.testing.assert_close((positions[1]-2*terms['local_s']+positions[0])/eps**2,terms['local_bias'],atol=2e-7,rtol=2e-4)
    # Base x/y translation cancels exactly, but base world-z remains.
    torch.testing.assert_close(terms['local_h'][:,:,:3],torch.diag(s.new_tensor([0,0,1])).expand(8,-1,-1))


def test_reference_snapshots_reset_release_and_replay():
    from test_b1z1_compliance_review import actual_urdf_task
    from legged_gym.envs.b1z1.compliance_task import attach_transition
    _,_,_,task=actual_urdf_task()
    task.target=None;task.capture_feedforward=True;task.capture_local_teacher=True
    task.begin()
    assert not task.interval['local_ref_valid'].any()
    previous=task.target.velocity.clone();position=task.target.position.clone()
    task.begin()
    assert task.interval['local_ref_valid'].all()
    torch.testing.assert_close(task.interval['local_ref_position_start'],position)
    torch.testing.assert_close(task.interval['local_ref_acceleration'],(task.target.velocity-previous)/task.env.dt)
    algorithm=SimpleNamespace(device='cpu',transition=SimpleNamespace(actor_physics={}))
    task.env.compliance=task;attach_transition(algorithm,task.env)
    stored={k:v.clone() for k,v in algorithm.transition.actor_physics.items()}
    task.reference.reset(torch.tensor([1]));task.env.ee_force_ext_world.zero_();task.begin()
    assert task.interval['local_ref_valid'].flatten().tolist()==[True,False,True]
    for k,v in stored.items():torch.testing.assert_close(v,algorithm.transition.actor_physics[k])
    assert not task.interval['feedforward_applied_force'].any()
    task.reference.recovering[0]=True;task.begin()
    assert not task.interval['local_ref_valid'][0]


def test_compact_cache_only_eligible_rows(monkeypatch):
    from rsl_rl.algorithms import b1z1_actor_physics as physics, b1z1_bard_pinn
    a,_,s,m,_,_,_,_=local_fixture()
    a.storage=SimpleNamespace(actor_physics={'state':s[None], 'local_ref_valid':torch.tensor([[[True],[False],[True]]]),
        'valid':torch.ones(1,3,1,dtype=torch.bool)},dones=torch.zeros(1,3,1,dtype=torch.bool),physics_invalid=torch.zeros(1,3,1))
    a.dynamics_backend=SimpleNamespace(batch_capacity=3)
    monkeypatch.setattr(b1z1_bard_pinn,'mechanics',lambda backend,b:SimpleNamespace(**{k:v[:len(b['dynamics_state'])] for k,v in vars(m).items()}))
    calls=[]
    def kin(backend,state,jac):
        calls.append(len(state));return {k:v[:len(state)] for k,v in vars(m).items() if k.startswith('local_')}
    monkeypatch.setattr(local,'kinematics',kin)
    physics.prepare(a,torch.tensor([1,2]))
    assert calls==[1] and len(a.actor_physics_cache.local_h)==2
    calls.clear();a.cfg['actor_phys_arm_feedforward_compliant_local_enabled']=False
    physics.prepare(a)
    assert not calls


def test_full_actor_objective_keeps_tracking_and_estimator_detachment():
    from rsl_rl.algorithms import b1z1_actor_physics as physics
    a,_,s,m,acc,ids,n,d=local_fixture()
    a.cfg.update(compliance_ee_velocity_weight=.25,compliance_ee_velocity_scale=.5)
    d.update(state=s,compliance_position=torch.zeros_like(m.local_s),compliance_velocity=torch.zeros_like(m.local_s))
    batch={'actor_phys_'+k:v for k,v in d.items()}
    batch.update(indices=torch.arange(len(s)),dones=torch.zeros(len(s),1,dtype=torch.bool),
        physics_invalid=s.new_zeros(len(s),1),nominal_torque=s.new_zeros(len(s),n))
    a.dynamics_backend.ee_pose_twist=lambda x,use_orientation=False:(a.dynamics_backend.ee_position(x),x[:,26:29]+x[:,44:47])
    actions=a.actor_critic.action_mean;ctx=a.actor_critic.last_context
    a.cfg['actor_phys_arm_feedforward_enabled']=False
    before,baseline=physics.objective(a,batch,actions,ctx)
    a.cfg['actor_phys_arm_feedforward_enabled']=True
    after,metrics=physics.objective(a,batch,actions,ctx)
    torch.testing.assert_close(after-before,metrics['arm_feedforward_weighted'])
    for k in ('vel','ee','ee_velocity','q','qd'):torch.testing.assert_close(baseline[k],metrics[k],atol=0,rtol=0)
    (after-before).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in a.actor_critic.torque_head.parameters())
    for module in (a.actor_critic.context_encoder,a.actor_critic.explicit_decoder,a.actor_critic.physics_decoder):
        assert all(p.grad is None for p in module.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='BARD needs CUDA')
def test_bard_articulated_bias_and_jacobian():
    from test_b1z1_bard_dynamics import URDF,DOF_NAMES,FOOT_NAMES,BardB1Z1DynamicsBackend
    from rsl_rl.algorithms.b1z1_bard_pinn import mechanics
    torch.manual_seed(14)
    backend=BardB1Z1DynamicsBackend(URDF,DOF_NAMES,FOOT_NAMES,'ee_gripper_link','trunk',device='cuda:0',batch_capacity=4)
    s=torch.zeros(4,180,device='cuda:0');s[:,6]=1
    s[:,3:7]=torch.nn.functional.normalize(s[:,3:7]+torch.randn(4,4,device=s.device)*.15,dim=-1)
    s[:,7:26]=torch.randn(4,19,device=s.device)*.2
    s[:,26:51]=torch.randn(4,25,device=s.device)*.2
    m=mechanics(backend,dict(rollout_initial_state=s[:,:51],dynamics_state=s))
    terms=local.kinematics(backend,s,m.ee_jacobian)
    # Independent FK second difference at a larger step avoids float32
    # cancellation, and covers actual articulated J-dot, not only rigid transport.
    eps=.02;positions=[]
    for t in (-eps,0,eps):
        shifted=integrate_pose(s,s[:,26:51],t)
        positions.append(task_position(backend.ee_position(shifted),shifted[:,:3],heading(shifted[:,3:7])))
    torch.testing.assert_close((positions[2]-positions[0])/(2*eps),(terms['local_h']@s[:,26:51,None]).squeeze(-1),atol=4e-4,rtol=4e-3)
    torch.testing.assert_close((positions[2]-2*positions[1]+positions[0])/eps**2,terms['local_bias'],atol=.003,rtol=.03)
