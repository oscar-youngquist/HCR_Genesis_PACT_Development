from types import SimpleNamespace as NS
import torch
from test_b1z1_directional_compliance import fixture
from legged_gym.envs.b1z1.pilot_interactions import InteractionCfg, Interactions, configure, mixture, envelope, descriptors
from legged_gym.envs.b1z1.compliance_bounds import feasible


def test_restore_new_section_from_old_archived_class():
    from legged_gym.envs.b1z1.pilot_interactions import prepare_restore
    from legged_gym.scripts.eval_b1z1_pilot_compliance import restore_sections
    cfg=NS()
    saved={'pilot_interactions':{'version':1,'enabled':True,'source_hashes':{'a':'b'},
        'mixture_start_iteration':24201,'smoke_full_mixture':True}}
    prepare_restore(cfg,saved);restore_sections(cfg,saved)
    assert cfg.pilot_interactions.mixture_start_iteration==24201
    assert cfg.pilot_interactions.source_hashes=={'a':'b'}
    cfg.pilot_interactions.sample_scenarios=False
    assert cfg.pilot_interactions.smoke_full_mixture
    import pytest
    with pytest.raises(ValueError,match='version'):
        prepare_restore(NS(),{'pilot_interactions':{'version':99}})


def test_offset_cap_combined_workspace_and_collision_release():
    r,k=fixture(2);r.cfg.max_offset_norm=.30;r.collect_diagnostics=True
    k['force'][:]=torch.tensor([100.,100.,0.])
    # A collision-limited path prefix may never be replaced by radial projection.
    def safe(candidate,previous):
        d=candidate-previous
        alpha=((.12-previous[:,1])/d[:,1].clamp_min(1e-12)).clamp(0,1)
        alpha=torch.where(d[:,1]>0,alpha,torch.ones_like(alpha))
        return previous+alpha[:,None]*d
    k['target_safe']=safe
    for _ in range(150):
        out=r.advance(**k)
        assert (r.delta.norm(dim=-1)<=.300002).all()
        assert feasible(out.position,k['center'],k['bounds']).all()
        assert (out.position[:,1]<=.120001).all()
        torch.testing.assert_close(r.rate,out.velocity-k['nominal_velocity'])
    assert r.cap_active.all()
    k['force'].zero_()
    for _ in range(200):out=r.advance(**k)
    assert r.delta.norm(dim=-1).max()<1e-5
    assert not r.cap_active.any()


def test_filtered_state_and_moving_nominal_recovery_are_capped():
    r,k=fixture(1);r.cfg.max_offset_norm=.10
    r.initialized.fill_(True);r.accepted[:]=torch.tensor([.6,.25,0.])
    r.delta[:,1]=.25;r.rate[:,1]=2
    k['nominal_velocity'][:,0]=.1
    out=r.advance(**k)
    assert r.recovering.all() and r.delta.norm(dim=-1).max()<=.100002
    torch.testing.assert_close(r.delta,out.position-k['nominal'])
    torch.testing.assert_close(r.rate,out.velocity-k['nominal_velocity'])
    assert not r.constraint_invalid.any()


def test_collision_reduction_cannot_enter_inner_radius_hole():
    r,k=fixture(1);r.cfg.max_offset_norm=.3;r.collect_diagnostics=True
    k['nominal'][:]=torch.tensor([.31,0.,0.])
    r.initialized.fill_(True);r.accepted[:]=torch.tensor([.2,.25,0.])
    r.delta.copy_(r.accepted-k['nominal'])
    k['force'][:]=torch.tensor([-22.,-50.,0.])
    k['target_safe']=lambda candidate,previous:(candidate+previous)/2
    out=r.advance(**k)
    assert feasible(r.diagnostics['equilibrium_geometry'],k['center'],k['bounds']).all()
    assert feasible(out.position,k['center'],k['bounds']).all()


def test_cap_outward_velocity_and_inward_recovery():
    r,k=fixture(1);r.cfg.max_offset_norm=.10;k['force'][:,1]=100
    for _ in range(150):out=r.advance(**k)
    assert (r.delta*r.rate).sum()<=1e-6
    k['force'][:,1]=-20
    old=r.delta.clone();out=r.advance(**k)
    assert r.delta[:,1]<old[:,1] and r.rate[:,1]<0


def test_ramp_and_three_second_recovery():
    c=InteractionCfg()
    torch.testing.assert_close(mixture(c,0,'cpu'),torch.tensor([0.,0.,0.,0.,1.]))
    torch.testing.assert_close(mixture(c,5000,'cpu'),torch.tensor(c.mixture_weights))
    c.smoke_full_mixture=True
    torch.testing.assert_close(mixture(c,0,'cpu'),torch.tensor(c.mixture_weights))
    t=torch.tensor([7.,8.,8.25,8.5,10.,10.25,10.5,13.5,20.])
    torch.testing.assert_close(envelope(t,torch.ones_like(t)*1.5,c),torch.tensor([0.,0.,.5,1.,1.,.5,0.,0.,0.]))


def fake_env(n=1000):
    c=InteractionCfg();c.smoke_full_mixture=True
    e=NS(num_envs=n,device='cpu',cfg=NS(pilot_interactions=c,commands=NS(heading_command=False)),
        training_iteration=0,dt=.02,commands=torch.zeros(n,6),episode_length_buf=torch.zeros(n),
        ee_goal_sphere=torch.tensor([.7,0.,0.]).repeat(n,1),traj_total_timesteps=torch.ones(n),
        collision_check=lambda ids:torch.zeros(len(ids),dtype=torch.bool))
    e.compliance=NS(profile=torch.zeros(n,12),bounds=torch.tensor([[.4,.9],[-1.25,1.04],[-1.2,1.2]]),
                    coverage=NS(weights=None))
    return e


def test_descriptors_profiles_coverage_and_partial_reset():
    torch.manual_seed(5);e=fake_env(10000);s=Interactions(e);s.reset(torch.arange(e.num_envs))
    counts=torch.bincount(s.family,minlength=5)/e.num_envs
    torch.testing.assert_close(counts,torch.tensor(s.cfg.mixture_weights),atol=.02,rtol=0)
    for family in (0,1):
        ids=(s.family==family).nonzero().flatten()
        assert all((s.axis[ids]==a).any() for a in range(3))
        assert (s.sign[ids]<0).any() and (s.sign[ids]>0).any()
        coef=e.compliance.profile[ids,2*s.axis[ids]+(s.sign[ids]>0).long()]
        assert (coef==0).all() if family==1 else ((coef>=.8)&(coef<=1)).all()
    ample=s.family==2
    assert (e.compliance.profile[ample,6:10]>0).all()
    assert (e.commands[s.controlled,:3]==0).all()
    end=s.cfg.onset_s+2*s.cfg.ramp_s+s.plateau+s.cfg.recovery_s
    assert (e.traj_total_timesteps[s.controlled]*e.dt>=end[s.controlled]-1e-5).all()
    before=s.family.clone();p=e.compliance.profile.clone();s.reset(torch.tensor([0,2]))
    torch.testing.assert_close(s.family[3:],before[3:]);torch.testing.assert_close(e.compliance.profile[3:],p[3:])


def test_distribution_validity_uses_current_not_previous_reset(monkeypatch):
    from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact import B1Z1PACT
    from legged_gym.envs.b1z1 import pilot_continuation
    monkeypatch.setattr(pilot_continuation,'measurements',lambda e: {})
    captured=[]
    e=NS(cfg=NS(termination=NS(termination_terms=[])),
        simulator=NS(termination_contact_indices=[],base_euler=torch.zeros(3,3),
            base_pos=torch.zeros(3,3),measured_heights=torch.zeros(3,1)),
        termination_contact_counter=torch.zeros(3),episode_length_buf=torch.tensor([0,0,11]),
        max_episode_length=10,reset_buf=torch.tensor([True,False,False]),extras={},
        compliance=NS(recovery_failed=torch.tensor([False,True,False]),
            recovery_rejections=torch.zeros(3),profile=torch.zeros(3,12)),
        interactions=NS(record=lambda:None,family=torch.zeros(3,dtype=torch.long)),
        candidate_recorder=NS(add=lambda snapshot,valid,profile:captured.append(valid.clone())))
    for name in ('fail_buf','contact_fail_buf','roll_fail_buf','pitch_fail_buf','height_min_fail_buf','height_max_fail_buf'):
        setattr(e,name,torch.zeros(3,dtype=torch.bool))
    B1Z1PACT.check_termination(e)
    assert captured[0].tolist()==[True,False,False]
    assert e.reset_buf.tolist()==[False,True,True]


def test_inheritance_and_facets_exclusion():
    from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg
    from legged_gym.envs.b1z1.b1z1_pact.ablation_configs import make_b1z1_pact_ablation_configs
    from legged_gym.envs.b1z1.b1z1_ppo_pos.b1z1_ppo_pos_config import B1Z1PPOPosCfg
    from legged_gym.envs.b1z1.b1z1_facet.b1z1_facet_config import B1Z1FACETCfg
    from legged_gym.envs.b1z1.b1z1_unifp.b1z1_unifp_config import B1Z1UniFPCfg
    for cls in [B1Z1PACTCfg,B1Z1PPOPosCfg]+[make_b1z1_pact_ablation_configs(v)[0] for v in (4,5,6,7,8,9,10,12,13,14,15)]:
        c=cls();configure(c)
        assert c.compliance.max_offset_norm==.3 and c.compliance.base_gain==[.06,.04,.015]
    c=make_b1z1_pact_ablation_configs(14)[0]();configure(c)
    assert not c.compliance.stiffness_gate and not c.compliance.workspace_gate
    assert not B1Z1FACETCfg().pilot_interactions.enabled
    assert not hasattr(B1Z1UniFPCfg(),'pilot_interactions')


def test_exact_paired_descriptors_and_reproducibility():
    c=InteractionCfg();c.smoke_full_mixture=True
    torch.manual_seed(17);a=descriptors(c,1000,'cpu',0)
    torch.manual_seed(17);b=descriptors(c,1000,'cpu',0)
    for key in a:torch.testing.assert_close(a[key],b[key])
    paired=a['family'].reshape(-1,2)
    y=paired[:,0]<2
    assert (paired[y].sum(-1)==1).all()
    for key in ('axis','sign','force','plateau','coefficient'):
        v=a[key].reshape(-1,2)
        torch.testing.assert_close(v[:,0],v[:,1])
    off=a['base_off'].reshape(-1,2)
    assert (off[:,0]!=off[:,1]).all()


def test_normal_clock_and_reference_air_time():
    from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact import B1Z1PACT
    c=InteractionCfg()
    e=NS(cfg=NS(pilot_interactions=c,rewards=NS(cycle_time=.64),commands=NS(
        lin_vel_x_clip=.1,lin_vel_y_clip=.1,ang_vel_yaw_clip=.1)),dt=.02,
        gait_indices=torch.zeros(2),get_walking_cmd_mask=lambda:torch.zeros(2,dtype=torch.bool),
        simulator=NS(foot_contacts=torch.ones(2,4,dtype=torch.bool)),last_contacts=torch.zeros(2,4,dtype=torch.bool),
        feet_air_time=torch.full((2,4),.3),commands=torch.tensor([[0.,0.,0.],[1.,0.,0.]]),
        compliance=NS(target=NS(base=torch.tensor([[.2,0.,0.],[0.,0.,0.]]))))
    for _ in range(8):B1Z1PACT._step_contact_targets(e)
    torch.testing.assert_close(e.gait_indices,torch.full((2,),.25))
    reward=B1Z1PACT._reward_feet_air_time(e)
    assert reward[0]>0 and reward[1]==0
    c.enabled=False;B1Z1PACT._step_contact_targets(e)
    assert not e.gait_indices.any()


def test_ideal_ample_protected_and_restricted_reference_controls():
    r,k=fixture(3);r.cfg.max_offset_norm=.3;r.cfg.base_gain=[.06,.04,.015]
    r.profile.fill_(.9);r.profile[1,:6]=0
    k['nominal'][2,0]=.895
    k['force'][:]=torch.tensor([[0.,50.,0.],[0.,50.,0.],[50.,0.,0.]])
    for _ in range(150):out=r.advance(**k)
    assert out.gates[0].max()<.01 and out.base[0].abs().max()<1e-6
    assert .20<r.delta[0].norm()<.25
    assert r.delta[1].norm()<1e-6
    assert out.gates[2,0]>.9 and out.base[2,0]>.1
    k['force'].zero_()
    for _ in range(150):out=r.advance(**k)
    assert r.delta.norm(dim=-1).max()<.001
    assert out.base.abs().max()<.002
