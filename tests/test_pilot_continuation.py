import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import torch
from test_b1z1_directional_compliance import fixture

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('candidate_test',ROOT/'legged_gym/envs/b1z1/pilot_continuation.py')
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)


def test_reference_large_demand_and_base_clipping():
    r,k=fixture(2);r.cfg.ee_gain=[.005]*3;r.cfg.base_gain=[.060,.040,.015]
    r.profile[:,:]=1;r.collect_diagnostics=True
    k['force'][:,2]=torch.tensor([30.,50.]);k['torso_wrench']=torch.tensor([[100.,100.,0.]]).repeat(2,1)
    for _ in range(100): out=r.advance(**k)
    torch.testing.assert_close(r.diagnostics['raw_displacement'][:,2],torch.tensor([.15,.25]))
    assert torch.all(out.base<=k['command_limits'][:,1])
    torch.testing.assert_close(r.eta,out.base-k['command'])


def test_pulse_plateau_and_release():
    c=mod.Candidate();age=torch.tensor([4.,5.,5.25,5.5,7.,7.25,7.5,8.])
    torch.testing.assert_close(mod.pulse(age,torch.full_like(age,1.5),c),torch.tensor([0.,0.,.5,1.,1.,.5,0.,0.]))


def test_clock_zero_commands_and_reset():
    class Base:
        def reset_idx(self,ids):self.gait_indices[ids]=0
    cls=mod.environment_class(Base)
    env=object.__new__(cls);env.cfg=NS(pilot_candidate={'continuous_clock':True},rewards=NS(cycle_time=.64))
    env.gait_indices=torch.zeros(3);env.dt=.02;env.candidate_ready=False
    for _ in range(8):env._step_contact_targets()
    torch.testing.assert_close(env.gait_indices,torch.full((3,),.25))
    env.reset_idx(torch.tensor([1]))
    torch.testing.assert_close(env.gait_indices,torch.tensor([.25,0.,.25]))


def test_air_time_mask_thresholds_not_nominal_norm():
    cfg=NS(lin_vel_x_clip=.1,lin_vel_y_clip=.2,ang_vel_yaw_clip=.3)
    assert mod.walking_mask(torch.tensor([[.09,.19,.29],[.11,0,0],[0,.21,0],[0,0,.31]]),cfg).tolist()==[False,True,True,True]


def test_air_time_uses_privileged_reference_only_for_reward():
    cls=mod.environment_class(object);env=object.__new__(cls)
    env.cfg=NS(pilot_candidate={'reference_air_time':True},commands=NS(
        lin_vel_x_clip=.1,lin_vel_y_clip=.1,ang_vel_yaw_clip=.1))
    env.simulator=NS(foot_contacts=torch.ones(2,4,dtype=torch.bool))
    env.last_contacts=torch.zeros(2,4,dtype=torch.bool)
    env.feet_air_time=torch.full((2,4),.3);env.dt=.02
    env.commands=torch.tensor([[0.,0.,0.],[1.,0.,0.]])
    env.compliance=NS(target=NS(base=torch.tensor([[.2,0.,0.],[0.,0.,0.]])))
    reward=env._reward_feet_air_time()
    assert reward[0]>0 and reward[1]==0
    assert not env.feet_air_time.any()


def test_candidate_isolation_and_contract():
    cfg=NS(compliance=NS(enabled=True,ee_gain=[.005]*3,base_gain=[.008,.008,.015]))
    mod.configure(cfg)
    assert cfg.compliance.base_gain==[.060,.040,.015]
    assert cfg.compliance.ee_gain==[.005]*3
    assert 'compute_observations' not in mod.environment_class(object).__dict__


def test_histograms_pool_samples_not_batch_quantiles(tmp_path):
    s=importlib.util.spec_from_file_location('candidate_dist',ROOT/'legged_gym/utils/pilot_candidate_diagnostics.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    d=m.Distributions(tmp_path,.02)
    for values in ([.1,.2,.3],[.9]):
        a=torch.tensor(values)[:,None].repeat(1,3)
        data=dict(applied_force_task=a,recruited_force_task=a,workspace_gate=a*0,
                  reference_stage_selected_coefficients=a*0+1,accepted_offset=a)
        d.add(data,torch.ones(len(a),dtype=torch.bool),None);d.flush(0)
    g=d.groups[('all','accepted_offset/0')]
    assert g['count']==4 and abs(g['total']/4-.375)<1e-6
    assert g['hist'].sum()==4


def test_resize_progress_preserves_counters_and_matching_rows():
    spec=importlib.util.spec_from_file_location('candidate_train',ROOT/'legged_gym/scripts/train_pilot_continuation.py')
    train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)
    state=dict(next_iteration=23201,total_timesteps=123456,
        progress={'environment':{'tradeoff_step_ctr':torch.arange(10)[:,None],'common_step_counter':987},
                  'simulator':{'_terrain_levels':torch.arange(10),'_env_origins':torch.arange(30).reshape(10,3)}})
    changed,indices=train.resize_progress(state,10,4)
    assert indices.tolist()==[0,2,5,7]
    assert changed['next_iteration']==23201 and changed['total_timesteps']==123456
    assert changed['progress']['environment']['common_step_counter']==987
    assert changed['progress']['environment']['tradeoff_step_ctr'].flatten().tolist()==indices.tolist()
    torch.testing.assert_close(changed['progress']['simulator']['_env_origins'],state['progress']['simulator']['_env_origins'][indices])
    assert state['progress']['simulator']['_env_origins'].shape==(10,3)
    assert train.allowed_change(dict(path='.grf_semantics.sensor_count',old=260,new=104),10,4)
    assert not train.allowed_change(dict(path='.grf_semantics.sensor_count',old=260,new=105),10,4)
    assert not train.allowed_change(dict(path='.grf_semantics.source',old='normal',new='total'),10,4)
