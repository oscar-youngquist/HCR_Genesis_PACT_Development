"""Reset diagnostics use real episode counts, original terrain, and legacy rules."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import math

import pytest
import torch

from rsl_rl.hard_pact_episode_logging import HardPACTEpisodeDiagnostics, log_hard_pact_episodes
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos import Go2HardPACTPos


def fixture(cls=Go2HardPACT, device='cpu'):
    env = cls.__new__(cls)
    env.device, env.num_envs, env.dt = device, 8, .02
    env.cfg = NS(env=NS(episode_diagnostics_enabled=True, episode_diagnostics_interval=2,
                        fail_to_terminal_time_s=.02),
        terrain=NS(curriculum=True, selected=False, mesh_type='trimesh',num_rows=10,num_cols=20,
                   terrain_proportions=[.1,.2,.2,.2,.2,.1],reset_out_of_bounds=False),
        termination=NS(termination_terms=['roll','pitch','height_min','height_max'],
                       roll_threshold=.7,pitch_threshold=1.,height_min=.2,height_max=1.5),
        commands=NS(curriculum=False,curriculum_patience_iterations=0))
    env.simulator = NS(terrain_kind_ids=torch.tensor([4,4,1,0,4,4,2,3],device=device),
        terrain_levels=torch.arange(8,device=device), _base_euler=torch.zeros(8,3,device=device),
        base_pos=torch.zeros(8,3,device=device), measured_heights=torch.zeros(8,4,device=device),
        link_contact_forces=torch.zeros(8,2,3,device=device),termination_contact_indices=[1])
    env.simulator.base_pos[:,2]=.4
    env.simulator.base_pos[3,2]=.1;env.simulator.base_pos[4,2]=1.6
    env.simulator._base_euler[0,0]=.8
    env.simulator._base_euler[1,0]=2*math.pi+.8
    env.simulator._base_euler[2,1]=1.2
    env.simulator.link_contact_forces[0,1,2]=11
    env.episode_length_buf=torch.arange(1,9,device=device)*100
    env.fail_buf=torch.tensor([2,2,2,2,2,0,2,0],device=device)
    env.time_out_buf=torch.tensor([False]*5+[True,True,False],device=device)
    env.non_failure_reset_buf=torch.zeros(8,dtype=torch.bool,device=device)
    env.max_episode_length=1000
    env.reset_buf=torch.ones(8,dtype=torch.bool,device=device)
    return env


def summary(env, parts):
    recorder=HardPACTEpisodeDiagnostics(env);recorder.complete_episode.fill_(True)
    for ids in parts:
        recorder.record(env,torch.tensor(ids,device=env.device))
    return recorder.metrics()


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_reason_overlap_episode_weighting_terrain_and_level(device):
    env=fixture(device=device)
    saved={k:v.clone() for k,v in vars(env.simulator).items() if torch.is_tensor(v)}
    rng=torch.get_rng_state().clone()
    metrics=summary(env,[list(range(8))])
    partitioned=summary(env,[[0],[1,2,3,4],[5,6,7]])
    for key in metrics:
        torch.testing.assert_close(metrics[key],partitioned[key],equal_nan=True,rtol=0,atol=0)
        assert not metrics[key].requires_grad
    p='EpisodeDiagnostics/'
    assert metrics[p+'all/all_count']==8
    assert metrics[p+'all/all_mean_length_s']==9
    assert metrics[p+'all/failure_count']==6 and metrics[p+'all/timeout_count']==1
    assert metrics[p+'all/other_count']==1 and metrics[p+'failure_timeout_overlap_count']==1
    assert metrics[p+'reason/contact_count']==1 and metrics[p+'reason/roll_count']==2
    for reason in ('pitch','height_min','height_max','unattributed'):
        assert metrics[p+'reason/'+reason+'_count']==1
    assert metrics[p+'terrain/discrete_obstacles/all_count']==4
    assert metrics[p+'terrain/discrete_obstacles/all_mean_length_s']==7
    assert metrics[p+'level/5/timeout_mean_length_s']==12
    assert torch.isnan(metrics[p+'level/9/all_mean_length_s'])
    for k,v in saved.items():torch.testing.assert_close(getattr(env.simulator,k),v,rtol=0,atol=0)
    assert torch.equal(rng,torch.get_rng_state())


def test_partial_initial_episode_and_report_cadence():
    env=fixture();r=env._episode_diagnostics=HardPACTEpisodeDiagnostics(env)
    ids=torch.arange(8);writer=Mock()
    r.begin_iteration(0);r.record(env,ids)
    assert r.counts.sum()==0 and r.excluded_initial==8
    r.record(env,ids)
    log_hard_pact_episodes(env,writer,0);writer.add_scalar.assert_not_called()
    r.begin_iteration(1);r.begin_iteration(1)
    log_hard_pact_episodes(env,writer,1)
    values={call.args[0]:call.args[1] for call in writer.add_scalar.call_args_list}
    assert len(values)==writer.add_scalar.call_count
    assert values['EpisodeDiagnostics/reporting_iterations']==2
    assert values['EpisodeDiagnostics/all/all_count']==8
    assert r.counts.sum()==0 and r.reporting_iterations==0 and r.complete_episode.all()
    count=writer.add_scalar.call_count
    log_hard_pact_episodes(env,writer,1)
    assert writer.add_scalar.call_count==count


@pytest.mark.parametrize('cls',[Go2HardPACT,Go2HardPACTPos])
def test_record_before_legacy_reset_and_disabled_parity(cls):
    env=fixture(cls);env._terrain_curriculum_iteration=0
    env.begin_command_curriculum_iteration()
    env._episode_diagnostics.complete_episode.fill_(True)
    def reset(self,ids):
        self.simulator.terrain_levels[ids]=9
        self.episode_length_buf[ids]=0;self.fail_buf[ids]=0
    env._legacy_task_class=NS(reset_idx=reset)
    env.reset_idx(torch.tensor([0]))
    m=env._episode_diagnostics.metrics()
    assert m['EpisodeDiagnostics/level/0/failure_count']==1
    assert m['EpisodeDiagnostics/level/9/failure_count']==0
    assert env.episode_length_buf[0]==0 and env.simulator.terrain_levels[0]==9
    env.cfg.env.episode_diagnostics_enabled=False
    env.begin_command_curriculum_iteration()
    assert env._episode_diagnostics is None
    env.reset_idx(torch.tensor([1]))
    writer=Mock();log_hard_pact_episodes(env,writer,49)
    writer.add_scalar.assert_not_called()


@pytest.mark.parametrize('cls',[Go2HardPACT,Go2HardPACTPos])
def test_legacy_termination_decisions_unchanged(cls):
    env=fixture(cls);reference=fixture(cls)
    env.fail_buf.zero_();reference.fail_buf.zero_()
    env._episode_diagnostics=HardPACTEpisodeDiagnostics(env)
    for _ in range(3):
        env.check_termination();reference.check_termination()
        env._episode_diagnostics.record(env,env.reset_buf.nonzero().flatten())
        for key in ('fail_buf','reset_buf','time_out_buf'):
            assert torch.equal(getattr(env,key),getattr(reference,key))


def test_column_family_resolution_and_unknown():
    env=fixture();del env.simulator.terrain_kind_ids
    env.simulator.terrain_types=torch.tensor([14]*8)  # discrete-obstacle column
    assert summary(env,[list(range(8))])['EpisodeDiagnostics/terrain/discrete_obstacles/all_count']==8
    env.cfg.terrain.curriculum=False  # randomized family not inferable from columns
    assert summary(env,[list(range(8))])['EpisodeDiagnostics/terrain/unknown/all_count']==8
