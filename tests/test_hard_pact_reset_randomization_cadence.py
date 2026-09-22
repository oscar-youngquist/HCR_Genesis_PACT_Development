from types import SimpleNamespace
import pytest
import torch
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import GO2HardPACTCfg
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos_config import GO2HardPACTPosCfg
from legged_gym.simulator.reset_randomization_cadence import ResetRandomizationCadence
from legged_gym.simulator.isaaclab_simulator_pact import IsaacLabSimulator_PACT


def test_independent_episode_counts_exact_threshold_and_repeated_windows():
    c=ResetRandomizationCadence(3,"cpu",100)
    ids=torch.arange(3)
    c.begin_reset(ids)
    assert c.episodes.tolist()==[0,0,0]
    assert c.select('mass',ids).tolist()==[0,1,2]
    for episode in range(1,201):
        c.begin_reset(torch.tensor([0]))
        selected=c.select('mass',torch.tensor([0]))
        assert selected.tolist()==([0] if episode in (100,200) else [])
    assert c.episodes.tolist()==[200,0,0]
    c.begin_reset(torch.tensor([1,2]))
    assert not c.select('mass',torch.tensor([1,2])).numel()
    assert c.episodes.tolist()==[200,1,1]


def test_changed_feature_pending_per_env_and_restarts_its_own_wait():
    c=ResetRandomizationCadence(2,"cpu",3)
    ids=torch.arange(2)
    c.update_ranges({'mass':(0,1),'friction':(.2,1.8)})
    c.begin_reset(ids)
    for name in c.ranges:c.select(name,ids)
    c.begin_reset(torch.tensor([0]))
    c.update_ranges({'mass':(0,2),'friction':(.2,1.8)})
    assert c.select('mass',torch.tensor([0])).tolist()==[0]
    assert not c.select('friction',torch.tensor([0])).numel()
    c.update_ranges({'mass':(0,2)})  # unchanged range publication is not a step
    assert not c.select('mass',torch.tensor([0])).numel()
    c.begin_reset(torch.tensor([1]))
    assert c.select('mass',torch.tensor([1])).tolist()==[1]
    for episode in (2,3,4):
        c.begin_reset(torch.tensor([0]))
        assert c.select('mass',torch.tensor([0])).tolist()==([0] if episode==4 else [])
    assert c.episodes.tolist()==[4,1]


@pytest.mark.parametrize('interval',[0,1,100])
def test_adapter_sampler_calls_and_legacy_fast_path(interval):
    sim=IsaacLabSimulator_PACT.__new__(IsaacLabSimulator_PACT)
    names=('friction','base_mass','com_displacement','joint_armature',
           'joint_friction','joint_damping','pd_gain')
    sim._cfg=SimpleNamespace(domain_rand=SimpleNamespace(reset_resample_episodes=interval,
        **{'randomize_'+n:True for n in names}))
    sim._num_envs=2;sim._device='cpu'
    calls=[]
    for name in names:
        setattr(sim,'_randomize_'+name,lambda ids,n=name:calls.append((n,ids.tolist())))
    for _ in range(2):
        c=sim._reset_sampling_cadence()
        if c is not None:c.begin_reset(torch.arange(2))
        sim._reset_domain_randomization(torch.arange(2))
    assert len(calls)==len(names)*(1 if interval>1 else 2)
    if interval<=1:
        assert not hasattr(sim,'_reset_cadence')
    else:
        sim._reset_cadence.update_ranges({'base_mass':(0,2)})
        sim._reset_domain_randomization(torch.tensor([1]))
        assert calls[-1]==('base_mass',[1]) and len(calls)==len(names)+1


def test_defaults_empty_reset_rng_and_invalid_settings():
    assert GO2HardPACTCfg.domain_rand.reset_resample_episodes==100
    assert GO2HardPACTPosCfg.domain_rand.reset_resample_episodes==100
    c=ResetRandomizationCadence(2,'cpu',100)
    before=torch.random.get_rng_state()
    c.begin_reset(torch.tensor([],dtype=torch.long))
    assert not c.started.any()
    c.begin_reset(torch.tensor([1]))
    c.select('mass',torch.tensor([1]))
    c.begin_reset(torch.tensor([0]))  # late initialization also counts as zero
    assert c.episodes.eq(0).all()
    assert torch.equal(before,torch.random.get_rng_state())
    for value in (-1,1.5):
        with pytest.raises(ValueError):ResetRandomizationCadence(2,'cpu',value)
