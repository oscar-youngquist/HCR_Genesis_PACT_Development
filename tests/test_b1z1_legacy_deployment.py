"""Scripted legacy policy contracts without importing the simulator/training packages."""
import importlib.util
from pathlib import Path
import sys
import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('legacy_export_test', ROOT/'legged_gym/scripts/export_b1z1_legacy_pact_jit.py')
export = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = export
spec.loader.exec_module(export)


@pytest.fixture
def model():
    torch.manual_seed(4)
    actor = export.reference_class()(81, 100, 17, 2592, latent_dim=64, film_v02_enabled=True).eval()
    settings = dict(base_soft=.1,ee_soft=.05,clip_obs=100.,clip_action=50.,position_scale=.25,
        angular_scale=.25,q_scale=1.,qd_scale=.05,dt=.02,cycle=.64,q_default=[.1]*19,
        kp=[40.]*19,kd=[1.]*19,ff_scale=[30.]*12+[10.]*5,
        command_scale=[2.,2.,.25,.5,1.,1.3],walking_threshold=[.05,.05,.1])
    cls = export.load_file('_legacy_pact_test_module',ROOT/'rsl_rl/b1z1_legacy_deployment.py').LegacyPACTDeployment
    return actor, torch.jit.script(cls(actor,settings).eval())


def test_native_sequence_parity(model):
    actor, jit = model
    assert export.validate(actor,jit)['max_action_abs_error'] < 1e-5


def test_observation_units_and_action_order(model):
    _, jit = model
    obs = jit.build_observation(torch.tensor([[.2,-.3]]),torch.ones(1,3),torch.full((1,19),.2),
        torch.ones(1,19),torch.zeros(1),torch.ones(1,34),torch.ones(1,6))
    torch.testing.assert_close(obs[:, :2],torch.tensor([[.2,-.3]]))
    torch.testing.assert_close(obs[:,2:5],torch.full((1,3),.25))
    torch.testing.assert_close(obs[:,5:22],torch.full((1,17),.1))
    torch.testing.assert_close(obs[:,39:41],torch.tensor([[0.,1.]]))
    torch.testing.assert_close(obs[:,-6:],jit.command_scale[None])
    target, ff = jit.action_targets(torch.ones(1,34))
    torch.testing.assert_close(target[0,17:],jit.q_default[17:])
    assert ff[0,17:].count_nonzero() == 0
    torch.testing.assert_close(ff[0,:12],torch.full((12,),30.))
    torch.testing.assert_close(ff[0,12:17],torch.full((5,),10.))


def test_controller_delay_and_limits(model):
    _, jit = model
    action = torch.ones(2,34)
    queue = torch.full((2,2,34),2.)
    executed, queue = jit.apply_delay(action,queue,torch.tensor([0,1]),torch.tensor([False,True]))
    torch.testing.assert_close(executed[0],action[0])
    assert executed[1].count_nonzero() == 0
    pd, ff, requested, clipped = jit.controller(action,torch.zeros(2,19),torch.ones(2,19),
        torch.full((2,19),1.2),torch.full((2,19),.8),torch.full((2,19),.9),
        torch.full((2,1),.6),torch.full((2,1),1.1),torch.full((19,),5.))
    target, raw_ff = jit.action_targets(action)
    torch.testing.assert_close(pd,(40*1.2*target-.8)*.6*.9)
    torch.testing.assert_close(ff,raw_ff*1.1*.9)
    torch.testing.assert_close(requested,pd+ff)
    torch.testing.assert_close(clipped,requested.clamp(-5,5))


def test_roundtrip_and_partial_reset(tmp_path,model):
    _,jit = model
    path = tmp_path/'policy.pt'
    torch.jit.save(jit,str(path))
    loaded = torch.jit.load(str(path))
    hist,track,_ = jit.initial_state(2)
    hist.fill_(1);track.fill_(2)
    obs = torch.randn(2,81)
    result = loaded.step(obs,hist,track,torch.tensor([True,False]))
    assert result[-2][0,:-81].count_nonzero() == 0
    assert result[-1][0,:4].count_nonzero() == 0
    torch.testing.assert_close(result[-2][1,:-81],hist[1,81:])
    for a,b in zip(result,jit.step(obs,hist,track,torch.tensor([True,False]))):
        torch.testing.assert_close(a,b)
    with pytest.raises((torch.jit.Error,RuntimeError)):
        loaded(torch.zeros(2,93),hist,track)
    assert not any(k in sys.modules for k in ('isaaclab','isaacsim'))
