from types import SimpleNamespace
import pytest
import torch

from rsl_rl.algorithms.hard_pact_action_channel_diagnostics import (
    ActionChannelUsage, action_channel_metrics,
)


def test_known_rms_bias_removal_update_and_invalid_rows():
    before=torch.tensor([[1.,3.],[3.,5.],[float('nan'),0.]])
    after=torch.tensor([[2.,4.],[4.,6.],[0.,0.]])
    m=action_channel_metrics(before,after)
    assert m['Action_Channel_Usage/mean_ff_torque_rms_nm']==pytest.approx(18**.5)
    assert m['Action_Channel_Usage/state_dependent_ff_rms_nm']==1
    assert m['Action_Channel_Usage/ff_mean_update_rms_nm']==1
    assert m['Action_Channel_Usage/sample_count']==2
    empty=action_channel_metrics(before[:0],after[:0])
    assert empty['Action_Channel_Usage/sample_count']==0
    assert empty['Action_Channel_Usage/mean_ff_torque_rms_nm'].isnan()
    constant=action_channel_metrics(after[:1].expand(3,-1),after[:1].expand(3,-1))
    assert constant['Action_Channel_Usage/state_dependent_ff_rms_nm']==0
    assert constant['Action_Channel_Usage/ff_mean_update_rms_nm']==0


def test_fixed_inputs_physical_scaling_no_rng_caches_or_gradients():
    class Encoder:
        def encode_with_features(self,h): return h,h*0,h
        def reparameterization_trick(self,mu,logvar,noise):
            assert noise.eq(0).all()
            return mu
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias=torch.nn.Parameter(torch.tensor(.5))
            self.context_encoder=Encoder()
            self.explicit_estimator=lambda h:SimpleNamespace(explicit_for_policy=h)
            self.std=torch.ones(24)
            self.distribution=object()
        def actor_forward(self,x):
            return x[:,:12]*0,(x[:,:1]+self.bias).expand(-1,12)
    actor=Actor();cache=actor.distribution
    shape=(2,3)
    fields={k:torch.ones(*shape,12)*v for k,v in (
        ('control_kp',10),('control_kd',1),('control_motor_strength',2),
        ('control_feedback_weight',3),('control_feedforward_weight',.25))}
    fields.update(pre_q=torch.zeros(*shape,19),pre_v=torch.zeros(*shape,18))
    storage=SimpleNamespace(step=2, observations=torch.zeros(*shape,12),
        observation_history=torch.ones(*shape,2),latent_noise=torch.ones(*shape,2),
        latent_boot_mask=torch.ones(*shape,1,dtype=torch.bool),hard_pact_fields=fields)
    rng=torch.random.get_rng_state()
    usage=ActionChannelUsage(actor,storage,lambda a:(a[:,:12],a[:,12:]*10),1.,4)
    assert usage.before.shape==(4,12)
    torch.testing.assert_close(usage.before,torch.full((4,12),2.5))
    assert usage.finish(actor)['Action_Channel_Usage/ff_mean_update_rms_nm']==0
    with torch.no_grad(): actor.bias.add_(.25)
    result=usage.finish(actor)
    assert result['Action_Channel_Usage/ff_mean_update_rms_nm']==1.25
    assert result['Action_Channel_Usage/mean_ff_torque_rms_nm']==3.75
    assert actor.distribution is cache and actor.bias.grad is None
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(not v.requires_grad for v in result.values())


@pytest.mark.parametrize('enabled,iteration',[(False,0),(True,1)])
def test_disabled_or_unscheduled_update_does_not_evaluate(monkeypatch,enabled,iteration):
    from rsl_rl.algorithms.ppo_hard_pact import PPO_HardPACT
    from rsl_rl.algorithms.hard_pact_actor_regularizers import ActorRegularizerConfig
    import rsl_rl.algorithms.hard_pact_action_channel_diagnostics as diagnostics
    def forbidden(*args,**kwargs): raise AssertionError('Unexpected diagnostic forward')
    monkeypatch.setattr(diagnostics,'ActionChannelUsage',forbidden)
    alg=PPO_HardPACT.__new__(PPO_HardPACT)
    alg.action_channel_usage_enabled=enabled;alg.action_channel_usage_interval=100
    alg.actor_regularizers=ActorRegularizerConfig()
    class StopUpdate(Exception): pass
    def stop(*args): raise StopUpdate
    alg._actor_physics_weight_at_iteration=stop
    with pytest.raises(StopUpdate): alg.update(None,None,None,iteration,None,None)
    assert alg.last_action_channel_usage=={}


def test_runner_emits_only_available_action_channel_metrics_once():
    from test_hard_pact_compact_logging import runner,emit
    obj=runner()
    assert not any(k.startswith('Action_Channel_Usage/') for k in emit(obj))
    obj.writer.reset_mock()
    obj.alg.last_action_channel_usage=action_channel_metrics(torch.ones(4,12),torch.ones(4,12)*2)
    keys=emit(obj)
    actual=[k for k in keys if k.startswith('Action_Channel_Usage/')]
    assert set(actual)==set(obj.alg.last_action_channel_usage)
    assert len(actual)==len(set(actual))
