"""Reference parity for selected-row collection; no simulator required."""
from copy import deepcopy
from types import SimpleNamespace as NS
import pytest
import torch
from legged_gym.envs.go2.go2_mule.go2_mule_isaaclab import IsaacLabMULEEnv
from legged_gym.envs.go2.go2_mule.go2_mule_config import resolve_go2_mule
from rsl_rl.beyond_robustness_sensors import BRActorObservationCache


def environment():
    c=resolve_go2_mule(environment_overrides={'env.num_envs':8})
    e=object.__new__(IsaacLabMULEEnv); e.input_schema=c.environment.schema
    e.compact_reset_min_envs=1 # Exercise compact math on a tiny CPU fixture.
    e.commands=torch.randn(8,4); e.previous_action=torch.randn(8,12)
    e.episode_id=torch.arange(8); e.clock=7; e._packet=None
    e.actor_observation_cache=BRActorObservationCache(8,'cpu',c.shared.noise,100.,123)
    e.actor_observation_cache.enabled=True
    q=torch.randn(8,4); q/=q.norm(dim=-1,keepdim=True)
    e.simulator=NS(_robot=NS(data=NS(root_link_quat_w=q,root_link_pos_w=torch.randn(8,3),
        root_link_ang_vel_w=torch.randn(8,3),root_link_lin_vel_w=torch.randn(8,3))),
        _global_gravity=torch.tensor([[0.,0.,-1.]]*8),dof_pos=torch.randn(8,12),dof_vel=torch.randn(8,12),
        payload_reaction_world=torch.randn(8,3),_measured_heights=torch.randn(8,e.input_schema.terrain_dim))
    return e


@pytest.mark.parametrize('ids',[torch.tensor([5,1]),torch.arange(8),torch.tensor([3])])
@pytest.mark.parametrize('compact',[True,False])
def test_partial_refresh_matches_reference_rng_ownership_and_work(ids,compact,monkeypatch):
    import legged_gym.envs.go2.go2_mule.go2_mule_isaaclab as adapter
    torch.manual_seed(17); optimized=environment(); reference=deepcopy(optimized)
    optimized.compact_reset_min_envs=1 if compact else 1024
    reference.collection_optimizations=False
    jac=torch.randn(8,4,3,3)+torch.eye(3)*3; torque=torch.randn(8,12)
    sizes=[]; estimate=adapter.estimate_foot_forces
    def counted(t,j): sizes.append(len(t)); return estimate(t,j)
    monkeypatch.setattr(adapter,'estimate_foot_forces',counted)
    for e in (optimized,reference):
        e.simulator.hard_pact_executed_torque=lambda:torque
        e.simulator.leg_jacobians_b=lambda ids=None:jac if ids is None else jac[ids]
        e._refresh()
    before=optimized.observe()
    for e in (optimized,reference):
        e.simulator.dof_vel[ids]+=2; e.episode_id[ids]+=1; e._refresh(ids)
    assert sizes==[8,8,len(ids) if compact else 8,8]
    a,b=optimized.observe(),reference.observe()
    for name in ('actor','critic','velocity_b','episode_id','supervision_valid'):
        torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=0,atol=0)
    torch.testing.assert_close(a.foot_force_estimate.force_b,b.foot_force_estimate.force_b)
    assert torch.equal(a.foot_force_estimate.valid,b.foot_force_estimate.valid)
    assert torch.equal(optimized.actor_observation_cache.generator.get_state(),reference.actor_observation_cache.generator.get_state())
    others=torch.ones(8,dtype=torch.bool); others[ids]=False
    torch.testing.assert_close(a.actor[others],before.actor[others],rtol=0,atol=0)
    optimized.actor_observation_cache.values.zero_(); torque.zero_(); jac.zero_()
    torch.testing.assert_close(optimized.observe().actor,a.actor,rtol=0,atol=0)


def test_material_transfers_skip_unchanged_rows(monkeypatch):
    # Exercise the real reset/readback path with the existing fake PhysX API.
    from legged_gym.simulator.isaaclab_simulator_mule import IsaacLabSimulatorMULE
    from test_mule_remaining_fixes import test_cadence_ineligible_reset_resamples_mass_but_resume_preserves_it
    original=IsaacLabSimulatorMULE._write_ball_properties; calls=[]
    def checked(self,ids,**kwargs):
        material_ids=kwargs['material_ids']; calls.append(material_ids.clone())
        views=[b.root_physx_view for b in self.balls]
        saved=[v.get_material_properties for v in views]
        if not len(material_ids):
            def forbidden(): raise AssertionError('Unchanged materials must not transfer')
            for v in views: v.get_material_properties=forbidden
        try: return original(self,ids,**kwargs)
        finally:
            for v,get in zip(views,saved): v.get_material_properties=get
    monkeypatch.setattr(IsaacLabSimulatorMULE,'_write_ball_properties',checked)
    test_cadence_ineligible_reset_resamples_mass_but_resume_preserves_it()
    assert calls[0].tolist()==[0,2] and calls[1].numel()==0
