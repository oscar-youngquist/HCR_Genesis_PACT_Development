"""Applied-only labels preserve substep frames and randomized-mechanics inputs."""
import inspect
from types import SimpleNamespace

import pytest
import torch

from test_go2_hard_pact_disturbances import _persistent_task
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos import Go2HardPACTPos
from rsl_rl.algorithms.ppo_hard_pact import PPO_HardPACT


@pytest.mark.parametrize("task_class", [Go2HardPACT, Go2HardPACTPos])
@pytest.mark.parametrize("loaded", [False, True])
def test_actual_ramped_wrench_only_with_mass_com_and_rotating_yaw(task_class, loaded):
    task = _persistent_task(task_class=task_class)
    task.obs_scales = SimpleNamespace(base_wrench=.01)
    task._realized_added_mass = torch.tensor([[3.]])
    task._realized_com_shift_body = torch.tensor([[.2, -.1, .05]])
    applied = []
    task.simulator.hard_pact_apply_base_wrench_world = lambda value: applied.append(value.clone())
    for suffix in ("sustained", "mass_com", "total", "sustained_yaw_scaled",
                   "mass_com_yaw_scaled", "yaw_scaled", "yaw_physical"):
        setattr(task, "_disturbance_interval_sum_"+suffix, torch.zeros(1, 6))
    task._disturbance_interval_count = torch.zeros(1, 1)
    world = torch.tensor([[8., -4., 2., 1., -2., 3.]]) if loaded else torch.zeros(1, 6)
    rotations = [torch.tensor([[0., 0., 0., 1.]]), torch.tensor([[0., 0., 2**-.5, 2**-.5]])]
    for factor, quat in zip((.25, .75), rotations):
        # The event sampler has already ramped/gated this buffer; do not label its unscaled target.
        task._current_sustained_wrench_world.copy_(factor*world)
        task._current_sustained_active_mask.fill_(loaded)
        task._current_base_quat_xyzw = lambda q=quat: q
        task._hard_pact_pre_physics_substep()
    result = task._end_disturbance_interval()
    yaw90 = world[:, [1, 0, 2, 4, 3, 5]] * torch.tensor([[1., -1., 1., 1., -1., 1.]])
    expected_yaw = (.25*world + .75*yaw90)/2
    torch.testing.assert_close(result["total_external_wrench_label_world"], torch.stack(applied).mean(0))
    torch.testing.assert_close(result["total_external_wrench_label_world"], .5*world)
    torch.testing.assert_close(result["total_external_wrench_label_yaw_scaled"], expected_yaw*.01)
    torch.testing.assert_close(result["total_external_wrench_label_yaw_normalized"],
        expected_yaw/torch.tensor(task.cfg.deployment_physics.wrench_scale))
    assert result["equivalent_mass_com_wrench_world"].abs().sum() > 0  # Diagnostic, not label.
    assert result["sustained_wrench_active_mask"].item() == loaded


def test_pinn_wiring_does_not_consume_mass_wrench_diagnostic():
    source = inspect.getsource(PPO_HardPACT._compute_bard_loss)
    assert "equivalent_mass_com_wrench_world" not in source
    assert "label_mass_wrench" not in source
    assert "total_wrench_world=applied_at_base[sl]" in source
    assert "applied_wrench_world=applied_at_base[sl]" in source
    # Actual randomized inertia is still supplied to BARD; the QP stays separate.
    assert '"added_base_mass": batch["realized_added_mass"].detach()' in source
    assert '"base_com_shift": batch["realized_com_shift_body"].detach()' in source


@pytest.mark.parametrize('composer_api',[True,False])
@pytest.mark.parametrize('enabled',[True,False])
def test_isaaclab_wrench_application_point_tracks_full_rotated_com(composer_api,enabled):
    from legged_gym.simulator.isaaclab_simulator_pact import IsaacLabSimulator_PACT
    sim=object.__new__(IsaacLabSimulator_PACT)
    sim._cfg=SimpleNamespace(domain_rand=SimpleNamespace(persistent_wrench_at_com=enabled))
    sim._base_link_index=1
    sim._default_com_pos=torch.tensor([[.02,.03,.04]])
    sim._base_com_bias=torch.tensor([[.10,-.02,.01]])
    # Torso rotated 90 degrees around world z; nonzero world translation.
    data=SimpleNamespace(body_link_pos_w=torch.tensor([[[0.,0.,0.],[1.,2.,3.]]]),
        body_link_quat_w=torch.tensor([[[1.,0.,0.,0.],[2**-.5,0.,0.,2**-.5]]]))
    calls=[]; resets=[]
    composer=SimpleNamespace(reset=lambda:resets.append(True),set_forces_and_torques=lambda **kw:calls.append(kw))
    sim._robot=SimpleNamespace(data=data)
    if composer_api: sim._robot.permanent_wrench_composer=composer
    else:
        sim._robot.set_external_force_and_torque=lambda force,torque,**kw:calls.append(dict(forces=force,torques=torque,**kw))
    wrench=torch.tensor([[3.,4.,-20.,1.,2.,3.]])
    original=wrench.clone()
    for change in (0.,.03):
        sim._base_com_bias[:,0]+=change
        sim.hard_pact_apply_base_wrench_world(wrench)
        call=calls[-1]
        assert call['is_global'] and call['body_ids']==[1]
        torch.testing.assert_close(call['forces'],wrench[:,:3,None].transpose(1,2))
        torch.testing.assert_close(call['torques'],wrench[:,3:,None].transpose(1,2))
        if enabled:
            offset=sim._default_com_pos+sim._base_com_bias
            world_offset=torch.stack((-offset[:,1],offset[:,0],offset[:,2]),-1)
            torch.testing.assert_close(call['positions'][:,0],data.body_link_pos_w[:,1]+world_offset)
            # Link-origin equivalent moment yields exactly the specified free
            # torque at CoM; don't pre-add another lever-arm moment in adapter.
            link_moment=wrench[:,3:]+torch.cross(world_offset,wrench[:,:3],dim=-1)
            torch.testing.assert_close(link_moment-torch.cross(world_offset,wrench[:,:3],dim=-1),wrench[:,3:])
        else: assert 'positions' not in call
    assert len(resets)==(2 if composer_api else 0)
    torch.testing.assert_close(wrench,original)


@pytest.mark.parametrize('task_class',[Go2HardPACT,Go2HardPACTPos])
def test_com_application_offset_is_owned_interval_metadata(task_class):
    task=_persistent_task(task_class=task_class)
    task.cfg.domain_rand.persistent_wrench_at_com=True
    task._realized_added_mass=torch.zeros(1,1)
    task._realized_com_shift_body=torch.zeros(1,3)
    offset=torch.tensor([[.12,.01,.05]])
    task.simulator.hard_pact_wrench_application_offset_body=lambda:offset
    for suffix in ('sustained','mass_com','total','sustained_yaw_scaled',
                   'mass_com_yaw_scaled','yaw_scaled','yaw_physical'):
        setattr(task,'_disturbance_interval_sum_'+suffix,torch.zeros(1,6))
    task._disturbance_interval_count=torch.ones(1,1)
    packet=task._end_disturbance_interval()
    offset.zero_()
    torch.testing.assert_close(packet['wrench_application_offset_body'],torch.tensor([[.12,.01,.05]]))
