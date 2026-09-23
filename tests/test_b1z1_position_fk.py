"""Local command-FK contracts, without a simulator or dynamics rollout."""
from types import SimpleNamespace
import pytest
import torch
from rsl_rl.algorithms import b1z1_actor_physics as physics


def fixture():
    actions = torch.zeros(2, 8, requires_grad=True)
    cfg = dict(actor_phys_pos_fk_enabled=True, actor_phys_pos_fk_weight=.4,
               actor_phys_pos_fk_huber_delta=1., actor_phys_pos_fk_deadband=0.,
               actor_phys_pos_fk_axis_weights=[1.,2.,1.], actor_phys_ee_scale=.1,
               actor_phys_arm_indices=[1,3], actor_phys_num_actions=4,
               actor_phys_arm_root_frame="arm_root", position_action_scale=[.2,.3,.4,.5],
               clip_actions=10.)
    # A differentiable kinematic fixture isolates routing independently of BARD.
    def fk(pos, quat, joints, reference, root):
        assert root == "arm_root"
        assert not pos.requires_grad and not quat.requires_grad
        point = torch.stack((joints[:,1], joints[:,3], joints[:,1]+joints[:,3]), -1)
        return point, reference
    a = SimpleNamespace(cfg=cfg, dynamics_backend=SimpleNamespace(commanded_ee_in_frame=fk))
    data = dict(fk_default=torch.zeros(2,5,requires_grad=True),
                fk_base_pos=torch.zeros(2,3,requires_grad=True),
                fk_base_quat=torch.tensor([[0.,0.,0.,1.]]*2,requires_grad=True),
                ee_target=torch.zeros(2,3,requires_grad=True))
    return a, actions, data


def test_matching_and_perturbed_command_gradients():
    a, actions, data = fixture()
    loss, _ = physics.position_fk(a, actions, data)
    assert loss.item() == 0
    with torch.no_grad():
        actions[:,1] = .5
    loss, metrics = physics.position_fk(a, actions, data)
    assert loss > 0 and metrics["pos_fk_finite_count"] == 2
    loss.backward()
    assert torch.isfinite(actions.grad).all()
    assert actions.grad[:,1].abs().sum() > 0
    assert actions.grad[:,[0,2,4,5,6,7]].count_nonzero() == 0
    assert all(value.grad is None for value in data.values())


def test_same_reference_deadband_and_metrics():
    a, actions, data = fixture()
    with torch.no_grad():
        actions[:,1] = .5
        data["ee_target"][:] = torch.tensor([.15,0.,.15])
    loss, metrics = physics.position_fk(a, actions, data)
    assert loss.abs() < 1e-10
    assert metrics["pos_fk_nonfinite_count"] == 0
    a.cfg["actor_phys_pos_fk_deadband"] = .01
    with torch.no_grad():
        data["ee_target"].add_(.005)
    assert physics.position_fk(a, actions, data)[0] == 0


def test_disabled_objective_is_identical():
    from test_b1z1_actor_physics import setup
    a, batch, actions, context = setup()
    before, _ = physics.objective(a, batch, actions, context)
    a.cfg.update(actor_phys_pos_fk_enabled=False, actor_phys_pos_fk_weight=1.)
    after, _ = physics.objective(a, batch, actions, context)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_combined_objective_adds_weighted_fk():
    from test_b1z1_actor_physics import setup
    a, batch, actions, context = setup()
    base_loss, _ = physics.objective(a, batch, actions, context)
    count = actions.shape[-1] // 2
    a.cfg.update(actor_phys_pos_fk_enabled=True, actor_phys_pos_fk_weight=.3,
                 actor_phys_arm_indices=[count-2,count-1], actor_phys_num_actions=count,
                 actor_phys_arm_root_frame="arm_root")
    state = batch["actor_phys_state"]
    batch.update(actor_phys_fk_default=state[:,156:175], actor_phys_fk_base_pos=state[:,:3],
                 actor_phys_fk_base_quat=state[:,3:7])
    a.dynamics_backend.commanded_ee_in_frame = lambda p, r, q, t, f: (
        torch.stack((q[:,count-2],q[:,count-1],q[:,count-1]),-1), t)
    loss, metrics = physics.objective(a,batch,actions,context)
    torch.testing.assert_close(loss, base_loss + metrics["pos_fk_weighted"])
    torch.testing.assert_close(metrics["loss"], loss.detach())
    assert metrics["pos_fk_finite_count"] == len(actions)
    a.cfg.update(actor_phys_pos_fk_enabled=True, actor_phys_pos_fk_weight=0.)
    after, _ = physics.objective(a, batch, actions, context)
    torch.testing.assert_close(base_loss, after, rtol=0, atol=0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires reserved GPU 1")
def test_bard_command_fk_cuda():
    import xml.etree.ElementTree as ET
    from legged_gym import LEGGED_GYM_ROOT_DIR
    from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg
    from legged_gym.dynamics.bard_b1z1_dynamics import BardB1Z1DynamicsBackend
    cfg = B1Z1PACTCfg()
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    path = cfg.asset.file.replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    arm_names = [name for name in cfg.asset.dof_names if name.startswith("z1_")]
    joint = next(j for j in ET.parse(path).getroot().findall("joint") if j.attrib["name"] == arm_names[0])
    root = joint.find("parent").attrib["link"]
    backend = BardB1Z1DynamicsBackend(path, cfg.asset.dof_names, cfg.asset.foot_name,
        cfg.asset.gripper_name, cfg.asset.base_name, device=device, batch_capacity=2)
    joints = torch.tensor([[cfg.init_state.default_joint_angles[n] for n in cfg.asset.dof_names]]*2,
                          device=device, requires_grad=True)
    pos = torch.zeros(2,3,device=device,requires_grad=True)
    quat = torch.tensor([[0.,0.,0.,1.]]*2,device=device,requires_grad=True)
    target = torch.zeros_like(pos,requires_grad=True)
    predicted, reference = backend.commanded_ee_in_frame(pos,quat,joints,target,root)
    (predicted-reference).square().mean().backward()
    assert torch.isfinite(joints.grad).all() and joints.grad.abs().sum() > 0
    assert pos.grad is None and quat.grad is None and target.grad is None
