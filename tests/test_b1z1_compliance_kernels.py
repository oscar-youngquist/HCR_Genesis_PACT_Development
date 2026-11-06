"""Fused kernels must preserve the previous probe/frame/exclusion arithmetic."""
import pytest
import torch
from test_b1z1_directional_compliance import geometry
from legged_gym.envs.b1z1.directional_compliance import (
    directional_probe, collision_approach, rotate_yaw, task_position, world_position)
from legged_gym.envs.b1z1.ee_collision import segment_box
from legged_gym.utils.math_utils import quat_apply, quat_rotate_inverse


def inputs(device, compiled):
    torch.manual_seed(71)
    g, q = geometry(device, compiled)
    q = q[:1].repeat(17, 1) + .04 * torch.randn(17, q.shape[-1], device=device)
    ee, j, arm, legs = g.forward(q)
    yaw = torch.linspace(-2., 2., len(q), device=device)
    roll = yaw * .12
    quat = torch.stack(((roll/2).sin()*(yaw/2).cos(), (roll/2).sin()*(yaw/2).sin(),
                        (roll/2).cos()*(yaw/2).sin(), (roll/2).cos()*(yaw/2).cos()), -1)
    base = torch.randn_like(ee)
    measured = task_position(quat_apply(quat, ee) + base, base, yaw)
    j = rotate_yaw(quat_apply(quat[:, None].expand(-1, j.shape[-1], -1).reshape(-1, 4),
        j.transpose(1, 2).reshape(-1, 3)).reshape(len(q), -1, 3), yaw, True).transpose(1, 2)
    previous = measured + .01 * torch.randn_like(measured)
    candidate = measured + .2 * torch.randn_like(measured)
    candidate[::3] = measured[::3]
    candidate[1::3] = task_position(quat_apply(quat, g.box_center.expand_as(ee))+base, base, yaw)[1::3]
    limits = q.new_tensor([-4., 4.]).expand(len(q), len(g.arm_ids), -1)
    rows = (candidate, previous, q[:, g.arm_ids], j, limits, measured,
            g.clearance_pairs(q[:, g.arm_ids], q, legs), arm, legs, base, quat, yaw, base)
    return g, q, rows


def legacy_predicate(g, full, rows, lower, upper, underground):
    candidate, previous, q, j, limits, measured, baseline, arm, legs, base, quat, yaw, center = rows
    def body(point):
        if point.ndim == 2:
            return quat_rotate_inverse(quat, world_position(point, base, yaw)-base)
        world = world_position(point, base[:, None], yaw[:, None])-base[:, None]
        orientation = quat[:, None].expand(*point.shape[:-1], 4)
        return quat_rotate_inverse(orientation.reshape(-1, 4), world.reshape(-1, 3)).reshape_as(point)
    dq, _, _ = directional_probe(j, q, limits, candidate-measured, g.cfg)
    fractions = torch.linspace(1/g.cfg.collision_probe_steps, 1, g.cfg.collision_probe_steps, device=q.device)
    minimum = g.clearance_pairs(q[:, None]+dq[:, None]*fractions[None, :, None], full, legs).amin(1)
    allowed = ((minimum >= 0) | (minimum > baseline+1e-6)).all(-1)
    start = torch.stack((measured, previous), 1)
    clearance = g.point_clearance(body(start), arm[:, None], legs[:, None])
    path = g.path_clearance(body(start), body(candidate[:, None].expand_as(start)), arm[:, None], legs[:, None])
    end = g.point_clearance(body(candidate), arm, legs)[:, None]
    allowed &= ((path >= 0) | ((clearance < 0) & (path >= clearance-1e-6) & (end > clearance+1e-6))).all(-1).all(-1)
    local = candidate-center
    allowed &= ~((local > lower) & (local < upper)).all(-1) & (local[:, 2] >= underground)
    return allowed & (segment_box(previous-center, local, (lower+upper)/2, (upper-lower)/2) >= 0)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_target_predicate_and_handoff_parity(device,monkeypatch):
    # Match the existing geometry parity suite's full-float32 convention.
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    g, full, rows = inputs(device, device.startswith('cuda'))
    # Full URDF FK retains the original matrix-product formulation as an oracle.
    arm, legs = rows[7:9]
    torch.testing.assert_close(g.arm_points(rows[2]), arm, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(rows[6], g.point_clearance(arm[:, -1], arm, legs), atol=2e-6, rtol=2e-5)
    lower, upper = full.new_full((3,), -2.), full.new_full((3,), -1.)
    g.cfg.collision_batch_size = 24  # Exercise multiple bounded batches and a singleton tail.
    expected = legacy_predicate(g, full, rows, lower, upper, -2.)
    assert expected.any() and (~expected).any()
    actual = g.target_path_allowed(*rows, lower=lower, upper=upper, underground=-2.)
    assert torch.equal(actual, expected)
    ids = torch.tensor([12, 3, 9, 0], device=device)
    assert torch.equal(g.target_path_allowed(*(x[ids] for x in rows), lower=lower, upper=upper, underground=-2.), expected[ids])
    _, previous, _, _, _, measured, _, arm, legs, base, quat, yaw, _ = rows
    directions = full.new_tensor([[.04,0,0],[0,-.04,0],[0,0,.04],[.02,-.02,.02]]).expand(len(full),-1,-1)
    expected_gate = torch.zeros_like(directions[..., 0])
    for origin in (measured, previous):
        start = origin[:, None].expand_as(directions)
        p = g.task_to_body(start, base, quat, yaw)
        end = g.task_to_body(start+directions, base, quat, yaw)
        expected_gate = torch.maximum(expected_gate, collision_approach(
            g.point_clearance(p, arm[:, None], legs[:, None]),
            g.path_clearance(p, end, arm[:, None], legs[:, None]), g.cfg.collision_margin))
    torch.testing.assert_close(g.handoff_paths(directions, previous, measured, arm, legs,
        base, quat, yaw, g.cfg.collision_margin), expected_gate, atol=3e-5, rtol=3e-5)


def test_cached_initial_clearance_preserves_reference():
    from test_b1z1_directional_compliance import fixture
    a, kw = fixture(); b, _ = fixture()
    b.profile.copy_(a.profile)
    expected = a.advance(**kw)
    actual = b.advance(**kw, collision_start=kw['collision_clearance'](kw['q']))
    for name in vars(expected):
        torch.testing.assert_close(getattr(actual,name),getattr(expected,name),rtol=0,atol=0)


def test_graph_buckets_only_duplicate_selected_rows(monkeypatch):
    g, _, rows = inputs('cpu', False)
    g.graph_target_checks = True
    g.cfg.collision_batch_size = 24
    seen = []
    def check(*args):
        candidate = args[0]
        seen.append(candidate.clone())
        return candidate[:, 0] > 105
    monkeypatch.setattr(g, '_target_path_allowed', check)
    selected = tuple(x[:15].clone() for x in rows)
    selected[0][:, 0] = torch.arange(100, 115)
    result = g.target_path_allowed(*selected, lower=torch.zeros(3), upper=torch.ones(3), underground=-2.)
    assert [len(x) for x in seen] == [8, 8]
    assert seen[-1][:, 0].tolist() == [108, 109, 110, 111, 112, 113, 114, 114]
    assert torch.equal(result, selected[0][:, 0] > 105)
