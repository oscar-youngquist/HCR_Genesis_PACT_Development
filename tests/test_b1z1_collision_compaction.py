"""Subset refinement preserves the full-batch path predicate and row identity."""
import pytest
import torch
from legged_gym.envs.b1z1.ee_collision import clip_prefix


@pytest.mark.parametrize('unsafe', [[], [3], [0, 2, 7], list(range(8))])
def test_refinement_only_visits_initially_unsafe_rows(unsafe):
    previous = torch.zeros(8, 3)
    candidate = torch.zeros_like(previous)
    candidate[:, 0] = .1
    candidate[unsafe, 0] = 2.
    limits = torch.linspace(.3, .9, 8)
    calls = []
    selections = []
    def predicate(end, start):
        return end[:, 0] <= limits
    def select(ids):
        selections.append(ids.clone())
        bound = limits.index_select(0, ids)
        def compact(end, start):
            calls.append(len(end))
            return end[:, 0] <= bound
        return compact
    expected = clip_prefix(candidate, previous, predicate)
    actual = clip_prefix(candidate, previous, predicate, select_path=select)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == ([len(unsafe)] * 4 if unsafe else [])
    assert len(selections) == bool(unsafe)
    if unsafe:
        assert selections[0].tolist() == unsafe
        # Four bisections retain a safe prefix within one 1/16 displacement bin.
        remaining = limits[unsafe] - actual[unsafe, 0]
        assert (remaining >= 0).all() and (remaining < 2. / 16).all()
    safe = predicate(candidate, previous)
    assert torch.equal(actual[safe], candidate[safe])
    assert torch.equal(previous, torch.zeros_like(previous))
    assert torch.equal(candidate[unsafe, 0], torch.full((len(unsafe),), 2.))


def test_adapter_geometry_subset_matches_full_batch(monkeypatch):
    from test_b1z1_compliance_review import actual_urdf_task
    from legged_gym.envs.b1z1 import compliance_task
    _, _, _, task = actual_urdf_task()
    task.env.cfg.compliance.collision_compact_refinement = True
    checked = []
    def compare(candidate, previous, predicate, *, select_path):
        # Reordered subset verifies geometry follows IDs, not compact row numbers.
        ids = torch.tensor([2, 0])
        compact = select_path(ids)
        for displacement in (0., .02, -.1):
            end = candidate.clone()
            end[:, 1] += torch.tensor([displacement, 0., -displacement])
            assert torch.equal(compact(end[ids], previous[ids]), predicate(end, previous)[ids])
        full = clip_prefix(candidate, previous, predicate)
        result = clip_prefix(candidate, previous, predicate, select_path=select_path)
        torch.testing.assert_close(result, full, rtol=0, atol=0)
        checked.append(True)
        return result
    monkeypatch.setattr(compliance_task, 'clip_prefix', compare)
    for _ in range(4):
        task.begin()
    assert checked


def test_default_adapter_keeps_full_batch_and_reference_parity(monkeypatch):
    from test_b1z1_compliance_review import actual_urdf_task
    from legged_gym.envs.b1z1 import compliance_task
    _, _, _, task = actual_urdf_task()
    assert not task.env.cfg.compliance.collision_compact_refinement
    assert not task.geometry.graph_target_checks
    calls = []
    def fixed(candidate, previous, predicate, *, select_path):
        assert select_path is None
        def checked(end, start):
            calls.append(len(end))
            assert len(end) == len(candidate)
            return predicate(end, start)
        return clip_prefix(candidate, previous, checked)
    monkeypatch.setattr(compliance_task, 'clip_prefix', fixed)
    for _ in range(4):
        task.begin()
    assert calls and set(calls) == {task.env.num_envs}


def test_fixed_refinement_shapes_do_not_follow_unsafe_counts():
    previous = torch.zeros(32, 3)
    for count in (0, 1, 3, 17, 32, 1, 0):
        candidate = previous.clone()
        candidate[:count, 0] = 2.
        calls = []
        def allowed(end, start):
            calls.append(end.shape)
            return end[:, 0] <= 1.
        actual = clip_prefix(candidate, previous, allowed)
        assert calls == [candidate.shape] * (5 if count else 1)
        expected = candidate.clamp_max(1.)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_recovery_signature_is_warmed_once_without_changing_reference(monkeypatch):
    from test_b1z1_compliance_review import actual_urdf_task
    _, _, _, task = actual_urdf_task()
    _, _, _, eager = actual_urdf_task()
    eager.reference.profile.copy_(task.reference.profile)
    eager.env.cfg.compliance.compile_geometry = False
    calls, eager_calls = [], []
    original = task.geometry.point_clearance
    eager_original = eager.geometry.point_clearance
    def checked(point, *args):
        calls.append(point.shape)
        return original(point, *args)
    monkeypatch.setattr(task.geometry, 'point_clearance', checked)
    def eager_checked(point, *args):
        eager_calls.append(point.shape)
        return eager_original(point, *args)
    monkeypatch.setattr(eager.geometry, 'point_clearance', eager_checked)
    for step in range(3):
        task.begin();eager.begin()
        assert task._recovery_geometry_ready
        assert len(calls) == len(eager_calls) + 1
        for name in vars(task.target):
            torch.testing.assert_close(getattr(task.target, name), getattr(eager.target, name), rtol=0, atol=0)
    # Startup warmup adds only a pure query; profile/filter progression is identical.
    assert calls
