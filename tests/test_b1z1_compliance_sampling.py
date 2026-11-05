"""Coverage/config tests; no simulator, learning, or changed force distribution."""
import pytest
import torch

from legged_gym.envs.b1z1.directional_compliance import ComplianceCfg, ComplianceReference
from legged_gym.envs.b1z1.compliance_sampling import ProfileSampler


def sample(cfg=None, n=30000, seed=123):
    torch.manual_seed(seed)
    sampler = ProfileSampler(cfg or ComplianceCfg(), 'cpu')
    profile, ids = sampler.sample(n)
    return profile, ids, sampler


def test_mixture_proportions_zeros_correlations_and_yaw():
    p, ids, sampler = sample()
    expected = torch.tensor(ComplianceCfg.family_weights)
    torch.testing.assert_close(torch.bincount(ids, minlength=6)/len(ids), expected, atol=.012, rtol=0)
    assert ((0 <= p) & (p <= 1)).all()
    assert p[ids == 0].eq(0).all()
    assert p[ids == 1, 6:].eq(0).all()
    assert p[ids == 1, :6].gt(0).any(-1).all()
    assert p[ids == 2, :6].eq(0).all()
    assert p[ids == 2, 6:].gt(0).any(-1).all()
    handoff = p[ids == 3]
    assert torch.equal(handoff[:, :4] > 0, handoff[:, 6:10] > 0)
    assert handoff[:, :4].gt(0).any(-1).all()
    assert (handoff[:, :4] != handoff[:, 6:10])[handoff[:, :4] > 0].float().mean() > .95
    arm = p[ids == 1, :6].reshape(-1, 3, 2)
    num_axes = arm.gt(0).any(-1).sum(-1)
    assert (num_axes == 1).any() and (num_axes > 1).any()
    both = arm.gt(0).all(-1)
    assert (arm[..., 0] == arm[..., 1])[both].float().mean() > .6
    planar = p[ids == 1, :4]
    both_planar = planar.gt(0).all(-1)
    assert (planar[both_planar, 0] == planar[both_planar, 2]).float().mean() > .3
    base = p[ids == 2, 6:]
    translation = base[:, :4].gt(0).any(-1)
    yaw = base[:, 4:].gt(0).any(-1)
    assert (~yaw[translation]).float().mean() > .6
    assert (yaw & ~translation).any() and (yaw & translation).any()
    uniform = p[ids == 5]
    assert not uniform.eq(0).any()
    stats = sampler.statistics(p, ids)
    assert stats['compliance_sampling/zero_coefficient_fraction'] > .4


def test_vertical_payload_ordering_and_reverse_patterns():
    cfg = ComplianceCfg()
    cfg.family_weights = [0, 0, 0, 0, 1, 0]
    cfg.payload_probability = 1.
    p, ids, _ = sample(cfg, 10000)
    assert ids.eq(4).all()
    downward_low = p[:, 4] < p[:, 5]
    assert .72 < downward_low.float().mean() < .78
    for rows, low_index, high_index in ((downward_low, 4, 5), (~downward_low, 5, 4)):
        assert p[rows, low_index].ge(.05).all() and p[rows, low_index].le(.15).all()
        assert p[rows, high_index].ge(.4).all() and p[rows, high_index].le(.65).all()
    assert p[:, :4].gt(0).any(-1).all()
    assert p[:, :4].le(.6).all()


def test_axis_probabilities_can_select_yaw_only():
    cfg = ComplianceCfg(); cfg.family_weights = [0, 0, 1, 0, 0, 0]
    cfg.base_axis_probability = [0, 0, 1]
    cfg.paired_direction_probability = 1.
    p, ids, _ = sample(cfg, 200)
    assert ids.eq(2).all() and p[:, :10].eq(0).all() and p[:, 10:].gt(0).all()


def test_vertical_payload_reference_response_not_universal_downward_stiffness():
    from test_b1z1_directional_compliance import fixture
    model, kw = fixture(4)
    model.profile.zero_()
    model.profile[:, 4:6] = torch.tensor([[.1,.5], [.1,.5], [.5,.1], [.5,.1]])
    kw['force'][:, 2] = torch.tensor([-10.,10.,-10.,10.])
    for _ in range(100): result = model.advance(**kw)
    displacement = result.position[:, 2]-kw['nominal'][:, 2]
    assert displacement[[0,2]].lt(0).all() and displacement[[1,3]].gt(0).all()
    torch.testing.assert_close(displacement[1], -5*displacement[0], atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(displacement[2], -5*displacement[3], atol=1e-6, rtol=1e-4)
    assert result.base.eq(0).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_legacy_uniform_exact_rng_and_bounds(dtype):
    cfg = ComplianceCfg(); cfg.sampling_mode = 'uniform'
    cfg.ee_profile_low = [.1, .3, 0, 0, .2, .9]
    cfg.ee_profile_high = [.1, .8, 0, .1, 1, 1]
    torch.manual_seed(71)
    expected = torch.tensor(cfg.ee_profile_low+cfg.base_profile_low, dtype=dtype)
    expected = expected + (torch.tensor(cfg.ee_profile_high+cfg.base_profile_high, dtype=dtype)-expected)*torch.rand(100, 12)
    next_draw = torch.rand(3)
    torch.manual_seed(71)
    p, ids = ProfileSampler(cfg, 'cpu', dtype).sample(100)
    assert torch.equal(p, expected) and ids.eq(5).all()
    assert torch.equal(torch.rand(3), next_draw)


def test_bounds_disabled_directions_fixed_and_exclusion():
    cfg = ComplianceCfg()
    cfg.ee_profile_low = [.1, 0, 0, 0, 0, 0]
    cfg.ee_profile_high = [.1, .8, 0, 0, 1, 1]
    cfg.base_profile_high = [0]*6
    with pytest.warns(UserWarning, match='Excluded'):
        p, ids, sampler = sample(cfg, 4000)
    assert p[:, 0].eq(.1).all() and p[:, 2:4].eq(0).all() and p[:, 6:].eq(0).all()
    assert not torch.isin(ids, torch.tensor([0, 2, 3])).any()
    assert sampler.effective_weights.sum() == 1
    assert p.ge(torch.tensor(cfg.ee_profile_low+cfg.base_profile_low)).all()
    assert p.le(torch.tensor(cfg.ee_profile_high+cfg.base_profile_high)).all()
    cfg.ee_profile_high = cfg.ee_profile_low.copy()
    cfg.base_profile_low = cfg.base_profile_high.copy()
    p, ids, _ = sample(cfg, 20)
    assert ids.eq(6).all()
    assert torch.equal(p, torch.tensor(cfg.ee_profile_low+cfg.base_profile_low).expand_as(p))


def test_incompatible_positive_weights_fail_clearly():
    cfg = ComplianceCfg(); cfg.family_weights = [1, 0, 0, 0, 0, 0]
    cfg.ee_profile_low = [.1]*6
    with pytest.warns(UserWarning, match='Excluded rejection'):
        with pytest.raises(ValueError, match='No compatible'):
            ProfileSampler(cfg, 'cpu')


@pytest.mark.parametrize('key,value', [('sampling_mode','bad'), ('family_weights',[0]*6),
    ('base_axis_probability',[.5,.5,1.1]), ('symmetric_gain_probability',-1),
    ('ee_active_range',[.8,.2]), ('payload_low_range',[.1,.6])])
def test_invalid_sampling_settings(key, value):
    cfg = ComplianceCfg(); setattr(cfg, key, value)
    with pytest.raises(ValueError): ProfileSampler(cfg, 'cpu')


def test_reproducibility_partial_reset_and_diagnostic_only_ids():
    cfg = ComplianceCfg()
    torch.manual_seed(2); first = ComplianceReference(cfg, 32, 'cpu')
    torch.manual_seed(2); second = ComplianceReference(cfg, 32, 'cpu')
    assert torch.equal(first.profile, second.profile)
    assert torch.equal(first.family_ids, second.family_ids)
    old_profile, old_ids = first.profile.clone(), first.family_ids.clone()
    first.delta.fill_(1); first.rate.fill_(2); first.eta.fill_(3)
    first.reset(torch.tensor([1, 7]))
    keep = torch.ones(32, dtype=torch.bool); keep[[1, 7]] = False
    assert torch.equal(first.profile[keep], old_profile[keep])
    assert torch.equal(first.family_ids[keep], old_ids[keep])
    for buffer, value in ((first.delta, 1), (first.rate, 2), (first.eta, 3)):
        assert buffer[keep].eq(value).all() and buffer[~keep].eq(0).all()
    assert first.profile.shape == (32, 12)
    first.reset(torch.empty(0, dtype=torch.long))


def test_sampled_family_reference_responses():
    from test_b1z1_directional_compliance import fixture
    p, ids, _ = sample(n=4000)
    for family in range(4):
        eligible = (ids == family)
        if family == 3:
            eligible &= p[:, 3] > 0  # Apply load along this profile's yielding direction.
        rows = eligible.nonzero(as_tuple=True)[0][:128]
        model, kw = fixture(len(rows)); model.profile.copy_(p[rows])
        # Interior arm workspace: matching lateral loads do not force handoff.
        kw['force'][:, 1] = 10
        result = model.advance(**kw)
        if family == 0:
            torch.testing.assert_close(result.position, kw['nominal'])
            assert result.base.eq(0).all()
        elif family == 1:
            assert result.base.eq(0).all()
            assert result.position[model.profile[:, 3] > 0, 1].gt(0).all()
        elif family == 2:
            torch.testing.assert_close(result.position, kw['nominal'])
            assert result.base[model.profile[:, 9] > 0, 1].gt(0).all()
        else:
            assert result.base.eq(0).all()  # positive EE gains exceed stiffness threshold
            assert result.position[model.profile[:, 3] > 0, 1].gt(0).all()
