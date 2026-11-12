"""Stage 4 standalone API and optional pinned upstream numerical comparison."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Union
import pytest
import torch
from torch import nn

from rsl_rl.facet_model import FACETModel, FACETVecNorm


def inputs(n=3):
    generator = torch.Generator().manual_seed(42)
    return [torch.randn(n, d, generator=generator) for d in (237, 52, 9)]


def test_model_api_isolation_and_statistics_roundtrip():
    model = FACETModel(237, 52)
    obs, priv, ext = inputs()
    hidden, init = torch.randn(3, 128), torch.tensor([True, False, True])
    model.update_normalizers(obs, priv, ext)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    first = model.student_action(obs, hidden, init)
    for _ in range(2):
        teacher = model.teacher_distribution(obs, priv*20, ext+10)
        assert teacher.mean.shape == (3, 17)
        assert teacher.log_prob(teacher.mean).shape == teacher.entropy().shape == (3,)
        assert model.value(obs, priv, ext).shape == (3, 1)
        for expected, actual in zip(first, model.student_action(obs, hidden, init)):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    copy = FACETModel(237, 52)
    copy.load_state_dict(before)
    for expected, actual in zip(first, copy.student_action(obs, hidden, init)):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    with torch.no_grad():
        model.teacher.std.fill_(-2)
    assert (model.teacher_distribution(obs, priv, ext).stddev == 1e-6).all()
    model.eval()
    with pytest.raises(RuntimeError, match='evaluation'):
        model.update_normalizers(obs, priv, ext)


def test_sequence_resets_match_steps_and_no_privileged_input():
    model = FACETModel(237, 52)
    obs, priv, ext = inputs(21)
    model.update_normalizers(obs, priv, ext)
    obs = obs.reshape(3, 7, -1)
    init = torch.zeros(3, 7, dtype=torch.bool)
    init[:, 0] = True
    init[1, 3], init[2, 5] = True, True
    hidden = torch.randn(3, 128)
    sequence = model.student_action(obs, hidden, init)
    steps, recon = [], []
    h = hidden
    for t in range(7):
        action, h, prediction = model.student_action(obs[:, t], h, init[:, t])
        steps.append(action)
        recon.append(prediction)
    torch.testing.assert_close(sequence[0], torch.stack(steps, 1), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(sequence[1], h)
    torch.testing.assert_close(sequence[2], torch.stack(recon, 1), atol=2e-6, rtol=2e-5)
    reset = model.estimate(obs[:, 0], hidden*100, init[:, 0])
    zero = model.estimate(obs[:, 0], torch.zeros_like(hidden), init[:, 0])
    for a, b in zip(reset, zero):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_vecnorm_initial_pseudocount_and_explicit_update():
    norm = FACETVecNorm(2)
    x = torch.tensor([[1., 2.], [3., 4.]])
    torch.testing.assert_close(norm(x), x/(1e-5**.5))
    assert norm.count == 1
    norm.update(x)
    torch.testing.assert_close(norm.sum, x.sum(0)/3)
    torch.testing.assert_close(norm.ssq, x.square().sum(0)/3)
    assert norm.count == 3


def upstream_objects():
    """Execute only isolated pure-PyTorch definitions, not framework imports.

    Set FACET_UPSTREAM_ROOT to the pinned checkout to rerun this comparison.
    No torchrl/tensordict/hydra installation or upstream runner is needed.
    """
    root = Path(os.environ.get('FACET_UPSTREAM_ROOT', '/tmp/facet-stage2-upstream'))
    if not (root/'active_adaptation/learning/ppo/ppo_facet.py').exists():
        pytest.skip('Pinned FACET source checkout unavailable; set FACET_UPSTREAM_ROOT')
    import subprocess
    assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == 'c0d6f5aa6146814803090b5f6f19aafb8ce6f792'
    namespace = dict(torch=torch, nn=nn, Union=Union, recurrent_mode=lambda: False,
        _DecoratorContextManager=__import__('contextlib').ContextDecorator,
        D=torch.distributions, constraints=torch.distributions.constraints)
    for path, names in (
        ('ppo/common.py', ('make_mlp', 'Actor')),
        ('modules/vecnorm.py', ('VecNorm',)),
        ('modules/distributions.py', ('IndependentNormal',)),
        ('ppo/ppo_facet.py', ('GRU', 'GRUModule'))):
        tree = ast.parse((root/'active_adaptation/learning'/path).read_text())
        selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        exec(compile(ast.Module(body=selected, type_ignores=[]), path, 'exec'), namespace)
    return SimpleNamespace(**namespace)


def test_pinned_upstream_encoders_actors_critic_gru_and_norm():
    up = upstream_objects()
    local = FACETModel(237, 52)
    obs, priv, ext = inputs()
    norms = [up.VecNorm(d, d, decay=1.) for d in (237, 52, 9)]
    normalized = [norm(x) for norm, x in zip(norms, (obs, priv, ext))]
    local.update_normalizers(obs, priv, ext)
    for norm, theirs, x in zip((local.obs_norm, local.priv_norm, local.wrench_norm), normalized, (obs, priv, ext)):
        torch.testing.assert_close(norm(x), theirs, rtol=0, atol=0)
    o, p, e = normalized
    def network(widths, out, x, weights):
        net = nn.Sequential(up.make_mlp(widths), nn.LazyLinear(out))
        net(x)
        net.load_state_dict(weights)
        return net(x)
    pf = network([128], 128, p, local.priv_encoder.state_dict())
    ef = network([32], 32, e, local.force_encoder.state_dict())
    torch.testing.assert_close(pf, local.priv_encoder(p), rtol=0, atol=0)
    torch.testing.assert_close(ef, local.force_encoder(e), rtol=0, atol=0)
    for actor in (local.teacher, local.student):
        trunk = up.make_mlp([256, 256, 256])
        x = torch.cat((o, pf, ef), -1)
        head = up.Actor(17)
        head(trunk(x))
        trunk.load_state_dict(actor.mlp.state_dict())
        head.actor_mean.load_state_dict(actor.mean.state_dict())
        head.actor_std.data.copy_(actor.std)
        mean, scale = head(trunk(x))
        dist = up.IndependentNormal(mean, scale)
        ours = actor(o, pf, ef)
        for a, b in ((mean, ours.mean), (dist.stddev, ours.stddev),
                     (dist.log_prob(mean), ours.log_prob(mean)), (dist.entropy(), ours.entropy())):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    expected = network([512, 256, 128], 1, torch.cat((e, o, p), -1), local.critic.state_dict())
    torch.testing.assert_close(expected, local.value(obs, priv, ext), rtol=0, atol=0)
    estimator = up.GRUModule(169, [128, 32, 9])
    h = torch.randn(3, 128)
    init = torch.zeros(3, 1, dtype=torch.bool)
    estimator(o, init, h)
    estimator.mlp.load_state_dict(local.estimator.mlp.state_dict())
    estimator.gru.gru.load_state_dict(local.estimator.gru.state_dict())
    estimator.gru.ln.load_state_dict(local.estimator.norm.state_dict())
    estimator.out.load_state_dict(local.estimator.out.state_dict())
    for a, b in zip(estimator(o, init, h), local.estimate(obs, h, init)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # The upstream sequence path DOES reset. Verify it with mid-sequence resets;
    # our explicit single-step reset fixes the upstream framework dependency.
    estimator.gru.forward.__globals__['recurrent_mode'] = lambda: True
    estimator.gru.forward.__globals__['einops'] = SimpleNamespace(repeat=lambda x, pattern, t: x[:, None].expand(-1, t, -1))
    sequence = o[:, None].repeat(1, 4, 1)
    resets = torch.zeros(3, 4, 1, dtype=torch.bool)
    resets[1, 2] = True
    theirs = estimator(sequence, resets, h[:, None].expand(-1, 4, -1))
    ours = local.estimator(sequence, h, resets)
    for a, b in zip(theirs[:3], ours[:3]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(theirs[-1][:, -1], ours[-1], rtol=0, atol=0)
