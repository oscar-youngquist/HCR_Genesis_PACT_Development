"""One-way PINN projection, shared dispatch, and thin-config inheritance."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

# Reuse the existing lightweight B1Z1 import setup (no simulator construction).
from test_b1z1_bard_pinn import auxiliary_backward
from rsl_rl.algorithms.pc_grad import PCGrad
from rsl_rl.algorithms.b1z1_bard_pinn import validate_projection_mode


@pytest.mark.parametrize("reference,physics,expected", [
    ([1., 0.], [2., 0.], [3., 0.]),
    ([1., 0.], [0., 2.], [1., 2.]),
    ([1., 0.], [-2., 3.], [1., 3.]),
    ([1., 0.], [-2., 0.], [1., 0.]),
    ([0., 0.], [2., 3.], [2., 3.]),
    ([1., 0.], [0., 0.], [1., 0.]),
    ([1e-9, 0.], [-2., 3.], [1e-9, 3.]),
    ([1e-14, 0.], [0., 1e-13], [1e-14, 1e-13]),
])
@pytest.mark.parametrize("balance", [False, True])
def test_projection(reference, physics, expected, balance):
    ref, phys, result = map(torch.tensor, (reference, physics, expected))
    before = ref.clone(), phys.clone()
    projected = result - ref
    if balance and projected.norm() > ref.norm():
        result = ref + projected * (ref.norm() / projected.norm())
    parameter = torch.nn.Parameter(torch.zeros_like(ref))
    # The new rule is a sum even with the default general PCGrad mean reduction.
    opt = PCGrad(torch.optim.SGD([parameter], lr=.1), owned_only=True)
    opt.pc_backward_pinn_conflict_only(
        [(parameter * ref).sum(), (parameter * phys).sum()], norm_balance=balance)
    assert torch.isfinite(parameter.grad).all()
    torch.testing.assert_close(parameter.grad, result, atol=1e-20, rtol=1e-5)
    torch.testing.assert_close(ref, before[0])
    torch.testing.assert_close(phys, before[1])
    torch.testing.assert_close(opt.last_objective_grads[0], ref)


def test_owned_only_unused_parameters_and_masks():
    ref_only, phys_only, unused, peer = [torch.nn.Parameter(torch.tensor(1.)) for _ in range(4)]
    peer.grad = torch.tensor(7.)
    opt = PCGrad(torch.optim.AdamW([ref_only, phys_only, unused], lr=.1), owned_only=True)
    opt.pc_backward_pinn_conflict_only(
        [2 * ref_only + peer, 3 * phys_only + peer], norm_balance=False)
    assert ref_only.grad == 2 and phys_only.grad == 3
    assert unused.grad is None and peer.grad == 7
    torch.testing.assert_close(opt.last_has_grads[0], torch.tensor([1., 0., 0.]))
    torch.testing.assert_close(opt.last_has_grads[1], torch.tensor([0., 1., 0.]))
    opt.step()
    assert unused == 1 and peer == 1


@pytest.mark.parametrize("mode", [None, "legacy", "conflict_only"])
@pytest.mark.parametrize("sign", [-1., 1.])
@pytest.mark.parametrize("balance", [False, True])
def test_dispatch(mode, sign, balance):
    cfg = dict(pinn_loss_weight=sign, pinn_projection_norm_balance=balance)
    if mode is not None:
        cfg["pinn_projection_mode"] = mode
    a = SimpleNamespace(cfg=cfg, pinn_weight=.25)
    opt = Mock()
    ref, phys = torch.tensor(2., requires_grad=True), torch.tensor(3., requires_grad=True)
    auxiliary_backward(a, opt, ref, phys)
    method = ("pc_backward_pinn_conflict_only" if mode == "conflict_only" else
              "pc_backward_ppgrad" if sign < 0 else "pc_backward_pinn")
    assert [call[0] for call in opt.mock_calls] == [method]
    args = getattr(opt, method).call_args
    assert args.args[0][0] is ref
    assert args.kwargs == ({"norm_balance": balance} if mode == "conflict_only" else {})
    args.args[0][1].backward()
    assert phys.grad == .25


@pytest.mark.parametrize("detached", [False, True])
def test_disabled_physics_reference_only(detached):
    p = torch.nn.Parameter(torch.tensor(1.))
    a = SimpleNamespace(cfg={"pinn_projection_mode": "conflict_only"}, pinn_weight=1. if detached else 0.)
    opt = PCGrad(torch.optim.SGD([p], lr=.1), owned_only=True)
    opt.pc_backward_pinn_conflict_only = Mock(side_effect=AssertionError("disabled physics"))
    auxiliary_backward(a, opt, 2*p, torch.tensor(3., requires_grad=not detached))
    assert p.grad == 2


def test_invalid_selector():
    with pytest.raises(ValueError, match="pinn_projection_mode"):
        validate_projection_mode({"pinn_projection_mode": "symmetric"})


@pytest.mark.parametrize("balanced", [False, True])
def test_legacy_projection_unchanged(balanced):
    p = torch.nn.Parameter(torch.zeros(2))
    opt = PCGrad(torch.optim.SGD([p], lr=.1), reduction="sum", owned_only=True)
    method = opt.pc_backward_ppgrad if balanced else opt.pc_backward_pinn
    # Legacy removes even aligned reference-direction physics, unlike the new rule.
    method([p[0], 2*p[0] + 3*p[1]])
    torch.testing.assert_close(p.grad, torch.tensor([1., 1. if balanced else 3.]))


@pytest.mark.parametrize("sign", [-1., 1.])
def test_actor_backward_dispatch(monkeypatch, sign):
    import test_b1z1_actor_physics as existing
    original_setup = existing.setup
    def setup():
        a, *rest = original_setup()
        a.cfg["pinn_projection_mode"] = "conflict_only"
        return a, *rest
    monkeypatch.setattr(existing, "setup", setup)
    existing.test_actor_uses_shared_projection(
        monkeypatch, sign, "pc_backward_pinn_conflict_only")


def test_ablations_inherit_and_preserve_override():
    from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfgPPO
    from legged_gym.envs.b1z1.b1z1_pact.ablation_configs import make_b1z1_pact_ablation_configs
    from legged_gym.utils.helpers import class_to_dict
    for variant in (4, 5, 6, 7, 8, 9, 10, 11):
        _, train = make_b1z1_pact_ablation_configs(variant)
        cfg = train()
        assert cfg.algorithm.pinn_projection_mode == "conflict_only"
        assert cfg.algorithm.pinn_projection_norm_balance is True
        cfg.algorithm.pinn_projection_mode = "legacy"
        cfg.algorithm.pinn_projection_norm_balance = False
        serialized = class_to_dict(cfg)["algorithm"]
        assert serialized["pinn_projection_mode"] == "legacy"
        assert serialized["pinn_projection_norm_balance"] is False
    assert B1Z1PACTCfgPPO.algorithm.pinn_projection_mode == "conflict_only"
