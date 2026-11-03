"""B1Z1 one-way projection port; no simulator or dynamics backend needed."""
from unittest.mock import Mock

import pytest
import torch

from rsl_rl.algorithms.pc_grad import PCGrad
from rsl_rl.algorithms.ppo_hard_pact import PPO_HardPACT
from legged_gym.envs.go2.go2_hard_pact.ablation_configs import make_hard_pact_variant_configs
from rsl_rl.hard_pact_ablations import HARD_PACT_VARIANTS
from test_hard_pact_auxiliary import make_algorithm, make_batch


@pytest.mark.parametrize("reference,physics,projected", [
    ([1., 0.], [2., 0.], [2., 0.]),
    ([1., 0.], [0., 2.], [0., 2.]),
    ([1., 0.], [-2., 3.], [0., 3.]),
    ([1., 0.], [-2., 0.], [0., 0.]),
    ([0., 0.], [2., 3.], [2., 3.]),
    ([1., 0.], [0., 0.], [0., 0.]),
    ([1e-16, 0.], [-2., 3.], [0., 3.]),
])
@pytest.mark.parametrize("balance", [False, True])
def test_projection_and_norm_balance(reference, physics, projected, balance):
    p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    ref, phys, expected = [torch.tensor(v, dtype=p.dtype) for v in (reference, physics, projected)]
    opt = PCGrad(torch.optim.SGD([p], lr=.1))  # Even default mean reduction must SUM these objectives.
    opt.pc_backward_pinn_conflict_only([p.dot(ref), p.dot(phys)], norm_balance=balance)
    if balance and expected.norm() > ref.norm():
        expected = expected * (ref.norm()/expected.norm())
    torch.testing.assert_close(p.grad, ref+expected, atol=1e-28, rtol=1e-12)
    torch.testing.assert_close(opt.last_objective_grads[0], ref)
    torch.testing.assert_close(opt.last_objective_grads[1], phys)
    assert torch.isfinite(p.grad).all()


def test_owned_parameters_and_unused_grad_none():
    p = torch.nn.Parameter(torch.ones(2))
    peer = torch.nn.Parameter(torch.tensor(2.))
    unused = torch.nn.Parameter(torch.tensor(3.))
    peer.grad = torch.tensor(7.)
    opt = PCGrad(torch.optim.AdamW([p, unused]), owned_only=True)
    opt.pc_backward_pinn_conflict_only([p.sum(), -p[0]+p[1]+peer], norm_balance=False,
                                      record_diagnostics=False)
    assert peer.grad == 7 and unused.grad is None
    assert opt.last_objective_grads is None
    before = unused.detach().clone()
    opt.step()
    torch.testing.assert_close(unused, before)


@pytest.mark.parametrize("balanced", [False, True])
def test_legacy_still_removes_aligned_component(balanced):
    p = torch.nn.Parameter(torch.zeros(2))
    opt = PCGrad(torch.optim.SGD([p], lr=.1), reduction="sum")
    method = opt.pc_backward_ppgrad if balanced else opt.pc_backward_pinn
    method([p[0], 2*p[0]+3*p[1]])
    torch.testing.assert_close(p.grad, torch.tensor([1., 1. if balanced else 3.]))


@pytest.mark.parametrize("weight", [-1., .25])
def test_projection_selection_does_not_change_existing_weight_schedule(weight):
    traces = []
    for mode in ("legacy", "conflict_only"):
        alg = PPO_HardPACT.__new__(PPO_HardPACT)
        alg.pinn_projection_mode = mode
        alg.pinn_weight_final, alg.pinn_weight = weight, 0.
        alg.pinn_init, alg.pinn_warmup_steps = 2, 5
        alg.console_debug = False
        trace = []
        for iteration in range(10):
            alg.num_pinn_updates = max(iteration-2, 0)
            trace.append(alg._update_pinn_weight_for_iteration(iteration))
        traces.append(trace)
    assert traces[0] == traces[1]


@pytest.mark.parametrize("mode", [None, "legacy", "conflict_only"])
@pytest.mark.parametrize("weight", [-1., 1.])
@pytest.mark.parametrize("physics", [False, True])
def test_shared_actor_encoder_decoder_dispatch(mode, weight, physics):
    alg = PPO_HardPACT.__new__(PPO_HardPACT)
    alg.pinn_weight_final = weight
    if mode is not None:
        alg.pinn_projection_mode = mode
    alg.pinn_projection_norm_balance = False
    for auxiliary in (False, True):
        opt = Mock()
        reference, pinn = torch.tensor(1.), torch.tensor(2.) if physics else None
        if auxiliary:
            alg._auxiliary_pcgrad_step(opt, [], reference, pinn)
        else:
            alg._physics_pcgrad_backward(opt, [reference, pinn] if physics else [reference])
        expected = ("pc_backward" if not physics else "pc_backward_pinn_conflict_only"
                    if mode == "conflict_only" else "pc_backward_pinn"
                    if weight >= 0 else "pc_backward_ppgrad")
        assert [c[0] for c in opt.method_calls] == [expected]
        if physics and mode == "conflict_only":
            assert opt.method_calls[0].kwargs["norm_balance"] is False


def test_config_inheritance_overrides_and_constructor_validation():
    for variant in HARD_PACT_VARIANTS:
        _, cls = make_hard_pact_variant_configs(variant, "isaaclab")
        cfg = cls()
        assert cfg.algorithm.pinn_projection_mode == "conflict_only"
        assert cfg.algorithm.pinn_projection_norm_balance is True
        cfg.algorithm.pinn_projection_mode = "legacy"
        cfg.algorithm.pinn_projection_norm_balance = False
        assert cls().algorithm.pinn_projection_mode == "conflict_only"
    with pytest.raises(ValueError, match="pinn_projection_mode"):
        make_algorithm(pinn_projection_mode="typo")
    assert make_algorithm().pinn_projection_mode == "legacy"


def test_real_owned_auxiliary_construction_backward_and_step():
    alg = make_algorithm(pinn_projection_mode="conflict_only", pinn_projection_norm_balance=False)
    owners = [alg.ppo_parameters, alg.auxiliary_encoder_parameters, alg.auxiliary_decoder_parameters]
    assert all(not ({id(p) for p in a} & {id(p) for p in b})
               for i, a in enumerate(owners) for b in owners[i+1:])
    with alg._frozen_auxiliary_decoders():
        aux = alg._compute_auxiliary_loss(*make_batch(), return_decoder_inputs=True)
        # A synthetic already-weighted physics objective exercises the actual
        # frozen-decoder graph and ownership, without a BARD/simulator allocation.
        alg._auxiliary_pcgrad_step(alg.encoder_pcgrad, owners[1], aux["loss"], .1*aux["grf"])
        assert all(p.grad is None for p in owners[0]+owners[2])
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in owners[1])
        alg.encoder_pcgrad.step()
    alg.encoder_pcgrad.zero_grad()
    dec = alg._compute_auxiliary_decoder_loss(**aux["decoder_inputs"])
    alg._auxiliary_pcgrad_step(alg.decoder_pcgrad, owners[2], dec["loss"], .1*dec["grf"])
    assert all(p.grad is None for p in owners[0]+owners[1])
    assert all(torch.isfinite(p.grad).all() for p in owners[2] if p.grad is not None)
    alg.decoder_pcgrad.step()
