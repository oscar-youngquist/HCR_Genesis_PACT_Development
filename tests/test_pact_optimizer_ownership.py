import pytest
import torch

from rsl_rl.algorithms.pc_grad import PCGrad
from rsl_rl.algorithms.ppo_pact import PPO_PACT
from rsl_rl.algorithms.ppo_pact_pos import PPO_PACT_Pos
from rsl_rl.modules import ActorCritic_PACT, ActorCritic_PACT_Pos, ContextDecoder
from rsl_rl.runners.pact_runner import _load_optimizer_with_optional_appended_group


def _parameter_ids(optimizer):
    return {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}


def _make_actor(actor_type):
    return actor_type(
        57, 288, 12, [32, 16, 8], [32, 16, 8], 57 * 20,
        16, 16, [32, 16], "elu", 1.0,
    )


def test_pact_actor_and_auxiliary_parameter_groups_are_disjoint():
    actor = _make_actor(ActorCritic_PACT)
    privileged_decoder = ContextDecoder(32, [16, 16, 16], 276)
    grf_decoder = ContextDecoder(32, [16, 16, 16], 12)
    algorithm = PPO_PACT(
        actor, privileged_decoder, 288, grf_decoder_network=grf_decoder,
        num_learning_epochs=1, num_mini_batches=1,
    )

    ppo_ids = _parameter_ids(algorithm.act_optimizer.optimizer)
    auxiliary_encoder_ids = _parameter_ids(algorithm.enc_optimizer)
    grf_auxiliary_ids = _parameter_ids(algorithm.grf_decoder_optimizer)
    context_ids = {id(parameter) for parameter in actor.context_encoder.parameters()}
    grf_ids = {id(parameter) for parameter in grf_decoder.parameters()}
    privileged_ids = {id(parameter) for parameter in privileged_decoder.parameters()}

    aux_ids = {id(p) for p in algorithm.aux_parameters}
    assert context_ids <= auxiliary_encoder_ids
    assert grf_ids <= grf_auxiliary_ids
    assert aux_ids == context_ids | grf_ids | privileged_ids
    assert ppo_ids.isdisjoint(aux_ids)
    assert len(aux_ids) == len(algorithm.aux_parameters)
    assert ppo_ids | context_ids == {id(p) for p in actor.parameters()}


def test_pact_pos_actor_and_auxiliary_parameter_groups_are_disjoint():
    actor = _make_actor(ActorCritic_PACT_Pos)
    privileged_decoder = ContextDecoder(32, [16, 16, 16], 276)
    grf_decoder = ContextDecoder(32, [16, 16, 16], 12)
    algorithm = PPO_PACT_Pos(
        actor, privileged_decoder, 288, grf_decoder_network=grf_decoder,
        num_learning_epochs=1, num_mini_batches=1,
    )

    ppo_ids = _parameter_ids(algorithm.act_optimizer.optimizer)
    auxiliary_encoder_ids = _parameter_ids(algorithm.enc_optimizer)
    context_ids = {id(parameter) for parameter in actor.context_encoder.parameters()}
    assert context_ids == auxiliary_encoder_ids
    assert ppo_ids.isdisjoint(context_ids)
    assert ppo_ids | context_ids == {id(p) for p in actor.parameters()}
    assert ppo_ids.isdisjoint({id(parameter) for parameter in grf_decoder.parameters()})
    assert ppo_ids.isdisjoint({id(parameter) for parameter in privileged_decoder.parameters()})

    # A loss may depend on encoder outputs without granting PPO ownership.
    encoder_parameters = list(actor.context_encoder.parameters())
    before = [p.detach().clone() for p in encoder_parameters]
    primary = sum(p.square().sum() for p in actor.parameters())
    cloning = sum(p.sum() for p in actor.parameters())
    algorithm.act_optimizer.pc_backward_ppgrad([primary, cloning])
    assert all(p.grad is None for p in encoder_parameters)
    algorithm.act_optimizer.step()
    for p, original in zip(encoder_parameters, before):
        assert torch.equal(p, original)
    algorithm.enc_optimizer.zero_grad()
    sum(p.square().sum() for p in encoder_parameters).backward()
    algorithm.enc_optimizer.step()
    assert any(not torch.equal(p, original) for p, original in zip(encoder_parameters, before))


def test_pcgrad_leaves_parameters_unused_by_all_objectives_at_none():
    active = torch.nn.Parameter(torch.tensor([2.0]))
    inactive = torch.nn.Parameter(torch.tensor([3.0]))
    optimizer = torch.optim.AdamW([active, inactive], lr=0.1, weight_decay=0.1)
    pcgrad = PCGrad(optimizer, reduction="sum")
    before = inactive.detach().clone()

    pcgrad.pc_backward([active.square(), 3.0 * active])
    assert active.grad is not None
    assert inactive.grad is None
    pcgrad.step()
    torch.testing.assert_close(inactive, before)


def test_pre_shared_decoder_optimizer_checkpoint_migrates():
    old_parameter = torch.nn.Parameter(torch.ones(2))
    old_optimizer = torch.optim.AdamW([{"params": [old_parameter], "name": "actor"}])
    old_state = old_optimizer.state_dict()

    actor_parameter = torch.nn.Parameter(torch.ones(2))
    grf_parameter = torch.nn.Parameter(torch.ones(2))
    current_optimizer = torch.optim.AdamW([
        {"params": [actor_parameter], "name": "actor"},
        {"params": [grf_parameter], "name": "ppo_grf_decoder"},
    ])
    _load_optimizer_with_optional_appended_group(current_optimizer, old_state)
    assert [group["name"] for group in current_optimizer.param_groups] == [
        "actor", "ppo_grf_decoder"
    ]


@pytest.mark.parametrize("pinn_direction,expected", [
    ([-2., 3.], [1., 3.]), ([2., 3.], [1., 3.]),
    ([0., 3.], [1., 3.]), ([2., 0.], [1., 0.]),
])
def test_aux_projection_protects_primary_and_isolates_parameter_group(pinn_direction, expected):
    shared = torch.nn.Parameter(torch.ones(2))
    decoder = torch.nn.Parameter(torch.ones(2))
    outside = torch.nn.Parameter(torch.ones(2))
    unused = torch.nn.Parameter(torch.ones(2))
    primary = shared[0] + 4 * decoder.sum() + outside.sum()
    pinn = (shared * torch.tensor(pinn_direction)).sum() + 5 * outside.sum()
    optimizer = PCGrad([torch.optim.SGD([shared, unused], lr=0.1),
                        torch.optim.Adam([decoder], lr=0.1)])
    info = optimizer.pc_backward_primary(primary, pinn)
    torch.testing.assert_close(shared.grad, torch.tensor(expected))
    torch.testing.assert_close(decoder.grad, torch.full((2,), 4.))
    assert outside.grad is None and unused.grad is None
    assert info["projected"] == 1.0
    # The added PINN contribution is orthogonal to the primary gradient.
    torch.testing.assert_close((shared.grad - torch.tensor([1., 0.]))[0], torch.tensor(0.))
    before = [p.detach().clone() for p in (shared, decoder)]
    optimizer.step()
    assert not torch.equal(shared, before[0])
    assert not torch.equal(decoder, before[1])
    optimizer.zero_grad()
    assert shared.grad is None and decoder.grad is None


def test_aux_projection_handles_zero_primary_and_inactive_pinn():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = PCGrad(torch.optim.SGD([parameter], lr=0.1))
    optimizer.pc_backward_primary(parameter.sum() * 0, parameter.sum())
    torch.testing.assert_close(parameter.grad, torch.ones(2))
    optimizer.pc_backward_primary(3 * parameter.sum())
    torch.testing.assert_close(parameter.grad, torch.full((2,), 3.))


@pytest.mark.parametrize("method", ["pc_backward", "pc_backward_pinn", "pc_backward_ppgrad"])
def test_restricted_actor_backward_preserves_original_projection(method):
    import random
    results = []
    for restricted in (False, True):
        parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
        encoder = torch.nn.Parameter(torch.tensor([2., 3.]))
        optimizer = PCGrad(torch.optim.Adam([parameter]), reduction="sum", restrict_backward=restricted)
        primary = (parameter * encoder).sum()
        pinn = (parameter * torch.tensor([-3., 1.]) * encoder).sum()
        random.seed(0)
        getattr(optimizer, method)([primary, pinn])
        results.append(parameter.grad.clone())
        if restricted:
            assert encoder.grad is None
    torch.testing.assert_close(*results)
