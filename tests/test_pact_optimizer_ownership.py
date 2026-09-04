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


def test_pact_optimizer_ownership_matches_shared_b1z1_plan():
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

    assert context_ids <= ppo_ids & auxiliary_encoder_ids
    assert grf_ids <= ppo_ids & grf_auxiliary_ids
    assert ppo_ids.isdisjoint(privileged_ids)


def test_pact_pos_overlaps_context_but_not_auxiliary_only_decoders():
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
    assert context_ids <= ppo_ids & auxiliary_encoder_ids
    assert ppo_ids.isdisjoint({id(parameter) for parameter in grf_decoder.parameters()})
    assert ppo_ids.isdisjoint({id(parameter) for parameter in privileged_decoder.parameters()})


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
