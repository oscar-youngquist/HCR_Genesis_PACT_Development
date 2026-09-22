"""Analytic B1Z1 force gradients, mass-label cancellation and ownership."""
from types import SimpleNamespace
import pytest
import torch
from test_b1z1_sampled_context import make_model
from rsl_rl.algorithms.b1z1_bard_pinn import losses, configure_optimizers
from rsl_rl.algorithms.b1z1_bard_pinn import restore_optimizers
from rsl_rl.algorithms.b1z1_bard_pinn import combined_pinn_loss
from rsl_rl.algorithms.b1z1_bard_pinn import auxiliary_backward
from rsl_rl.algorithms.b1z1_bard_pinn import physical_metrics
from unittest.mock import Mock


@pytest.mark.parametrize("kind,units", [("inverse", ("N", "Nm")), ("rollout", ("mps", "radps"))])
def test_physical_mae_units_and_detachment(kind, units):
    residual = torch.tensor([2.] * 3 + [-3.] * 3 + [4.] * 12 + [-5.] * 7).repeat(2, 1).requires_grad_()
    metrics = {}
    physical_metrics(kind, residual.square().mean(), residual, metrics)
    for block, value, unit in (("base_linear", 2., units[0]), ("base_angular", 3., units[1]),
                               ("legs", 4., units[1]), ("arm_gripper", 5., units[1])):
        assert metrics[f"{kind}/{block}_mae_{unit}"].item() == value
    assert all(not value.requires_grad for value in metrics.values())


@pytest.mark.parametrize("configured,scheduled,method", [
    (1., .25, "pc_backward_pinn"), (-1., .25, "pc_backward_ppgrad"),
    (-1., 0., "pc_backward"), (0., 0., "pc_backward")])
def test_projection_sign_and_warmup(configured, scheduled, method):
    algorithm = SimpleNamespace(cfg={"pinn_loss_weight": configured}, pinn_weight=scheduled)
    optimizer = Mock()
    supervised = torch.tensor(2., requires_grad=True)
    physics = torch.tensor(3., requires_grad=True)
    auxiliary_backward(algorithm, optimizer, supervised, physics)
    assert [call[0] for call in optimizer.mock_calls] == [method]
    objectives = getattr(optimizer, method).call_args.args[0]
    assert objectives[0] is supervised
    if scheduled:
        objectives[1].backward()
        torch.testing.assert_close(physics.grad, torch.tensor(scheduled))
    else:
        assert len(objectives) == 1


@pytest.mark.parametrize("inverse_weight,rollout_weight", [(1., 1.), (2., .5), (0., 1.), (1., 0.)])
def test_component_weights_and_outer_weight(inverse_weight, rollout_weight):
    inverse = torch.tensor(2., requires_grad=True)
    rollout = torch.tensor(3., requires_grad=True)
    loss = .01 * combined_pinn_loss(inverse, rollout, {
        "pinn_inverse_weight": inverse_weight, "pinn_rollout_weight": rollout_weight})
    torch.testing.assert_close(loss, torch.tensor(.01 * (2 * inverse_weight + 3 * rollout_weight)))
    loss.backward()
    torch.testing.assert_close(inverse.grad, torch.tensor(.01 * inverse_weight))
    torch.testing.assert_close(rollout.grad, torch.tensor(.01 * rollout_weight))


def test_analytic_force_gradients_and_mass_cancellation():
    model = make_model(False)
    n = 3
    context = model.decode_context(model.context_encoder(torch.randn(n, 162)))
    initial = torch.zeros(n, 51)
    initial[:, 6] = 1
    state = torch.zeros(n, 180)
    state[:, 6] = 1
    state[:, 26:51] = .1
    state[:, 76:88] = 1
    nominal = torch.randn(n, 19, requires_grad=True)
    torque = torch.randn(n, 19, requires_grad=True)
    fixed = SimpleNamespace(mass_matrix=torch.eye(25).repeat(n, 1, 1), bias=torch.zeros(n, 25),
        foot_jacobians=torch.ones(n, 4, 3, 25), ee_jacobian=torch.randn(n, 6, 25),
        base_jacobian=torch.randn(n, 6, 25))
    batch = dict(rollout_initial_state=initial, dynamics_state=state,
        nominal_torque=nominal, interval_torque=torque, mass_wrench=torch.zeros(n, 6),
        dones=torch.zeros(n, 1), physics_invalid=torch.zeros(n, 1))
    cfg = dict(grf_scale=1., ee_force_scale=1., base_wrench_scale=[1.]*6, dt=.02)
    measured = {}
    original = losses(model, context, batch, fixed, cfg, metrics=measured)
    assert measured["valid_samples"] == n
    assert "inverse/base_linear_mae_N" in measured
    assert "rollout/arm_gripper_mae_radps" in measured
    assert all(not value.requires_grad for key, value in measured.items() if key != "valid_samples")
    mass = torch.randn(n, 6)
    shifted = losses(model, {**context, "base_wrench": context["base_wrench"] + mass},
                     {**batch, "mass_wrench": mass}, fixed, cfg)
    for a, b in zip(original, shifted):
        torch.testing.assert_close(a, b)
    sum(original).backward()
    assert nominal.grad is None and torque.grad is None
    for module in (model.context_encoder, model.physics_decoder.force, model.physics_decoder.grf):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
    assert all(p.grad is None for p in model.explicit_decoder.parameters())


def test_disjoint_ownership():
    model = make_model(False)
    decoder = torch.nn.Linear(8, 188)
    actor, context = model.get_optim_groups()
    a = SimpleNamespace(actor_critic=model, privileged_decoder=decoder,
                        learning_rate=.001, cfg={})
    configure_optimizers(a, actor, [*context, {"params": list(decoder.parameters())}])
    owners = [set(map(id, values)) for values in
              (a.ppo_parameters, a.enc_parameters, a.decoder_parameters)]
    assert not (owners[0] & owners[1] or owners[1] & owners[2] or owners[0] & owners[2])
    assert set(map(id, model.context_encoder.parameters())) == owners[1]
    assert set(map(id, model.explicit_decoder.parameters())) <= owners[2]
    assert set(map(id, model.physics_decoder.parameters())) <= owners[2]
    # Save/restore all three independent moment sets under the new partition.
    optimizers = (a.actor_optimizer.optimizer, a.auxiliary_optimizer, a.decoder_optimizer)
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    states = [optimizer.state_dict() for optimizer in optimizers]
    checkpoint = dict(zip(("actor_optimizer", "auxiliary_optimizer", "decoder_optimizer"), states))
    checkpoint["optimizer_partition_version"] = 2
    for optimizer in optimizers:
        optimizer.state.clear()
    restore_optimizers(a, checkpoint)
    for optimizer, state in zip(optimizers, states):
        assert len(optimizer.state) == len(state["state"])
        assert all(value["step"].item() == 1 for value in optimizer.state.values())
