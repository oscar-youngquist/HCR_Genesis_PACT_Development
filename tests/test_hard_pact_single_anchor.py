"""Single-anchor schedule/deployment and real certified solver regression."""
from dataclasses import replace
from unittest import mock

import pytest
import torch

from rsl_rl.algorithms.hard_pact_qp import (
    HardPACTQPConfig, held_correction_torque, qp_substep_anchors,
)
from legged_gym.envs.go2.go2_hard_pact.deployment import qp_update_contract
from test_hard_pact_reduced_qp import solver as make_qp, inputs as qp_data


def test_contract_shared_execution_and_default():
    assert HardPACTQPConfig().qp_update_mode == "every_substep"
    for mode, anchors in (("every_substep", [0, 1, 2, 3]),
                          ("two_anchor_held_correction", [0, 2]),
                          ("single_anchor_held_correction", [0])):
        contract = qp_update_contract(mode, 4, 2000)
        assert contract["physics_substep_anchors"] == anchors
        assert contract["training_warmup_iterations"] == 2000
        assert not contract["held_commands_are_freshly_qp_certified"]
        assert contract["ppo_projection_loss_multiplier"] == 1
    assert qp_substep_anchors("single_anchor_held_correction", 1) == (0,)
    with pytest.raises(ValueError):
        qp_substep_anchors("two_anchor_held_correction", 3)
    with pytest.raises(ValueError):
        qp_substep_anchors("invalid", 4)
    nominal = torch.tensor([[0.2, 3., -3., float("nan")]])
    previous = torch.tensor([[0., 0.95, -0.95, 0.]])
    result = held_correction_torque(
        nominal, torch.zeros_like(nominal), previous, torch.ones(4), 10., .01,
        sanitize=True,
    )
    torch.testing.assert_close(result, torch.tensor([[.1, 1., -1., 0.]]))


def test_anchor_solver_primal_gradient_and_fallback_parity():
    data = qp_data(2)
    data["tau_nom"].fill_(0.2)
    data["tau_nom"].requires_grad_()
    reference = make_qp()
    anchor = make_qp(qp_update_mode="single_anchor_held_correction")
    old = reference.solve(**data, differentiable=True)
    new = anchor.solve(**data, differentiable=True)
    for name in ("tau_safe", "force_world", "qdd", "stage", "differentiated_mask"):
        torch.testing.assert_close(getattr(old, name), getattr(new, name), rtol=0, atol=0)
    grad_old, = torch.autograd.grad(old.tau_safe.sum(), data["tau_nom"], retain_graph=True)
    grad_new, = torch.autograd.grad(new.tau_safe.sum(), data["tau_nom"])
    torch.testing.assert_close(grad_old, grad_new, rtol=0, atol=0)
    assert torch.isfinite(grad_new).all() and grad_new.abs().sum() > 0
    # Recovery stays inside the one anchor invocation. Analytic rows cannot
    # contribute solver gradients, even though their held commands are safe.
    anchor.cfg = replace(anchor.cfg, exception_capture_enabled=False)
    with mock.patch("rsl_rl.algorithms.hard_pact_qp.QPFunction", side_effect=RuntimeError("forced")):
        fallback = anchor.solve(**data, differentiable=True)
    assert (fallback.stage == 2).all()
    assert not fallback.differentiated_mask.any()
    previous = fallback.tau_safe
    correction = previous - data["tau_nom"].detach()
    for _ in range(3):
        executed = held_correction_torque(
            data["tau_nom"].detach() + .1, correction, previous,
            anchor.torque_limits, anchor.cfg.torque_rate_limit_nm_s, .02,
            sanitize=True,
        )
        assert torch.isfinite(executed).all()
        assert (executed.abs() <= anchor.torque_limits).all()
        assert ((executed - previous).abs() <= anchor.cfg.torque_rate_limit_nm_s * .02 + 1e-12).all()
        previous = executed
