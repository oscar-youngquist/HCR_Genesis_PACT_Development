"""Scalar, absolute-iteration warmup; no extra network or solver evaluations."""
from types import SimpleNamespace

import pytest
import torch

from rsl_rl.algorithms.ppo_hard_pact import PPO_HardPACT
from test_hard_pact_auxiliary import make_algorithm


def schedule(duration):
    alg = PPO_HardPACT.__new__(PPO_HardPACT)
    alg.qp_config = SimpleNamespace(warmup_iterations=200)
    alg.actor_physics_warmup_iterations = duration
    return alg


@pytest.mark.parametrize('iteration,expected', [(0,0.), (200,0.), (700,.5), (1200,1.), (1500,1.)])
def test_endpoints_and_resume(iteration, expected):
    alg = schedule(1000)
    value = alg._actor_physics_weight_at_iteration(iteration)
    assert value == pytest.approx(expected)
    # No counter/history to restore: absolute iteration plus config suffices.
    assert value == schedule(1000)._actor_physics_weight_at_iteration(iteration)
    assert schedule(0)._actor_physics_weight_at_iteration(iteration) == 1.


@pytest.mark.parametrize('factor', [0., .5, 1.])
def test_only_combined_actor_secondary_gradient_is_scaled(factor):
    alg = schedule(1000)
    alg.actor_physics_warmup_factor = factor
    p = torch.tensor(2., requires_grad=True)
    primary = p.square()
    secondary = .25*p.square() + .1*p**3  # existing individually weighted terms
    losses = [primary, secondary]
    scaled = alg._warmup_actor_losses(losses)
    assert scaled[0] is primary
    if factor == 1.: assert scaled is losses
    assert alg._warmup_actor_losses([primary])[0] is primary
    grad = torch.autograd.grad(scaled[1], p, retain_graph=True)[0]
    expected = torch.autograd.grad(secondary, p)[0]*factor
    torch.testing.assert_close(grad, expected)


@pytest.mark.parametrize('duration', [-1, .5, float('nan'), float('inf')])
def test_invalid_duration(duration):
    with pytest.raises(ValueError, match='actor_physics_warmup_iterations'):
        make_algorithm(actor_physics_warmup_iterations=duration)


def test_physical_errors_count_weighted_without_loss_weights():
    alg=PPO_HardPACT.__new__(PPO_HardPACT)
    alg._actor_loss_sums={}
    alg._record_actor_loss('velocity_xy/error_m_s',torch.tensor(2.,requires_grad=True),1)
    alg._record_actor_loss('velocity_xy/error_m_s',torch.tensor(4.),3)
    alg._record_actor_loss('velocity_xy/error_m_s',torch.tensor(float('nan')),0)
    alg._record_actor_regularizer('fk',{'loss':torch.tensor(.5),'error_m':torch.tensor(.02)},4,8)
    alg._record_actor_regularizer('load',{'loss':torch.tensor(.5),'shortfall_nm':torch.tensor(3.)},2,8)
    alg._finish_actor_loss_logging(torch.tensor(0.))
    assert alg.last_actor_loss_metrics['Loss/actor_pinn/velocity_xy/error_m_s']==3.5
    assert alg.last_actor_loss_metrics['Loss/actor_pinn/fk/error_m']==pytest.approx(.02)
    assert alg.last_actor_loss_metrics['Loss/actor_pinn/load/shortfall_nm']==3.
    assert all(not v.requires_grad for v in alg.last_actor_loss_metrics.values())
