"""Cosine KL endpoints, disabled behavior, resume and actual loss weighting."""
import pytest
import torch
from test_hard_pact_auxiliary import make_algorithm,make_batch
from rsl_rl.algorithms.vae_kl_schedule import cosine_vae_beta


def test_cosine_endpoints_monotonicity_resume_and_disabled():
    kwargs=dict(vae_kl_initial_weight=.2,vae_kld_weight=5.,
                vae_kl_warmup_start=100,vae_kl_warmup_iterations=1000)
    alg=make_algorithm(**kwargs)
    for iteration,expected in ((0,.2),(100,.2),(600,2.6),(1100,5.),(5000,5.)):
        assert alg._vae_beta_for_iteration(iteration)==pytest.approx(expected)
    values=[alg._vae_beta_for_iteration(i) for i in range(100,1101,10)]
    assert values==sorted(values)
    resumed=make_algorithm(**kwargs)
    assert resumed._vae_beta_for_iteration(789)==alg._vae_beta_for_iteration(789)
    assert cosine_vae_beta(50,.2,5.,100,0)==5.
    with pytest.raises(ValueError,match='nonnegative'):
        make_algorithm(vae_kl_warmup_iterations=-1)


def test_only_kl_weight_changes_raw_loss_and_ppo_settings_stay_unchanged():
    alg=make_algorithm(vae_kl_initial_weight=.2,vae_kld_weight=1.,vae_kl_warmup_iterations=10)
    args=make_batch()
    desired_kl,rate=alg.desired_kl,alg.learning_rate
    outputs=[]
    for iteration in (0,5,10):
        alg.current_vae_beta=alg._vae_beta_for_iteration(iteration)
        torch.manual_seed(123)
        outputs.append(alg._compute_auxiliary_loss(*args))
    for out in outputs[1:]:
        torch.testing.assert_close(out['kl'],outputs[0]['kl'],rtol=0,atol=0)
    torch.testing.assert_close(outputs[2]['loss']-outputs[0]['loss'],.8*outputs[0]['kl'])
    assert (alg.desired_kl,alg.learning_rate)==(desired_kl,rate)
