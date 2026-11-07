"""B1Z1 leg exploration profiles without changing legacy PACT checkpoints."""
import pytest
import torch

from legged_gym.envs.go2.go2_hard_pact.ablation_configs import make_hard_pact_variant_configs
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import GO2HardPACTCfg, GO2HardPACTCfgPPO
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos_config import GO2HardPACTPosCfgPPO
from legged_gym.envs.go2.go2_hard_pact.deployment import calculate_physics_head_gains
from rsl_rl.hard_pact_ablations import HARD_PACT_VARIANTS
from rsl_rl.modules import ActorCritic_HardPACT, ActorCritic_HardPACT_Pos
from rsl_rl.modules.hard_pact_action_std import initialize_action_std


@pytest.mark.parametrize('pos', [False, True])
def test_policy_profile_bounds_gradient_and_checkpoint_compatibility(pos):
    cfg = (GO2HardPACTPosCfgPPO if pos else GO2HardPACTCfgPPO).policy
    cls = ActorCritic_HardPACT_Pos if pos else ActorCritic_HardPACT
    gains = calculate_physics_head_gains(GO2HardPACTCfg())
    actor = cls(actor_layers=[16,16], critic_layers=[16,16], cenet_enc_layers=[16,16],
                cenet_in_dim=1140, init_noise_std=cfg.init_noise_std,
                min_noise_std=cfg.min_noise_std, max_noise_std=cfg.max_noise_std,
                grf_scale_n=gains.grf_scale_n, wrench_scale=gains.wrench_scale_n_nm,
                wrench_qp_clip=gains.wrench_qp_clip_n_nm)
    torch.testing.assert_close(actor.std, torch.tensor([.8, 1., 1.]* (4 if pos else 8)))
    assert not any('_std_clip' in k for k in actor.state_dict())
    saved = {k: v.clone() for k, v in actor.state_dict().items()}
    with torch.no_grad():
        actor.std[::2] = -1
        actor.std[1::2] = 10
    actor._clip_std()
    torch.testing.assert_close(actor.std[::2], torch.tensor(cfg.min_noise_std)[::2])
    torch.testing.assert_close(actor.std[1::2], torch.full_like(actor.std[1::2], 1.1))
    actor.std.sum().backward()
    assert actor.std.grad.eq(1).all()
    actor.load_state_dict(saved, strict=True)
    torch.testing.assert_close(actor.std, saved['std'])
    actor.to(dtype=torch.float64)
    assert actor._std_clip_lwr.dtype == actor.std.dtype == actor._std_clip_upr.dtype
    actor._set_std_clip_lwr(.3)
    assert actor._std_clip_lwr.eq(torch.tensor(.3, dtype=torch.float32).double()).all()


@pytest.mark.parametrize('variant', HARD_PACT_VARIANTS)
def test_all_ablations_inherit_profile(variant):
    _, cfg = make_hard_pact_variant_configs(variant, 'isaaclab')
    assert cfg.policy.init_noise_std == [.8, 1., 1.] * 8
    assert cfg.policy.min_noise_std == [.15, .25, .25] * 8
    assert cfg.policy.max_noise_std == 1.1


@pytest.mark.parametrize('initial,lower,upper', [([1,2], .1, 1.1),
    (1., 2., 1.1), (float('nan'), .1, 1.1), (1., 0., 1.1)])
def test_bad_profiles_rejected(initial, lower, upper):
    with pytest.raises(ValueError):
        initialize_action_std(torch.nn.Module(), 24, initial, lower, upper)
