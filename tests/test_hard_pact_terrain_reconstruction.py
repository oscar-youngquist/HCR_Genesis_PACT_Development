"""Terrain targets retain critic scaling and train both privileged decoders."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from test_hard_pact_auxiliary import make_algorithm
from test_hard_pact_pos_auxiliary import _small_algorithm
from legged_gym.envs.go2.go2_hard_pact.deployment import (
    RECONSTRUCTION_DIM, RECONSTRUCTION_INDICES,
)
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import GO2HardPACTCfgPPO
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos_config import GO2HardPACTPosCfgPPO
from rsl_rl.modules.actor_critic_hard_pact import ContextDecoder


@pytest.mark.parametrize("pos", [False, True])
def test_actual_transition_selection_preserves_scaled_terrain(pos):
    alg = (_small_algorithm if pos else make_algorithm)()
    alg.num_priv_obs = 300  # Legacy frame plus appended disturbance fields.
    alg.reconstruction_indices = RECONSTRUCTION_INDICES
    captured = []
    alg.storage = SimpleNamespace(add_transitions=lambda tr: captured.append(tr.obs_targets.clone()))
    alg.actor_critic.reset = Mock()
    frames = torch.arange(600, dtype=torch.float32).repeat(2, 1) / 100
    original = frames.clone()
    fields = {
        "interval_executed_torque": torch.zeros(2, 12),
        "total_external_wrench_label_yaw_normalized": torch.zeros(2, 6),
        "sustained_wrench_active_mask": torch.zeros(2, 1, dtype=torch.bool),
    }
    args = (torch.zeros(2), torch.zeros(2, dtype=torch.bool),
            {"hard_pact_transition": fields}, torch.zeros(2, 12), frames, torch.zeros(2, 11))
    alg.process_env_step(*args, *((None,) * 4 if not pos else ()))
    target = captured[0]
    assert target.shape == (2, 276)
    torch.testing.assert_close(target[:, :61], frames[:, 300:361])
    torch.testing.assert_close(target[:, 61:133], frames[:, 373:445])
    torch.testing.assert_close(target[:, 133:], frames[:, 445:588])
    torch.testing.assert_close(frames, original)
    cfg = GO2HardPACTPosCfgPPO if pos else GO2HardPACTCfgPPO
    assert cfg.policy.cenet_dec_out_dim == RECONSTRUCTION_DIM == 276


@pytest.mark.parametrize("pos", [False, True])
def test_terrain_contributes_to_actual_auxiliary_objective(pos):
    torch.manual_seed(18)
    alg = (_small_algorithm if pos else make_algorithm)()
    alg.decoder = ContextDecoder(input_dim=27, layers=[32, 24, 16], decode_dim=RECONSTRUCTION_DIM)
    outputs = []
    hook = alg.decoder.register_forward_hook(lambda module, args, output: outputs.append(output))
    target = torch.zeros(2, RECONSTRUCTION_DIM)
    target[:, 133:] = 0.75
    valid = torch.ones(2, 1, dtype=torch.bool)
    history = torch.randn(2, 570 if pos else 1140)
    explicit, grf, torque = torch.zeros(2, 11), torch.zeros(2, 12), torch.zeros(2, 12)
    if pos:
        result = alg._compute_vae_loss(history, grf, target, explicit, valid,
                                      executed_torque_target=torque,
                                      wrench_target=torch.zeros(2, 6),
                                      wrench_active_mask=torch.zeros_like(valid))
        loss = result[-1]["privileged_reconstruction"]
    else:
        result = alg._compute_auxiliary_loss(history, target, explicit, grf, valid, torque, None)
        loss = result["privileged"]
    hook.remove()
    reconstruction = outputs[0]
    torch.testing.assert_close(loss, (reconstruction - target).square().mean())
    reconstruction.retain_grad()
    loss.backward()
    torch.testing.assert_close(reconstruction.grad[:, 133:],
                               2 * (reconstruction.detach()[:, 133:] - .75) / target.numel())
    assert reconstruction.grad[:, 133:].abs().sum() > 0
    grad = alg.actor_critic.context_encoder.ce_out_mean.weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
