from pathlib import Path

import pytest
import torch
from torch import nn

from rsl_rl.algorithms.ppo_pact import PPO_PACT
from rsl_rl.algorithms.ppo_pact_pos import PPO_PACT_Pos
from rsl_rl.storage.rollout_storage_pact import RolloutStoragePACT


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    "legged_gym/envs/go1/go1_pact/go1_pact_config.py",
    "legged_gym/envs/go1/go1_pact_pos/go1_pact_pos_config.py",
    "legged_gym/envs/go2/go2_pact/go2_pact_config.py",
    "legged_gym/envs/go2/go2_pact_pos/go2_pact_pos_config.py",
)


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, history):
        batch = history.shape[0]
        latent = history[:, :2] * self.scale
        explicit = history[:, 2:3] * self.scale
        return torch.zeros(batch, 2), torch.zeros(batch, 2), latent, explicit


class _Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_encoder = _Encoder()


@pytest.mark.parametrize("algorithm_type", [PPO_PACT, PPO_PACT_Pos])
def test_privileged_target_excludes_only_grfs(algorithm_type):
    algorithm = algorithm_type.__new__(algorithm_type)
    algorithm.privileged_grf_start_index = 61
    target = torch.arange(2 * 288, dtype=torch.float32).reshape(2, 288)

    actual = algorithm._privileged_decode_target(target, 12)
    expected = torch.cat((target[:, :61], target[:, 73:]), dim=-1)

    assert actual.shape == (2, 276)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("algorithm_type", [PPO_PACT, PPO_PACT_Pos])
def test_auxiliary_loss_trains_separate_privileged_and_grf_decoders(algorithm_type):
    algorithm = algorithm_type.__new__(algorithm_type)
    algorithm.actor_critic = _Actor()
    algorithm.decoder = nn.Linear(3, 276)
    algorithm.grf_decoder = nn.Linear(15, 12)
    algorithm.privileged_grf_start_index = 61
    algorithm.grf_reconstruction_loss_weight = 1.0
    algorithm.vae_beta = 0.1
    algorithm.dof_tau_observation_scale = 0.01
    history = torch.randn(5, 3)
    privileged = torch.randn(5, 288)
    grfs = privileged[:, 61:73].clone()
    explicit = torch.randn(5, 1)
    mask = torch.ones(5, 1)
    nominal_torque = torch.randn(5, 12)

    result = algorithm._compute_vae_loss(
        history, grfs, privileged, explicit, mask, nominal_torque
    )
    loss, _, _, grf_loss, _, _, privileged_target, _ = result
    assert privileged_target.shape == (5, 276)
    assert grf_loss.isfinite()
    loss.backward()
    assert algorithm.decoder.weight.grad.abs().sum() > 0
    assert algorithm.grf_decoder.weight.grad.abs().sum() > 0
    assert algorithm.actor_critic.context_encoder.scale.grad.abs() > 0


def test_pinn_grf_gate_uses_decoder_below_threshold_and_simulator_above():
    algorithm = PPO_PACT.__new__(PPO_PACT)
    algorithm.actor_critic = _Actor()
    algorithm.grf_decoder = nn.Linear(15, 12, bias=False)
    algorithm.grf_observation_scale = 0.01
    algorithm.dof_tau_observation_scale = 0.01
    algorithm.pinn_grf_reconstruction_mse_threshold = 1e-8
    algorithm.last_pinn_grf_reconstruction_mse = float("nan")
    algorithm.last_pinn_grf_replacement_fraction = 0.0

    history = torch.randn(4, 3)
    nominal_torque = torch.randn(4, 12)
    with torch.no_grad():
        _, _, latent, explicit = algorithm.actor_critic.context_encoder(history)
        decoder_input = algorithm._grf_decoder_input(
            torch.cat((latent, explicit), dim=-1), nominal_torque
        )
        target = algorithm.grf_decoder(decoder_input)
    jacobian = torch.randn(4, 18, 12)
    simulator_force = torch.randn(4, 18)
    mask = torch.ones(4, 1)

    selected, mse, used = algorithm._select_pinn_contact_forces(
        history, target, mask, nominal_torque, jacobian, simulator_force
    )
    expected = torch.einsum("bnk,bk->bn", jacobian, target / 0.01)
    assert used.item()
    assert mse.item() == pytest.approx(0.0)
    torch.testing.assert_close(selected, expected)
    selected.square().mean().backward()
    assert algorithm.grf_decoder.weight.grad is not None
    assert algorithm.grf_decoder.weight.grad.abs().sum() > 0
    assert algorithm.actor_critic.context_encoder.scale.grad is not None

    algorithm.pinn_grf_reconstruction_mse_threshold = 0.0
    selected, _, used = algorithm._select_pinn_contact_forces(
        history, target, mask, nominal_torque, jacobian, simulator_force
    )
    assert not used.item()
    torch.testing.assert_close(selected, simulator_force)


@pytest.mark.parametrize("relative_path", CONFIGS)
def test_all_go1_go2_pact_configs_define_split_decoders(relative_path):
    source = (ROOT / relative_path).read_text()
    assert "cenet_dec_out_dim = 57 + (50 + 38) + 143 - 12" in source
    assert "privileged_grf_start_index = 61" in source
    assert "separate_grf_decoder = True" in source
    assert "grf_dec_input_dim = cenet_dec_input_dim + 12" in source
    assert "grf_dec_out_dim = 12" in source
    assert "dof_tau = 0.01" in source


@pytest.mark.parametrize("algorithm_type", [PPO_PACT, PPO_PACT_Pos])
def test_grf_decoder_torque_condition_is_detached(algorithm_type):
    context = torch.randn(4, 3, requires_grad=True)
    nominal_torque = torch.randn(4, 12, requires_grad=True)
    decoder = nn.Linear(15, 12, bias=False)

    decoder_input = algorithm_type._grf_decoder_input(
        context, nominal_torque, dof_tau_observation_scale=0.01
    )
    torch.testing.assert_close(decoder_input[:, :3], context)
    torch.testing.assert_close(decoder_input[:, 3:], nominal_torque.detach() * 0.01)
    prediction = decoder(decoder_input)
    prediction.square().mean().backward()

    assert context.grad is not None and context.grad.abs().sum() > 0
    assert decoder.weight.grad is not None and decoder.weight.grad.abs().sum() > 0
    assert nominal_torque.grad is None


@pytest.mark.parametrize("algorithm_type", [PPO_PACT, PPO_PACT_Pos])
@pytest.mark.parametrize("action_dim", [12, 24])
def test_nominal_torque_condition_uses_physical_feedforward_plus_pd(
    algorithm_type, action_dim
):
    actions = torch.randn(3, action_dim, requires_grad=True)
    observations = torch.randn(3, 57, requires_grad=True)
    default_pose = torch.linspace(-0.2, 0.2, 12)

    def action_func(value):
        return value[:, :12] + default_pose, 2.0 * value[:, 12:24]

    def feedback_func(q_des, q_pos, q_vel):
        return 3.0 * (q_des - q_pos) - 0.5 * q_vel

    actual = algorithm_type._nominal_torque_from_action(
        actions, observations, action_func, feedback_func,
        default_pose, 0.25,
    )
    transformed_actions = actions.detach()
    if action_dim == 12:
        transformed_actions = torch.cat(
            (transformed_actions, torch.zeros_like(transformed_actions)), dim=-1
        )
    q_des, tau_ff = action_func(transformed_actions)
    q_pos = observations.detach()[:, 9:21] + default_pose
    q_vel = observations.detach()[:, 21:33] / 0.25
    expected = tau_ff + feedback_func(q_des, q_pos, q_vel)

    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


def test_contact_jacobian_maps_canonical_grfs_to_generalized_force():
    batch, nv = 3, 18
    blocks = [torch.randn(batch, nv, 3) for _ in range(4)]
    canonical_map = torch.cat(blocks, dim=-1)
    canonical_grf = torch.randn(batch, 4, 3)
    mapped = torch.einsum("bnk,bk->bn", canonical_map, canonical_grf.flatten(1))
    direct = sum(
        torch.einsum("bnk,bk->bn", block, canonical_grf[:, foot])
        for foot, block in enumerate(blocks)
    )
    torch.testing.assert_close(mapped, direct)


def test_pact_storage_preserves_contact_map_for_pinn_reconstruction():
    storage = RolloutStoragePACT(
        2, 1, [57], [288], [288], [1140], [24], [15], [12], [18], "cpu",
        store_contact_jacobian=True,
    )
    transition = RolloutStoragePACT.Transition()
    transition.observations = torch.zeros(2, 57)
    transition.critic_observations = torch.zeros(2, 288)
    transition.observation_history = torch.zeros(2, 1140)
    transition.dones = torch.zeros(2, dtype=torch.uint8)
    transition.explicit_labels = torch.zeros(2, 15)
    transition.grf_targets = torch.zeros(2, 12)
    transition.obs_targets = torch.zeros(2, 288)
    transition.actions = torch.zeros(2, 24)
    transition.rewards = torch.zeros(2)
    transition.values = torch.zeros(2, 1)
    transition.actions_log_prob = torch.zeros(2)
    transition.action_mean = torch.zeros(2, 24)
    transition.action_sigma = torch.ones(2, 24)
    transition.prev_obs = torch.zeros(2, 57)
    transition.prev_obs_hist = torch.zeros(2, 1140)
    transition.pprev_obs = torch.zeros(2, 57)
    transition.pprev_obs_hist = torch.zeros(2, 1140)
    transition.wb_contact_forces = torch.zeros(2, 18)
    transition.wb_contact_jacobian = torch.randn(1, 18, 12).expand(2, -1, -1).clone()
    transition.wb_mass_mat = torch.eye(18).expand(2, -1, -1)
    transition.wb_bias_vec = torch.zeros(2, 18)
    transition.torso_acc = torch.zeros(2, 6)

    storage.add_transitions(transition)
    batch = next(storage.mini_batch_generator(1, 1))
    torch.testing.assert_close(batch[17], transition.wb_contact_jacobian)


def test_legacy_pact_path_keeps_full_decoder_target_and_skips_map_storage():
    algorithm = PPO_PACT.__new__(PPO_PACT)
    algorithm.actor_critic = _Actor()
    algorithm.decoder = nn.Linear(3, 288)
    algorithm.grf_decoder = None
    algorithm.privileged_grf_start_index = 61
    algorithm.grf_reconstruction_loss_weight = 1.0
    algorithm.vae_beta = 0.0
    result = algorithm._compute_vae_loss(
        torch.randn(2, 3), torch.randn(2, 12), torch.randn(2, 288),
        torch.randn(2, 1), torch.ones(2, 1),
    )
    assert result[6].shape == (2, 288)
    assert result[3].item() == 0.0

    storage = RolloutStoragePACT(
        1, 1, [57], [288], [288], [1140], [24], [15], [12], [18], "cpu"
    )
    assert storage.wb_contact_jacobians is None
