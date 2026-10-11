"""Applied-only wrench semantics and deterministic B1Z1 deployment."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from test_b1z1_sampled_context import make_model
from test_b1z1_actor_physics import setup
from test_b1z1_pact_privileged_force_order import B1Z1PACTPrivilegedForceOrderTests
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact import B1Z1PACT
from legged_gym.envs.b1z1.b1z1_pact_pos.b1z1_pact_pos import B1Z1PACTPos
from legged_gym.envs.b1z1.b1z1_ppo_pos.b1z1_ppo_pos import B1Z1PPOPos
from legged_gym.envs.b1z1.force_task_utils import observe_pact_rejection_quality
from rsl_rl.b1z1_pact_ablations import B1Z1_PACT_ABLATIONS
from rsl_rl.algorithms import b1z1_actor_physics as physics


@pytest.mark.parametrize("variant", ["legacy_pos", *B1Z1_PACT_ABLATIONS])
def test_deterministic_inference_and_stochastic_training(variant, monkeypatch):
    model = make_model(variant == "legacy_pos")
    if variant != "legacy_pos":
        features = B1Z1_PACT_ABLATIONS[variant]
        # Construct the shared model with each factory's action/conditioning mode.
        model = type(model)(81, 40, 17, 162, latent_dim=8,
            actor_layers=[32, 16], critic_layers=[32, 16], context_layers=[32, 16],
            explicit_decoder_layers=[16], force_decoder_layers=[16], grf_decoder_layers=[16],
            action_mode=features.action_mode, conditioning_mode=features.conditioning_mode)
    obs, history = torch.randn(3, 81), torch.randn(3, 162)
    rng = torch.random.get_rng_state()
    context = model.context_encoder.forward_inf(history)
    torch.testing.assert_close(context["z"], context["mean"], rtol=0, atol=0)
    one = model.act_inference(obs, history)
    torch.testing.assert_close(one, model.act_inference(obs, history), rtol=0, atol=0)
    assert torch.equal(rng, torch.random.get_rng_state())
    position, torque = model.actor_forward(obs, model.decode_context(context))
    expected = position if variant != "legacy_pos" and features.action_mode == "position" else torch.cat((position, torque), -1)
    torch.testing.assert_close(one, expected, rtol=0, atol=0)
    monkeypatch.setattr(model, "act_inference", Mock(side_effect=AssertionError("rollout used inference")))
    monkeypatch.setattr(model.context_encoder, "forward_inf", Mock(side_effect=AssertionError("rollout used inference")))
    action = model.act(obs, history)
    noise, mean = model.last_context["latent_noise"].clone(), model.action_mean.clone()
    rng = torch.random.get_rng_state()
    model.update_distribution(obs, history, latent_noise=noise)
    torch.testing.assert_close(mean, model.action_mean, rtol=0, atol=0)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert not torch.equal(action, model.act(obs, history))
    assert not torch.equal(noise, model.last_context["latent_noise"])
    assert not torch.equal(model.context_encoder(history)["z"], model.context_encoder(history)["z"])


@pytest.mark.parametrize("cls", [B1Z1PACT, B1Z1PACTPos, B1Z1PPOPos])
@pytest.mark.parametrize("active", [False, True])
def test_applied_only_privileged_labels_and_force_quality(cls, active):
    env = B1Z1PACTPrivilegedForceOrderTests._make_env(cls)
    env.simulator._added_base_mass = torch.tensor([[5.]])
    env.simulator._base_com_bias = torch.tensor([[.1, .2, .3]])
    env.simulator._added_gripper_mass = torch.tensor([[2.]])
    env.bard_mass_wrench_labels = True  # Even an obsolete caller flag cannot alter labels.
    env.get_mass_wrench_label = Mock(side_effect=AssertionError("obsolete label read"))
    if not active:
        env.base_force_ext_world.zero_()
        env.base_torque_ext_world.zero_()
        env.ee_force_ext_world.zero_()
    env.base_wrench_scale = torch.tensor([.01]*3 + [.02]*3)
    target = env.get_privileged_force_observation()
    applied = torch.cat((env.base_force_ext_world, env.base_torque_ext_world), -1)
    torch.testing.assert_close(target[:, 12:18], applied * env.base_wrench_scale)
    env.device = "cpu"
    env.obs_scales.base_force = .01
    env._staged_force_curriculum = SimpleNamespace(observe_forces=Mock())
    observe_pact_rejection_quality(env, {"base_wrench": target[:, 12:18], "ee_force": target[:, 18:]})
    args = env._staged_force_curriculum.observe_forces.call_args.args
    torch.testing.assert_close(args[1], env.base_force_ext_world)
    env.get_mass_wrench_label.assert_not_called()


def test_actor_rollout_uses_applied_wrench_once(monkeypatch):
    a, batch, actions, context = setup()
    batch["actor_phys_mass_wrench"].fill_(100.)  # Old snapshots must not cancel applied force.
    applied = torch.tensor([[2., -3., 4., 5., 6., 7.]]).expand(3, -1)
    context["base_wrench"] = applied * .01
    captured = []
    rollout = physics.differentiable_bard_rollout_loss
    def spy(**kwargs):
        captured.append(kwargs["applied_wrench_world"].detach().clone())
        return rollout(**kwargs)
    monkeypatch.setattr(physics, "differentiable_bard_rollout_loss", spy)
    loss, _ = physics.objective(a, batch, actions, context)
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], applied)
    loss.backward()
    assert any(p.grad is not None for p in a.actor_critic.position_head.parameters())
    assert all(p.grad is None for p in a.actor_critic.physics_decoder.parameters())


@pytest.mark.parametrize("active", [False, True])
def test_explicit_labels_exclude_randomized_gravity(active, monkeypatch):
    from importlib import import_module
    module = import_module(B1Z1PACT.__module__)
    class LabelsComplete(Exception):
        pass
    class LabelProbe(B1Z1PACT):
        @property
        def _mass_params_buf(self):
            # Stop after real observation/explicit construction, before critic packing.
            raise LabelsComplete
    env = B1Z1PACTPrivilegedForceOrderTests._make_env(LabelProbe)
    zero = lambda n: torch.zeros(1, n)
    env.simulator.__dict__.update(ee_pos=torch.tensor([[.5, 0., 0.]]),
        dof_pos=zero(19), default_dof_pos=zero(19), dof_vel=zero(19),
        base_ang_vel=zero(3), base_lin_vel=zero(3), foot_contacts=zero(4),
        feet_pos=torch.zeros(1, 4, 3), height_around_feet=torch.zeros(1, 4, 1),
        _added_base_mass=torch.tensor([[5.]]), _base_com_bias=torch.ones(1, 3),
        _added_gripper_mass=torch.tensor([[2.]]))
    if not active:
        env.base_force_ext_world.zero_()
        env.base_torque_ext_world.zero_()
    env.cfg = SimpleNamespace(env=SimpleNamespace(num_observations=81, num_explicit_recon_obs=23),
                              rewards=SimpleNamespace(foot_height_offset=0.))
    env.obs_scales.__dict__.update(dof_pos=1., dof_vel=1., ang_vel=1.)
    env.obs_buf, env.last_obs_buf, env.llast_obs_buf = zero(81), zero(81), zero(81)
    env.actions, env.commands, env.commands_scale = zero(34), zero(6), 1.
    env.base_velocity_scale = env.ee_sphere_scale = 1.
    env.add_noise = False
    env.get_ee_goal_spherical_center = lambda yaw: zero(3)
    env._get_phase = lambda: torch.zeros(1)
    env.compute_ref_state = lambda: None
    env.get_body_orientation = lambda: zero(2)
    monkeypatch.setattr(module, "get_force_adjusted_ee_target",
                        lambda env: SimpleNamespace(effective_target=env.simulator.ee_pos))
    with pytest.raises(LabelsComplete):
        env.compute_observations()
    expected = torch.cat((env.base_force_ext_world, env.base_torque_ext_world), -1)
    torch.testing.assert_close(env.explicit_labels_buf[:, 6:12], expected * env.base_wrench_scale)
