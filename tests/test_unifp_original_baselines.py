"""Small CPU checks only; no simulator initialization or training job."""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
import legged_gym.envs  # Establish the repository's task/runner import ordering.
from legged_gym.utils import task_registry
from legged_gym.envs.b1z1.b1z1_unifp_original.b1z1_unifp_original_config import B1Z1UniFPOriginalCfg
from legged_gym.envs.b1z1.b1z1_unifp_reject.b1z1_unifp_reject import B1Z1UniFPReject
from rsl_rl.modules.actor_critic_unifp_original import ActorCriticUniFPOriginal
from rsl_rl.algorithms.ppo_unifp_original import PPO_UniFPOriginal
from rsl_rl.runners.unifp_original_runner import OnPolicyRunnerUniFPOriginal


def model():
    torch.set_num_threads(1)
    return ActorCriticUniFPOriginal(2336, 30)


def widths(module):
    return [(layer.in_features, layer.out_features) for layer in module if isinstance(layer, nn.Linear)]


def test_architecture_and_decoder_independence():
    net = model()
    assert widths(net.adaptation_encoder_module) == [(2336,512),(512,256),(256,128),(128,64)]
    assert widths(net.adaptation_decoder_module) == [(64,128),(128,64),(64,12)]
    assert widths(net.actor_body) == [(137,512),(512,256),(256,128),(128,17)]
    assert widths(net.critic_body) == [(30,512),(512,256),(256,128),(128,1)]
    assert torch.equal(net.std, torch.ones(17))
    h = torch.randn(2,2336)
    before = net.act_inference(h).detach().clone()
    with torch.no_grad():
        net.adaptation_decoder_module[-1].bias.add_(100)
    assert torch.equal(before, net.act_inference(h))
    assert net.last_prediction.shape == (2,12)
    assert not any(word in key for key in net.state_dict() for word in ("logvar", "privileged", "mean_head"))
    with pytest.raises(RuntimeError, match="architecture/schema"):
        net.load_state_dict({})


def test_gradients_and_tiny_update():
    net = model()
    calls = []
    alg = PPO_UniFPOriginal(net, num_learning_epochs=1, num_mini_batches=1,
                           decision_callback=lambda x: calls.append(x.clone()))
    h, c, labels = torch.randn(2,2336), torch.randn(2,30), torch.randn(2,12)
    net.update_distribution(h)
    net.action_mean.square().mean().backward()
    assert net.actor_body[0].weight.grad.abs().sum() > 0
    assert net.adaptation_encoder_module[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in net.adaptation_decoder_module.parameters())
    net.zero_grad(set_to_none=True)
    loss, blocks = alg.adaptation_loss(h, labels)
    loss.backward()
    assert set(blocks) == set(net.schema)
    assert net.adaptation_encoder_module[0].weight.grad.abs().sum() > 0
    assert net.adaptation_decoder_module[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in net.actor_body.parameters())
    alg.init_storage(2,2,[2336],[30],[12],[0],[17])
    with torch.no_grad():
        for _ in range(2):
            alg.act(h,c,labels)
            alg.process_env_step(torch.ones(2),torch.zeros(2),{})
        alg.compute_returns(c)
    result = alg.update(0)
    assert all(torch.isfinite(torch.tensor(x)) for x in result[:3])
    assert len(calls) == 2
    assert not hasattr(alg, "kl_controller")


def rejection_env():
    env = B1Z1UniFPReject.__new__(B1Z1UniFPReject)
    env.num_envs, env.device, env.dt = 2, "cpu", .02
    env.cfg = B1Z1UniFPOriginalCfg()
    env.cfg.commands.ee_impedance_force_filter_tau = 0.
    env.cfg.commands.base_impedance_force_filter_tau = 0.
    env.obs_scales = SimpleNamespace(ee_force=.1, base_force=.02)
    for name in ("estimated_ee_force_local", "estimated_base_force_local",
                 "filtered_ee_force_local", "filtered_base_force_local",
                 "current_Fxyz_gripper_cmd", "current_Fxyz_base_cmd"):
        setattr(env,name,torch.zeros(2,3))
    env.commands = torch.zeros(2,15)
    return env


def test_rejection_schema_scaling_reset_and_filter():
    env = rejection_env()  # No simulator or true-force buffers exist on this fixture.
    pred = torch.zeros(2,12)
    pred[:,6:9], pred[:,9:12] = 2., 3.
    env.set_impedance_force_estimates(pred)
    env._apply_external_impedance_compensation()
    assert torch.allclose(env.commands[:,9:12], torch.full((2,3),-20.))
    assert torch.allclose(env.commands[:,12:15], torch.full((2,3),-150.))
    # The target's external + command force is zero for an exact settled estimate.
    assert torch.allclose(env.commands[:,9:12] + pred[:,6:9]/.1, torch.zeros(2,3))
    assert torch.allclose(env.commands[:,12:15] + pred[:,9:12]/.02, torch.zeros(2,3))
    env._reset_impedance_force_filters(torch.tensor([0]))
    assert env.estimated_ee_force_local[0].count_nonzero() == 0
    assert env.filtered_base_force_local[0].count_nonzero() == 0
    assert env.filtered_base_force_local[1].count_nonzero() == 3
    env.set_impedance_force_estimates(torch.zeros_like(pred))
    env._apply_external_impedance_compensation()
    assert env.commands[:,9:15].count_nonzero() == 0
    env.cfg.commands.ee_impedance_force_filter_tau = .3
    env.set_impedance_force_estimates(pred)
    env._apply_external_impedance_compensation()
    assert torch.allclose(env.commands[:,9:12],torch.full((2,3),-20*.02/.32))
    assert not env.force_command_stream_enabled


def test_single_pass_training_and_inference_callback():
    env, net = rejection_env(), model()
    count = []
    hook = net.adaptation_encoder_module.register_forward_hook(lambda *args: count.append(1))
    runner = OnPolicyRunnerUniFPOriginal.__new__(OnPolicyRunnerUniFPOriginal)
    runner.env = env
    runner.alg = PPO_UniFPOriginal(net, decision_callback=runner._publish_estimates)
    h = torch.randn(2,2336)
    with torch.no_grad():
        runner.alg.act(h,torch.zeros(2,30),torch.zeros(2,12))
    estimate = env.estimated_ee_force_local.clone()
    runner.get_inference_policy()(h)
    assert len(count) == 2
    assert torch.equal(estimate,env.estimated_ee_force_local)
    hook.remove()


def test_registration_config_and_separate_labels():
    assert task_registry.task_classes["b1z1_unifp_reject"] is B1Z1UniFPReject
    assert task_registry.task_classes["b1z1_unifp"].__name__ == "B1Z1UniFP"
    a, b = B1Z1UniFPOriginalCfg(), B1Z1UniFPOriginalCfg()
    assert (a.env.num_obs_hist,a.env.num_priv_stack,a.env.num_pred_obs) == (32,3,12)
    a.commands.ee_impedance_force_filter_tau = 9
    assert b.commands.ee_impedance_force_filter_tau != 9
    labels = torch.arange(40.).reshape(2,20)
    result = rejection_env().original_adaptation_target(labels)
    assert torch.equal(result, labels[:,[0,1,2,3,4,5,9,10,11,6,7,8]])
    assert labels.shape == (2,20)
