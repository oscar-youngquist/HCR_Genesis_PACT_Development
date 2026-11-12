"""Matched PILOT selectors and information-matched concat; CPU, no simulator."""
import copy

import pytest
import torch

from legged_gym.envs.b1z1.b1z1_pact.ablation_configs import make_b1z1_pact_ablation_configs
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg, B1Z1PACTCfgPPO
from legged_gym.envs.b1z1.compliance_task import configure_algorithm
from legged_gym.utils.helpers import class_to_dict as _class_to_dict, _json_config_value
from rsl_rl.modules.actor_critic_b1z1_pact import prepare_tracking_condition
from test_b1z1_directional_compliance import fixture
from test_b1z1_film_v02 import model


def class_to_dict(value):
    # The configuration includes NumPy arrays; compare resolved values, not identity.
    return _json_config_value(_class_to_dict(value))


@pytest.mark.parametrize('variant', [13, 14, 15])
def test_exact_configuration_delta_and_isolation(variant):
    ec, tc = make_b1z1_pact_ablation_configs(variant)
    e, t = ec(), tc()
    expected_e, expected_t = class_to_dict(B1Z1PACTCfg()), class_to_dict(B1Z1PACTCfgPPO())
    expected_e.update(ablation_variant=variant, action_mode='coupled')
    expected_t['ablation_variant'] = variant
    expected_t['policy']['action_mode'] = 'coupled'
    expected_t['algorithm']['representation_pinn_enabled'] = True  # Full runner's implicit default.
    if variant in (13, 14):
        expected_e['compliance'].update(joint_gate=False, collision_gate=False)
    if variant == 14:
        expected_e['compliance'].update(stiffness_gate=False, workspace_gate=False)
    if variant == 15:
        expected_t['algorithm'].update(actor_phys_pos_fk_enabled=False,
                                      actor_phys_arm_feedforward_enabled=False)
    expected_t['runner'].update(experiment_name=t.runner.experiment_name, run_name=t.runner.run_name)
    assert class_to_dict(e) == expected_e  # Includes actual loads, bounds, gains, profiles, schedules.
    assert class_to_dict(t) == expected_t
    before_e, before_t = class_to_dict(B1Z1PACTCfg()), class_to_dict(B1Z1PACTCfgPPO())
    e.compliance.family_weights[0] = 999
    e.control.stiffness.clear()
    t.policy.actor_layers[0] = 1
    assert class_to_dict(B1Z1PACTCfg()) == before_e
    assert class_to_dict(B1Z1PACTCfgPPO()) == before_t
    assert class_to_dict(ec()) == expected_e
    assert class_to_dict(tc()) == expected_t


@pytest.mark.parametrize('variant', [13, 14])
def test_recruitment_preserves_constraints_profiles_and_loads(variant):
    m, k = fixture()
    cfg, _ = make_b1z1_pact_ablation_configs(variant)
    m.cfg = cfg().compliance
    k['force'][:, :2] = 1000
    force = k['force'].clone()
    profiles = m.profile.clone()
    calls = []
    def hold(candidate, previous):
        calls.append(candidate.clone())
        return previous.clone()
    k['target_safe'] = hold
    r = m.advance(**k)
    assert len(calls) == 2  # Equilibrium and integrated path remain independently checked.
    torch.testing.assert_close(r.position, k['nominal'])
    assert r.velocity.eq(0).all()
    torch.testing.assert_close(k['force'], force)
    torch.testing.assert_close(m.profile, profiles)
    assert r.joint.eq(0).all() and r.collision.eq(0).all()
    if variant == 14:
        assert r.gates.eq(0).all() and r.base.eq(0).all()
    # Without the collision obstacle, workspace bounds still limit a sustained load.
    del k['target_safe']
    for _ in range(30):
        r = m.advance(**k)
    assert (r.position.norm(dim=-1) <= .900001).all()


def test_no_ee_recruitment_keeps_ee_yielding_and_torso_response():
    ec, _ = make_b1z1_pact_ablation_configs(14)
    m, k = fixture()
    m.cfg = ec().compliance
    k['force'][:, 1] = 30
    k['torso_wrench'] = torch.tensor([10., -15., 4.]).expand(4, -1)
    r = m.advance(**k)
    assert r.position[:, 1].gt(0).all()
    assert r.gates.eq(0).all() and r.ee_wrench.eq(0).all()
    assert r.base[:, 0].gt(0).all() and r.base[:, 1].lt(0).all() and r.base[:, 2].gt(0).all()
    torch.testing.assert_close(r.combined_wrench, k['torso_wrench'])
    other, k2 = fixture()
    other.cfg = ec().compliance
    k2['torso_wrench'] = k['torso_wrench']
    torch.testing.assert_close(other.advance(**k2).base, r.base)


def test_workspace_variant_retains_stiffness_trigger():
    ec, _ = make_b1z1_pact_ablation_configs(13)
    m, k = fixture()
    m.cfg = ec().compliance
    m.profile[:, :6] = 0
    k['force'][:, 1] = 10
    assert m.advance(**k).base[:, 1].gt(0).all()
    m, k = fixture()
    m.cfg = ec().compliance
    k['nominal'][:, 0] = .895
    k['force'][:, 0] = 30
    assert m.advance(**k).workspace[:, 0].min() > .9


@pytest.mark.parametrize('local_teacher', [False, True])
def test_channel_off_skips_teachers_but_keeps_rollout_gradients(monkeypatch, local_teacher):
    from test_b1z1_actor_physics import setup
    from rsl_rl.algorithms import b1z1_actor_physics as physics
    a, batch, actions, context = setup()
    ec, tc = make_b1z1_pact_ablation_configs(15)
    cfg = configure_algorithm(class_to_dict(tc().algorithm), ec())
    assert cfg['actor_phys_enabled'] and cfg['representation_pinn_enabled']
    for key in ('actor_phys_pos_fk_enabled', 'actor_phys_arm_feedforward_enabled'):
        a.cfg[key] = cfg[key]
        assert not cfg[key]
    a.cfg.update(actor_phys_pos_fk_weight=1., actor_phys_arm_feedforward_weight=1.,
                 actor_phys_arm_feedforward_compliant_local_enabled=local_teacher)
    def forbidden(*args, **kwargs):
        pytest.fail('Disabled channel teacher evaluated')
    monkeypatch.setattr(physics, 'position_fk', forbidden)
    monkeypatch.setattr(physics.b1z1_actor_feedforward, 'objective', forbidden)
    loss, _ = physics.objective(a, batch, actions, context)
    loss.backward()
    for head in (a.actor_critic.position_head, a.actor_critic.torque_head):
        assert head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in a.actor_critic.context_encoder.parameters())


def test_concat_exact_preprocessing_and_unmodulated_latent():
    film, concat = model(), model(mode='concat')
    errors = torch.randn(3, 30)
    torch.testing.assert_close(film.film.prepare_condition(errors), prepare_tracking_condition(errors), rtol=0, atol=0)
    obs, hist = torch.randn(3, 81), torch.randn(3, 162)
    captured = []
    hook = concat.actor_trunk.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
    with torch.no_grad():
        concat.act(obs, hist)
    hook.remove()
    ctx = concat.last_context
    common, _ = concat._actor_inputs(obs, ctx)
    torch.testing.assert_close(captured[0][:, :-40], common)
    torch.testing.assert_close(captured[0][:, 81:89], ctx['z'])
    torch.testing.assert_close(captured[0][:, -40:], film.film.prepare_condition(ctx['film_tracking_history']), rtol=0, atol=0)
    assert not ctx['film_tracking_history'].requires_grad


def test_concat_v02_is_the_registered_default():
    from legged_gym.utils import task_registry
    training = task_registry.train_cfgs['b1z1_pact_ab5_coupled_concat']
    p = training.policy
    assert p.conditioning_mode == 'concat' and p.concat_conditioning_version == 2
    assert p.film_history_steps == 5 and p.film_identity_loss_weight == 0
    assert not training.algorithm.representation_pinn_enabled
    assert not training.algorithm.actor_phys_enabled
    # History is independent of the modulation switch; concat never applies FiLM.
    m = model(enabled=False, mode=p.conditioning_mode,
              concat_conditioning_version=p.concat_conditioning_version)
    assert m.tracking_history_enabled and m.film_history_dim == 30 and m.film is None
    assert m.actor_trunk[0][0].in_features == 81 + 8 + 23 + 40


def test_concat_checkpoint_version_and_legacy_explicit_loading():
    legacy = model(mode='concat', concat_conditioning_version=1)
    current = model(mode='concat')
    with pytest.raises(ValueError, match='fresh training'):
        current.load_state_dict(legacy.state_dict(), strict=False)
    legacy.load_state_dict(copy.deepcopy(legacy.state_dict()))
    assert legacy.film_history_dim == 0
    current.load_state_dict(copy.deepcopy(current.state_dict()))
    with pytest.raises(ValueError, match='contract'):
        legacy.load_state_dict(current.state_dict())
    wrong = copy.deepcopy(current.state_dict())
    wrong['_concat_contract'][1] = 4
    with pytest.raises(ValueError, match='contract'):
        current.load_state_dict(wrong)
    assert not any('_concat_contract' in k for k in model().state_dict())


def test_concat_inference_reset_and_readonly_replay():
    m = model(mode='concat').eval()
    obs, hist = torch.randn(2, 81), torch.randn(2, 162)
    with torch.no_grad():
        first = m.act_inference(obs, hist)
        stored = m.last_context['film_tracking_history'].clone()
        for _ in range(3):
            m.act_inference(obs + 1, hist)
        running = m._film_history.clone()
        ctx = m.decode_context(m.context_encoder.forward_inf(hist))
        m._condition_history(obs, ctx, stored, False)
        torch.testing.assert_close(torch.cat(m.actor_forward(obs, ctx, ctx), -1), first)
        torch.testing.assert_close(m._film_history, running)
        m.reset()
        torch.testing.assert_close(m.act_inference(obs, hist), first, rtol=0, atol=0)
