"""Focused physical GRF gating, exclusive gradients, and config/metric checks."""
from unittest.mock import patch

import pytest
import torch

from rsl_rl.modules.hard_pact_physics import (
    GRFSwingConfig, GRFSwingMetricsAccumulator, gate_grf_for_qp, log_qp_swing_grf,
)
from rsl_rl.algorithms.hard_pact_qp_diagnostics import QPIterationDiagnostics
from rsl_rl.hard_pact_logging import collect_force_decoder_scalars
from rsl_rl.algorithms.ppo_hard_pact import _yaw_local_to_world
from test_go2_hard_pact_physics_heads import (
    _small_actor, ActorCritic_HardPACT, ActorCritic_HardPACT_Pos,
    GO2HardPACTCfg, GO2HardPACTPosCfg, build_deployment_contract,
    calculate_physics_head_gains, build_hard_pact_start_checkpoint,
)
from test_hard_pact_auxiliary import make_algorithm, make_batch
from test_hard_pact_pos_auxiliary import _small_algorithm


def assert_grf_only_gradients(actor, loss):
    gradients = torch.autograd.grad(loss, list(actor.parameters()), allow_unused=True)
    grf_sum = 0.
    for (name, _), gradient in zip(actor.named_parameters(), gradients):
        if name.startswith("physics_estimator.grf_head."):
            assert gradient is not None and torch.isfinite(gradient).all()
            grf_sum += gradient.abs().sum().item()
        else:
            assert gradient is None, name
    assert grf_sum > 0.


def test_physical_gate_exact_xyz_boundary_and_gradients():
    raw = torch.arange(-12., 12.).reshape(2, 4, 3).requires_grad_()
    before = raw.detach().clone()
    contact = torch.tensor([[.1, .5, .9, .49], [.99, .2, .5, .01]], requires_grad=True)
    cfg = GRFSwingConfig(enabled=True)
    swing = contact.detach() < .5
    with patch("torch.sigmoid", side_effect=AssertionError("second sigmoid")):
        gated = gate_grf_for_qp(raw, contact, cfg)
    assert gated[swing].eq(0).all()
    torch.testing.assert_close(gated[~swing], raw[~swing], rtol=0, atol=0)
    torch.testing.assert_close(raw, before, rtol=0, atol=0)
    gated.sum().backward()
    assert contact.grad is None
    torch.testing.assert_close(raw.grad, (~swing)[..., None].expand_as(raw).float(), rtol=0, atol=0)
    # The frame changes but foot selection and all-vector zeros do not.
    quat = torch.tensor([[0., 0., 2**-.5, 2**-.5]]).expand(2, -1)
    torch.testing.assert_close(_yaw_local_to_world(gated, quat),
        gate_grf_for_qp(_yaw_local_to_world(raw, quat), contact, cfg), rtol=0, atol=0)


@pytest.mark.parametrize("actor_class", [ActorCritic_HardPACT, ActorCritic_HardPACT_Pos])
def test_consistency_routes_only_to_grf_and_preserves_rng(actor_class):
    torch.manual_seed(13)
    actor = _small_actor(actor_class=actor_class)
    heads = actor.physics_estimator
    heads.grf_swing = GRFSwingConfig(True, 1., .4)  # All predicted feet swing.
    _, _, latent, explicit = actor.cenet_enc_forward(torch.randn(3, 1140))
    pos, ff = actor.actor_forward(torch.cat((torch.randn(3, 57), latent, explicit), -1))
    torque = 3. * pos + ff  # Deliberately connected to the actor and encoder.
    raw = heads.predict_grf(latent, explicit, torque)
    rng = torch.get_rng_state()
    loss, stats = heads.swing_grf_auxiliary(latent, explicit, torque, raw, torch.ones(3, 1))
    torch.testing.assert_close(torch.get_rng_state(), rng)
    expected = .4 * (heads.grf_to_physical(raw) / heads.grf_scale_n).square().sum() / 12
    torch.testing.assert_close(loss, expected)
    assert stats["swing_count"] == 12
    assert_grf_only_gradients(actor, loss)


def test_no_swing_zero_loss_and_disabled_no_extra_forward():
    actor = _small_actor()
    heads = actor.physics_estimator
    latent, explicit, torque = torch.randn(2, 16), torch.rand(2, 11), torch.randn(2, 12)
    raw = heads.predict_grf(latent, explicit, torque)
    heads.grf_swing = GRFSwingConfig(True, 0., 1.)
    loss, stats = heads.swing_grf_auxiliary(latent, explicit, torque, raw, torch.ones(2, 1))
    assert loss.isfinite() and loss == 0 and stats["swing_count"] == 0
    loss.backward()
    assert all(p.grad is not None and p.grad.eq(0).all() for p in heads.grf_head.parameters())
    heads.grf_swing = GRFSwingConfig()
    with patch.object(heads, "predict_grf", side_effect=AssertionError("disabled forward")):
        zero, stats = heads.swing_grf_auxiliary(latent, explicit, torque, raw, torch.ones(2, 1))
        assert zero == 0 and stats is None
        assert gate_grf_for_qp(raw, explicit[:, 3:7], heads.grf_swing) is raw


@pytest.mark.parametrize("position_only", [False, True])
def test_real_auxiliary_phase_adds_only_exclusive_loss(position_only):
    torch.manual_seed(17)
    alg = _small_algorithm() if position_only else make_algorithm()
    physics = alg.actor_critic.physics_estimator
    alg._grf_swing_metrics = GRFSwingMetricsAccumulator(GRFSwingConfig(True, 1., .3))
    if position_only:
        labels = torch.zeros(3, 11)
        args = (torch.randn(3, 570), torch.randn(3, 12), torch.randn(3, 133),
                labels, torch.ones(3, 1), torch.randn(3, 12), torch.randn(3, 6),
                torch.zeros(3, 1, dtype=torch.bool))
        def run():
            torch.manual_seed(91)
            result = alg._compute_vae_loss(*args)
            return result[0], result[-1]
    else:
        batch = make_batch(3)
        inputs = alg._compute_auxiliary_loss(*batch, return_decoder_inputs=True)["decoder_inputs"]
        def run():
            result = alg._compute_auxiliary_decoder_loss(**inputs)
            return result["loss"], result
    baseline, old_terms = run()
    physics.grf_swing = GRFSwingConfig(True, 1., .3)
    total, terms = run()
    consistency = terms["grf_swing_consistency_loss"]
    torch.testing.assert_close(total, baseline + consistency, rtol=0, atol=0)
    for name in ("grf", "wrench_active", "wrench_neutral"):
        torch.testing.assert_close(old_terms[name], terms[name], rtol=0, atol=0)
    assert_grf_only_gradients(alg.actor_critic, consistency)
    metrics = collect_force_decoder_scalars(alg._grf_swing_metrics.finalize())
    assert set(metrics) == {
        "physics/grf/swing_fraction", "physics/grf/swing_raw_norm_n",
        "physics/grf/swing_removed_norm_n", "physics/grf/swing_consistency_loss",
    }
    assert all(torch.isfinite(v) and not v.requires_grad for v in metrics.values())


def test_statistics_partition_mask_and_physical_zero_target():
    heads = _small_actor().physics_estimator
    heads.grf_swing = GRFSwingConfig(True, .5, .7)
    latent, explicit, torque = torch.zeros(4, 16), torch.ones(4, 11), torch.arange(48.).reshape(4, 12)/48
    explicit[:, 3] = .1
    explicit[1, 4] = .2
    valid = torch.tensor([[1], [1], [1], [0]])
    latent[-1] = torque[-1] = float("nan")
    sums = []
    with patch.object(heads, "predict_grf", side_effect=lambda z, e, tau: tau), \
         patch.object(heads, "grf_to_physical", side_effect=lambda raw: raw * heads.grf_scale_n + 10.):
        for partitions in ((slice(None),), (slice(0, 1), slice(1, 4))):
            accumulator = GRFSwingMetricsAccumulator(heads.grf_swing)
            for rows in partitions:
                _, stats = heads.swing_grf_auxiliary(latent[rows], explicit[rows], torque[rows], torque[rows], valid[rows])
                accumulator.add(stats)
            sums.append(accumulator.finalize())
    for name in sums[0]:
        torch.testing.assert_close(sums[0][name], sums[1][name])
    swing = explicit[:3, 3:7] < .5
    normalized_physical = (torque[:3] + 10./heads.grf_scale_n).reshape(3, 4, 3)
    expected = .7 * normalized_physical[swing].square().sum() / swing.sum()
    torch.testing.assert_close(sums[0]["grf_swing_consistency_loss"], expected)
    torch.testing.assert_close(sums[0]["grf_swing_fraction"], torch.tensor(4./12))
    physical = normalized_physical * heads.grf_scale_n.reshape(4, 3)
    qp_stats = QPIterationDiagnostics()
    for rows in (slice(0, 1), slice(1, 3)):
        log_qp_swing_grf(qp_stats, physical[rows], explicit[rows, 3:7], heads.grf_swing)
    metrics = qp_stats.finalize(torch.tensor(0.))
    torch.testing.assert_close(metrics["grf_swing/fraction"], torch.tensor(4./12))
    torch.testing.assert_close(metrics["grf_swing/raw_norm_n"], physical.norm(dim=-1)[swing].mean())
    torch.testing.assert_close(metrics["grf_swing/removed_norm_n"], physical.norm(dim=-1).where(swing, 0.).mean())


def test_config_contract_and_strict_pos_migration():
    pos = _small_actor(actor_class=ActorCritic_HardPACT_Pos)
    hard = _small_actor()
    keys = set(hard.state_dict())
    for cfg_class, actor in ((GO2HardPACTPosCfg, pos), (GO2HardPACTCfg, hard)):
        cfg = cfg_class()
        assert GRFSwingConfig.from_task(cfg) == GRFSwingConfig()
        cfg.deployment_physics.grf_swing_gating_enabled = True
        cfg.deployment_physics.grf_swing_contact_threshold = .6
        cfg.deployment_physics.grf_swing_loss_weight = .2
        actor.physics_estimator.grf_swing = GRFSwingConfig.from_task(cfg)
        contract = build_deployment_contract(cfg, actor, calculate_physics_head_gains(cfg))
        assert contract["schema_version"] == 10
        assert contract["grf_swing_gating"]["enabled"]
        assert contract["grf_swing_gating"]["contact_probability_threshold"] == .6
        assert contract["grf_swing_gating"]["consistency_loss_weight"] == .2
    checkpoint = build_hard_pact_start_checkpoint(pos.state_dict(), {}, 3)
    hard.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert set(hard.state_dict()) == keys
    z, e, tau = torch.randn(2, 16), torch.rand(2, 11), torch.randn(2, 12)
    outputs = [a.physics_estimator.grf_to_qp_physical(
        a.physics_estimator.predict_grf(z, e, tau), e[:, 3:7]) for a in (pos, hard)]
    torch.testing.assert_close(*outputs, rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [{"threshold": -1.}, {"threshold": float("nan")}, {"loss_weight": -.1}])
def test_invalid_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        GRFSwingConfig(**kwargs)
