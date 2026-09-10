"""Regressions for the live Isaac Lab contact-index and zero inverse-gate bugs."""
import ast
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import torch

from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos import Go2HardPACTPos
from rsl_rl.algorithms.hard_pact_bard import measured_contact_generalized_force
from rsl_rl.algorithms.ppo_hard_pact import _yaw_local_to_world
from test_hard_pact_auxiliary import make_algorithm
from test_go2_hard_pact_bard import BardGo2Dynamics, URDF


def test_support_contacts_use_sensor_order_and_match_pos():
    contact = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0]])
    forces = torch.zeros(4, 8, 3)
    forces[:, [0, 2, 4, 6], 2] = contact * 30.
    sim = SimpleNamespace(
        base_pos=torch.zeros(4, 3),
        feet_pos=torch.tensor([[[.2, -.15, 0], [.2, .15, 0], [-.2, -.15, 0], [-.2, .15, 0]]]).repeat(4, 1, 1),
        feet_vel=torch.ones(4, 4, 3),
        link_contact_forces=forces, feet_indices=[1, 3, 5, 7],
        feet_contact_indices=[0, 2, 4, 6],
    )
    cfg = SimpleNamespace(rewards=SimpleNamespace(
        support_polygon_sigma=.01, contact_force_threshold=5., max_contact_force=20.,
    ))
    for cls in (Go2HardPACT, Go2HardPACTPos):
        env = cls.__new__(cls)
        env.simulator, env.cfg = sim, cfg
        torch.testing.assert_close(env._reward_support_polygon(), torch.tensor([1., 1., 0., 0.]))
        torch.testing.assert_close(env._reward_foot_slip(), 4. * contact.sum(1))
        torch.testing.assert_close(env._reward_feet_contact_forces(), 10. * contact.sum(1))
        torch.testing.assert_close(env._feet_contact_mask(), contact.bool())
    # Genesis aliases the two index spaces: identical canonical input yields
    # exactly the same reward, with no articulation indexing changes needed.
    sim.feet_indices = sim.feet_contact_indices
    torch.testing.assert_close(env._reward_support_polygon(), torch.tensor([1., 1., 0., 0.]))


def test_every_inherited_contact_tensor_access_uses_sensor_indices():
    # Includes critic contacts, support/air-time/slip/stumble, VHIP and edge
    # helpers, not just the reward that initially exposed the bug.
    tree = ast.parse(inspect.getsource(Go2PACT))
    accesses = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            if node.value.attr in ("link_contact_forces", "_link_contact_forces", "link_contact_states"):
                names = {x.attr for x in ast.walk(node.slice) if isinstance(x, ast.Attribute)}
                assert "feet_indices" not in names
                accesses.append(names)
    assert sum("feet_contact_indices" in names for names in accesses) >= 16


def test_measured_contact_projection_frame_scale_order_and_detachment():
    # Nonidentity yaw: normalized FR Fx becomes world Fy. Nonuniform scale
    # catches accidental critic scaling; distinct feet/J columns catch order.
    q = torch.tensor([[0., 0., 0., 0., 0., 2**-.5, 2**-.5]])
    labels = torch.arange(1., 13.).reshape(1, 4, 3).requires_grad_()
    scale = torch.tensor([100., 200., 300.])
    world = _yaw_local_to_world(labels * scale, q[:, 3:7])
    expected_world = torch.stack((-labels[..., 1]*200., labels[..., 0]*100., labels[..., 2]*300.), -1)
    torch.testing.assert_close(world, expected_world, rtol=1e-6, atol=1e-3)
    jac = torch.zeros(1, 4, 3, 18)
    for foot in range(4):
        jac[:, foot, :, 6+3*foot:9+3*foot] = torch.eye(3)
    jac.requires_grad_()
    actual = measured_contact_generalized_force(jac, world)
    torch.testing.assert_close(actual[:, 6:], expected_world.flatten(1), rtol=1e-6, atol=1e-3)
    assert not actual.requires_grad and labels.grad is None and jac.grad is None


def test_real_ppo_inverse_update_ignores_zero_legacy_buffer_and_trains_heads():
    alg = make_algorithm(num_learning_epochs=1, num_mini_batches=1)
    alg.bard_enabled = alg.bard_inverse_enabled = True
    alg.bard_rollout_enabled = False
    # Exercise the live run's positive-weight (unbalanced) PCGrad path.
    alg.pinn_init, alg.pinn_weight_final = -1, .01
    alg.pinn_warmup_steps, alg.num_pinn_updates = 1, 1
    alg.physics_dynamics = BardGo2Dynamics(URDF, device="cpu", batch_capacity=4)
    alg.init_storage(4, 2, [57], [95], [133], [1140], [24], [11], [12], [18])
    storage = alg.storage
    storage.max_action_delay = 0
    zeros = lambda width: torch.zeros(2, 4, width)
    q = zeros(19)
    q[..., 2], q[..., 6] = .4, 1.
    q[..., 7:] = torch.tensor([0., .8, -1.5] * 4)
    fields = {
        "pre_q": q, "pre_v": zeros(18), "post_v": zeros(18),
        "control_dt": torch.full((2, 4, 1), .02),
        "interval_executed_torque": zeros(12),
        "standardized_action_noise": zeros(24),
        "delayed_action_source_valid": torch.ones(2, 4, 1, dtype=torch.bool),
        "realized_added_mass": zeros(1), "realized_com_shift_body": zeros(3),
        "joint_armature": zeros(12), "joint_friction": zeros(12),
        "joint_stiffness": zeros(12), "joint_damping": zeros(12),
        "equivalent_mass_com_wrench_world": zeros(6),
        "total_external_wrench_label_yaw_normalized": zeros(6),
        "sustained_wrench_active_mask": zeros(1).bool(),
        **{name: zeros(1).bool() for name in ("push_event_mask", "reset_mask", "timeout_mask", "teleport_mask")},
    }
    storage.hard_pact_fields = fields
    for t in range(2):
        obs, hist, critic = torch.randn(4, 57), torch.randn(4, 1140), torch.randn(4, 95)
        with torch.no_grad():
            alg.act(obs, critic, hist, obs, hist, obs, hist)
        tr = alg.transition
        for name, value in (
            ("observations", obs), ("observation_history", hist), ("critic_observations", critic),
            ("actions", tr.actions), ("actions_log_prob", tr.actions_log_prob[:, None]),
            ("mu", tr.action_mean), ("sigma", tr.action_sigma),
            ("latent_noise", tr.latent_noise), ("latent_boot_mask", tr.latent_boot_mask),
        ):
            getattr(storage, name)[t].copy_(value)
    storage.advantages.normal_()
    storage.grf_targets[..., 2::3] = .3
    assert not storage.wb_contact_forces.any()  # The real Isaac Lab failure condition.
    measured, gradients = [], []
    from rsl_rl.algorithms.hard_pact_bard import corrected_bard_inverse_dynamics_loss

    def checked_loss(**kwargs):
        measured.append(kwargs["measured_generalized_contact_force"])
        result = corrected_bard_inverse_dynamics_loss(**kwargs)
        weights = [alg.actor_critic.context_encoder.ce_out_mean.weight,
                   alg.actor_critic.physics_estimator.grf_head[-1].weight,
                   alg.actor_critic.physics_estimator.wrench_head[-1].weight]
        grads = torch.autograd.grad(result.loss, weights, retain_graph=True)
        gradients.extend(g.detach() for g in grads)
        assert result.loss > 0 and torch.isfinite(result.loss)
        return result

    with patch("rsl_rl.algorithms.ppo_hard_pact.corrected_bard_inverse_dynamics_loss", side_effect=checked_loss):
        alg.update(lambda a: (a[:, :12], a[:, 12:]), lambda q, p, v: q-p-v,
                   .02, 0, torch.zeros(12), 1.)
    assert measured and all(x.abs().sum() > 0 and not x.requires_grad for x in measured)
    assert gradients and all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
