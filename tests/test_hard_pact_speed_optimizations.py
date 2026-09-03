"""Focused regression tests for HardPACT's iteration-scoped fast paths."""

import os
from types import SimpleNamespace

os.environ.setdefault("SIMULATOR", "genesis_pact")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_hard_pact_speed_tests")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_hard_pact_speed_tests")

import torch

from rsl_rl.algorithms.pc_grad import PCGrad
from rsl_rl.algorithms.ppo_hard_pact import PPO_HardPACT
from rsl_rl.storage.rollout_storage_pact import RolloutStoragePACT


class _FakeContext:
    def __init__(self, q, v, post, randomized, need_qp):
        batch = q.shape[0]
        offset = randomized.reshape(batch, 1, 1)
        self.mass_matrix = (
            torch.eye(18).expand(batch, -1, -1) + offset
        ).detach()
        self.bias = torch.full((batch, 18), float(need_qp), device=q.device)
        self.foot_jacobians = torch.zeros(batch, 4, 3, 18, device=q.device)
        self.base_jacobian = torch.zeros(batch, 6, 18, device=q.device)
        self.foot_acceleration_bias = (
            torch.zeros(batch, 4, 3, device=q.device) if need_qp else None
        )
        self.pre_v_canonical = v.detach()
        self.post_v_canonical = None if post is None else post.detach()


class _FakeDynamics:
    batch_capacity = 4

    def __init__(self):
        self.calls = []
        self.default_joint_position = None

    def build_context(
        self, q, v, *, parameters, post_v_world=None,
        mass_com_wrench_world=None, need_qp=False, **_kwargs,
    ):
        added = parameters.get("added_base_mass")
        randomized = (
            torch.zeros(q.shape[0], device=q.device)
            if added is None else added.reshape(-1)
        )
        self.calls.append((q.shape[0], bool(need_qp), bool(parameters)))
        return _FakeContext(q, v, post_v_world, randomized, need_qp)


def _flat_transition_fields(steps=2, envs=3):
    shape = (steps, envs)
    zeros = lambda *tail: torch.zeros(*shape, *tail)
    q = zeros(19)
    q[..., 6] = 1.0
    return {
        "pre_q": q,
        "pre_v": zeros(18),
        "post_v": torch.ones(*shape, 18),
        "realized_added_mass": torch.arange(1, steps * envs + 1.0).reshape(
            *shape, 1
        ),
        "realized_com_shift_body": zeros(3),
        "joint_armature": zeros(12),
        "joint_friction": zeros(12),
        "joint_stiffness": zeros(12),
        "joint_damping": zeros(12),
        "equivalent_mass_com_wrench_world": zeros(6),
        "sampled_qp_q": q.clone(),
        "sampled_qp_v": zeros(18),
    }


def test_rollout_cache_is_indexable_invalidated_and_separates_mechanics():
    algorithm = PPO_HardPACT.__new__(PPO_HardPACT)
    algorithm.physics_dynamics = _FakeDynamics()
    algorithm.storage = SimpleNamespace(
        hard_pact_fields=_flat_transition_fields()
    )
    algorithm.bard_inverse_enabled = True
    algorithm.bard_rollout_enabled = True
    algorithm.hard_pact_features = SimpleNamespace(execution_qp=True)
    algorithm.hard_pact_qp = object()
    algorithm._rollout_actual_mechanics = None
    algorithm._rollout_deployment_qp_mechanics = None

    algorithm._prepare_rollout_mechanics_cache(torch.zeros(12))
    actual = algorithm._rollout_actual_mechanics
    deployment = algorithm._rollout_deployment_qp_mechanics
    assert actual.kind == "actual"
    assert deployment.kind == "deployment"
    assert not torch.equal(actual.mass_matrix, deployment.mass_matrix)
    assert torch.equal(
        deployment.mass_matrix, torch.eye(18).expand(6, -1, -1)
    )
    order = torch.tensor([5, 0, 3])
    assert torch.equal(
        actual.index(order).mass_matrix, actual.mass_matrix[order]
    )
    # Six rows at capacity four: one actual and one deployment pass, each in
    # exactly two chunks. Repeated indexing invokes no dynamics work.
    assert len(algorithm.physics_dynamics.calls) == 4
    actual.index(order)
    assert len(algorithm.physics_dynamics.calls) == 4
    algorithm._clear_rollout_mechanics_cache()
    assert algorithm._rollout_actual_mechanics is None
    assert algorithm._rollout_deployment_qp_mechanics is None


def test_cached_and_uncached_chunk_mechanics_have_identical_values_and_vjp():
    """Indexing the iteration cache must not alter rollout solve semantics."""
    algorithm = PPO_HardPACT.__new__(PPO_HardPACT)
    algorithm.physics_dynamics = _FakeDynamics()
    fields = _flat_transition_fields(steps=2, envs=3)
    flat = algorithm._flat_fields(fields)
    parameters = {"added_base_mass": flat["realized_added_mass"]}
    cached = algorithm._materialize_mechanics_cache(
        kind="actual", q=flat["pre_q"], v=flat["pre_v"],
        parameters=parameters, post_v=flat["post_v"],
    )
    order = torch.tensor([5, 1, 4])
    indexed = cached.index(order)
    rebuilt = algorithm._materialize_mechanics_cache(
        kind="actual", q=flat["pre_q"][order], v=flat["pre_v"][order],
        parameters={"added_base_mass": parameters["added_base_mass"][order]},
        post_v=flat["post_v"][order],
    )
    for name in (
        "mass_matrix", "bias", "foot_jacobians", "base_jacobian",
        "pre_v_canonical", "post_v_canonical",
    ):
        torch.testing.assert_close(
            getattr(indexed, name), getattr(rebuilt, name), rtol=0, atol=0
        )

    left_force = torch.randn(3, 18, requires_grad=True)
    right_force = left_force.detach().clone().requires_grad_(True)
    left = indexed.as_context(algorithm.physics_dynamics).forward_dynamics(
        left_force
    )
    right = rebuilt.as_context(algorithm.physics_dynamics).forward_dynamics(
        right_force
    )
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    left.square().sum().backward()
    right.square().sum().backward()
    torch.testing.assert_close(left_force.grad, right_force.grad, rtol=0, atol=0)


def test_hard_pact_storage_omits_only_unused_legacy_mechanics():
    common = dict(
        num_envs=2, num_transitions_per_env=3, obs_shape=[57],
        critic_obs_shape=[95], sinle_critc_obs_shape=[10],
        obs_hist_shape=[1140], actions_shape=[24], explicit_shape=[11],
        grf_shape=[12], wb_shape=[18], device="cpu",
    )
    hard = RolloutStoragePACT(**common, store_legacy_pinn_dynamics=False)
    legacy = RolloutStoragePACT(**common)
    assert hard.wb_contact_forces is not None
    assert hard.wb_mass_mats is None
    assert hard.wb_bias_vecs is None
    assert hard.torso_accelerations is None
    assert legacy.wb_mass_mats.shape == (3, 2, 18, 18)
    assert legacy.wb_bias_vecs.shape == (3, 2, 18)
    assert legacy.torso_accelerations.shape == (3, 2, 6)


def test_pcgrad_diagnostics_schedule_and_clone_gate():
    algorithm = PPO_HardPACT.__new__(PPO_HardPACT)
    algorithm.pcgrad_diagnostics_enabled = True
    algorithm.pcgrad_diagnostics_start_iteration = 7
    algorithm.pcgrad_diagnostics_interval = 5
    assert not algorithm._pcgrad_diagnostics_due(6)
    assert algorithm._pcgrad_diagnostics_due(7)
    assert not algorithm._pcgrad_diagnostics_due(8)
    assert algorithm._pcgrad_diagnostics_due(12)
    algorithm.pcgrad_diagnostics_enabled = False
    assert not algorithm._pcgrad_diagnostics_due(12)

    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    wrapper = PCGrad(torch.optim.SGD([parameter], lr=0.1))
    wrapper.pc_backward(
        [parameter.square().sum(), (parameter - 1.0).square().sum()],
        record_diagnostics=False,
    )
    assert wrapper.last_objective_grads is None
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_zero_delay_reuses_current_policy_result():
    algorithm = PPO_HardPACT.__new__(PPO_HardPACT)
    algorithm.actor_critic = SimpleNamespace(std=torch.ones(24))
    algorithm.action_clip = 10.0
    algorithm.storage = SimpleNamespace(max_action_delay=0)
    algorithm._policy_raw_action_from_noise = lambda *_args: (_ for _ in ()).throw(
        AssertionError("zero-delay replay evaluated the actor twice")
    )
    current = torch.randn(2, 24, requires_grad=True)
    observation = torch.zeros(2, 57)
    transition = {
        "standardized_action_noise": torch.randn(2, 24),
        "delayed_action_source_valid": torch.ones(2, 1, dtype=torch.bool),
    }

    def transform(action):
        return action[:, :12], action[:, 12:]

    result = algorithm._replay_action_path(
        current, observation, transition, transform,
        lambda desired, _q, _v: desired, torch.zeros(12), 1.0,
    )
    torch.testing.assert_close(result["raw_action"], result["delayed_action"])
    result["nominal_torque"].sum().backward()
    assert current.grad is not None and current.grad.abs().sum() > 0


def test_mechanics_materialization_contains_no_explicit_cuda_sync():
    # This protects the normal per-chunk path from accidental `.item()` or
    # synchronize reintroduction. Iteration-boundary timing remains allowed.
    import inspect

    source = inspect.getsource(PPO_HardPACT._materialize_mechanics_cache)
    assert ".item(" not in source
    assert "synchronize(" not in source
    assert "bool(" not in source
