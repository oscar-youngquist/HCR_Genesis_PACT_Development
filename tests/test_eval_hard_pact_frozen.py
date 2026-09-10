"""Small forward-only tests for frozen evaluation (no training/backward)."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.eval_hard_pact_frozen import (
    PREFIX, VARIANTS, Moments, Observer, apply_config, extend_evaluation_timeout, frozen_policy_actions, make_scenario, parse_args, tensor_hash, write_json,
)


def test_cli_defaults_and_exact_variant_matrix(tmp_path):
    checkpoint, config = tmp_path / "model.pt", tmp_path / "config.json"
    checkpoint.touch()
    config.touch()
    args = parse_args(["--checkpoint", str(checkpoint), "--resolved-config", str(config),
                       "--output-dir", str(tmp_path / "out")])
    assert (args.num_envs, args.duration, PREFIX, args.sample_actions) == (64, 20., 100, False)
    assert VARIANTS == {
        "pre_qp": None, "analytic": None,
        "qp_every_substep": "every_substep",
        "qp_random_one_substep": "random_one_substep",
    }


def test_aggregate_invalid_and_partitioned_values(tmp_path):
    whole, parts = Moments(), Moments()
    data = torch.tensor([[1., -3.], [float("nan"), float("inf")], [5., 7.]])
    mask = torch.tensor([True, False, True])
    whole.add("error", data, mask)
    parts.add("error", data[:1], mask[:1])
    parts.add("error", data[1:], mask[1:])
    assert whole.result() == parts.result()
    result = whole.result()["error"]
    assert result["count"] == 4 and result["nonfinite_count"] == 0
    assert result["mean_abs"] == 4. and result["rms"] == pytest.approx(21. ** .5)
    whole.add("unavailable", data[1])
    assert whole.result()["unavailable"]["mean"] is None
    assert whole.result()["unavailable"]["nonfinite_count"] == 2
    write_json(tmp_path / "test.json", {"metrics": whole.result(), "missing": float("inf")})
    assert json.loads((tmp_path / "test.json").read_text())["missing"] is None


def test_resolved_config_is_independent_of_defaults_and_original():
    original = SimpleNamespace(control=SimpleNamespace(decimation=4), mapping={"a": 1})
    target = copy.deepcopy(original)
    values = {"control": {"decimation": 2}, "mapping": {"a": 3}}
    apply_config(target, values)
    target.mapping["a"] = 4
    assert original.control.decimation == 4 and original.mapping == {"a": 1}
    assert values["mapping"]["a"] == 3 and target.control.decimation == 2


@pytest.mark.parametrize("variant", ["pre_qp", "analytic"])
def test_prefix_then_analytic_uses_actual_previous_torque_and_resets(variant):
    # Deliberately different raw and executed torque; projection must not use
    # a pre-clipped or stale buffer as its rate center. No dynamics/QP work.
    n = 2
    simulator = SimpleNamespace(
        _torques=torch.zeros(n, 12), _kp_scale=torch.ones(n, 12),
        _kd_scale=torch.ones(n, 12), _motor_strength=torch.ones(n, 12),
        feedback_torques=torch.ones(n, 12),
        hard_pact_executed_torque=lambda: simulator._torques.clamp(-2., 2.),
        hard_pact_set_executed_torque=lambda tau: setattr(simulator, "_torques", tau),
        _hard_pact_pre_physics_substep=lambda: None,
        _hard_pact_grf_post_physics_substep=lambda: None,
    )
    env = SimpleNamespace(
        num_envs=n, device="cpu", simulator=simulator,
        cfg=SimpleNamespace(sim=SimpleNamespace(dt=.01)),
        _hard_pact_q_d=torch.ones(n, 12), _hard_pact_tau_ff=torch.ones(n, 12),
        _hard_pact_previous_substep_torque=torch.full((n, 12), 50.),
        _canonical_joint_state=lambda: (torch.zeros(n, 12), torch.zeros(n, 12)),
        _get_pinn_feedback=lambda *args: torch.ones(n, 12),
        _yaw_local_to_world=lambda force, quat: force,
        _current_base_quat_xyzw=lambda: torch.tensor([[0., 0., 0., 1.]]).repeat(n, 1),
        check_termination=lambda: None, reset_idx=lambda ids: None,
    )
    heads = SimpleNamespace(predict_grf=lambda *args: torch.zeros(n, 12), grf_to_physical=lambda value: value)
    actor = SimpleNamespace(physics_estimator=heads, cenet_z=torch.zeros(n, 16), cenet_torso_velo=torch.zeros(n, 11))
    qp = SimpleNamespace(torque_limits=torch.full((12,), 2.), cfg=SimpleNamespace(torque_rate_limit_nm_s=10.),
                         solve=lambda **kwargs: pytest.fail("QP must not run for these baselines"))
    observer = Observer(env, actor, qp, SimpleNamespace(variant=variant, trace_window=2))
    with torch.no_grad():
        observer.step = 99
        simulator._torques.fill_(50.)
        observer.pre()
        assert simulator.hard_pact_executed_torque().eq(2.).all()  # prefix is untouched
        observer.reset(torch.arange(n))
        for k in range(4):
            observer.step, observer.k = 100, k
            simulator._torques = torch.full((n, 12), 50. + k)
            observer.pre()
            expected = (k + 1) * .1 if variant == "analytic" else 2.
            torch.testing.assert_close(simulator.hard_pact_executed_torque(), torch.full((n, 12), expected))
    metrics = observer.metrics.result()
    assert metrics["evaluation/torque_violation"]["mean"] == 0.
    if variant == "analytic":
        assert metrics["evaluation/rate_violation"]["mean"] == 0.
    else:
        assert metrics["evaluation/rate_violation"]["mean"] > 0.  # violations are exposed, not hidden


def test_tensor_hash_checks_buffers_and_weights_without_mutation():
    layer = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    before = tensor_hash(layer.state_dict())
    with torch.no_grad():
        layer(torch.ones(3, 2))
    assert tensor_hash(layer.state_dict()) == before
    assert all(p.grad is None and not p.requires_grad for p in layer.parameters())


def test_seeded_sampled_evaluation_never_clamps_checkpoint_weights():
    from test_hard_pact_auxiliary import make_modules
    actor, _ = make_modules()
    actor.eval().requires_grad_(False)
    actor.std.fill_(0.001)  # act() would mutate this parameter via _clip_std()
    before = tensor_hash(actor.state_dict())
    obs, hist = torch.randn(2, 57), torch.randn(2, 1140)
    with torch.no_grad():
        torch.manual_seed(32)
        first = frozen_policy_actions(actor, obs, hist, True)
        torch.manual_seed(32)
        second = frozen_policy_actions(actor, obs, hist, True)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert tensor_hash(actor.state_dict()) == before
    assert not first.requires_grad and torch.isfinite(first).all()


def test_traces_bound_four_environments_and_activation_failure_windows():
    from collections import deque
    observer = Observer.__new__(Observer)
    observer.args = SimpleNamespace(trace_window=2)
    observer.obs, observer.history, observer.actions = torch.zeros(8, 57), torch.zeros(8, 1140), torch.zeros(8, 24)
    observer.env = SimpleNamespace(commands=torch.zeros(8, 4))
    observer.ring, observer.traces, observer.trace_until = deque(maxlen=3), {}, -1
    observer.first_failure = torch.full((8,), -1)
    for step in range(160):
        observer.step = step
        if step in (110, 120, 130, 140):
            observer.first_failure[(step - 110) // 10] = step
        observer.substeps = [{"torque": torch.zeros(4, 12)} for _ in range(4)]
        observer.finish_control()
    expected = {i for center in (100, 110, 120, 130, 140) for i in range(center - 2, center + 3)}
    assert set(observer.traces) == expected
    assert len(observer.traces) <= 5 * (2 * observer.args.trace_window + 1)
    assert all(record["observation"].shape[0] == 4 for record in observer.traces.values())


def test_timeout_and_out_of_bounds_are_censored_not_physical_failures():
    observer = Observer.__new__(Observer)
    observer.first_failure = torch.full((4,), -1)
    observer.first_episode_censored = torch.ones(4, dtype=torch.bool)
    observer.step, observer.reasons, observer.metrics = 110, {}, Moments()
    observer._termination = lambda: None
    observer.sim = SimpleNamespace(
        link_contact_forces=torch.zeros(4, 1, 3), termination_contact_indices=[0],
        base_pos=torch.zeros(4, 3), measured_heights=torch.zeros(4, 1),
        base_lin_vel=torch.zeros(4, 3), base_ang_vel=torch.zeros(4, 3),
    )
    observer.env = SimpleNamespace(
        reset_buf=torch.tensor([True, True, True, False]),
        time_out_buf=torch.tensor([True, False, False, False]),
        non_failure_reset_buf=torch.tensor([False, False, True, False]),
        cfg=SimpleNamespace(termination=SimpleNamespace(termination_terms=[])),
        episode_length_buf=torch.ones(4), dt=.02, commands=torch.zeros(4, 4),
    )
    observer.termination()
    assert observer.first_episode_censored.tolist() == [True, False, True, True]
    assert observer.first_failure.tolist() == [110, 110, 110, -1]


def test_evaluation_timeout_covers_prefix_without_changing_training_config():
    training_config = SimpleNamespace(env=SimpleNamespace(episode_length_s=20.))
    env = SimpleNamespace(dt=.02, max_episode_length=1000, cfg=copy.deepcopy(training_config))
    steps = extend_evaluation_timeout(env, 20.)
    assert steps == 1100 and env.max_episode_length == 1102
    assert env.max_episode_length_s == pytest.approx(22.04)
    assert training_config.env.episode_length_s == 20.
    # A user-specified longer episode limit must not be shortened either.
    env.max_episode_length = 2000
    extend_evaluation_timeout(env, 20.)
    assert env.max_episode_length == 2000


def test_scenario_uses_existing_samplers_without_mutating_environment_or_rng():
    from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
    from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import GO2HardPACTCfg
    from legged_gym.envs.go2.go2_hard_pact.domain_rand_curriculum import HardPACTDomainRandCurriculum
    cfg = GO2HardPACTCfg()
    cfg.commands.heading_command = False
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.persistent_disturbance = True
    cfg.domain_rand.persistent_force_interval_range_s = [0.02, 0.02]
    cfg.domain_rand.persistent_force_duration_range_s = [0.08, 0.08]
    cfg.domain_rand.persistent_force_probability = 1.
    env = Go2HardPACT.__new__(Go2HardPACT)
    env.cfg, env.num_envs, env.device, env.dt, env.common_step_counter = cfg, 2, "cpu", .02, 0
    env.domain_rand_curriculum = HardPACTDomainRandCurriculum(cfg, 1)
    env.commands = torch.zeros(2, 4)
    env.command_ranges = {name: getattr(cfg.commands.ranges, name) for name in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")}
    for name in ("_persistent_wrench_target_world", "_current_sustained_wrench_world"):
        setattr(env, name, torch.zeros(2, 6))
    env._persistent_component_active = torch.zeros(2, 2, dtype=torch.bool)
    env._current_sustained_active_mask = torch.zeros(2, 1, dtype=torch.bool)
    for name in ("_persistent_start_step", "_persistent_end_step", "_persistent_duration_steps", "_persistent_next_event_step"):
        setattr(env, name, torch.zeros(2, 2, dtype=torch.long))
    env.simulator = SimpleNamespace(**{name: torch.zeros(2, width) for name, width in (
        ("push_timeouts", 1), ("vert_timeouts", 1), ("wrench_timeouts", 1), ("_rand_push_vels", 3), ("_rand_wrench_vels", 3))})
    before = tensor_hash({k: v for k, v in vars(env).items() if torch.is_tensor(v)})
    rng = torch.random.get_rng_state()
    first, second = make_scenario(env, 20, 4), make_scenario(env, 20, 4)
    assert tensor_hash(first) == tensor_hash(second)
    assert tensor_hash({k: v for k, v in vars(env).items() if torch.is_tensor(v)}) == before
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    assert first["wrench_world"].abs().sum() > 0 and first["push_delta_world"].eq(0).all()
