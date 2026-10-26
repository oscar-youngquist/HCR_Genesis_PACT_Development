"""CPU-only contracts; no simulator/registry imports or checkpoint execution."""
import importlib.util
from pathlib import Path
import copy

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_exp1", ROOT / "legged_gym/scripts/eval_b1z1_exp1.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
u = evaluation.util


def scenario(trial=0, condition="combined-disturbance", load_range="out"):
    return u.Scenario(42, trial, condition, load_range, "mixed", evaluation.EVAL_CONFIG)


def test_nested_config_isolation():
    class Parent:
        class commands:
            ranges = [1, 2]
    class Child(Parent):
        pass
    a, b = u.materialize(Child()), u.materialize(Child())
    a.commands.ranges[0] = 5
    assert b.commands.ranges == Parent.commands.ranges == [1, 2]
    with pytest.raises(ValueError, match="Unknown"):
        evaluation.apply_overrides(a, {"typo": True})


def test_cli_and_family():
    a = evaluation.parser().parse_args(["--run", "a", "--ckpt", "123", "--output-dir", "/tmp/test"])
    assert a.ckpt == "123" and a.gpu == "cuda:0"
    for name in ("b1z1_pact", "b1z1_ppo_pos", *[f"b1z1_pact_ab{i}" for i in range(4, 11)]):
        assert evaluation.simulator_family(name) == "isaaclab_b1z1_pact"
    for name in ("b1z1_unifp", "b1z1_unifp_original", "b1z1_unifp_reject"):
        assert evaluation.simulator_family(name) == "isaaclab_b1z1_unifp"


@pytest.mark.parametrize("condition", ["nominal", "ee-disturbance", "combined-disturbance"])
@pytest.mark.parametrize("load_range", ["in", "out"])
def test_scenario_bounds(condition, load_range):
    s = scenario(condition=condition, load_range=load_range)
    bounds = np.array([50] * 3 + [35] * 3 + [10] * 3) if load_range == "in" else np.array([70] * 3 + [55] * 3 + [20] * 3)
    for time in (0., 2., 5., 8., 10.):
        load = s.at(time)[2]
        assert (np.abs(load) <= bounds).all()
        if condition == "nominal" or time < 2 or time >= 8:
            assert not load.any()
        elif condition == "ee-disturbance":
            assert not load[3:].any() and load[:3].any()
        else:
            assert load[:3].any() and load[3:6].any() and load[6:].any()
    if load_range == "out" and condition != "nominal":
        assert any(s.metadata["events"][0]["exceeds_training_bounds"])


def test_keyed_scenarios_reset_order_and_exact_count():
    forward = {i: scenario(i).metadata for batch in u.batches(7, 3) for i in batch}
    reverse = {i: scenario(i).metadata for i in reversed(range(7))}
    assert forward == reverse
    assert list(u.batches(7, 3)) == [[0, 1, 2], [3, 4, 5], [6]]


def test_metric_denominators_and_censored_recovery():
    config = copy.deepcopy(evaluation.EVAL_CONFIG)
    m = u.TrialMetrics(10, 1., scenario(), config)
    m.add(1., [0., 0., 0.], [0., 0.], 0., 0, violations=(1, 2, 3), denominators=(10, 10, 10))
    m.add(2., [.1, 0., 0.], [0., 0.], 0., 1)
    m.add(3., [9., 9., 9.], [9., 9.], 9., 1, valid=False)
    result = m.finish("height")
    assert result["S_track"] == .1
    assert result["ee_error_m"] == .05 and result["valid_steps"] == 2
    assert result["actuator_saturation_frequency"] == .3
    assert result["event_metrics"][0]["iae_m_s"] is None
    assert not result["event_metrics"][0]["recovered"]


def test_writer_bounded_numeric_terminal_rows(tmp_path):
    w = u.NumericWriter(tmp_path, 2)
    for i in range(5):
        w.append(dict(action=np.ones(2) * i, successor=np.ones(2) * (i + 1), terminal=np.bool_(i == 4)))
        assert len(w.rows) < 2
    w.close()
    parts = [np.load(tmp_path / item["file"], allow_pickle=False) for item in w.chunks]
    assert sum(len(p["terminal"]) for p in parts) == 5
    assert parts[-1]["terminal"][-1]
    np.testing.assert_array_equal(parts[-1]["successor"][-1], [5, 5])
    assert "prediction" not in w.schema  # Missing estimates are not fabricated.


def test_inference_logging_parity_and_rng():
    import torch
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(3, 2)
        def act_inference(self, obs, history):
            return self.layer(obs)
    actor = Actor().eval().requires_grad_(False)
    before = evaluation.model_hash(actor)
    obs = torch.ones(2, 3)
    rng = torch.random.get_rng_state().clone()
    a = actor.act_inference(obs, None)
    evaluation.numpy_value(a.clone())
    b = actor.act_inference(obs, None)
    assert torch.equal(a, b) and torch.equal(rng, torch.random.get_rng_state())
    assert before == evaluation.model_hash(actor)


def test_rng_isolation():
    import torch
    before = torch.random.get_rng_state().clone()
    with evaluation.isolated_rng(123, "cpu"):
        first = torch.randn(4)
    with evaluation.isolated_rng(123, "cpu"):
        second = torch.randn(4)
    assert torch.equal(first, second)
    assert torch.equal(before, torch.random.get_rng_state())


def test_registered_inference_contracts_and_strict_loading(monkeypatch):
    import os
    import torch
    from types import SimpleNamespace
    monkeypatch.setenv("SIMULATOR", "isaaclab_b1z1_pact")
    import legged_gym.envs
    from legged_gym.utils import task_registry
    from rsl_rl.b1z1_pact_ablations import B1Z1_PACT_ABLATIONS
    tasks = [key for key in task_registry.env_cfgs if key.startswith("b1z1_")]
    assert {v.task_name for v in B1Z1_PACT_ABLATIONS.values()} <= set(tasks)
    torch.set_num_threads(1)
    for task in tasks:
        c = task_registry.env_cfgs[task].env
        training = u.materialize(task_registry.train_cfgs[task])
        env = SimpleNamespace(num_obs=c.num_observations, num_privileged_obs=c.num_privileged_obs,
                              num_crit_obs_stack=c.num_priv_stack, num_actions=c.num_actions,
                              num_obs_hist=c.num_obs_hist, num_exp_labels=c.num_explicit_recon_obs,
                              num_pred_obs=getattr(c, "num_pred_obs", 0), device="cpu",
                              cfg=task_registry.env_cfgs[task])
        actor = evaluation.build_actor(env, training)
        obs = torch.zeros(2, env.num_obs)
        history = torch.zeros(2, env.num_obs * env.num_obs_hist)
        before = evaluation.model_hash(actor)
        rng = torch.random.get_rng_state().clone()
        a = evaluation.inference(actor, obs, history)
        if hasattr(actor, "decode_context"):
            # Same deterministic extra pass used for logging; it cannot affect actions.
            actor.decode_context(actor.context_encoder.forward_inf(history))
        b = evaluation.inference(actor, obs, history)
        assert a.shape == (2, getattr(c, "num_policy_actions", c.num_actions)), task
        assert torch.equal(a, b) and torch.equal(rng, torch.random.get_rng_state()), task
        assert before == evaluation.model_hash(actor), task
        with pytest.raises(RuntimeError):
            evaluation.build_actor(env, training, {"model_state_dict": {}})


def test_checkpoint_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMULATOR", "isaaclab_b1z1_pact")
    run = tmp_path / "run"
    run.mkdir()
    for number in (2, 10):
        (run / f"model_{number}.pt").touch()
    assert evaluation.resolve_checkpoint(str(run), "2", "unused").name == "model_2.pt"
    assert evaluation.resolve_checkpoint(str(run), "-1", "unused").name == "model_10.pt"
    assert evaluation.resolve_checkpoint(str(run), "model_2.pt", "unused").name == "model_2.pt"
    with pytest.raises(FileNotFoundError):
        evaluation.resolve_checkpoint(str(run), "99", "unused")


def test_motor_scaling_and_substep_clipping_record():
    import torch
    from types import SimpleNamespace as NS
    a = evaluation.EvaluationAdapter.__new__(evaluation.EvaluationAdapter)
    s = NS(default_dof_pos=torch.zeros(1, 2), dof_pos=torch.zeros(1, 2), dof_vel=torch.zeros(1, 2),
           _num_learned_actions=2, _cfg=NS(control=NS(action_scale=.5)),
           feedback_torques=torch.tensor([[10., 20.]]), feedforward_torques=torch.tensor([[5., -2.]]),
           _motor_strength=torch.tensor([[2., .5]]), torque_limits=torch.tensor([10., 20.]),
           dof_pos_limits=torch.tensor([[-1., 1.], [-1., 1.]]), dof_vel_limits=torch.ones(2),
           _gripper_index=1, _base_link_index=0,
           feet_indices=torch.arange(4),
           _robot=NS(data=NS(body_com_pos_w=torch.zeros(1, 4, 3), body_link_pos_w=torch.zeros(1, 4, 3))))
    s.combined_feedback_torques = s.feedback_torques * .8
    s.combined_feedforward_torques = s.feedforward_torques * 1.2
    s.unclipped_torques = s.combined_feedback_torques + s.combined_feedforward_torques
    s._sensor_foot_forces_world = lambda: torch.ones(1, 4, 3)
    a.sim, a.substeps = s, []
    a.body_masses = torch.ones(1, 4)
    a.compute_torques = lambda actions: s.unclipped_torques * s._motor_strength
    a.sensor_update = lambda *args, **kwargs: None
    requested = a.torque(torch.ones(1, 4))
    s.executed_torques = requested.clamp(-1.1 * s.torque_limits, 1.1 * s.torque_limits)
    a.contact_update(.005)
    row = a.substeps[0]
    torch.testing.assert_close(row["torque_request_motor_scaled"], torch.tensor([[28., 6.8]]))
    torch.testing.assert_close(row["executed_torque"], torch.tensor([[11., 6.8]]))
    torch.testing.assert_close(row["pd_weighted_motor"] + row["ff_weighted_motor"], requested)
    assert row["saturation"].tolist() == [[True, False]]


def test_nominal_target_and_application_flags(monkeypatch):
    monkeypatch.setenv("SIMULATOR", "isaaclab_b1z1_pact")
    import legged_gym.envs
    from legged_gym.utils import task_registry
    args = evaluation.parser().parse_args(["--run", "x", "--output-dir", "/tmp/x"])
    for task in ("b1z1_pact", "b1z1_ppo_pos", "b1z1_unifp_original", "b1z1_unifp_reject"):
        original = task_registry.env_cfgs[task]
        before = getattr(original, "use_force_shifted_target", True)
        cfg = evaluation.configure(original, args)
        assert cfg.use_force_shifted_target is False
        assert cfg.commands.apply_ee_external_forces and cfg.commands.apply_base_external_torques
        assert getattr(original, "use_force_shifted_target", True) == before


def test_recovery_dwell_and_full_window():
    c = copy.deepcopy(evaluation.EVAL_CONFIG)
    c.update(recovery_dwell_seconds=.2, recovery_window_seconds=.4)
    m = u.TrialMetrics(100, .1, scenario(), c)
    for time, error in ((2., .2), (2.1, 0.), (2.2, 0.), (2.3, 0.)):
        m.add(time, [error, 0, 0], [0, 0], 0, 1)
    e = m.finish("")["event_metrics"][0]
    assert e["recovered"] and e["recovery"] == pytest.approx(.1)
    assert e["iae_m_s"] == pytest.approx(.02)
