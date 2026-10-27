"""Visual launcher contracts without importing Isaac Lab or constructing a simulator."""
import importlib.util
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("visual_exp2", ROOT / "legged_gym/scripts/play_b1z1_exp2.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


@pytest.mark.parametrize("approach,task,backend", [
    ("PILOT", "b1z1_pact", "isaaclab_b1z1_pact"),
    ("pos", "b1z1_ppo_pos", "isaaclab_b1z1_pact"),
    ("unifp", "b1z1_unifp_original", "isaaclab_b1z1_unifp"),
    ("b1z1_pact_ab7_position_full", "b1z1_pact_ab7_position_full", "isaaclab_b1z1_pact"),
])
def test_dispatch_and_single_robot(approach, task, backend):
    args = p.parser().parse_args(["--approach", approach, "--run", "example", "--profile", "pulling", "--ckpt", "123"])
    p.validate(args)
    assert args.task == task and args.num_envs == 1
    assert p.exp1.simulator_family(args.task) == backend
    assert args.ckpt == "123" and args.follow_robot and not args.headless


@pytest.mark.parametrize("profile", ["pushing", "pulling", "carrying"])
def test_reuses_current_exp2_schedules(profile):
    args = p.parser().parse_args(["--run", "example", "--profile", profile])
    p.validate(args)
    visual = p.exp2.tasks.Scenario(args.seed, 0, profile, "loaded", p.exp2.EVAL_CONFIG)
    expected = p.exp2.Experiment2().scenario(args, 0)
    for t in [0., *p.exp2.EVAL_CONFIG["load_times_s"], p.exp2.EVAL_CONFIG["stop_s"]]:
        for a, b in zip(visual.at(t), expected.at(t)):
            np.testing.assert_array_equal(a, b)


def test_visual_adapter_skips_recording_not_control():
    calls = []
    owner = SimpleNamespace(compute_torques=lambda *a, **k: calls.append((a, k)) or 7,
                            sensor_update=lambda *a, **k: calls.append((a, k)) or 8)
    assert p.VisualAdapter.torque(owner, "actions", motor=True) == 7
    assert p.VisualAdapter.contact_update(owner, .01) == 8
    assert calls == [(("actions",), {"motor": True}), ((.01,), {})]


def test_independent_profile_elevations():
    cfg = copy.deepcopy(p.exp2.EVAL_CONFIG)
    cfg["profiles"]["pushing"]["elevation_rad"] = .2
    cfg["profiles"]["pulling"]["elevation_rad"] = -.1
    cfg["profiles"]["carrying"]["elevation_rad"] = .3
    p.exp2.tasks.validate(cfg, cfg["finish_s"])
    scenarios = {name: p.exp2.tasks.Scenario(0, 0, name, "loaded", cfg) for name in cfg["profiles"]}
    for name, scenario in scenarios.items():
        assert scenario.at(0.)[1][1] == cfg["profiles"][name]["elevation_rad"]
    for name in ("pushing", "pulling"):
        assert scenarios[name].at(cfg["finish_s"])[1][1] == cfg["profiles"][name]["elevation_rad"]
    cfg["profiles"]["pushing"]["elevation_rad"] = .4
    for name, scenario in scenarios.items():
        updated = p.exp2.tasks.Scenario(0, 0, name, "loaded", cfg)
        assert (updated.metadata["motion_hash"] != scenario.metadata["motion_hash"]) == (name == "pushing")


def test_invalid_profile_elevation():
    cfg = copy.deepcopy(p.exp2.EVAL_CONFIG)
    cfg["profiles"]["pulling"]["elevation_rad"] = float("nan")
    with pytest.raises(ValueError):
        p.exp2.tasks.validate(cfg, cfg["finish_s"])


def test_relative_heading_feedback_and_wrap():
    cfg = p.exp2.EVAL_CONFIG
    initial = 1.2
    yaw = np.array([initial, initial+.2, initial-.2, initial+2*np.pi-.2])
    result = p.exp2.tasks.heading_yaw_rate(yaw, initial, cfg)
    np.testing.assert_allclose(result, [0., -.6*.2/np.pi, .6*.2/np.pi, .6*.2/np.pi], atol=1e-14)
    assert np.max(np.abs(p.exp2.tasks.heading_yaw_rate(np.linspace(-20, 20, 100), initial, cfg))) <= .6


def test_heading_recording_uses_both_endpoint_states():
    scenario = p.exp2.tasks.Scenario(0, 0, "pushing", "loaded", p.exp2.EVAL_CONFIG)
    scenario.set_heading(1.)
    quat = lambda yaw: np.array([0., 0., np.sin(yaw/2), np.cos(yaw/2)])
    row = dict(base_quaternion_t=quat(1.), base_quaternion_xyzw=quat(1.2))
    p.exp2.Experiment2.augment_row(row, scenario, 0., .02)
    assert row["command_nominal_t"][2] == pytest.approx(0.)
    assert row["command_nominal"][2] == pytest.approx(-.6*.2/np.pi)


def test_heading_adapter_capture_and_policy_command(monkeypatch):
    torch = pytest.importorskip("torch")
    cfg = p.exp2.EVAL_CONFIG
    scenario = p.exp2.tasks.Scenario(0, 0, "pushing", "loaded", cfg)
    adapter = object.__new__(p.exp2.Adapter)
    adapter.config, adapter.scenarios, adapter.step_index = cfg, [scenario], -1
    quat = lambda yaw: torch.tensor([[0., 0., np.sin(yaw/2), np.cos(yaw/2)]])
    adapter.sim = SimpleNamespace(base_quat=quat(1.))
    adapter.env = SimpleNamespace(commands=torch.zeros(1, 6))
    monkeypatch.setattr(p.exp1.EvaluationAdapter, "publish_targets", lambda self, time: None)
    adapter.publish_targets(0.)
    assert scenario.heading == pytest.approx(1.)
    assert adapter.env.commands[0, 2] == 0.
    adapter.step_index = 0
    adapter.sim.base_quat = quat(1.2)
    adapter.publish_targets(.02)
    assert scenario.heading == pytest.approx(1.)
    assert adapter.env.commands[0, 2].item() == pytest.approx(-.6*.2/np.pi)


@pytest.mark.parametrize("arguments", [["--speed", "0"], ["--episodes", "0"], ["--max-steps", "0"], ["--force-arrow-scale", "nan"]])
def test_invalid_controls(arguments):
    with pytest.raises(ValueError):
        p.validate(p.parser().parse_args(["--run", "example", *arguments]))
