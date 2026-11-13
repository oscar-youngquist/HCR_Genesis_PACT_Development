"""Complete BR CPU runner integration on a small synthetic auto-reset vector env."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import os
import subprocess
import sys
import json

import pytest
import torch

from rsl_rl.beyond_robustness_config import BeyondRobustnessConfig as Config, PHASES
from rsl_rl.beyond_robustness_schema import BeyondRobustnessInputSchema as Schema, InputField
from rsl_rl.beyond_robustness_env import BRObservation, BRStep
from rsl_rl.runners import BeyondRobustnessRunner as Runner
from rsl_rl.utils.adaptive_entropy import performance_gap_entropy
from test_beyond_robustness import cpu_threads, changed


class SyntheticBR:
    """Not payload physics: deterministic named signals, partial failures and timeouts."""
    supports_exact_resume = True

    def __init__(self, schema, n=3):
        self.input_schema, self.num_envs, self.device = schema, n, torch.device("cpu")
        self.rng = torch.Generator().manual_seed(71)
        self.q = torch.zeros(n, 12)
        self.previous = torch.zeros_like(self.q)
        self.age = torch.zeros(n, dtype=torch.long)
        self.clock = 0
        self.resets = 0
        self.curriculum_stage = 1
        self.progress = .2
        self.last_packet = None

    def _reset(self, ids):
        self.q[ids] = torch.randn(len(ids), 12, generator=self.rng) * .1
        self.previous[ids] = 0
        self.age[ids] = 0

    def reset(self):
        self.resets += 1
        self._reset(torch.arange(self.num_envs))
        return self.observe()

    def observe(self):
        schema, n = self.input_schema, self.num_envs
        prop = {f.name: torch.zeros(n, f.width) for f in schema.fields("observations")}
        prop.update(joint_position=self.q.clone(), joint_velocity=self.previous * .1,
                    previous_action=self.previous.clone(), command_B=torch.tensor([.5, 0., .2]).repeat(n, 1))
        prop["gravity_B"][:, 2] = -1
        obs = schema.pack("observations", prop)
        state = {f.name: torch.ones(n, f.width) * .03 for f in schema.state_fields}
        state["proprioception"] = obs
        state["linear_velocity_B"] = self.q[:, :3] * .1
        params = {f.name: torch.ones(n, f.width) for f in schema.parameter_fields}
        load = torch.zeros(n, 7)
        load[:, 0] = self.age * .1
        load[:, 3] = self.previous[:, 0] * .02
        load[:, 6] = 2 + self.progress
        packet = BRObservation(obs, schema.pack("state", state), schema.pack("parameters", params),
                               load, self.age != 1)
        self.last_packet = packet
        return packet

    def step(self, actions):
        # In-place validity mutation emulates a resettable simulator workspace.
        if self.last_packet is not None:
            self.last_packet.supervision_valid.fill_(False)
        self.previous.copy_(actions)
        self.q += .01 * actions
        self.age += 1
        self.clock += 1
        done = self.age >= torch.arange(2, 2 + self.num_envs)
        timeout = done & (torch.arange(self.num_envs) != 1)
        terminal = self.observe()
        self._reset(done.nonzero(as_tuple=False).flatten())
        reward = torch.exp(-actions.square().mean(-1))
        return BRStep(self.observe(), reward, done, timeout, terminal, done.clone(),
                      {"lin_vel_tracking": .4, "ang_vel_tracking": .2, "terrain_level": 2.})

    def curriculum_state_dict(self):
        return dict(stage=self.curriculum_stage, progress=self.progress)

    def load_curriculum_state_dict(self, data):
        self.curriculum_stage, self.progress = data["stage"], data["progress"]

    def state_dict(self):
        return dict(q=self.q.clone(), previous=self.previous.clone(), age=self.age.clone(),
                    clock=self.clock, resets=self.resets, rng=self.rng.get_state(), curriculum=self.curriculum_state_dict())

    def load_state_dict(self, data):
        for key in ("q", "previous", "age"): getattr(self, key).copy_(data[key])
        self.clock, self.resets = data["clock"], data["resets"]
        self.rng.set_state(data["rng"])
        self.load_curriculum_state_dict(data["curriculum"])


class Writer:
    def __init__(self): self.scalars, self.text = [], []
    def add_scalar(self, name, value, step): self.scalars.append((name, value, step))
    def add_text(self, name, value, step): self.text.append((name, value, step))
    def flush(self): pass


def make(phase="teacher_student", schema=None, log_dir=None, **kwargs):
    schema = schema or Schema(terrain_points_x=(0., .2), terrain_points_y=(0.,))
    config = Config(input_schema=schema, training_phase=phase, history_length=3,
                    num_envs=3, num_steps_per_env=3, num_learning_epochs=1,
                    num_mini_batches=2, auxiliary_epochs=1, **kwargs)
    env = SyntheticBR(schema)
    return Runner(env, config, log_dir, writer=Writer())


@pytest.mark.parametrize("phase", PHASES)
def test_complete_runner_updates_logging_and_ownership(phase):
    r = make(phase)
    before = {name: deepcopy(getattr(r.alg.actor_critic, name).state_dict()) for name in
              ("actor", "critic", "privileged_encoder", "history_encoder", "load_estimator")}
    with patch.object(r.alg.history, "append", wraps=r.alg.history.append) as append:
        logs = r.learn(2)
        assert append.call_count == 6  # bootstrap never appends
    assert r.interactions == 18 and r.control_steps == 6 and r.alg.iteration == 2
    for name in before:
        assert changed(before[name], getattr(r.alg.actor_critic, name)) == (name != "privileged_encoder" or phase == "teacher_student")
    for metrics in logs:
        assert all(torch.isfinite(torch.tensor(x)) for x in metrics.values())
        assert "load/load_mae/com_x_m" in metrics and "load/load_rmse/mass_kg" in metrics
        assert metrics["train/entropy_schedule_updated"] == 1
    assert r.env.curriculum_stage == 1  # learning mode never chooses physical curriculum
    assert [entry[1] for entry in r.writer.text] == [phase, phase]
    policy = r.get_inference_policy()
    current = r.env.observe()
    assert policy(current.observations, r.alg.history.frames.flatten(1)).shape == (3, 12)


@pytest.mark.parametrize("phase", PHASES)
def test_runner_replay_validity_and_timeout_alignment(phase):
    r = make(phase)
    captured = []
    original = r.alg.bootstrap_values
    def bootstrap(*args, **kwargs):
        before = r.alg.history.frames.clone()
        rng = torch.get_rng_state().clone()
        value = original(*args, **kwargs)
        assert torch.equal(before, r.alg.history.frames) and torch.equal(rng, torch.get_rng_state())
        if kwargs.get("env_ids") is not None:
            captured.append((r.alg.storage.step, kwargs["env_ids"].clone(), value.clone()))
        return value
    with patch.object(r.alg, "bootstrap_values", side_effect=bootstrap):
        r.collect_rollout()
    assert r.alg.storage.supervision_valid[0].all()  # not overwritten by env.step's mask mutation
    assert not r.alg.storage.supervision_valid[1].any()
    assert r.alg.storage.histories[2, 0, :-45].eq(0).all()
    assert captured and captured[0][0] == 1
    for k, ids, values in captured:
        actions = r.alg.storage.actions[k, ids]
        reward = torch.exp(-actions.square().mean(-1))
        torch.testing.assert_close(r.alg.storage.rewards[k, ids, 0], reward + r.alg.gamma * values[:, 0])
    for batch in r.alg.storage.batches(2, 2):
        dist, _ = r.alg.actor_critic.evaluate_batch(batch, phase)
        logp = dist.log_prob(batch["actions"]).sum(-1)
        torch.testing.assert_close(logp, batch["log_prob"].flatten(), rtol=0, atol=2e-6)
        assert (batch["state"][:, :45] == batch["observations"]).all()


@pytest.mark.parametrize("missing", ["packet", "mask", "row"])
def test_timeout_requires_explicit_pre_reset_inputs(missing):
    r = make()
    step = r.env.step
    def broken(actions):
        result = step(actions)
        if missing == "packet": result.terminal = None
        if missing == "mask": result.terminal_valid = None
        if missing == "row": result.terminal_valid.zero_()
        return result
    with patch.object(r.env, "step", side_effect=broken), pytest.raises(ValueError, match="terminal"):
        r.learn(1)


def test_schedule_switch_resume_and_no_optimizer_reset(tmp_path):
    r = make(student_reinforcement_start_iteration=2)
    assert r.cfg.training_phase == "teacher_student"
    r.learn(1)
    r.save(tmp_path / "boundary.pt")
    restored = make(student_reinforcement_start_iteration=2)
    restored.load(tmp_path / "boundary.pt", resume_mode="exact")
    logs = restored.learn(2)
    assert [x["train/phase_student_reinforcement"] for x in logs] == [0., 1.]
    # Switching itself neither rebuilds optimizers nor changes weights/moments.
    opt = restored.alg.optimizer
    state = deepcopy(opt.state_dict())
    model = deepcopy(restored.alg.actor_critic.state_dict())
    restored.set_training_phase("teacher_student")
    restored.alg.apply_phase_schedule()
    assert restored.alg.optimizer is opt and restored.alg.training_phase == "student_reinforcement"
    for key, value in model.items(): torch.testing.assert_close(value, restored.alg.actor_critic.state_dict()[key], rtol=0, atol=0)
    for key, values in state["state"].items():
        for name, value in values.items(): torch.testing.assert_close(value, opt.state_dict()["state"][key][name], rtol=0, atol=0)
    restored.save(tmp_path / "continued" / "student.pt")
    next_runner = make(student_reinforcement_start_iteration=2)
    next_runner.load(tmp_path / "continued" / "student.pt", resume_mode="exact")
    assert next_runner.alg.training_phase == "student_reinforcement"
    assert next_runner.alg.entropy_coef == restored.alg.entropy_coef


def test_default_no_switch_and_explicit_student():
    r = make()
    r.alg.iteration = 9999
    r.alg.apply_phase_schedule()
    assert r.alg.training_phase == "teacher_student"
    r.set_training_phase("student_reinforcement")
    assert r.alg.training_phase == "student_reinforcement"
    r.collect_rollout()
    with pytest.raises(RuntimeError, match="completed"):
        r.set_training_phase("teacher_student")


def test_exact_resume_restores_environment_and_next_update(tmp_path):
    r = make(student_reinforcement_start_iteration=1)
    r.learn(1)
    r.env.curriculum_stage, r.env.progress = 2, .7
    r.save(tmp_path / "exact.pt")
    expected = r.learn(1)
    restored = make(student_reinforcement_start_iteration=1)
    restored.load(tmp_path / "exact.pt", resume_mode="exact")
    assert restored.env.curriculum_stage == 2 and restored.env.progress == .7
    actual = restored.learn(1)
    # Wall timings are observations, not part of deterministic continuation.
    strip=lambda logs:[{k:v for k,v in log.items() if not k.endswith('_wall_s')} for log in logs]
    assert strip(actual) == strip(expected)
    for key, value in r.alg.actor_critic.state_dict().items():
        torch.testing.assert_close(value, restored.alg.actor_critic.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(r.env.q, restored.env.q, rtol=0, atol=0)
    assert torch.equal(r.alg.reset_pending, restored.alg.reset_pending)


def test_fresh_resume_clears_stale_history_but_retains_optimization_and_curriculum(tmp_path):
    r = make()
    r.learn(1)
    r.env.curriculum_stage = 2
    r.save(tmp_path / "fresh.pt")
    restored = make()
    restored.load(tmp_path / "fresh.pt")  # safe default
    assert restored.alg.iteration == 1 and restored.interactions == r.interactions
    assert restored.env.resets == 1 and restored.env.curriculum_stage == 2
    assert restored.alg.history.frames.eq(0).all() and restored.alg.reset_pending.all()
    assert restored.episode_lengths.eq(0).all() and restored.env.age.eq(0).all()
    for key, state in r.alg.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, restored.alg.optimizer.state_dict()["state"][key][name], rtol=0, atol=0)
    restored.collect_rollout()
    assert restored.alg.storage.histories[0, :, :-45].eq(0).all()


def test_exact_resume_refuses_incomplete_state(tmp_path):
    r = make()
    r.env.supports_exact_resume = False
    r.learn(1)
    r.save(tmp_path / "not_exact.pt")
    with pytest.raises(ValueError, match="complete"):
        make().load(tmp_path / "not_exact.pt", resume_mode="exact")


def test_configurable_schema_pack_and_runner_validation(tmp_path):
    base = Schema()
    parameters = tuple(replace(f, width=1) if f.name == "motor_strength" else f for f in base.parameter_fields)
    schema = Schema.from_terrain(dict(measure_heights=True, measured_points_x=[-.1, .1], measured_points_y=[-.2, 0., .2]),
                                 parameter_fields=parameters)
    assert schema.state_dim == 45 + 3 + 6 + 12 + 12 + 12
    assert schema.parameter_dim == base.parameter_dim - 11
    r = make(schema=schema, log_dir=tmp_path)
    r.learn(1, save_interval=1)
    assert r.alg.actor_critic.privileged_encoder[0].in_features == schema.privileged_dim
    assert r.alg.storage.privileged_observations.shape[-1] == schema.privileged_dim
    assert (tmp_path / "model_1.pt").exists()
    assert (tmp_path / "beyond_robustness_resolved_config.json").exists()
    with pytest.raises(ValueError, match="fields"):
        schema.pack("parameters", {"arbitrary_padding": torch.zeros(3, 62)})
    cfg = replace(r.cfg, input_schema=base)
    with pytest.raises(ValueError, match="schema"):
        Runner(r.env, cfg)
    # No terrain samples when disabled, not a zero-filled paper-sized vector.
    assert Schema.from_terrain(dict(measure_heights=False)).state_dim == 84


def test_adaptive_entropy_inheritance_formula_and_resume(tmp_path):
    cfg = Config()
    alg = SimpleNamespace(**cfg.resolved())
    alg.adaptive_ent_bounds = [.003, .02]
    alg.adaptive_ent_lin_threshold, alg.adaptive_ent_ang_threshold = .8, .4
    alg.adaptive_ent_ter_threshold, alg.adaptive_ent_softmax_temp = 7., 1.5
    env_cfg = SimpleNamespace(env=SimpleNamespace(num_envs=3, num_obs_hist=3),
                             terrain=SimpleNamespace(measure_heights=False))
    train_cfg = SimpleNamespace(algorithm=alg, policy=SimpleNamespace(init_noise_std=1., min_noise_std=.1, max_noise_std=2.),
                               runner=SimpleNamespace(num_steps_per_env=3))
    inherited = Config.from_hardpact(env_cfg, train_cfg)
    for name in ("use_adaptive_entropy", "adaptive_ent_lin_threshold", "adaptive_ent_ang_threshold",
                 "adaptive_ent_ter_threshold", "adaptive_ent_softmax_temp"):
        assert getattr(inherited, name) == getattr(alg, name)
    assert inherited.adaptive_ent_bounds == tuple(alg.adaptive_ent_bounds)
    for metrics in ({}, dict(lin_vel_tracking=.4, ang_vel_tracking=.1, terrain_level=2.),
                    dict(lin_vel_tracking=1., ang_vel_tracking=1., terrain_level=10.)):
        thresholds = (.75, .35, 6.)
        gaps = torch.tensor([max(0, t - metrics.get(k, 0.)) / t for t, k in
                            zip(thresholds, ("lin_vel_tracking", "ang_vel_tracking", "terrain_level"))], dtype=torch.float32)
        expected = .005 + (torch.softmax(gaps / 2., dim=0) * gaps).sum().item() * (.01 - .005)
        assert performance_gap_entropy(metrics, (.005, .01), thresholds, 2.) == expected
    r = make()
    initial = r.alg.entropy_coef
    assert not r.alg.update_adaptive_entropy_coef({}) and r.alg.entropy_coef == initial
    assert not r.alg.update_adaptive_entropy_coef(dict(lin_vel_tracking=float("nan"), ang_vel_tracking=1., terrain_level=1.))
    logs = r.learn(2)
    assert logs[0]["train/entropy_coefficient"] == initial
    assert logs[1]["train/entropy_coefficient"] == logs[0]["train/entropy_coefficient_next"]
    disabled = make(use_adaptive_entropy=False)
    assert not disabled.alg.update_adaptive_entropy_coef(dict(lin_vel_tracking=0., ang_vel_tracking=0., terrain_level=0.))


def test_runner_import_and_registry_without_simulator():
    env = dict(os.environ)
    env.pop("SIMULATOR", None)
    script = ("import sys; from rsl_rl.utils.runner_registry import runner_registry; "
              "cls = runner_registry.get_runner_class('BeyondRobustnessRunner'); "
              "from rsl_rl.runners import BeyondRobustnessRunner; "
              "assert cls is BeyondRobustnessRunner; "
              "assert runner_registry.get_runner_class('BeyondRobustnessRunner') is BeyondRobustnessRunner; "
              "assert 'legged_gym' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_load_errors_are_valid_sample_weighted_not_minibatch_weighted():
    r = make(auxiliary_learning_rate=0.)
    r.collect_rollout()
    s = r.alg.storage
    valid = s.supervision_valid.flatten()
    h, target = s.histories.flatten(0, 1)[valid], s.load_labels.flatten(0, 1)[valid]
    with torch.no_grad():
        error = r.alg.actor_critic.load_estimator(h) - target
        mae, rmse = error.abs().mean(0), error.square().mean(0).sqrt()
    metrics = r.alg.update()
    names = ("com_x_m", "com_y_m", "com_z_m", "velocity_x_m_s", "velocity_y_m_s", "velocity_z_m_s", "mass_kg")
    for k, name in enumerate(names):
        assert metrics[f"load_mae/{name}"] == pytest.approx(mae[k].item(), abs=1e-6)
        assert metrics[f"load_rmse/{name}"] == pytest.approx(rmse[k].item(), abs=1e-6)


def test_resolved_json_roundtrip_and_semantic_schema_mismatch(tmp_path):
    r = make()
    assert Config(**json.loads(json.dumps(r.cfg.resolved()))) == r.cfg
    r.learn(1)
    r.save(tmp_path / "schema.pt")
    # Same widths but different order are incompatible, not an implicit remapping.
    wrong = replace(r.cfg.input_schema, joint_order=tuple(reversed(r.cfg.input_schema.joint_order)))
    other = make(schema=wrong)
    with pytest.raises(ValueError, match="contract"):
        other.load(tmp_path / "schema.pt")
