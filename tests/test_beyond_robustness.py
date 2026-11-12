"""BR Stage 1: real network/PPO/optimizer tests on CPU, no simulator or QP."""
from copy import deepcopy
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rsl_rl.beyond_robustness_config import BeyondRobustnessConfig as Config, PHASES
from rsl_rl.modules import ActorCriticBeyondRobustness as Model
from rsl_rl.algorithms import PPOBeyondRobustness as Algorithm


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(old)


def make(phase="teacher_student", **changes):
    cfg = Config(history_length=3, num_envs=3, num_steps_per_env=3,
                 num_learning_epochs=2, num_mini_batches=2, auxiliary_epochs=2,
                 training_phase=phase, **changes)
    return Algorithm(Model(cfg), cfg)


def inputs(n=3):
    return [torch.randn(n, width) for width in (45, 264, 62, 7)]


def collect(alg):
    for k in range(alg.config.num_steps_per_env):
        o, s, p, labels = inputs(alg.storage.num_envs)
        alg.act(o, s, p, labels)
        done = torch.zeros(alg.storage.num_envs, dtype=torch.bool)
        done[0] = k == 1
        alg.process_env_step(torch.randn(alg.storage.num_envs), done)
    alg.compute_returns(*inputs(alg.storage.num_envs))


def changed(before, module):
    return any(not torch.equal(before[k], v) for k, v in module.state_dict().items())


def has_grad(module):
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    return any(torch.count_nonzero(g) > 0 for g in grads)


def test_dimensions_normalization_and_student_inputs():
    model = Model(Config(history_length=4))
    widths = {"actor": [84, 512, 256, 128, 12], "critic": [365, 512, 256, 128, 1],
              "privileged_encoder": [326, 512, 256, 128, 32],
              "history_encoder": [180, 512, 256, 128, 32],
              "load_estimator": [180, 512, 256, 64, 7]}
    for name, expected in widths.items():
        layers = [m for m in getattr(model, name) if isinstance(m, nn.Linear)]
        assert [layers[0].in_features] + [m.out_features for m in layers] == expected
        assert sum(isinstance(m, nn.ELU) for m in getattr(model, name)) == 3
    o, s, p, labels = inputs()
    h = torch.randn(3, 180)
    torch.testing.assert_close(model.teacher_latent(s, p).norm(dim=-1), torch.ones(3))
    student = model.act_inference(o, h)
    dist = model.distribution(o, h, "teacher_student", s, p)
    assert dist.sample().shape == (3, 12)
    assert dist.log_prob(dist.sample()).sum(-1).shape == (3,)
    assert dist.entropy().sum(-1).shape == (3,)
    # Even supplied privileged values must have no effect in student mode.
    torch.testing.assert_close(student, model.distribution(o, h, "student_reinforcement",
                                                        s * float("nan"), p * float("nan")).mean)
    with torch.no_grad():
        for param in model.privileged_encoder.parameters(): param.fill_(float("nan"))
        for param in model.critic.parameters(): param.fill_(float("nan"))
    torch.testing.assert_close(student, model.act_inference(o, h))


@pytest.mark.parametrize("phase", PHASES)
def test_actor_uses_estimated_load_not_labels(phase):
    a = make(phase)
    collect(a)
    batch = next(a.storage.batches(1))
    before, _ = a.actor_critic.evaluate_batch(batch, phase)
    altered = dict(batch, load_labels=batch["load_labels"] + 100)
    after, _ = a.actor_critic.evaluate_batch(altered, phase)
    torch.testing.assert_close(before.mean, after.mean, rtol=0, atol=0)
    with torch.no_grad(): a.actor_critic.load_estimator[-1].bias.add_(1)
    changed_dist, _ = a.actor_critic.evaluate_batch(batch, phase)
    assert not torch.equal(before.mean, changed_dist.mean)


@pytest.mark.parametrize("phase", PHASES)
def test_gradient_ownership_and_likelihood_replay(phase):
    a = make(phase)
    collect(a)
    for batch in a.storage.batches(2, 3):
        dist, _ = a.actor_critic.evaluate_batch(batch, phase)
        logp = dist.log_prob(batch["actions"]).sum(-1)
        torch.testing.assert_close(logp, batch["log_prob"].squeeze(-1), atol=2e-6, rtol=0)
        ratio = (logp - batch["log_prob"].squeeze(-1)).exp()
        torch.testing.assert_close(ratio, torch.ones_like(ratio), atol=2e-6, rtol=0)
        kl = torch.distributions.kl_divergence(torch.distributions.Normal(batch["mu"], batch["sigma"]), dist).sum(-1)
        assert kl.abs().max() < 1e-6
    a.actor_critic.zero_grad(set_to_none=True)
    loss, _ = a.ppo_loss(batch)
    loss.backward()
    assert has_grad(a.actor_critic.actor) and has_grad(a.actor_critic.critic)
    assert has_grad(a.actor_critic.privileged_encoder) == (phase == "teacher_student")
    assert has_grad(a.actor_critic.history_encoder) == (phase == "student_reinforcement")
    assert not has_grad(a.actor_critic.load_estimator)
    a.actor_critic.zero_grad(set_to_none=True)
    matching, load = a.auxiliary_losses(batch)
    if matching is not None:
        matching.backward()
        assert has_grad(a.actor_critic.history_encoder)
        assert not has_grad(a.actor_critic.privileged_encoder)
    load.backward()
    assert has_grad(a.actor_critic.load_estimator)
    assert not has_grad(a.actor_critic.actor) and not has_grad(a.actor_critic.critic)


def test_squared_l2_not_feature_mean_and_squared_load_weights():
    a = make()
    collect(a)
    b = next(a.storage.batches(1))
    matching, load = a.auxiliary_losses(b)
    predicted = a.actor_critic.load_estimator(b["history"])
    weights = torch.tensor([3, 3, 3, 1, 1, 1, 10])
    torch.testing.assert_close(load, ((predicted - b["load_labels"]) * weights).square().sum(-1).mean())
    z = a.actor_critic.teacher_latent(b["state"], b["parameters"])
    torch.testing.assert_close(matching, (a.actor_critic.history_encoder(b["history"]) - z).square().sum(-1).mean())
    b["valid"][:] = False
    b["history"][:] = float("nan")
    assert a.auxiliary_losses(b) == (None, None)


@pytest.mark.parametrize("phase", PHASES)
def test_actual_finite_updates_and_inactive_encoder_unchanged(phase):
    a = make(phase)
    before = {key: deepcopy(getattr(a.actor_critic, key).state_dict()) for key in
              ("actor", "critic", "privileged_encoder", "history_encoder", "load_estimator")}
    collect(a)
    metrics = a.update()
    assert all(torch.isfinite(torch.tensor(x)) for x in metrics.values())
    for key in before:
        expected = not (phase == "student_reinforcement" and key == "privileged_encoder")
        assert changed(before[key], getattr(a.actor_critic, key)) == expected
    assert a.iteration == 1 and a.storage.step == 0
    assert all(torch.isfinite(p).all() for p in a.actor_critic.parameters())


def test_timestamp_copies_partial_resets_and_complete_shuffle():
    a = make()
    snapshots = []
    for k in range(3):
        values = [torch.full((3, width), float(k + 1)) for width in (45, 264, 62, 7)]
        a.act(*values)
        snapshots.append(deepcopy(a.pending))
        for value in values: value.fill_(-900)  # emulate simulator workspace mutation
        a.process_env_step(torch.ones(3), torch.tensor([k == 0, False, False]))
    assert torch.equal(a.storage.histories[1, 0, :-45], torch.zeros(90))
    assert torch.equal(a.storage.histories[1, 1, -90:-45], torch.ones(45))
    for k in range(3):
        assert torch.equal(a.storage.load_labels[k], snapshots[k]["load_labels"])
        assert torch.equal(a.storage.observations[k], snapshots[k]["observations"])
        assert torch.equal(a.storage.histories[k], snapshots[k]["history"])
    batches = list(a.storage.batches(4))  # 9 rows, remainder must not be discarded
    assert sum(len(b["actions"]) for b in batches) == 9
    for b in batches:
        assert torch.equal(b["observations"][:, 0], b["load_labels"][:, 0])
        assert torch.equal(b["observations"][:, 0], b["history"][:, -1])
        assert torch.equal(b["state"][:, 0], b["parameters"][:, 0])


def test_timeouts_bootstrap_terminal_not_reset_state_and_cut_gae():
    a = make()
    a.act(*inputs())
    with pytest.raises(ValueError, match="terminal_values"):
        a.process_env_step(torch.ones(3), torch.tensor([True, True, False]),
                           timeouts=torch.tensor([True, False, False]))
    a.process_env_step(torch.ones(3), torch.tensor([True, True, False]),
                       timeouts=torch.tensor([True, False, False]), terminal_values=torch.tensor([7., 999., 999.]))
    for _ in range(2):
        a.act(*inputs())
        a.process_env_step(torch.ones(3) * 100, torch.zeros(3, dtype=torch.bool))
    a.compute_returns(*inputs())
    torch.testing.assert_close(a.storage.returns[0, 0, 0], torch.tensor(1 + a.gamma * 7))
    torch.testing.assert_close(a.storage.returns[0, 1, 0], torch.tensor(1.))
    assert a.storage.supervision_valid[0].all()  # labels belong to valid pre-terminal t
    assert not a.storage.histories[1, :2, :-45].any()


@pytest.mark.parametrize("phase", PHASES)
def test_terminal_bootstrap_is_history_and_rng_side_effect_free(phase):
    a = make(phase)
    a.act(*inputs())
    before = a.history.frames.clone()
    o, s, p, labels = inputs()
    rng = torch.get_rng_state().clone()
    h = a.history.preview(o, a.reset_pending).flatten(1)
    z = a.actor_critic.latent(h, phase, s, p)
    expected = a.actor_critic.evaluate(s, p, labels, z)
    actual = a.bootstrap_values(o, s, p, labels)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(a.history.frames, before)
    assert torch.equal(torch.get_rng_state(), rng)


def test_auxiliary_empty_update_preserves_optimizer_moments():
    a = make()
    collect(a)
    a.update()  # initialize Adam moments before testing skipped updates
    before = deepcopy(a.actor_critic.load_estimator.state_dict())
    history = deepcopy(a.actor_critic.history_encoder.state_dict())
    states = deepcopy(a.load_optimizer.state_dict())
    history_states = deepcopy(a.history_optimizer.state_dict())
    collect(a)
    a.storage.supervision_valid.zero_()
    metrics = a.update()
    assert metrics["latent_matching"] == metrics["load_estimation"] == 0
    assert not changed(before, a.actor_critic.load_estimator)
    assert not changed(history, a.actor_critic.history_encoder)
    for key, state in states["state"].items():
        for field, val in state.items(): torch.testing.assert_close(val, a.load_optimizer.state_dict()["state"][key][field])
    for key, state in history_states["state"].items():
        for field, val in state.items(): torch.testing.assert_close(val, a.history_optimizer.state_dict()["state"][key][field])


@pytest.mark.parametrize("phase", PHASES)
def test_checkpoint_roundtrip_and_exact_next_update(tmp_path, phase):
    a = make(phase)
    collect(a)
    a.update()
    path = tmp_path / "br.pt"
    a.save(path)
    b = make(phase)
    assert b.load(path) == a.iteration
    assert b.training_phase == phase
    for key, value in a.actor_critic.state_dict().items():
        torch.testing.assert_close(value, b.actor_critic.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(a.history.frames, b.history.frames)
    rng = torch.get_rng_state()
    collect(a)
    metrics_a = a.update()
    torch.set_rng_state(rng)
    collect(b)
    metrics_b = b.update()
    assert metrics_a == metrics_b
    for key, value in a.actor_critic.state_dict().items():
        torch.testing.assert_close(value, b.actor_critic.state_dict()[key], rtol=0, atol=0)
    for name in ("optimizer", "history_optimizer", "load_optimizer"):
        for key, state in getattr(a, name).state_dict()["state"].items():
            for field, val in state.items():
                torch.testing.assert_close(val, getattr(b, name).state_dict()["state"][key][field], rtol=0, atol=0)


def test_phase_switch_checkpoint_and_boundary_guard(tmp_path):
    a = make()
    a.act(*inputs())
    with pytest.raises(RuntimeError): a.set_training_phase("student_reinforcement")
    with pytest.raises(RuntimeError): a.save(tmp_path / "partial.pt")
    a.pending = None
    collect(a)
    a.update()
    teacher = deepcopy(a.actor_critic.privileged_encoder.state_dict())
    a.set_training_phase("student_reinforcement")
    collect(a)
    a.update()
    assert not changed(teacher, a.actor_critic.privileged_encoder)
    a.save(tmp_path / "student.pt")
    b = make()
    b.load(tmp_path / "student.pt")
    assert b.training_phase == "student_reinforcement"


def test_inherit_allowlisted_config_only():
    cfg = Config()
    algorithm = SimpleNamespace(**cfg.resolved(), hard_pact_qp={"enabled": True}, pinn_loss_weight=900)
    env = SimpleNamespace(env=SimpleNamespace(num_obs_hist=9, num_envs=123, num_observations=57))
    policy = SimpleNamespace(init_noise_std=list(range(1, 25)), min_noise_std=.1, max_noise_std=30.)
    train = SimpleNamespace(algorithm=algorithm, policy=policy, runner=SimpleNamespace(num_steps_per_env=11))
    inherited = Config.from_hardpact(env, train)
    assert inherited.history_length == 9 and inherited.num_envs == 123 and inherited.num_steps_per_env == 11
    assert inherited.init_noise_std == tuple(range(1, 13))
    assert inherited.auxiliary_learning_rate == cfg.auxiliary_learning_rate
    policy.init_noise_std[0] = 999
    assert inherited.init_noise_std[0] == 1
    assert not hasattr(inherited, "hard_pact_qp") and not hasattr(inherited, "pinn_loss_weight")


def test_checkpoint_rejects_legacy_or_incompatible_contract(tmp_path):
    a = make()
    path = tmp_path / "incompatible.pt"
    torch.save({"model_state_dict": a.actor_critic.state_dict()}, path)
    with pytest.raises(ValueError, match="schema"):
        a.load(path)
    data = a.checkpoint_state()
    data["input_contract"]["actor_load"] = "ground_truth"
    torch.save(data, path)
    with pytest.raises(ValueError, match="contract"):
        a.load(path)


def test_import_without_simulator_selection_or_imports():
    env = dict(os.environ)
    env.pop("SIMULATOR", None)
    result = subprocess.run([sys.executable, "-c", "import sys; from rsl_rl.algorithms import PPOBeyondRobustness; "
                             "assert 'legged_gym' not in sys.modules; assert 'isaaclab' not in sys.modules; "
                             "assert 'genesis' not in sys.modules"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
