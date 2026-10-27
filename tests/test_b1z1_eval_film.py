"""Evaluation-only FiLM capture: actual inference, no simulator or checkpoint."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from rsl_rl.modules.actor_critic_b1z1_pact import ActorCriticB1Z1PACT
from rsl_rl.modules.actor_critic_b1z1_pact_pos import ActorCriticB1Z1PACTPos

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_film", ROOT / "legged_gym/scripts/eval_b1z1_exp1.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def model(mode="film", action_mode="coupled", legacy=False):
    cls = ActorCriticB1Z1PACTPos if legacy else ActorCriticB1Z1PACT
    kwargs = {} if legacy else dict(conditioning_mode=mode, action_mode=action_mode)
    return cls(81, 40, 17, 162, latent_dim=8, actor_layers=[32, 16],
               critic_layers=[32, 16], context_layers=[32, 16],
               explicit_decoder_layers=[16], force_decoder_layers=[16],
               grf_decoder_layers=[16], **kwargs).eval()


@pytest.mark.parametrize("action_mode,legacy", [("coupled", False), ("position", False), ("position", True)])
def test_capture_parity_and_chunk_roundtrip(action_mode, legacy, tmp_path):
    actor = model(action_mode=action_mode, legacy=legacy)
    # Nonidentity modulation tests gamma's residual-scale convention.
    with torch.no_grad():
        actor.film.network[-1].bias.copy_(torch.linspace(-.2, .3, 32))
    obs, history = torch.randn(2, 81), torch.randn(2, 162)
    weights = {k: v.clone() for k, v in actor.state_dict().items()}
    calls = []
    handle = actor.film.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        baseline = evaluation.inference(actor, obs, history)
        rng = torch.random.get_rng_state().clone()
        action, records = evaluation.recorded_inference(actor, obs, history)
    handle.remove()
    assert len(calls) == 2  # One FiLM call per inference, not a recording replay.
    torch.testing.assert_close(action, baseline, rtol=0, atol=0)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert len(records) == 5
    assert records["film_condition_t"].shape == (2, 6)
    torch.testing.assert_close(records["film_features_post_t"],
                               records["film_features_pre_t"] * (1+records["film_gamma_t"]) + records["film_beta_t"])
    assert all(not value.requires_grad for value in records.values())
    assert not actor.film._forward_hooks and not actor.film.network._forward_hooks
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    writer = evaluation.util.NumericWriter(tmp_path, chunk_rows=1)
    writer.append({k: v[0].numpy() for k, v in records.items()})
    writer.close()
    with np.load(tmp_path / "transitions-000000.npz", allow_pickle=False) as stored:
        for key, value in records.items():
            np.testing.assert_array_equal(stored[key][0], value[0].numpy())


@pytest.mark.parametrize("mode", ["none", "concat"])
def test_nonfilm_omits_records(mode):
    actor = model(mode)
    obs, history = torch.randn(2, 81), torch.randn(2, 162)
    with torch.no_grad():
        baseline = evaluation.inference(actor, obs, history)
        action, records = evaluation.recorded_inference(actor, obs, history)
    assert records == {}
    torch.testing.assert_close(action, baseline, rtol=0, atol=0)


def test_hook_cleanup_on_exception(monkeypatch):
    actor = model()
    def fail(*args):
        raise RuntimeError("inference failed")
    monkeypatch.setattr(actor, "act_inference", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        evaluation.recorded_inference(actor, None, None)
    assert not actor.film._forward_hooks and not actor.film.network._forward_hooks
