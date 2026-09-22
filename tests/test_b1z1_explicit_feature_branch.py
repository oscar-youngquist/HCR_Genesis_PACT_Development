"""HardPACT-style deterministic explicit estimates and reconstruction gradients."""
import pytest
import torch
from test_b1z1_sampled_context import make_model


@pytest.mark.parametrize("pos", [False, True])
def test_explicit_independent_of_latent_noise(pos):
    model = make_model(pos)
    history = torch.randn(4, 162)
    first = model.decode_context(model.context_encoder(history, latent_noise=torch.zeros(4, 8)))
    second = model.decode_context(model.context_encoder(history, latent_noise=torch.ones(4, 8)))
    assert not torch.equal(first["z"], second["z"])
    torch.testing.assert_close(first["explicit_condition"], second["explicit_condition"], rtol=0, atol=0)
    contact = first["explicit_condition"][:, 6:10]
    torch.testing.assert_close(contact, .01 + .98 * first["foot_contact_logits"].sigmoid())
    assert ((contact >= .01) & (contact <= .99)).all()
    first["explicit_condition"].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.context_encoder.trunk.parameters())
    assert any(p.grad is not None for p in model.explicit_decoder.parameters())
    assert all(p.grad is None for p in model.context_encoder.latent_mean.parameters())
    assert all(p.grad is None for p in model.context_encoder.latent_logvar.parameters())
    assert all(p.grad is None for p in model.context_encoder.mean_hidden.parameters())
    assert all(p.grad is None for p in model.context_encoder.logvar_hidden.parameters())


@pytest.mark.parametrize("pos", [False, True])
def test_hidden_latent_branches_and_bounded_logvar(pos):
    encoder = make_model(pos).context_encoder
    history = torch.randn(4, 162)
    context = encoder(history, latent_noise=torch.ones(4, 8))
    features = encoder.trunk(history)
    torch.testing.assert_close(context["mean"], encoder.latent_mean(encoder.mean_hidden(features)))
    torch.testing.assert_close(context["logvar"], encoder.latent_logvar(encoder.logvar_hidden(features)))
    assert isinstance(encoder.latent_logvar[-1], torch.nn.Hardtanh)
    assert (context["logvar"].abs() <= 5).all()
    context["z"].square().mean().backward()
    for branch in (encoder.mean_hidden, encoder.logvar_hidden):
        assert branch[0].in_features == branch[0].out_features == encoder.feature_dim
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters())


@pytest.mark.parametrize("pos", [False, True])
def test_privileged_reconstruction_trains_explicit_only_in_encoder_phase(pos):
    model = make_model(pos)
    decoder = torch.nn.Linear(8 + 14, 5)
    context = model.decode_context(model.context_encoder(torch.randn(4, 162)))
    decoder(torch.cat((context["z"], context["explicit_condition"]), -1)).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.explicit_decoder.parameters())
    model.zero_grad(set_to_none=True)
    detached = {k: v.detach() for k, v in context.items()}
    decoded = model.decode_context(detached)
    decoder(torch.cat((decoded["z"], decoded["explicit_condition"]), -1)).square().mean().backward()
    assert all(p.grad is None for p in model.context_encoder.parameters())
    assert all(p.grad is None for p in model.explicit_decoder.parameters())
