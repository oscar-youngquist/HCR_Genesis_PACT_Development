# B1Z1 PACT explicit estimation

Both coupled PACT and PACT-Pos use this HardPACT-style flow:

```text
history -> deterministic trunk features h
  h -> mean/logvar -> sampled z -> actor and physics decoders
  h -> explicit estimator -> e = [base velocity, spherical EE position,
                                 bounded contact probabilities, foot heights]
  [z, e] -> next privileged-state reconstruction
  [z, stopgrad(e)] -> force decoder
  [z, stopgrad(e), stopgrad(torque)/scale] -> GRF decoder
```

The explicit head uses two 128-wide hidden layers by default. It bypasses VAE
sampling and the KL bottleneck; the history trunk still receives gradients from
all its branches. Each mean/logvar branch now has its own feature-width linear
hidden layer and activation before its output projection, matching HardPACT's
branch structure. Log variance remains bounded by Hardtanh(-5, 5).
The existing latent dimensions, trunk widths, sampling, force
targets, reconstruction target layout and actor/FiLM inputs are unchanged.
Contacts use `0.01 + 0.98 * sigmoid(logits)` consistently for policy and decoder
conditioning. Contact supervision remains BCE with the original raw logits.

The encoder optimizer owns the history encoder and explicit estimator together.
The decoder optimizer owns privileged reconstruction, force and GRF heads.
During the encoder step those decoder parameters are frozen, but reconstruction
gradients reach both z and e. The decoder step consumes the same detached z/e;
it does not recompute or update the explicit branch. Both auxiliary optimizers
use `adaptation_learning_rate = 2e-4`. PPO conditioning remains detached.

Checkpoint compatibility: the explicit input width and privileged decoder input
width changed, and the latent branches gained hidden layers, so old weights
require an intentional conversion or fresh training.
Strict model loading is retained. Optimizer partition version is now 3; older
optimizer moments are not silently assigned to the new ownership groups.

This supersedes older documents describing a z-only explicit/privileged decoder
or placing the explicit head in the decoder optimizer. No UniFP changes are made.
