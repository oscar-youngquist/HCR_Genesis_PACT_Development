# B1Z1 PACT FiLM V02

Enable in `B1Z1PACTCfgPPO.policy` (inherited by current PACT-based ablations):

```python
film_v02_enabled = True
film_history_steps = 5
```

The default is `False`, retaining the original post-trunk FiLM and checkpoint
parameter layout. `none`/`concat` ablations ignore V02. The separate legacy
PACT-Pos implementation is unchanged; current position-only PACT ablations use
the shared PACT model and support V02.

Each control step forms six native-normalized errors: commanded minus estimated
base velocity/yaw rate and spherical EE position. Store the latest five frames,
oldest first, including the current frame. Apply LayerNorm across the six entries
of each frame, flatten the history, and predict latent-width gamma/beta:

```python
z_actor = z * (1 + gamma) + beta
```

Only the actor receives `z_actor`. Explicit/force/GRF/privileged reconstruction
and representation PINNs retain the original latent. FiLM no longer modulates
post-trunk features in V02. LayerNorm parameters belong to FiLM's actor optimizer.

Missing history after reset is zero-padded. Rollout collection snapshots the
complete error history alongside latent noise; PPO and delayed-action physics
reconstruction replay that snapshot without advancing online history. Inference
advances history once per action, so deployment must call `actor.reset(dones)`
at episode boundaries (or `actor.reset()` for all robots). Evaluation and the
B1Z1 PACT/Exp2 play paths do this automatically. Runtime history is not checkpoint
state; resumed training starts with fresh environments/history.

V02 is an architecture change: train a new policy and use the same enabled config
when loading it. Old V01 checkpoints remain supported with the flag off; they
are not silently converted. Auxiliary objectives and PPO/PINN schedules are unchanged.

FiLM evaluation fields now refer to latent features for V02. `film_condition_t`
stores the raw flattened tracking history and `film_condition_normalized_t`
stores its LayerNorm output. `prediction_z` remains unmodulated. Offline replay
of V02 must supply the saved error history; observations alone do not encode the
exact past predictions used at collection time.
