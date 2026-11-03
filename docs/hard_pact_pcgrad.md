# HardPACT Conflict-Only Projection

The shared HardPACT algorithm config now defaults to:

```python
pinn_projection_mode = "conflict_only"
pinn_projection_norm_balance = True
```

All thin HardPACT variants inherit these settings. They do not enable disabled
physics objectives. The runner already forwards the complete algorithm config
and saves its resolved values, so no new runner/checkpoint state is required.

For the already-weighted physics gradient, remove its component opposing the
reference gradient only when their dot product is negative. Aligned and
orthogonal components remain. The reference is never projected. Optional norm
balancing then caps the projected physics norm to the reference norm; updates
sum the reference and scaled projected physics, never average them.

The reference is PPO for actor/critic optimization and the supervised loss
for each of the independently owned encoder and decoder optimizers. Actor
physics here is HardPACT's existing QP objective; representation physics is
its existing auxiliary physics objective. Detach boundaries, update ordering,
loss definitions, weights and warmup schedules are unchanged.

Set `pinn_projection_norm_balance = False` to disable norm capping. In that
case a zero reference retains the physics gradient. With balancing enabled,
a zero reference suppresses it, matching the prior balanced behavior.

Set `pinn_projection_mode = "legacy"` to restore the old sign-selected rule:
nonnegative `pinn_loss_weight` uses unconditional orthogonal projection;
negative values additionally enable norm balancing. Missing selectors in
direct/old algorithm callers also retain legacy behavior. Explicit invalid
modes fail. The existing negative-weight schedule convention is deliberately
unchanged; changing the projection selector does not change activation timing.

The standalone Pos pretraining algorithm and unrelated legacy algorithms are
unchanged. Existing PCGrad entry points retain their previous behavior.

## Focused Checks

```bash
SIMULATOR=isaaclab python -m pytest -q -p no:cacheprovider \
  tests/test_pc_grad.py tests/test_hard_pact_conflict_projection.py
```

These cover synthetic gradients, balancing/legacy dispatch, inherited config,
optimizer ownership, unused parameters, and real small encoder/decoder
construction/backward/optimizer steps without simulation or BARD allocation.
