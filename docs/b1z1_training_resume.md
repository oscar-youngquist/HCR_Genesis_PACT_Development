# B1Z1 PACT training continuation

New `B1Z1PACTRunner` checkpoints automatically include versioned learning state.
This covers PACT and variants using this runner, not UniFP or FACET runners.
Existing evaluator weight keys are unchanged. Only load trusted checkpoint files.

Saved state includes both networks, all three named optimizer partitions and
moments, adaptive learning rate/entropy/KL state, PINN progress, spectral power
iteration vectors, force reliability gates, force/domain curricula, reward scales,
command ranges, per-environment tradeoff progress/clean weights, terrain levels
and origins when available, completed-episode reward/length windows used by gates,
next PPO iteration, cumulative interactions/time,
Python/NumPy/PyTorch RNG, and the training GPU RNG. The envelope records resolved
configuration, optimizer parameter ownership, GRF semantics and source Git SHA.
Saves use a same-directory temporary file and atomic replacement; interrupted
writes do not replace an existing checkpoint. Save only after a completed PPO
and auxiliary update, not mid-rollout.

## Resume

From the repository root, using the same seed and environment count as the source:

```bash
bash legged_gym/scripts/b1z1_pact_lab.sh --gpu=cuda:0 --seed=1 \
  --num_envs=4096 --resume --load_run=/absolute/path/to/source/run \
  --ckpt=12000 --max_iterations=5000
```

`--max_iterations` retains the existing **additional updates** convention: this
example adds 5,000 updates starting at the checkpoint's saved next iteration.
Periodic `model_12000.pt` stores next iteration 12,001; use the saved iteration,
not merely its filename, when calculating a remaining budget. Omitting `--ckpt`
uses the existing configured checkpoint selection (normally latest). Unity uses
the same arguments with `b1z1_pact_unity.sh`; preserve scheduler GPU visibility.
The existing registry creates a new output run for continuation.

Training/environment contracts must match, including environment count, seed,
conditioning/action semantics, compliance, losses, gains and curricula. Budget,
save cadence and output naming may differ. An intentional weights-only transfer
uses `runner.load(path, load_optimizer=False)` and keeps fresh training progress;
it is not a full resume. Old checkpoints still load with a warning because missing
RNG/adaptive/curriculum state cannot be reconstructed retroactively. Running jobs
must restart on this code to begin producing complete learning-state checkpoints.

## Physical-state limitation

As with HardPACT learning continuation, this is **not bitwise simulator resume**.
The first resumed collection starts fresh physical episodes, compliant references,
observation/FiLM histories, contact/delay state and action-replay queues. Learning
progress is restored around reset callbacks; partial episode reward evidence and
unfinished trajectories are discarded. Fresh episode initialization consumes the
restored RNG stream, so the ensuing physics is not an uninterrupted trajectory.
No simulator contact/solver state or in-progress rollout is serialized.

## CPU verification

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests python -m pytest \
  tests/test_b1z1_training_checkpoint.py -q
```

Tests cover next-update/RNG parity on small networks with real Adam optimizers,
contract and moment preflight, legacy/weights-only behavior, atomic-write failure,
completed-update enforcement, and fresh-history reset without curriculum loss.
They do not establish simulator or Unity continuation parity.
