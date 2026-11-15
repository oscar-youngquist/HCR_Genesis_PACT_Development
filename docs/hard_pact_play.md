# HardPACT Isaac Lab playback with deployment QP

The existing `play_go2_hard_pact_pos.py` player now supports coupled HardPACT
policies too. `play_go2_hard_pact.sh` is a thin alias to the same launcher.
Visualization is enabled unless `--headless` is supplied.

Enable either `GO2HardPACTCfg.play.qp_enabled = True` in
`go2_hard_pact_config.py`, or pass `--play-qp`. `--no-play-qp` overrides the
config and explicitly disables QP execution. Default is off; training never
reads this play-only switch.

```bash
cd /home/oyoungquist/Research/Genesis_Development/HCR_HardPact
bash legged_gym/scripts/play_go2_hard_pact.sh \
  --task go2_hard_pact_rss_full_isaaclab --gpu cuda:0 --num_envs 4 \
  --load_run /absolute/path/to/checkpoint_run --ckpt 3500 --play-qp
```

Choose the task matching the checkpoint (for config-driven training use
`go2_hard_pact_isaaclab`). `--load_run` supports runs outside this worktree;
`--ckpt -1` selects the latest. The launcher defaults coupled Isaac Lab play
to `lr_lab_cupiqp`; `HARD_PACT_CONDA_ENV` overrides that environment. The shared
training runtime initializer prepares cuPIQP's Warp before Isaac Sim startup.

The QP is built from the **current task's** `algorithm.hard_pact_qp`, including
CLI solver overrides, rather than a saved training schedule. Only deployment
execution settings are overridden: `enabled=True`, `every_substep`, no warmup
or objective/correction curriculum, execution alpha 1. Configured component
weights, horizons, constraints, tolerances and recovery/fallback remain intact.
Actual actuator/joint limits come from the simulator. Training/checkpoint
configurations and learned weights are not modified by this override.

Each policy interval computes actor/latent/explicit/GRF/wrench predictions at
the existing policy-rate boundary. Fresh mechanics/PD torque are used by the
shared environment QP path at **each physics substep**: normally four primary
problems per environment per 20 ms interval, plus any required recovery solves.
Corrections are applied through the existing execution conversion, not added
as another PD torque. Reset handling, certification and fallback are shared
with training. This is simulator playback, not a standalone hardware exporter.

An explicit opt-in can enable execution for a coupled no-QP ablation without
changing its trained network or algorithm flags. Position-only/HardPACTPos
and non-Isaac-Lab QP playback are rejected; their ordinary QP-off player remains.

## Bounded validation

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_hard_pact_play_qp.py tests/test_hard_pact_qp_modes.py
bash -n legged_gym/scripts/play_go2_hard_pact{,_pos}.sh

HARD_PACT_PLAY_STEPS=3 HARD_PACT_PLAY_CHECK_FINITE=1 \
  bash legged_gym/scripts/play_go2_hard_pact.sh \
  --task go2_hard_pact_rss_full_isaaclab --gpu cuda:1 --headless --num_envs 2 \
  --load_run /absolute/path/to/checkpoint_run --ckpt 3500 --play-qp
```

Focused tests: 10 passed (config immutability, activation/opt-out, launcher forwarding, four solves,
reset/replay and existing mode regressions). Physical GPU 1, cuPIQP, checkpoint
`Nov13_01-39-36_hard_pact_rss_full_isaaclab/model_3500.pt`: 2 environments × 3
control steps completed with finite action/observation/reward/torque checks and
four solves per non-reset row/interval. This headless check does not validate
interactive rendering, long-horizon stability, or hardware deployment.
Explicit `--no-play-qp` also completed two control steps with finite checks.
