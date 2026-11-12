# RSS HardPACT ablations

Shared implementation on `rescaled_qp_exps`; selectors do not copy task logic.
Registry: `rsl_rl/hard_pact_ablations.py`. Every RSS variant disables both
representation PINNs; all supervised estimator/decoder objectives remain.
Current HardPACT observations/history, rewards, randomization, curricula,
training budget, solver settings and enabled coefficients are inherited.

| Variant | Actions | FK | FF support | Execute QP | Post-QP learning |
|---|---|---|---|---|---|
| pos_ppo | 12 position | off | off | off | off |
| coupled_ppo | 24 coupled | off | off | off | off |
| guided_no_qp | coupled | on | on | off | off |
| qp_no_post | coupled | on | on | on | metrics only |
| qp_imitation | coupled | on | on | on | detached torque targets |
| full | coupled | on | on | on | differentiable |
| no_fk | coupled | off | on | on | differentiable |
| no_ff | coupled | on | off | on | differentiable |
| no_channels | coupled | off | off | on | differentiable |
| no_qp_wrench | coupled | on | on | on | differentiable |

Each task is `go2_hard_pact_rss_<variant>_isaaclab`. Legacy tasks, including
the legacy named `full` (representation PINNs enabled), are unchanged.
RSS uses isolated nested mutable configs and per-variant experiment/run names.

## Gradient and eligibility details

- Position-only uses a genuine 12-dimensional Normal, entropy and raw-action
  likelihood. The existing 24-wide action history/controller receives zero FF
  placeholders, not extra sampled actions. No learned FF torque is executed.
  Checkpoints with 24-D std cannot be silently loaded into this distribution;
  use a matching RSS position checkpoint or an explicitly converted bootstrap.
- FK uses existing delayed command / realized stance-foot targets. RSS support
  guidance now uses **all physics-valid, delay-source-valid control transitions**,
  predicted stance and feasible support targets at their pre-control state.
  It no longer depends on the selected QP substep, QP success, differentiability,
  or QP warmup. All RSS variants share this eligibility rule and the existing
  actor-loss warmup. This intentionally corrects the old success-mask coupling.
- Support targets and estimator inputs are detached. Channel guidance reaches
  only the corresponding actor branch and shared actor trunk, not estimators.
- No-post still evaluates QP metrics. Imitation keeps primary/recovery
  torque-normalized conditional means and recovery coefficient, with detached
  solved targets; stance/velocity/slack quantities are metrics only. Full keeps
  existing owned actor/encoder/decoder PCGrad paths. Solver **numerical PPO mode**
  is identical even when autograd is off; no implicit solver VJP is retained.
- `no_qp_wrench` zeros only the QP-model wrench at rollout/replay/deployment
  boundaries. Disturbances, supervision and FF-support wrench remain intact.
- Resolved JSON, run metadata and deployment contract record selection, seed,
  configuration, Git revision, checkpoint source and action/wrench conventions.

## Local launch

From this checkout (replace `full` with any table row):

```bash
./legged_gym/scripts/go2_hard_pact_rss_full_lab.sh \
  --headless --gpu cuda:1 --seed 1 --num_envs 4096
```

Wrappers forward existing `--max_iterations`, `--resume`, `--load_run` and
`--checkpoint` arguments. They select the task; conflicting `--task` is rejected.
Use `--dry-run` to inspect without activating Conda or importing a simulator.
`HARD_PACT_CONDA_ENV` overrides environment selection; otherwise QP tasks use
the configured cuPIQP environment and non-QP tasks do not request a QP runtime.
Use `--smoke --num_envs 4` for the normal launcher’s one-iteration smoke.

## Unity / Job Composer

The matching `go2_hard_pact_rss_<variant>_unity.sh` uses container Python
(`/opt/env/bin/python`, configurable with `HARD_PACT_PYTHON`), skips Conda,
preserves scheduler `CUDA_VISIBLE_DEVICES`, and uses logical `cuda:0`.
The shared container helper mounts isolated outputs/caches and validates scripts.

```bash
export REPO=/path/to/checkout IMAGE=/path/to/isaaclab-hcr.sif
export RUN_DIR=/path/to/unique/rss-full-seed1
export OMNI_KIT_ACCEPT_EULA=YES  # after reviewing NVIDIA's EULA
# Inside a GPU allocation; do not replace scheduler CUDA_VISIBLE_DEVICES.
bash "$REPO/containers/isaaclab/run.sh" script \
  go2_hard_pact_rss_full_unity.sh --seed 1 --num_envs 4096 --dry-run
# Remove --dry-run to execute.
```

Configurable submission (not submitted by this patch):

```bash
export RUN_ROOT=/path/to/isolated/runs
export TRAIN_SCRIPT=go2_hard_pact_rss_full_unity.sh
sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT \
  --export=ALL "$REPO/containers/isaaclab/job_composer.sbatch" \
  --seed 1 --num_envs 4096 --max_iterations 4000
```

No extra training variant is needed for deployment-only QP: reuse a
`guided_no_qp` checkpoint with explicitly enabled deployment QP settings.

## Focused validation

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests python -m pytest -q -p no:cacheprovider \
  tests/test_hard_pact_rss.py tests/test_go2_hard_pact_ablations.py \
  tests/test_hard_pact_actor_regularizers.py tests/test_hard_pact_auxiliary.py \
  tests/test_go2_hard_pact_physics_heads.py tests/test_hard_pact_action_replay.py \
  tests/test_hard_pact_qp_modes.py tests/test_hard_pact_ppo_latent_replay.py \
  tests/test_hard_pact_two_stage_auxiliary.py

CUDA_VISIBLE_DEVICES=1 SIMULATOR=isaaclab PYTHONPATH=.:tests python \
  tests/smoke_hard_pact_rss.py --task go2_hard_pact_rss_full_isaaclab \
  --gpu cuda:0 --headless --num_envs 4
```

The bounded real smoke overrides terrain size, BARD capacity (32), rollout length (4), epochs (1),
minibatches (1), QP shard (100%) and warmups (0); never changes production configs.
It validates actual registered tasks, finite observations/actions/rewards/GRFs,
torque limits, zero FF for position-only, solver counts and owned PPO/auxiliary
gradients. `RSS_SMOKE_PASS` is required: Isaac Lab shutdown can otherwise mask
an exception's process exit code. Unity execution requires a real SIF/cluster;
host dry runs alone are not container runtime validation.

### Results (2026-11-12)

- Above regression command with `/home/oyoungquist/.conda/envs/lr_lab_cupiqp/bin/python`:
  **154 passed, 19 subtests passed, 12.42 s**; one existing hppfcl deprecation warning.
  RSS-only tests cover every variant's real PPO update on synthetic mechanics;
  matched-forward checks exercise qpth CPU and cuPIQP CUDA at float64
  `atol=rtol=1e-10`, plus detached-target versus full solver derivatives.
- `bash -n` passed for the main launcher, RSS shared launcher, all 20 wrappers,
  container helper and Job Composer template. All 20 wrapper dry runs and all
  ten container allowlist dry runs passed without launching jobs.
- Real GPU smoke command above passed for `pos_ppo`, `guided_no_qp`, `full`,
  `qp_no_post`, `qp_imitation`, `no_qp_wrench`: 4 environments × 4 control steps,
  one PPO update, finite gradients and actuator bounds. QP variants ran 32
  rollout rows and 16 PPO rows; non-QP variants ran zero. Position-only ran a
  12-D distribution and executed zero FF. Logs: `/tmp/rss_*_smoke.log`.
- Hardware/runtime: physical GPU 1, RTX 4090; Isaac Sim 5.1 / Isaac Lab,
  PyTorch 2.7.0+cu128, `lr_lab_cupiqp`. These are correctness smokes, not timing
  benchmarks. Channel-target eligibility was empty in these very short reset
  rollouts; nonzero FK/support gradients and routing are verified separately by
  focused feasible-target tests. Other four variants have synthetic PPO coverage,
  not real simulator smoke coverage. Unity/SIF execution remains untested.

Core changes: immutable RSS registry, thin configs/registration, actor action
width, runner storage/provenance, shared physical conversion, PPO guidance and
imitation routing, central QP wrench switch, deployment metadata and logging.
Four existing test fixtures were updated for current orientation, recovery-slack,
configurable latent-width and cached-mechanics contracts; tolerances were not relaxed.
