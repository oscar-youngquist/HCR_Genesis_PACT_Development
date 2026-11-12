# PILOT Differential-Compliance Ablations

All selectors inherit the current PACT physical task and PPO runner. No new
environment implementation, reward, controller or force sampler is introduced.
IDs 13-15 were unused at HEAD f082e612. Their task names are also their experiment
and run names. Current full defaults use coupled actions, FiLM V02 and both PINNs.

| ID / task suffix | Only behavioral differences from full PILOT |
| --- | --- |
| 13 / `ab13_workspace` | `compliance.joint_gate=False`, `collision_gate=False`; workspace and stiffness triggers retained |
| 14 / `ab14_no_ee_recruitment` | All four EE recruitment gates false; EE yielding and direct torso force/moment compliance retained |
| 15 / `ab15_no_channel_physics` | `actor_phys_pos_fk_enabled=False`, `actor_phys_arm_feedforward_enabled=False` |

The prefix is `b1z1_pact_`. Bounds, EE target-path collision feasibility, filtering,
profiles, observations, gains, actual forces, limits, randomization and schedules
remain inherited. In #14, only the EE term of the reference wrench is zero;
base profiles are not zeroed. In #15 the FF master flag skips both original and
local teachers. Representation and actor rollout/task/limit losses and ordinary
rewards are unchanged. These switches do not certify learned robot safety.

## Information-Matched Concat (#5)

`b1z1_pact_ab5_coupled_concat` uses `policy.concat_conditioning_version=2`.
At each policy decision, the common estimates produce detached, observation-scaled
errors in the existing order: base vx/vy/yaw rate, EE spherical radius/elevation/
azimuth. Each of the five oldest-to-newest frames is transformed identically to
FiLM V02: `[e_base/(L1_base+s_base), e_ee/(L1_ee+s_ee), tanh(L1_base), tanh(L1_ee)]`.
Current soft scales are 0.10 and 0.05 in observation units. Four zero frames pad
episode startup; partial resets clear only those rows. The 40-vector is appended
to the common input, including its original force estimates and **unmodulated**
latent. No FiLM, identity penalty or physics losses are enabled for #5.

Storage retains raw 30-D history alongside latent/action noise. Shuffled PPO and
delayed physics replay reuse the source history without advancing it. Inference
advances once per call/control boundary and must call `actor.reset(dones)` after
episode termination. Full FiLM preprocessing/weights and unconditioned #4 are
unchanged.

The new state dict records `_concat_contract=[2,history_steps]`; old #5 weights
are rejected even with `strict=False`. **Start corrected #5 from scratch.** To
load an old checkpoint deliberately, set `train_cfg.policy.concat_conditioning_version=1`
before `task_registry.make_alg_runner(..., train_cfg=train_cfg)`, or pass
`concat_conditioning_version=1` when constructing `ActorCriticB1Z1PACT` for offline
loading. Use that checkpoint's original observation/configuration dimensions.
Version 1 retains the old current-error/duplicate-force concatenation, not the
information-matched benchmark. Never remove the contract buffer to bypass checks.

At resolved default dimensions (93 actor observations, 2,976 history values,
2,155 critic inputs, 64-D latent), actor+heads+std counts are 276,548 (FiLM) versus
281,796 (concat). ActorCritic totals are 5,004,648 versus 5,009,896. The separate
privileged reconstruction decoder is identical and excluded from these totals.
Concat adds 20,480 first-layer weights but removes 15,232 FiLM parameters: +5,248.
This is information matching, not exact parameter-count matching.

## Launch And Evaluate

Activate the existing Isaac Lab Python environment. From the repository root:

```bash
bash legged_gym/scripts/b1z1_pact_ab13_workspace_lab.sh --seed=0
bash legged_gym/scripts/b1z1_pact_ab14_no_ee_recruitment_lab.sh --seed=0
bash legged_gym/scripts/b1z1_pact_ab15_no_channel_physics_lab.sh --seed=0
bash legged_gym/scripts/b1z1_pact_ab5_coupled_concat_lab.sh --seed=0
# Unity uses the corresponding _unity.sh; existing wildcard allowlists suffice.
TRAIN_SCRIPT=b1z1_pact_ab13_workspace_unity.sh
TRAIN_ARGS=(--seed=1 --num_envs=4096 --max_iterations=70000)
/bin/bash "$TOOLS/run.sh" script "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
```

No resume is enabled by these scripts; explicit arguments override inherited
defaults. Scheduler GPU visibility is preserved. Each variant has an
`eval_<task>_three_seeds.sh RUN_DIRECTORY CHECKPOINT_NUMBER [OUTPUT_BASE]` launcher.
It requires an explicit newly trained checkpoint: Exp1's three conditions, seeds
0/1/2, 100 trials, 25 robots, in-range mixed motion, 15 s; then Exp2's three loaded
profiles, seeds 0/1/2, 50 trials, 25 robots, 28 s. `EVAL_GPU` and
`EXP2_OUTPUT_BASE` overrides are supported. Existing evaluator family dispatch
and registration already resolve these selectors; the PACT visual player now
accepts registered coupled variants too.

Training archives the inherited PACT config, selector, feature matrix and
`b1z1_pact_resolved_config.json` (including runtime effective algorithm settings).

## Verification

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 python -m pytest \
  tests/test_b1z1_recruitment_ablations.py tests/test_b1z1_film_v02.py \
  tests/test_b1z1_new_ablation_launchers.py tests/test_b1z1_eval_film.py -q
SIMULATOR=isaaclab_b1z1_pact PYTHONPATH=. python \
  tests/smoke_b1z1_recruitment_ablations.py \
  --task b1z1_pact_ab14_no_ee_recruitment --gpu cuda:0 --headless
```

The smoke uses two robots, four steps and one PPO update, test-local active PINN
schedules, eager geometry, and diagnostic held EE/torso disturbances. It checks
finite outputs, exact physical force sums, #14's wrench separation and reset
inference repeatability. It does not measure trained performance, 4,096-env
throughput, or long-horizon feasibility. Production schedules are untouched.

Verified locally on 2026-11-12 (RTX 4090, cuda:0, `lr_lab_cupiqp`):

- Focused selector/conditioning/launcher/evaluation-capture tests: 48 passed,
  including registered concat V02 defaults; those plus actor-physics tests with
  GPU access: 83 passed.
- Broader relevant CPU regressions: 202 passed, six CUDA skips, three existing
  failures reproduced independently from pristine HEAD f082e612. They are the
  stale basic-Pos-as-variant-6 assertion (current baseline uses 12) and two
  sampled-context tests expecting legacy PCGrad instead of conflict-only. These
  unrelated expectations and production behavior were not changed.
- Isaac Lab smokes passed for #5/#13/#14/#15, including finite PPO updates,
  deterministic reset inference and unchanged physical load sums. #14 had zero
  EE reference recruitment with nonzero direct torso input. #15 retained actor
  gradients and representation losses without FK/FF teacher metrics.
- Runner construction + four-step rollout/update + inference took approximately
  1.47/2.16/2.13/2.11 seconds respectively, excluding simulator initialization.
  These tiny cold smokes are not a training-throughput benchmark.
- All changed Python files parse, all changed/new shell launchers pass `bash -n`,
  and `git diff --check` passes. Unity container dispatch was mocked, not submitted
  to Slurm. No trained-policy performance evaluation was performed.
