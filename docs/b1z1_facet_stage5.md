# FACET Stage 5: Teacher PPO and Student Fine-Tuning

`b1z1_facet` now selects `B1Z1FACETRunner`, `FACETPPO` and the Stage 4
`FACETModel`. No PACT actor, auxiliary decoder, BARD backend or PCGrad object is
constructed. Execution still uses 17 position actions and the shared Pos PD path.
This implements training plumbing, not evidence of learned recovery/performance.

## Protocol and Exceptions

Configuration sections are independent copies. Tests compare them with current
PILOT and reject differences outside an explicit allowlist.

| Setting | FACET |
|---|---|
| Robot, limits, action scaling/clipping/delay, held wrist/gripper targets | Existing B1Z1 position-action execution |
| PD gains, by actual joint name | All 12 leg joints: Kp 100, Kd 2; Z1 joints: Kp 40, Kd 1 |
| Terrain, nominal goals/collision rejection, timing, randomization, termination, applied disturbances/curriculum | PILOT settings; terrain curriculum remains off |
| Signed compliance / UniFP force-target shift / learned FF / PINNs | Disabled/not constructed |
| Reference impedance, temporal tracking, observations, model | FACET Stages 2-4, unchanged |
| Rollout / epochs / minibatches | 24 / 5 / 4 |
| PPO | lr 3e-4, adaptive KL .012, gamma .99, lambda .95, clip .2, clipped value loss, value coefficient 1, gradient cap 1 |
| Optimizer / advantages | PILOT AdamW: 1e-6 decay on weights and biases, zero on std; full-rollout sample-SD advantage normalization |
| Adaptive entropy | FACET base-velocity/yaw-rate rewards; thresholds 1.125/0.70 preserve PILOT success fractions; remaining entropy settings inherited; terrain threshold completed |
| Estimator | Separate Adam, lr 5e-4; 2 epochs; every other zero-indexed teacher update (0,2,...), every student update; same minibatch count |
| EMA estimator | Tau .04, initialized from online estimator; updated once after estimator epochs |
| Budget / saves / student start | 70,000 total completed updates / every 1,000 / after 60,000 |

Sources: pinned upstream [ppo_facet.py](https://github.com/btx0424/active-adaptation/blob/c0d6f5aa6146814803090b5f6f19aafb8ce6f792/active_adaptation/learning/ppo/ppo_facet.py)
and [quadruped.py](https://github.com/btx0424/active-adaptation/blob/c0d6f5aa6146814803090b5f6f19aafb8ce6f792/active_adaptation/assets/quadruped.py).
Estimator cadence, EMA and losses follow upstream; PPO optimizer/protocol follows
local PILOT as requested. Only upstream PD gains are imported, not other actuator
or physics settings. MIT attribution remains in `rsl_rl/FACET_LICENSE.txt`.

### Reward Audit

Raw shared reward coefficients and curricula remain copied, not mutated. FACET
registration determines which terms actually contribute. All active coefficients
and exclusion names are saved in `facet_resolved_config.json`; optimized scales
there already include control dt. Retained terms use the original coefficients,
curriculum bounds and cosine schedule, with dt applied exactly once.

| Treatment | Reward names |
|---|---|
| Replace with six Stage 3 temporal objectives | `tracking_lin_vel_force_world`, `tracking_ang_vel`, `tracking_ee_force_world`; keep original weighted episode sums solely for nominal diagnostics, not schedule inputs |
| Exclude conflicting nominal/stillness/recruitment shaping | `no_physical_progress`, `stand_still`, `stand_still_contact`, `ref_dof_leg`, `feet_contact_number`, `walking_ref_dof`, `walking_ref_swing_dof`, `arm_progress_before_torso`, `early_torso_tilt` |
| Exclude nonexistent learned-FF channels | `tracking_arm_feedforward`, `torque_cancellation`, `{leg,arm}_feedforward_{action_rate,action_smoothness,torques}`; Pos compatibility torque slots are NOT FF actions |
| Replace PILOT-only compliance velocity reward | `compliance_ee_velocity_weight` is not registered; FACET EE velocity temporal reward remains active |
| Retain safety/limits | `collision`, `dof_pos_limits`, `torque_limits`, `alive` |
| Retain actuator tracking/regularization | `tracking_dof_pos` tracks issued PD targets, not nominal task pose; `{leg,arm}_feedback_{action_rate,action_smoothness,torques}` |
| Retain effort/smoothness | `dof_acc`, `dof_vel`, `joint_power`, `joint_power_dist`, and their `_arm` counterparts |
| Retain base/posture stability | `hip_pos`, `base_height`, `lin_vel_z`, `ang_vel_xy`, `roll`, `orientation` |
| Retain foot/support regularization | `front_foot_overreach`, `rear_foot_overreach`, `support_polygon`, `vhip_angle`, `vhip_angular_acc`, `feet_drag`, `feet_regulation`, `feet_pos_xy`, `stumble`, `feet_contact_forces`, `feet_air_time`, `foot_clearance_terrain_aware` |

The six FACET weights/bandwidths/horizons are unchanged. Nominal diagnostic
formulas/units are unchanged, even when virtual references move under loading.
No untrained 9-D wrench reconstruction is used for force-curriculum decisions.

### FACET Reward-Based Schedules

Entropy uses `rew_facet_base_velocity` and `rew_facet_yaw_rate`. Episode sums
already include control dt and are divided by prescribed episode seconds;
their maxima are the reward weights, 1.5 and 1.0. Default entropy thresholds
are therefore 1.125 (75%) and 0.70 (70%), without another dt multiplier.
These preserve relative thresholds, not identical success probabilities under
the different FACET objectives. Explicit algorithm threshold overrides still apply.

Command-range advancement uses `facet_base_velocity` episode sums against its
own dt-scaled weight. Domain-randomization advancement uses the same reward,
converted to the inherited simulator's reward units (multiply by 2/1.5 with
default weights), preserving its relative minimum/recovery thresholds. Neither
uses nominal tracking diagnostics. Time-based reward curricula are unchanged.

Force advancement uses `rew_facet_ee_position / weights['ee_position']` with
`facet_tracking.force_curriculum_reward_fraction = 0.75`. This is an explicit
reward-fraction criterion, not a conversion of nominal L1 error into meters.
The shared gate internally smooths its complementary dimensionless deficit;
logs expose `ForceCurriculum/facet_ee_reward_fraction_ema` and its threshold.
Missing/zero-weight rewards cannot satisfy the performance gate. Existing
survival/roll checks, patience, latest-start fallback, and ramp timing remain.
Short/failed episodes retain the prescribed-horizon denominator.

Checkpoints tag the new gate semantics. Loading an older checkpoint clears the
incompatible nominal-error EMA/patience with a warning, preserving already
latched force exposure and ramp progress. Legacy entropy thresholds are rescaled
once; subsequent saves retain their FACET units. Running Python processes do
not pick up this change until restarted/resumed. Other methods are unchanged.

## Collection, Optimization and Handoff

Packets remain 237 student / 52 privileged / 9 wrench values by default (see
[Stage 4 schema](b1z1_facet_stage4.md)). Raw Gaussian samples and their summed
log probabilities are stored, before environment clipping and action delay.
The action history stores clipped issued positions before delay, never PD torque.

Normalizers incorporate the first observation batch once. Statistics are frozen
through each complete collection/update; exact normalized inputs are cached.
Raw sufficient statistics update the normalizers only for the next teacher
rollout. No repeated minibatch normalization updates or double teacher/student
updates occur. All three normalizers remain frozen throughout student training.

Teacher PPO updates teacher actor, critic and both encoders. Online estimator
supervision is 128-D privileged-feature MSE plus 32-D force-feature MSE against
detached current teacher encoder outputs. The separate last nine output rows have
no supervision. Chronological 24-step sequences retain their initial GRU state;
robots are shuffled, never timesteps. Reset masks clear state before processing
the new observation; padded steps do not advance recurrence or count in the loss.
Upstream's first-observation mask contributes zero loss but retains its valid-step
denominator. Both online and EMA recurrent states advance during teacher collection.

Complete terminal successor packets are captured after physics/reward evaluation
but BEFORE reset. Timeout-only transitions bootstrap from this packet's critic
value. True physical failures bootstrap zero, even if timeout coincides. Both
boundaries stop GAE and recurrent propagation. No reset-state or preceding-state
value substitutes for the terminal value. Terminal packet reads return copies.

`algorithm.student_start_iteration` counts **completed PPO updates**. Updates
0-59,999 use teacher rollouts. After the final teacher update, actor weights/std
copy exactly once to student. The same AdamW optimizer retains critic moments,
LR and entropy state; matching actor Adam moments/step counters transfer to student.
Teacher actor and both encoders freeze. Student collection uses EMA features;
student PPO uses cached detached features, never gradients through the estimator.
Estimator supervision continues against frozen teacher encoders. EMA is fixed
during collection/PPO reevaluation and updated after separate estimator training.
Maintained EMA recurrent state supplies the handoff; no environment/reference or
schedule reset occurs. No additional readiness/recovery gate is imposed.

`transition_60000.pt` archives the handoff; `model_60000.pt` is also saved for
normal checkpoint discovery. Later `model_<iteration>.pt` files remain selectable
as latest. Checkpoints preserve models, EMA, normalizers, both optimizers,
completed-update index, phase/copy status, LR/entropy, force/domain curricula,
reward scales, command ranges and counters. Resume never recopies a trained
student. Because simulator state is not saved, resumed episodes, observation
histories and both recurrent states restart consistently instead of loading stale
hidden states. `learn(max_iterations)` is a TOTAL budget, including prior updates.

TensorBoard includes phase, estimator latent errors, handoff action RMS difference
and EMA latent errors, actual force-curriculum exposure, existing episode tracking/
termination diagnostics and FACET horizon errors. These diagnose tracking/recovery;
the iteration trigger itself is not a claim of successful recovery.

## Commands

Run from the repository root in the existing Isaac Lab Python environment:

```bash
SIMULATOR=isaaclab_b1z1_pact python legged_gym/scripts/train.py --task b1z1_facet --gpu cuda:0 --headless
SIMULATOR=isaaclab_b1z1_pact python legged_gym/scripts/train.py --task b1z1_facet --gpu cuda:0 --headless --resume --load_run RUN_FOLDER --checkpoint 60000
```

Explicit existing CLI/config overrides remain supported. The student-start and
estimator settings live only in `B1Z1FACETCfgPPO.algorithm`. Standalone deployment
can use the saved model and EMA; runner `get_inference_policy()` accepts only
`(student_observation, hidden, is_init)` and returns `(mean_action, next_hidden)`.
Legacy PACT play/export interfaces are not FACET deployment adapters.

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 python -m pytest tests/test_b1z1_facet*.py -q
PYTHONPATH=. OMP_NUM_THREADS=1 python tests/smoke_b1z1_facet_training.py --gpu cuda:0 --output-dir /tmp/facet-stage5-smoke
PYTHONPATH=. OMP_NUM_THREADS=1 python tests/smoke_b1z1_facet.py --gpu cuda:0 --output-dir /tmp/facet-stage5-parity
SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 python tests/benchmark_b1z1_facet_training.py --device cuda:0 --num-envs 4096
```

The training smoke uses eight robots, three teacher and three student updates,
0.8-second episodes and save cadence three; production settings are not changed.
It verifies finite parameters/losses, intended updates, frozen targets, exact
handoff copy, real PD gains, timeout packets and strict student resume. Separate
matched-action fixtures explicitly match gains/mechanics/forces and compare
references/rewards/observation switches, including physical failures.

## Measured Checks (RTX 4090, GPU 0)

- FACET CPU suite: 59 passed, including upstream Stage 4 module parity tests.
- Existing position-only tests additionally revealed one pre-existing mismatch:
  `test_configuration_inheritance_and_registration` expects `film`, but Pos is
  configured `none`. Protected test/configuration left unchanged; one FK CUDA
  test skipped in the CPU-only invocation.
- Isaac Lab six-update training smoke passed twice; final run total learning/
  checkpoint time 5.84 s, collection 0.78-1.02 s, supervised updates 0.09-0.15 s
  and the teacher update without estimator 0.05 s (eight robots, not a large-scale
  throughput claim).
- Five matched-action Isaac Lab fixtures passed, with terminal/partial-reset
  coverage and nonzero explicitly matched EE/base forces and moments.
- Synthetic 4096-robot, 24-step benchmark, two warmups then three measurements:
  median teacher collection 29.94 ms/update 158.03 ms; student collection
  28.41 ms/update 151.49 ms; peak allocated memory 891.43 MiB. Estimator epochs
  included in both phases. Excludes simulator, reference/observation assembly,
  logging and file I/O; it is not full-training iteration time.

No full training run or trained-task-performance evaluation was performed.

## Training Logging

FACET writes `Policy/mean_noise_std` and `Policy/position_noise_std` (identical
for its position-only actions), plus `Policy/min_noise_std` and
`Policy/max_noise_std`. These are effective Gaussian standard deviations in
policy-action units, including the distribution's 1e-6 floor, for the updated
actor that collected the rollout. They are not joint-position errors or torques.
`Train/mean_reward` and `Train/mean_episode_length` average the last 100 completed
episodes; partial episodes are not counted as completed returns.

The terminal uses a PACT-style iteration panel with FACET PPO/estimator losses,
policy std, episode metrics, phase, throughput, elapsed time and ETA. Intermittent
estimator losses are omitted when not computed; no fictitious PINN losses are
printed. At handoff it labels both the completed rollout phase and next phase.
Reporting changes take effect on the next process launch, not in an already
running Python process.
