# PILOT candidate continuation

This opt-in experiment uses `pilot_continuation.environment_class`, not a new
registered policy or changes to other methods. Run the dedicated launcher, or
pass `--candidate` to the existing compliance evaluator. The original evaluation
remains in `analysis/pilot_compliance_rss/full_instrumented`.

## Configuration and deployment

`Candidate` in `legged_gym/envs/b1z1/pilot_continuation.py` defines the explicit
settings. `--candidate-config JSON` can override its fields for training.
EE gain stays 0.005 m/N, base planar gains become 0.060/0.040, yaw stays saved.
The reference generator, signed sum of torso/recruited loads, total-command
clipping and eta anti-windup are unchanged.

The gait clock now advances every control step, even at zero nominal commands,
with period 0.64 s. Only native episode reset clears it. Deployment must use
this same clock: observation dimensions/order are unchanged, but clock semantics
are intentionally different. The training-only air-time reward keeps its weight
and contact logic; its walking mask uses the compliant reference and the existing
three per-component thresholds. No privileged target enters actor observations.

At reset, 25% of episodes select the strong subset. Forward/upward directions
are equally likely, their positive coefficients are uniform 0.8-1, and the
opposite direction is exactly zero. Of these episodes, 25% receive the opposite
protected load. Magnitudes are uniform 40-50 N, multiplied by the **restored**
native curriculum scale. Starting at 5 s, pulses repeat every 8 s, with 0.5 s
ramps and independently sampled 1.5-2 s flat plateaus, followed by zero force.
The existing physical enable gate remains authoritative. Native torso streams
remain active; no additional physical torso force is applied.

Selected first goals use an interior spherical command [0.55,0.2,0] or, for half
the forward cases, [0.85,0,0]. Bounds and the existing nominal path rejection
are checked; rejected proposals retain the original goal. Native interpolation
duration is retained; the first hold lasts through the first pulse and one
second of recovery so early resampling cannot remove the intended headroom.
Subsequent nominal sampling is unchanged. These are reference-space opportunities, not certificates
of robot tracking or 25 cm feasible travel. Actual exposure must be measured.
The remaining 75% retain native profile/force/goal sampling.

## Resume contract

The launcher selects the highest numeric checkpoint in the exact Nov19 run.
It preserves the original file, optimizer ownership/moments, RNG, adaptive
learning state and mature force curriculum. Only `pilot_candidate` and planar
base gains are allowed contract migrations. The shared validator still checks
model shapes, optimizer moments, progress tensors and checkpoint boundaries.
New checkpoints store the candidate contract. Fresh physical episodes are
started, not bitwise simulator replay. Existing local reward-threshold edits
are retained and included in the source diff.

The saved model_23200 checkpoint's next iteration is 23201 and its batch was
10,240 environments x 24 steps. After a recorded OOM at that size, the user
explicitly selected 4,096 environments. `--num-envs` defaults to 4096. Per-env
progress tensors use identical deterministic indices floor(i*old/new) to retain
coverage across saved terrain types; global iteration/interaction counters and
all optimizer tensors remain unchanged. This is fresh-episode continuation with
a changed rollout batch, not identical optimization or simulator replay.
1,000 further updates now collect 98,304,000 transitions, not 245,760,000.

```bash
export SIMULATOR=isaaclab_b1z1_pact PYTHONPATH=.
export OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
python legged_gym/scripts/train_pilot_continuation.py \
  --gpu cuda:1 --num-envs 4096 --iterations 1000 --output-dir analysis/pilot_candidate/new_training

python legged_gym/scripts/eval_b1z1_pilot_compliance.py --candidate \
  --gpu cuda:1 --headless --num-envs 64 --seeds 0 1 2 \
  --episodes-per-condition 20 --calibration-episodes 20 \
  --conditions nominal up_yield_up_force up_yield_down_force \
    down_yield_up_force down_yield_down_force ample_base_on ample_base_off \
    restricted_base_on restricted_base_off \
  --output-dir analysis/pilot_candidate/start_evaluation
# Repeat the identical evaluator arguments with --run <new_training> for the
# final checkpoint. Pair by saved manifest/pair_hash, not seed numbers alone.
python legged_gym/scripts/analyze_pilot_candidate.py <evaluation-directory>
python legged_gym/scripts/plot_b1z1_pilot_compliance.py <evaluation-directory>
python legged_gym/scripts/analyze_pilot_reference_stages.py <evaluation-directory>
```

## Numeric diagnostics

Snapshots are detached at terminal successor capture, before resets. Reference
stages reuse the existing instrumentation; filters are never advanced twice.
Task coordinates are heading-aligned base-relative xy and world-height z;
velocities include native transport terms. Wrenches are [Fx,Fy,tau_z] about the
native base reference; they must not be treated as a homogeneous N-valued vector.
Per-axis force magnitudes are N, moments Nm, base xy m/s, yaw rad/s. Wrench/base
norms use only xy; yaw moments/rates stay separate, never mixed into those norms.

`distributions.csv` and `distribution_histograms.npz` retain pooled counts, sums,
maxima, quantile **bin bounds**, overflow and >=15/20/25 cm offset exposure.
Quantiles are not averages of batch quantiles. Offset bins are 1 mm; quantiles
are unavailable when they fall beyond the histogram range. Counts/durations use
finite pre-failure robot-control samples. Empty categories are omitted, not zero.
Masks distinguish force-active yielding, exact-zero protected coefficients,
signed axes, strong coefficients, recruitment, and workspace gate >0.5 versus
<=0.5. The original stage analysis additionally retains per-trial/seed statistics.

`candidate_trials.csv` adds pre-load-drift-corrected forward response, recovery,
base clipping and unloaded swing fraction. Swing fraction is a stepping proxy,
not an independently validated gait-event classifier. All failure-inclusive
outcomes remain in the original evaluator's trial summaries. Native raw forces,
recruited forces, direct/combined wrenches, gate contributions, requested and
accepted base references, gait phase and successor states are retained in NPZ.

Do not pool original settings, candidate settings at the starting checkpoint,
and final candidate training results. They measure different comparisons.
