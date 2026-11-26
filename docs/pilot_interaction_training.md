# PACT-family bounded interaction training

`B1Z1PACTCfg.pilot_interactions` enables these settings in the ordinary registered
PACT environment. All PACT-derived ablation configs and PPO position baseline
inherit them; explicit recruitment/loss ablations still override their respective
gates/objectives. UniFP does not inherit this section. FACET explicitly disables
it and retains its own formulation. The independent legacy `b1z1_pact_pos` task
does not derive its environment/configuration from B1Z1PACT and is unchanged.

## References and observations

- `max_offset_norm=0.30` m bounds reference-minus-nominal in heading-relative XY,
  world-Z task coordinates. The raw equilibrium, filtered step, and recovery
  path are bounded. First-exit clipping along validated path prefixes preserves
  the intersection with workspace and EE-only collision guidance, rather than
  projecting a collision-safe endpoint onto a different ray.
- Recovery anchors must also lie in the offset ball. An empty/invalid intersection
  marks the affected environment for existing recovery-failure termination;
  its numerical placeholder is not a certified reference. Such terminal samples
  are excluded by existing actor-PINN done masks. There is no silent relaxation.
- Accepted position/velocity update both filter states. At the cap, outward
  offset velocity is stopped; inward recovery remains possible. If moving nominal
  and workspace velocity requirements conflict, the event is explicitly invalid.
- EE gain remains 0.005 m/N; planar base gains are 0.060/0.040 (m/s)/N. The yaw
  gain stays 0.015. Existing signed summation, filtering, total-command clipping,
  and post-clip eta correction are unchanged.
- The 0.64 s phase advances even at zero nominal command and resets only on
  episode reset. Deployment must use the same phase convention. Observation
  dimensions/order and policy architecture are unchanged.
- Feet-air-time retains its original weight/contact calculation. Its activation
  uses compliant reference commands with existing per-component walking
  thresholds, only in the reward. No privileged reference enters observations.
  Existing no-progress threshold 0.10 m/s is retained; no reward weight is raised.

## Sampling

Final episode weights: yielding/protected/ample/restricted/native =
25/25/15/15/20 percent. Production linearly mixes from entirely native sampling
over 5,000 global PPO updates, starting at `mixture_start_iteration`. The explicit
legacy migration launcher starts that ramp at the saved next iteration; subsequent
continuations retain the stored start. Native external-force curriculum state is
restored independently and multiplies controlled forces too.

Adjacent resetting rows share force magnitude/sign/direction, plateau, coefficient,
and goal descriptors. Yielding/protected counterparts differ in the loaded signed
EE coefficient; they use identical physical-initialization distributions, not
identical simulator states. Partial/odd resets may leave an unpaired descriptor.
Custom unequal yielding/protected weights retain shared descriptors but probabilistic
family assignment. Descriptor/family IDs are diagnostic only.

Coefficients are 0.8-1, forces 40-50 N, plateaus 1.5-2 s, with 0.5 s ramps and
onset at 8 s. All axes and signs occur in yielding/protected/ample families.
Restricted pairs use outward +x at radius 0.88 m, with enabled/disabled planar
base profiles. Ample controls keep permissive planar base compliance and zero
nominal velocity. Controlled episodes suppress direct torso streams and velocity
kicks; native episodes retain the original combined disturbance distribution.

Each controlled episode has one pulse. The first nominal goal is held until
at least three seconds after ramp-down; no unrelated base-command resampling
occurs during controlled episodes. Existing goal path rejection remains active;
rejected setups are logged, not relabeled successful. Geometry headroom does not
guarantee near-zero joint/collision recruitment in the learned physical state.
Use measured gate/base-response diagnostics to verify the intended controls.

## Serialization and continuation

All settings and source hashes are in the checkpoint environment contract.
Normal full-state resume remains strict. Old checkpoint migration is explicit
via `--production-interactions`: only the new versioned section, cap, existing
candidate/gain migration fields and proportional environment-count GRF totals
are permitted. Model/optimizer ownership, moments, schedules, physical parameters,
and RNG validation remain active. Original files are never rewritten. Episode
states restart, so this is not bitwise physical replay.
The explicit migration refreshes current implementation hashes and lists those
changes in its audit report; it does not retain stale hashes from the source run.

Fresh Unity training (inside the existing Slurm allocation/setup):

```bash
TRAIN_ARGS=(--seed=1 --num_envs=10240 --max_iterations=27500)
/bin/bash "$TOOLS/run.sh" script b1z1_pact_unity.sh "${TRAIN_ARGS[@]}"
```

Compatible existing new-contract runs may use the usual `--resume --load_run`
and `--ckpt` with matching resolved settings. For explicit old-to-new migration,
or continuing an archived resolved configuration without guessing its settings:

```bash
/bin/bash "$TOOLS/run.sh" script b1z1_pact_continue_unity.sh \
  --run /workspace/repo/analysis/pilot_candidate/rerun_20261125_194754/training \
  --checkpoint 24201 --num-envs 4096 --iterations 50 \
  --output-dir /output/pilot_interactions_continuation
```

Paths must exist inside the existing binds; do not change scheduler CUDA visibility.
`cuda:0` inside the allocation is the assigned GPU. The launcher defaults to a
50-update short run, not a full training run. It uses the ordinary environment,
not the previous experimental environment subclass. Seed identity is restored.
The fresh command preserves the current 10,240-environment/27,500-update Unity
budget; the local integration smoke explicitly uses 4,096 environments.

Local integration smoke (test override bypasses only the new mixture ramp):

```bash
SIMULATOR=isaaclab_b1z1_pact PYTHONPATH=. OMP_NUM_THREADS=1 \
TORCHINDUCTOR_COMPILE_THREADS=1 python legged_gym/scripts/train_pilot_continuation.py \
  --production-interactions --smoke-full-mixture --gpu cuda:1 \
  --run analysis/pilot_candidate/rerun_20261125_194754/training \
  --checkpoint 24201 --num-envs 4096 --iterations 50 \
  --output-dir analysis/pilot_interactions_smoke_final
```

`resume_report.json` records exact checkpoint hash, migration, optimizer restoration,
family counts, sums/denominators, gradient-finiteness counts and maximum offset.
The smoke override is serialized and must not be mistaken for production ramping.
Training scalar/distribution exports remain available through the continuation
launcher. Normal training logs family fractions and separately counted cap events.
Small deterministic evaluation uses the existing compliance evaluator, without
`--candidate`; its scenario sampler is disabled while clock/cap semantics remain.
Do not interpret short-run tracking changes as convergence or policy quality.

## Verification scope

The focused CPU suite covers inheritance/configuration isolation, cap/workspace/
collision intersections, signed and protected responses, paired descriptors,
partial resets, phase progression, air-time activation and current-interval
diagnostic validity. Unity dispatch is tested with a mocked container runtime;
no Unity job is submitted. GPU smoke coverage is PILOT only, not independently
trained runs of every inherited ablation.

Actual interaction-family IDs are recorded separately from the legacy compliance
profile-sampler IDs. Controlled profiles overwrite sampled coefficients, so the
legacy profile-family label must not be used to infer the interaction scenario.

## Executed results

See [machine-readable validation](pilot_interaction_validation.json) for exact
commands, checkpoint hashes, source hashes, counts and denominators.

- CPU regressions: 100 passed, 2 skipped; shell syntax and Python parsing passed.
- `analysis/pilot_interactions_smoke_verified`: 50 updates, 4,096 environments,
  GPU 1 (RTX 4090), restored optimizer and mature force curriculum (scale 1).
  All exported scalars and checked actor/critic gradients were finite. All five
  families were sampled. Maximum accepted offset was 0.299669 m; no cap violations.
  Warm iteration median was 6.24 s (5.60-9.17 s after the first five updates).
- Yielding-family force-active samples exceeded 15/20 cm in 69.3%/26.9% of samples.
  These include ramps, not just plateaus. None exceeded 25 cm in that family;
  native samples did. Five failure events occurred in native episodes.
- Ample controls were imperfect: 3.17% of loaded samples had a gate above 0.05,
  and 8.40% had a base-reference component above 0.02. Interior nominal geometry
  alone does not establish unrestricted joint/collision capability in execution.
- A subsequent diagnostic-mask fix uses current failure/timeout flags rather
  than stale reset flags. Do not use the preceding 50-update histogram exports
  as authoritative pre-failure quantiles. Its independent family counters and
  cap checks remain usable. Reference dynamics/rewards/actions were unchanged.
- `analysis/pilot_interactions_resume_verified`: one additional update verified
  the reporting fix and continuation from 24251 to 24252. A failed first attempt
  exposed old archived classes lacking the new section; restoration is now
  explicitly tested and both continuation/evaluation use the corrected path.
- `analysis/pilot_interactions_evaluation`: 24 deterministic trials, evaluation
  seed 0, checkpoint 24252. Policy hashes matched before/after; force alignment
  and reference advancement checks passed. No cap exceedance occurred among
  16,131 valid samples; maximum offset was 0.249999 m.
- Five roll failures remain: four restricted-pose trials failed before useful
  loaded measurements, plus one ample/base-disabled trial. These are retained,
  not replaced with successful episodes. The 30/50 N calibration references
  reached about 15/25 cm, versus measured displacements about 13.7/21.8 cm.
  Protected 50 N trials requested zero offset but measured about 10.5 cm motion.

These checks establish integration and expose limitations, not convergence or
successful restricted-arm recruitment. No full training or Unity job was run.
Only PILOT received GPU smoke coverage; inherited variants received config and
reference regression coverage.

To repeat the small evaluation (use a new output directory):

```bash
SIMULATOR=isaaclab_b1z1_pact PYTHONPATH=. OMP_NUM_THREADS=1 \
TORCHINDUCTOR_COMPILE_THREADS=1 python legged_gym/scripts/eval_b1z1_pilot_compliance.py \
  --run analysis/pilot_interactions_resume_verified --checkpoint 24252 \
  --gpu cuda:1 --headless --num-envs 24 --seeds 0 \
  --episodes-per-condition 2 --calibration-episodes 2 \
  --conditions nominal up_yield_up_force up_yield_down_force \
    down_yield_up_force down_yield_down_force ample_base_on ample_base_off \
    restricted_base_on restricted_base_off calibration_protected_50n \
  --output-dir "analysis/pilot_interactions_evaluation_$(date +%Y%m%d_%H%M%S)"
```
