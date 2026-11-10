# FACET Stage 1

Stage 2 now adds diagnostic-only references; see [Stage 2](b1z1_facet_stage2.md).
The Stage 1 physical/configuration contracts below remain unchanged.

`b1z1_facet` is selectable scaffolding, not the FACET method yet. It subclasses
`B1Z1PPOPos` without overriding execution. Its 17 learned position actions use
the baseline UniFP-matched PD gains, scales, clipping, delay, and default-held
wrist/gripper joints. The compatibility torque-history block still records
executed torque; it is not a learned feedforward output.

The environment and PPO sections are independent nested copies made with the
existing UniFP-original copying helper. Robot/assets, terrain, nominal command
sampling/interpolation/rejection, disturbances/curriculum, domain randomization,
terminations, and controller settings are unchanged. Only the registration entry
point is modified outside new FACET files.

## Intentional differences

- `compliance.enabled=False`; no signed-compliance profiles/references.
- `use_force_shifted_target=False`, `action_mode="position"`, representation and
  actor PINNs disabled, arm feedforward PINN disabled (already baseline values).
- Experiment name `b1z1_facet`, run name `facet_stage1`.
- The current baseline runner/model are reused: `B1Z1PACTRunner` with its
  position-only actor and `conditioning_mode="none"`. No new network, teacher,
  student, dynamics backend, or reward is implemented.

Disabling compliance prevents its runtime addition of 12 profile entries:
actor observation width is 81 instead of 93; critic-state/reconstruction width
is 232 instead of 244; single-frame privileged width is 419 instead of 431.
History/critic stacking is unchanged. Existing compliance-only reward overrides
also no longer suppress the nominal baseline's standing, gait/reference-pose,
and arm-progress rewards. Position-only setup continues to drop the FF reward;
disabled compliance drops its EE-velocity reward. These are existing code paths,
not FACET reward implementations. Old compliance checkpoints are not compatible
with these reduced observation widths.

## Launch

From the repository root in the installed Isaac Lab Python environment:

```bash
SIMULATOR=isaaclab_b1z1_pact PYTHONPATH=. python legged_gym/scripts/train.py \
  --task b1z1_facet --gpu cuda:0 --headless
```

Default outputs use `logs/b1z1/b1z1_facet/<timestamp>_facet_stage1/`.
This command runs the temporary baseline PPO training, not FACET training stages.

## Verification

```bash
SIMULATOR=isaaclab PYTHONPATH=.:tests OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_b1z1_facet.py tests/test_b1z1_ppo_pos.py
PYTHONPATH=. OMP_NUM_THREADS=1 python tests/smoke_b1z1_facet.py \
  --gpu cuda:1 --output-dir /tmp/facet-stage1-smoke
```

The smoke harness uses two independent Isaac Lab processes, two robots, six
control intervals each. It explicitly sets root/joint states, commands, actions,
non-unit gain/motor factors and delays of 0/2 control intervals. Test-only plane
terrain, noise off, and disabled random sampling make mechanics reproducible;
the production defaults are unmodified. Effective masses, inertias, CoMs,
materials, controller factors, actions, and applied world-frame EE/torso forces
and torso couples are compared, not inferred to match from random seeds.
Nominal EE trajectories remain active; the compliant baseline's reference does
not affect physics when identical externally supplied actions are replayed.

The harness compares successor states and interval-mean executed torques, verifies
zero FF contributions, and initializes the real FACET runner for a finite
17-action inference smoke. It writes logs, resolved test configs, NPZ traces,
and `parity.json`. Inputs must match exactly; physical outputs use `atol=rtol=1e-4`
for independent GPU solves. No optimizer steps or long training runs are used.
This verifies plumbing/physical parity, not learned performance or all randomized
training scenarios.

Stage 1 validation on the installed Isaac Lab environment (`lr_lab_cupiqp`,
GPU `cuda:1`): the matched smoke passed with zero maximum difference for every
recorded comparison, including torques, actual disturbances and successor states.
Artifacts: `/tmp/facet-stage1-smoke-matched/`. All six FACET CPU cases passed.
The existing Pos suite has one pre-existing stale assertion expecting FiLM;
the current baseline config uses `conditioning_mode="none"`. Its other five
tests passed; that protected test was left unchanged. No training was run.
