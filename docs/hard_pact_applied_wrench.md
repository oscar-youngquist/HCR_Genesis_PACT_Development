# Base-Origin Applied External Wrench Targets (v3)

HardPACT, its thin variants and HardPACTPos now supervise the wrench decoder
with the externally applied training force and its equivalent moment about the
**base-link origin used by the BARD/Pinocchio base Jacobian**. The source is
the actual persistent-wrench buffer **after event ramps/gating**, not the
unramped event target. With no applied load, the label is zero even with
randomized mass and CoM. The decoder no longer represents inertial-parameter
mismatch.

For an application point P and base origin O, `M_O = M_P + (p_P-p_O) cross F`.
This application/label shift uses the actual rotated offset each physics substep,
before yaw conversion and control-interval averaging. Yaw-local axes do not
change the reference point. Normalization, active/neutral masks and transition
validity remain unchanged.
Instantaneous velocity kicks still have no force label and retain their
existing physics-validity exclusion. Physical randomization is unchanged.

The historical `total_external_wrench_label_*` field names and widths remain
for storage/critic compatibility; "total" now means total **applied** external
loading. `equivalent_mass_com_wrench_*` stays a separate legacy diagnostic,
not a decoder target or a PINN correction. Existing target-based diagnostics
and privileged label blocks consume the corrected fields automatically.

Both inverse and rollout PINNs rotate the prediction and use it directly.
They must NOT add another application-point-to-base moment shift.
No mass/CoM wrench is subtracted. Realized mass/CoM and passive parameters
remain in the actual BARD mechanics. Gradient routing is unchanged. The
inverse helper retains its historical argument names and accepts an ignored
`mass_com_wrench_world` compatibility argument, but production no longer
supplies it.

QP objectives, constraints, nominal mechanics and execution are unchanged.
Their input remains the learned wrench. They must not be interpreted as
receiving an estimate of inertial mismatch under the new supervision.

Deployment schema 18, wrench target semantics 3, and the persistent physics-head
`wrench_origin_semantics=3` marker reject older contracts/heads, including raw
weights-only/non-strict loads. v2 heads predicted application-point moments;
older heads may also include a mass/gravity proxy. None can be relabeled by
changing metadata: a decoder without the realized offset as an input cannot
generally be converted exactly. Continue historical runs with their historical
code, or explicitly retrain/migrate and validate before using this convention.
No automatic migration or running-process change is performed.

## Focused Validation

### Isaac Lab randomized-CoM application

New HardPACT/HardPACTPos configs enable
`domain_rand.persistent_wrench_at_com = True`. Persistent forces are applied at
the torso's nominal CoM plus its current randomized displacement, rotated into
world space each physics substep. Supplied torques are free moments about that
point. The adapter explicitly composes `M + r cross F` about the link origin
using the same helper as labels, then submits force/torque **without positions**.
No second lever-arm correction is added. Setting the flag to
`False` restores the previous Isaac Lab link-origin application. Missing flags
in legacy configurations also retain that behavior. Genesis already uses
`link_com` and is unchanged. This does not enable persistent wrenches in BR/MULE
or change instantaneous velocity pushes.

The label adapter includes both nominal and randomized CoM offsets. With the
flag disabled Isaac Lab applies at the link origin, so no shift is needed.
Genesis applies at `link_com`; its adapter reads `get_links_pos(ref='link_com')`
and `ref='link_origin'` and shifts the label accordingly. Genesis still receives
the original command; Isaac Lab receives its equivalent link-origin wrench.
Offset replay metadata is diagnostic only.
QP, FF support guidance, PPO replay and deployment all consume the base-origin
prediction directly, preserving their existing normalization/clamping rules.

### Isaac Lab explicit-moment application fix (2026-11-25)

Application semantics `base_origin_explicit_moment_v2` correct an installed
Isaac Lab 0.54.2 / Isaac Sim 5.1.0.0 bug: the Warp kernel
`set_forces_and_torques_at_position` assigns requested torque, then overwrites it
with `r cross F` when positions are supplied. The adapter avoids that branch;
the installed package is not modified. Labels retain applied-load semantics and
normalization; this fixes physical delivery, not inertial-mismatch supervision.

Verified call chain: the composer converts world axes to **link-local axes**;
`Articulation.write_data_to_sim()` passes these buffers to PhysX
`ArticulationView.apply_forces_and_torques_at_position` with `is_global=False`
and `position_data=None`. The installed API documents the omitted position as
the **link transform origin, not CoM**. Hence submitting the equivalent moment
about that origin is necessary, and adds the lever arm exactly once.
`composer.reset()` refreshes its cached live pose and clears old loads;
`set`, not `add`, replaces commands. Zero commands clear the force and moment.

The installed deprecated `set_external_force_and_torque` wrapper delegates to
the same composer and supports world-frame, no-position calls. It is checked
separately in the startup audit. Older no-composer versions are covered by mock
adapter regression tests only: they must support `is_global` and the documented
no-position link-origin convention. No silent frame/CoM assumption or missing-
argument workaround is provided for incompatible APIs.

Old results with CoM-position application can have labels containing moments
which physics did not receive. They are not corrected-result caches. New frozen
evaluation manifests record application semantics; use new output directories.
Checkpoint tensor dimensions and labels are unchanged, but same-checkpoint
trajectories can change because explicit moments now physically act. A resumed
run must document this behavior change; no automatic retraining/migration occurs.

Focused CPU checks: `tests/test_hard_pact_applied_wrench.py` and
`tests/test_hard_pact_momentum_observer.py`. GPU sanity artifacts are under
`diagnostics/wrench_application_fix/`, using the **production path without**
`--momentum-sanity-correct-wrench`. Standing/walking use 10 environments, seed3,
300 controls, normal foot contact, and known 5 N / 0.5 Nm pulses. The startup
audit checks both application modes, signed axes, combined loads, randomized
offsets, replacement and clearing against composer buffers (absolute tolerance
2e-5 N/Nm), without stepping physics during the audit.

For subsequent corrected-load comparisons, diagnostic acceptance is deliberately
permissive: the startup buffer audit allows 0.01 N/Nm; the full-rollout delivery
check allows `0.01 + 0.001 * abs(expected component)` in N or Nm. Raw errors and
the tolerance are saved, and nonfinite values still fail. These are evaluation
checks only, not production training, QP solver, or feasibility tolerances.

Executed validation: 81 focused tests passed; standing and walking each completed
300 controls on physical GPU 1 with zero resets. Both 160-case startup audits
had maximum buffer error 2.861e-6 N/Nm; the direct installed position-kernel
probe reproduced a missing 0.5 Nm moment. The independent whole-body,
all-contact/oracle-state momentum observer recovered steady force components
approximately 5.000 N. For requested 0.5 Nm moments (X/Y/Z), standing recovered
0.49995/0.50018/0.49999 Nm; walking recovered 0.50150/0.48837/0.50759 Nm.
Walking has contact/discretization residuals; buffer agreement alone was not
used as physical validation. Nominal mass/CoM/inertia and zero joint passive
parameters were verified by readback, with an unrestrained base and normal
foot-ground contact. This is a short application sanity check, not a policy
performance or randomized-dynamics guarantee.

Exact test, GPU collection, and offline analysis commands:
`diagnostics/wrench_application_fix/commands.sh` (run from repository root).
Each run's `manifest.json` records checkpoint/config hashes and semantics;
`standing_analysis/` and `walking_analysis/` contain response CSVs, independent
momentum validation, and plots. `diagnostics/wrench_application_fix/RESULTS.txt`
records the implementation revision and limitations.

## Randomized inertia and liquid reactions

This change does **not** reinterpret randomized robot mass/CoM as an external
wrench. Actual randomized mechanics remain in representation PINNs; nominal
mechanics remain in the QP/support allocator. Thus inertial model mismatch is
still a limitation, distinct from the now-correct external-force moment arm.
Adding a nominal-model residual wrench would require a separate, explicit target
definition and removal of double-counted mechanics in PINNs.

The `liquid_wrench_surrogate` recorder already stores liquid-on-robot reaction
moments about a recorded base-link origin. Future integration must retain its
sign, recorded reference pose, and impulse/duration semantics, rotate body axes
to yaw-local, and shift origins only if they differ. Its 200-Hz labels use a fixed
endpoint origin, not an instantaneous moving-origin moment. Do not add payload
gravity/CoM moments again to a measured total reaction. No liquid adapter or
liquid-data conversion is implemented in this patch.

```bash
SIMULATOR=isaaclab python -m pytest -q -p no:cacheprovider \
  tests/test_hard_pact_applied_wrench.py tests/test_go2_hard_pact_disturbances.py \
  tests/test_go2_hard_pact_bard.py::CorrectedInverseLossTests \
  tests/test_go2_hard_pact_bard.py::RolloutLossTests \
  tests/test_hard_pact_contact_indexing_and_inverse_gate.py
```

The tiny CPU PPO/BARD update uses nonzero randomized mechanics and NaN legacy
mass-wrench diagnostics to ensure neither PINN consumes that diagnostic. It
also checks that both losses receive the same applied wrench and that the
existing encoder/decoder versus actor gradient routing is preserved.

## Fixed robot CoM and realized-inertia FF support (November 2026)

HardPACT and inherited comparison configurations use fixed torso offsets
X/Y `[-0.05, 0.05] m`, Z `[0, 0.05] m`. Set
`domain_rand.com_displacement_{x,y,z}_{min,max}=0.05`,
`com_displacement_z_positive=True`, `com_displacement_z_min_pos=0.0`,
and `use_mass_com_curriculum=False`. Here min/max are initial/final
*upper magnitudes*, not the two signed endpoints. A positive-Z floor equal
to its initial upper bound is rejected. Added mass remains `[-1,1] kg`;
joint-dynamics settings are unchanged.

FF support now uses detached realized **robot** mass/CoM gravity `g(q)`
(zero velocity/acceleration, no passive dynamics) and the stored applied,
world-frame, base-origin external-wrench label. Allocation remains
`C f = (g - J_b.T W)_base`, `tau_support = (g - J.T f - J_b.T W)_joints`.
The world label is the existing interval-averaged applied load; the support
geometry is the existing replay state. No external load is added to robot
mass, and no robot gravity residual is added to the decoder label. This is
quasistatic support supervision, not a desired-acceleration target. Targets
are recomputed under `no_grad`; no cross-epoch target cache is used.
QP mechanics remain nominal, including after support-target computation.

### Disturbance sequencing and resume

`algorithm.hard_pact_qp.intervention_ramp_iterations=8000` counts gate-open
substep-ramp iterations. Existing actor-loss/correction ramps and substep
gates are unchanged. The HardPACT runner then latches **actual** completion,
collects settling evidence, and advances disturbance strength only after
both settling and `domain_rand.push_warmup` (currently 10000) have elapsed.
Settings change after the complete PPO update, for the next rollout.

New `domain_rand` keys:

- `disturbance_sequence_enabled=True` (HardPACT; independent of objective gates).
- `disturbance_settling_iterations=1000`.
- `disturbance_gate_window_iterations=200` (complete non-overlapping windows).
- `disturbance_gate_min_env_seconds=1000.0` (minimum exposure per window;
  every iteration also needs finite tracking evidence and a positive count).
- `disturbance_survival_ratio=1.20`, `disturbance_failure_allowance=0.0`
  (absolute allowance in failures/environment-second).

Existing tracking settings are reused: `reward_ema_alpha=0.05`,
`best_reward_quantile=0.9`, `recovery_ratio=0.9`, `min_reward_to_step=0.50`.
The tracking reference is the maximum completed-window EMA quantile;
the failure reference is the minimum complete-window failure rate.
Thresholds are `max(min_reward_to_step, recovery_ratio*tracking_reference)`
and `disturbance_survival_ratio*failure_reference + disturbance_failure_allowance`.
Missing/invalid evidence closes the gate. Window decisions apply to subsequent
iterations; neither reference becomes more permissive after deterioration.
Each transition contributes once to failures and simulated exposure, excluding
timeouts and terrain relocations. `step_interval=10` and
`disturbance_progress_delta=0.005` require about 2000 eligible iterations.

No-QP HardPACT ablations collect settling evidence without waiting for a QP.
Fixed-sampling QP modes wait for their correction/actor ramps. RSS variants
inherit these settings. BR/MULE inherit fixed CoM/mass bounds and disturbance
step size, but explicitly disable this HardPACT-only sequence; their existing
physical-payload triggers, gates and optimization phases remain intact.

At the existing QP diagnostic cadence, `curriculum/disturbance/` reports CoM
bounds in metres, progress, phase, EMA, references/thresholds and settling count.
`pause_reason`: 0 waiting for QP, 1 settling, 2 insufficient evidence,
3 tracking, 4 failure rate, 5 open, 6 warmup, 7 step interval, 8 complete.
Phase IDs are 0 joint dynamics, 1 mass/CoM, 2 disturbance, 3 complete.

Checkpoints include the completion latch, partial windows, best references,
gate and schedule state. Same-config resume restores them exactly. Full
training-contract validation still rejects changed physical/config contracts;
old checkpoints are not silently migrated into these new bounds. At the
low-level curriculum compatibility path, a missing sequence starts fresh
evidence/settling, never inferred progress from an iteration number. Stage
transfer clears gate evidence but retains attained difficulty and timing.
Weights-only initialization remains distinct from full continuation.

Focused validation (no simulator or long training):

```bash
SIMULATOR=isaaclab PYTHONPATH=. python -m pytest -q \
  tests/test_hard_pact_disturbance_sequence.py \
  tests/test_go2_hard_pact_domain_rand_curriculum.py \
  tests/test_hard_pact_actor_regularizers.py \
  tests/test_hard_pact_applied_wrench.py \
  tests/test_hard_pact_rss.py \
  tests/test_hard_pact_intervention_curriculum.py
```

Use the existing `lr_lab_cupiqp` environment. On hosts with a conflicting
system C++ runtime, prepend `LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"`.
This includes bounded tensor/BARD and synthetic PPO checks, not Isaac Lab
simulation or a training-performance comparison.
