# Applied-Only External Wrench Targets

HardPACT, its thin variants and HardPACTPos now supervise the wrench decoder
with only the externally applied training force and moment. The source is
the actual persistent-wrench buffer **after event ramps/gating**, not the
unramped event target. With no applied load, the label is zero even with
randomized mass and CoM. The decoder no longer represents inertial-parameter
mismatch.

Substep averaging, world/yaw transforms, application/reference points,
normalization, active/neutral masks and transition validity remain unchanged.
Instantaneous velocity kicks still have no force label and retain their
existing physics-validity exclusion. No physical randomization or force
application changes were made.

The historical `total_external_wrench_label_*` field names and widths remain
for storage/critic compatibility; "total" now means total **applied** external
loading. `equivalent_mass_com_wrench_*` stays a separate legacy diagnostic,
not a decoder target or a PINN correction. Existing target-based diagnostics
and privileged label blocks consume the corrected fields automatically.

Both inverse and rollout PINNs rotate the prediction and retain the existing
application-point-to-base moment shift, then use that applied wrench directly.
No mass/CoM wrench is subtracted. Realized mass/CoM and passive parameters
remain in the actual BARD mechanics. Gradient routing is unchanged. The
inverse helper retains its historical argument names and accepts an ignored
`mass_com_wrench_world` compatibility argument, but production no longer
supplies it.

QP objectives, constraints, nominal mechanics and execution are unchanged.
Their input remains the learned wrench. They must not be interpreted as
receiving an estimate of inertial mismatch under the new supervision.

New deployment contracts and frozen evaluation metadata identify applied-only
target semantics as version 2. Older checkpoints were trained on different
(applied plus mass/CoM gravity) labels. Tensor shapes remain loadable, but their
predictions are **not semantically interchangeable**: do not interpret or
resume them as applied-only estimators without accounting for retraining.

## Focused Validation

### Isaac Lab randomized-CoM application

New HardPACT/HardPACTPos configs enable
`domain_rand.persistent_wrench_at_com = True`. Persistent forces are applied at
the torso's nominal CoM plus its current randomized displacement, rotated into
world space each physics substep. Supplied torques are free moments about that
point. Isaac Lab's composer handles the equivalent link-origin moment; no
second lever-arm correction is added during application. Setting the flag to
`False` restores the previous Isaac Lab link-origin application. Missing flags
in legacy configurations also retain that behavior. Genesis already uses
`link_com` and is unchanged. This does not enable persistent wrenches in BR/MULE
or change instantaneous velocity pushes.

Decoder labels remain applied force/moment at the application point. New
Isaac Lab transitions carry the complete body-frame application offset for
the existing PINN application-point-to-base conversion (including nominal CoM,
not just random displacement). Old transitions retain their existing fallback.
QP prediction/solver settings and deployment behavior are not changed here.
This changes the training disturbance's moment arm; continuing an old run with
the new flag is a deliberate physical-setting change, not identical replay.
Fresh configs enable it; a resumed run may retain its saved configuration.

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
