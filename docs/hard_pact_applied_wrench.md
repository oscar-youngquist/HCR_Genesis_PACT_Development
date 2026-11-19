# Base-Origin Applied External Wrench Targets (v3)

HardPACT, its thin variants and HardPACTPos now supervise the wrench decoder
with the externally applied training force and its equivalent moment about the
**base-link origin used by the BARD/Pinocchio base Jacobian**. The source is
the actual persistent-wrench buffer **after event ramps/gating**, not the
unramped event target. With no applied load, the label is zero even with
randomized mass and CoM. The decoder no longer represents inertial-parameter
mismatch.

For an application point P and base origin O, `M_O = M_P + (p_P-p_O) cross F`.
This label-only shift uses the actual rotated offset each physics substep,
before yaw conversion and control-interval averaging. Yaw-local axes do not
change the reference point. Normalization, active/neutral masks and transition
validity remain unchanged.
Instantaneous velocity kicks still have no force label and retain their
existing physics-validity exclusion. No physical randomization or force
application changes were made.

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
point. Isaac Lab's composer handles the equivalent link-origin moment; no
second lever-arm correction is added during application. Setting the flag to
`False` restores the previous Isaac Lab link-origin application. Missing flags
in legacy configurations also retain that behavior. Genesis already uses
`link_com` and is unchanged. This does not enable persistent wrenches in BR/MULE
or change instantaneous velocity pushes.

The label adapter includes both nominal and randomized CoM offsets. With the
flag disabled Isaac Lab applies at the link origin, so no shift is needed.
Genesis applies at `link_com`; its adapter reads `get_links_pos(ref='link_com')`
and `ref='link_origin'` and shifts the label accordingly. Physics still receives
the original, unshifted command. Offset replay metadata is diagnostic only.
QP, FF support guidance, PPO replay and deployment all consume the base-origin
prediction directly, preserving their existing normalization/clamping rules.

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
