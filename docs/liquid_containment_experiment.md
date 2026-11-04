# Bounded liquid-containment experiments

Run from the repository in `genesis_lr` on `cuda:0`. The production launcher and both visualization paths now default to the recommended
`planes_10mm_4` settings: 10 mm particles, four substeps and 5 mm contact softness.
The experiment runner provides explicit alternative profiles; its `current` profile
retains the historical 10 mm/eight-substep SDF baseline. No installed Genesis files are changed.

```bash
python -m liquid_collection.containment \
  --output runs/my_containment_trial --profile planes_8mm_4 \
  --robot go2 --payload high_fill --max-seconds 900
```

The default trial uses two environments for 11 seconds: one second settling followed by ten seconds of forward, stop, reverse, and alternating lateral/yaw commands, clipped to the saved policy's training ranges. It uses play's termination settings and saved joint-dynamics midpoints. Commands are applied before the policy observation is built. An early robot termination cannot count as a full-duration containment pass.

Use `--seed` with fresh values for independent repeats. `--payload random` uses the existing bounded sampler; `--payload viscous_limit` tests the approved upper density, viscosity and surface-tension bounds with the viscous case’s geometry. Mount bounds never expand.

Every native substep is checked for escaped, nonfinite, inactive, detached and domain-contact particles. Those checks are reduced into the normal 200 Hz saved stream; raw particle trajectories are not saved. Escapes terminate a candidate early. Inner-wall penetration is reported separately, with a maximum allowed value of particle radius plus 1 mm. The experiment never clips or teleports particles to make containment pass.

The `planes_*` profiles retain the six translucent boxes, their mass, cavity dimensions, mount and rigid contacts. Six inward-facing analytic planes provide fluid-only contact, eliminating box seams and finite-wall normal reversals. Box fluid coupling is disabled to prevent duplicate forces. A reserved contact-mask bit isolates the planes from robot and terrain rigid contacts, and a runtime audit checks masks and shell mass. Contact softness defaults to half the particle size. The planes remain attached to the moving robot; they are not a fixed world-space particle boundary.

Use short scaling probes before choosing a production resolution, particularly for 50 environments:

```bash
python -m liquid_collection.containment \
  --output runs/my_scaling_reference --reference \
  --robot go2 --payload high_fill --num-envs 50 --timing-steps 50 --max-seconds 900
python -m liquid_collection.containment \
  --output runs/my_scaling_candidate --profile planes_8mm_4 \
  --robot go2 --payload high_fill --num-envs 50 --timing-steps 50 --max-seconds 900
```

These short probes measure throughput and memory, not ten-second containment. The reference uses the pre-swap mesh, 12 mm particles, two substeps and no recorder. Candidate timings include the 200 Hz recorder. Compare the same robot, payload, environment count and probe duration; a short settling probe does not establish long-run throughput across every maneuver. Metrics separate initialization, first step and subsequent steps, and include process GPU-memory sampling as well as PyTorch memory. Reducing particle spacing increases particle count approximately cubically, so two-environment timings cannot establish 50-environment feasibility.

Each fresh output directory retains the exact job, criteria, source hashes, checkpoint/config identities, resolved contacts, sampled terrain, command events, validity data, runtime metrics and worker log. Summarize directories under a common root with:

```bash
python -m liquid_collection.containment_report runs/liquid_containment_20261103
```

A containment pass is limited to its tested payloads, policies and motion duration. It does not approve the surrogate's physics losses or a full collection campaign. Repeat the format/physics validation with any adopted contact settings before making those claims.
