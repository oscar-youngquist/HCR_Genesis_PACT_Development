# PILOT Figure 1 diagnostic milestone

Evaluation-only branch `eval/pilot-figure1-milestone`, based on `2bf835d1`.
This is a physical-interaction diagnostic, not a trained-performance claim.

## Reproduction

From the repository root in the Isaac Lab Python environment:

```bash
bash legged_gym/scripts/eval_pilot_figure1.sh --validate-only --output /tmp/pilot-preflight
bash legged_gym/scripts/eval_pilot_figure1.sh --scenario nominal --gpu cuda:1 --headless --trials 3 --horizon 20 --output analysis/figure1_nominal
bash legged_gym/scripts/eval_pilot_figure1.sh --scenario support-follow --gpu cuda:1 --headless --trials 3 --horizon 20 --output analysis/figure1_support
bash legged_gym/scripts/eval_pilot_figure1.sh --scenario adjust-hold --gpu cuda:1 --headless --trials 3 --horizon 20 --output analysis/figure1_hold
```

Use `--checkpoint 21400` for explicit selection, otherwise the highest numeric
checkpoint in the exact requested run is selected. No substitute run is allowed.
`--scenario nominal` creates no payload/partner prims. `--scenario all` reuses
one scene, with its props parked away from the robot during nominal trials.
Use `--config path.json` for overrides of the documented `CONFIG` dictionary in
`legged_gym/utils/pilot_figure1.py`. Output directories must be new.
`--video --video-fps 10` enables cameras explicitly even headless. Capture uses
render-only updates and checks that the control counter and FiLM history do not
change. Capture failure is an error, not a successful empty video.

## Provenance and limitations

Located checkpoint: `Nov13_09-58-29_b1z1_pact_improved/model_21400.pt`.
SHA256: `e5e96acf00b00f19ddb3ee13e82ef7e18de660085c6943b5d8b70f6385d5252d`.
The archived configuration and strict actor loader are reused; hashes and resolved
configuration are exported. The archived Python configuration inherits current
compliance defaults, so it is not a fully frozen historical configuration.
The corrected Z1 asset can differ physically from the training asset. These are
explicit limitations even when weight shapes and finite inference pass.

The grasp is force-only: no orientation constraint or human hand simulation.
A 1.5 kg dynamic box is coupled to the robot TCP and a kinematic partner proxy
by bounded 150 N/m, 10 Ns/m springs, capped at 45 N. Gravity acts on the box
once. Robot/box collisions are filtered to avoid duplicate contact transfer;
box/floor contacts remain physical. The robot is never kinematically anchored.
Springs are evaluated at control boundaries and held over physics substeps.
The same force acting **on the robot**, in world newtons, is passed to compliance
and the simulator. Equal/opposite box forces and their current lever moments
are refreshed each substep. No additional payload weight or torso load is added.

Profiles use EE `[-x,+x,-y,+y,-z,+z]`, then base
`[-x,+x,-y,+y,-yaw,+yaw]`. Fixed reset profiles are held for each trial.
Nominal base velocity is zero. At 4 s the payload attaches; support-follow
lifts the partner by 0.15 m at 6 s and returns at 8 s. Guidance runs 10--16 s,
then holds until 20 s. Support-follow guides forward; adjust-hold guides laterally
with all base compliance zero. Arm planar gains stay 0.6: restriction is not
manufactured by changing the compliance gates. Nominal EE height stays fixed.
Random force events, kicks and command/profile resampling are disabled locally;
training defaults and policy weights are not changed.

## Recording and interpretation

`metadata.json` contains configuration, frames, joint ordering and checkpoint
identity. Each scenario/trial has compressed `trajectory.npz` and `summary.json`.
Rows are control intervals, with interval-start observations/actions/loads and
physical successor state before reset. `substep_interaction` contains actual
application points and box force/moment bookkeeping; see the metadata field
dictionary. Unavailable pre-attachment geometry is NaN, not zero measurement.
Clipping comes from `compliance.target.clipped`. Failures are retained and not
padded. `COMPLETE` means evaluation ended, not that behavior checks passed.
An `ERROR.txt` without COMPLETE indicates an interrupted/failed evaluation.

Behavior tolerances are predefined in CONFIG. Gate activation plus base motion
is only a screening diagnostic, not evidence that decreasing directional
capability caused successful recruitment. Inspect tracking, force directions,
guidance-relative motion and failure records together. Initial transients are
included in whole-trial errors; phase metrics use the specified time windows.

## Validation status

CPU tests cover exact checkpoint selection, force signs/capping/rotations,
equal/opposite substep loads and moments, profile independence and schedule
endpoints and missing-phase/guidance-relative summaries. Tests also reconstruct
recorded handle-force and lever-moment balance independently.

An initial GPU-1, 8-second nominal trial completed 400 steps with unchanged
weights and no termination, but mean nominal EE error was 0.212 m and maximum
base drift 0.297 m: it failed the predefined 0.15 m criteria.
Do not describe this checkpoint as validated for the demonstration.

The installed RGB annotator raised an Omniverse dependency dtype error. The
evaluation uses Kit viewport capture instead, explicitly enabling the viewport
updates that Lab disables headless. A 1-second no-prop camera check succeeded;
its PNG was visually inspected. The 20-second nominal video run reproduced the
non-rendered trial's reported errors exactly. No simulator package was patched
or upgraded. Numerical trials and videos, including failed behavior checks, are
retained under `analysis/pilot_figure1_milestone/`. Earlier capture failures remain
in their separate output folders for traceability.

Final GPU-1 validation completed nine numerical 20-second trials (three seeds
per scenario) and three 20-second video trials. All finished without termination
and preserved policy hashes. All three videos contain 200 nonblank 960x640 frames;
representative interaction PNGs were visually inspected. The final CPU suite
passed 32 tests; shell syntax and `git diff --check` passed.

The demonstration **failed its behavioral criteria**. Support-follow carried
7.15--8.80 N downward during the support window, but all three trials failed
upward yielding and recovery. Guidance-relative base progress was
0.143/0.051/0.065 m; joint-gate increases were only 0.0006/0/0.031, so this is
not evidence of increasing capability-driven recruitment. Adjust-hold had
0.181/0.118/0.138 m maximum base drift; two passed the 0.15 m station threshold.
All failed the whole-trial arm-displacement threshold, which includes settling.
No gates, policy weights, or tolerances were changed to manufacture success.

Outputs:
- `numeric/`: three seeds per scenario, compressed trajectories and original summaries.
- `videos/`: one filmed trial per scenario, MP4 and PNGs, with matching trajectories.
- `validation_report.json`: all 12 trials, additional guidance-relative metrics,
  coupling audits, video counts, unavailable diagnostics and limitations.

Recorded force balance residuals were below 1e-5 N and lever-moment residuals
below 5e-4 Nm (world-position subtraction uses float32). These audit the explicit
composer inputs, not independent force-sensor or momentum measurements.
Nominal render/non-render states and actions were identical. Interaction runs
had different preceding trial histories and are not asserted bitwise identical.
A separate rotated-robot simulator trial and deliberate terminal-failure smoke
were not run; rotated force algebra is covered by CPU tests. Historical dynamics
compatibility and trained behavior remain unverified. No training was launched.

```bash
SIMULATOR=isaaclab_b1z1_pact PYTHONPATH=. python -m pytest -p no:cacheprovider \
  tests/test_pilot_figure1.py tests/test_b1z1_compliance_review.py tests/test_b1z1_torso_compliance.py -q
```
