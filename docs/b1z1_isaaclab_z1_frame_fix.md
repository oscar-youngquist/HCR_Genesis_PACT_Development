# IsaacLab Z1 asset frame correction

## Cause and evidence

IsaacLab loads `asset.isaacgym_file` through `prepare_lab_urdf()` in
`legged_gym/simulator/b1z1_lab_assets.py`. The original `b1z1.urdf` applies an
extra +90-degree X rotation to the nine Z1 mesh visuals, their inertial frames,
and the two gripper mesh collisions. The DAE vertices are already in the
link-local convention expected by Lab. The arm's primitive collision shapes
use separate, intentional transforms.

The preexisting USD at `/tmp/IsaacLab/usd_20260923_173519_2201/b1z1.usd`
contains the rotated geometry. For example, link03's mesh Z bounds change
from approximately [-0.0325, 0.0915] m in the DAE to [-0.0398, 0.0225] m
after import, exchanging its Y/Z extent. The gripper collision meshes are
rotated too. This is evidence from the imported asset, not just the screenshot.

The archived configuration in
`logs/b1z1/b1z1_pact_lab/Sep23_17-35-31_b1z1_pact_improved/b1z1_pact_config.py`
selects this Gym URDF for Lab, while BARD uses `b1z1_genesis.urdf`, whose arm
inertial rotations are zero. Thus the configured simulation and dynamics model
also used different arm inertia orientations.

## Correction

The shared Lab asset preparation now removes the redundant quarter-turn from
Z1 visuals and mesh collisions, and matches the existing Genesis/BARD inertia
frame convention. It preserves joint origins/axes/limits, COM translations,
mass values, inertia coefficients, and primitive collision geometry. Source
URDFs are not modified. A new cache version prevents reusing the old converted
asset. All B1Z1 variants using this shared Lab backend receive the correction.

The correction is unconditional: there is no legacy-loading option.
Already-running simulators retain their loaded assets until restart.

## Training implications

This was **not exclusively a rendering defect**: gripper collision geometry
and arm inertia orientation were affected. Existing checkpoint weights remain
loadable, but evaluation with corrected physics is a domain change. Reevaluate
tracking and contact behavior; retrain with corrected assets for clean
experimental comparisons. The magnitude of the learning impact cannot be
inferred from the screenshot or asset inspection alone.

Headless training and visual playback use the same unconditional robot spawn
and URDF preparation in `IsaacLabSimulatorB1Z1`; the headless flag does not
select different collision or inertia assets. The cached USD explicitly stores
the rotated gripper collision geometry and arm principal inertia axes. Runtime
mass randomization scales imported default inertias rather than correcting
their orientation. Turning rendering off therefore did not avoid this issue.

For results intended to represent the corrected robot, retraining is
recommended. The existing policy may still work or provide a useful warm
start, but neither is guaranteed, and fine-tuning is not equivalent to a clean
training comparison. A post-fix replay is needed to measure performance loss;
no new training or GPU simulation was launched for this asset audit.

Joint transforms and the EE kinematic chain are unchanged; no compensating
offset should be added to EE targets or forward kinematics. Matching the
existing BARD inertia convention establishes internal consistency, not an
independent physical calibration of the manufacturer's inertia data.

## Focused validation

`tests/test_b1z1_lab_assets.py` checks corrected conversion consistency, unchanged
source files and joint transforms, preservation of primitive collisions and
mass/COM data, corrected mesh frames, agreement with BARD inertia origins,
cache reuse, and correction idempotence. These are asset-level tests;
they do not establish post-fix policy performance or replace a visual replay.
