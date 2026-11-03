# Go1/Go2 liquid-payload simulation

## Launch

Run from the repository root. The launcher activates `genesis_lr` (override with
`GENESIS_PACT_ENV`) and selects the Genesis liquid simulator automatically.

```bash
# Visible robot, sampled velocity commands
bash legged_gym/scripts/play_pact_water.sh \
  --robot go1 --num_envs 1 --gpu cuda:0 --follow_robot \
  --liquid_type water --liquid_volume 4

# Go2, several robots, explicit checkpoint and video
bash legged_gym/scripts/play_pact_water.sh \
  --robot go2 --num_envs 4 --follow_robot \
  --checkpoint /path/to/model_8000.pt --record_frames /path/to/video.mp4

# Headless evaluation with streamed CSV metrics
bash legged_gym/scripts/play_pact_water.sh \
  --robot go2 --headless --num_envs 16 --duration 60 \
  --fixed_cmd 0.5 0 0 --log_path /path/to/results.csv
```

Both robots accept identical options. `--use_joystick` enables manual commands;
`--terrain_type plane|rough` selects terrain; `--seed` controls reproducibility.
`--wall_opacity` controls container opacity (default .25).
`--steps` overrides duration. `--smoke` checks finite states, tank attachment,
actual joint properties/default pose, and partial fluid reset.

The older `play_go1_pact_water_exp.sh` and `play_go2_pact_water.sh` launchers
forward all arguments to this shared interface.

To view a continuously sampled payload using the collection pipeline's bounds:

```bash
bash legged_gym/scripts/visualize_liquid_collection.sh --num-envs 2
```

This samples a new configuration on each launch and prints its replay seed.
Use `--seed 1` to reproduce a sample, `--robot go1` to change robot, or
`--terrain training` for randomized training terrain. See the
[collection visual preview](liquid_surrogate_collection.md#visual-preview)
for camera, opacity and headless frame options.

## Checkpoints and evaluation settings

Use `--checkpoint FILE`, or `--load_run RUN_DIRECTORY --ckpt ITERATION`;
`--ckpt -1` selects the latest model. Default runs are configured in each robot's
liquid config: Go1 uses `May17_21-20-40_pact_100hz_spec_smartcurr_original_jointrand_hightau`;
Go2 uses `Aug01_18-27-22_pact_100hz_spec_smartcurr_stricterer`.

The player strictly loads supported legacy coupled PACT policies (57 actor
observations, 12 position + 12 feedforward outputs). Network dimensions/history
are inferred from weights. Selected settings from the adjacent saved training
config restore joint order/default pose, controller gains/scales/timing,
observation normalization, activation, and termination settings. Supply
`--config FILE` if that config is elsewhere. Missing saved configs produce a
warning and retain current task settings; activation can be overridden explicitly.
No optimizers or training curricula run during evaluation.

Joint dynamics use fixed midpoints of saved training ranges, preferring final
curriculum bounds. For features enabled in training, the simulator's existing
randomization APIs apply `[midpoint, midpoint]` at initialization/reset. Pushes
and additional randomized base mass/CoM are disabled for liquid evaluation.

Edit `PLAY_RESET_TERMINATION` near the top of
[play_pact_water.py](../legged_gym/scripts/play_pact_water.py) for manual
angle/height thresholds, contact termination, failure delay, episode length, or
out-of-bounds resets. `None` retains the saved/task value; explicit values win.
Startup prints the resolved policy, joint dynamics, and reset settings.

## Liquid configuration and physics

CLI presets support `water`, `oil`, and `gas` (gasoline-like **liquid**), with
nominal volumes `2, 4, 6, 8, 10, 12` litres. `--liquid_tank` supports `default`;
`wide`, `tall`, and `offset` are available for the 10-litre water preset.

Edit shared presets in
[liquid_payload_configs.py](../legged_gym/scripts/liquid_payload_configs.py):
`rho` controls density, `mu` viscosity, `gamma` surface tension,
`scale_x/y/z` tank dimensions, `offset` the liquid fill-box inset, and optional
`mount_offset` horizontal placement. Changes affect both robots. Volumes are
preset labels: actual particle mass depends on geometry, density and sampling.

Shared resolution, internal substeps, lid overlap and visual settings live in
[liquid_payload_config.py](../legged_gym/envs/base/liquid_payload_config.py).
Robot-specific configs set mounting height. Smaller `particle_size` increases
particle count/cost and may require more internal substeps.
The current defaults match collection: 10 mm particles and eight substeps.

```text
policy interval       = control.dt
PD update interval    = control.dt / control.decimation
internal physics dt   = control.dt / control.decimation / liquid.substeps
```

`liquid.substeps` controls Genesis's internal integration/collision substeps.
PD torque is held across them; increasing this setting does not increase the
PD update frequency.

The tank body is 20 × 15 × 20 cm before scaling, with nominal 1.5 cm walls/bottom.
Six rigid box parts attach to the torso. Wall corners overlap by 2 mm, the lid
overlap is 7 mm, and wall density compensates for the extra corner volume.
Contacts use 2 mm SDF cells and softness equal to half the particle size.
Genesis SPH particles fill an interior box and interact through fluid dynamics
and rigid–fluid collision coupling, transmitting loads through the mounted tank.
This is a closed-tank sloshing simulation. Translucent walls surround opaque blue
SPH particles; visual opacity does not change the fluid or contact material.

On episode reset, only affected environments restore their body-relative fluid
layout at the new robot pose and clear fluid velocity. Other environments retain
their fluid state.

## Code paths

The unified player uses `GenesisSimulator_PACT_Liquid`, which extends the legacy
PACT inference adapter with payload creation, rendering and resets:
[genesis_simulator_pact_liquid.py](../legged_gym/simulator/genesis_simulator_pact_liquid.py).
It reuses standard Go1/Go2 PACT observations and control.

The original standalone Go1 liquid implementation remains in
`play_exp_water.py` / `genesis_simulator_pact_water.py`; use the unified launcher
for the current interface. Separately, the older batch experiment scripts'
`--disturbance_type payload` randomizes rigid added mass/CoM; it does not enable
SPH liquid transport.
