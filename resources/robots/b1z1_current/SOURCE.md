# B1Z1 Visual Wholebody Asset

This asset was copied from:

- https://github.com/Ericonaldo/visual_wholebody/tree/main/low-level/resources/robots/b1z1
- Source commit: `869104c31953718f30ad20675e5291fcb5c5ea23`

The B1Z1 task points to `urdf/b1z1.urdf`, matching the upstream
`low-level/legged_gym/envs/manip_loco/b1z1_config.py` asset path.

Local compatibility note:

- Corrected the `base_static_joint` origin from `xyz="0.3 0 0.09>>"` to
  `xyz="0.3 0 0.09"` so Isaac Gym can parse the numeric origin field.
