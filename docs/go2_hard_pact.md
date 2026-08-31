# Go2 HardPACT

This implementation adds one simulator-neutral Go2 HardPACT core, thin coupled
and position-pretraining task classes, and Genesis and Isaac Lab adapters. The
backend is selected with `SIMULATOR`; there are no per-backend environment or
configuration folders. The legacy `go2_pact` and `go2_pact_pos` registrations
are unchanged.

## Canonical contract

- Joint and foot order: `FR, FL, RR, RL`, verified by exact joint/link names.
- Policy/task quaternion order: XYZW. BARD and Genesis WXYZ states are converted
  at their boundaries.
- Generalized velocity/wrench order: base linear, base angular, then 12 joints.
- Base twists are converted to world coordinates in transitions and to body
  coordinates in actor/estimator labels.
- GRF and QP wrench coordinates are yaw-local, in N and Nm. Validation wrenches
  are also retained in world coordinates.
- Control `dt` is checked against physics `dt * decimation` at startup.
- QP contact normals are always the fixed gravity normal `-g/|g|` in the QP
  force frame. Terrain maps and estimated surface normals never enter the
  actor, physics heads, QP, or 79-D reconstruction target.

The actor input is ordinary concatenation of the 57-D observation, deterministic
16-D history mean, and 11-D explicit estimator. The privileged decoder target
uses the named schema in `schema.py`; its first 33 fields are the non-force next
state and its remaining 46 fields are system-identification values. The GRF and
base-wrench heads are separate deployment-available estimators.

## Backend capabilities

| Capability | Genesis | Isaac Lab |
|---|---:|---:|
| Terrain representation | heightfield | triangle mesh |
| Ideal total-torque tracking | yes | yes |
| Substep GRF accumulation | yes | yes |
| Sustained base wrench | yes | yes |
| Reset/static domain randomization | yes | yes |
| Runtime domain-randomization curriculum | yes | no |
| Motor-strength randomization | yes | no |
| Joint-stiffness randomization | yes | no |

Requested, supported, active, and effective domain-randomization ranges are
written to run metadata. Unsupported flags are disabled before simulator
creation; no backend claims nominal values as active randomization.

Isaac Gym HardPACT is intentionally not registered: its Python 3.8 environment
cannot install the official BARD backend. Existing unrelated Isaac Gym tasks
are unchanged.

## Dependencies

The differentiable reference QP imports `qpth`. The rigid-body losses import the
official BARD package only when `bard.enabled=True`.

```bash
conda run -n genesis_lr pip install --no-deps qpth
conda run -n genesis_lr pip install 'git+https://github.com/YueWang996/bard-pytorch-dynamics.git@d272de7'
conda run -n lr_lab pip install qpth
conda run -n lr_lab pip install 'git+https://github.com/YueWang996/bard-pytorch-dynamics.git@d272de7'
```

The Genesis environment pins NumPy 2.1.2, while qpth 0.0.18 advertises an older
NumPy constraint; retain the environment's NumPy pin when installing qpth.

### Isaac Lab environment

The requested environment is `lr_lab` with Python 3.11, Isaac Lab v2.3.2,
CUDA PyTorch 2.7, the current workspace installed editable, qpth 0.0.18, and
BARD 0.4.3. The relevant setup is:

```bash
conda create -y -n lr_lab python=3.11
conda run -n lr_lab pip install --upgrade "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git /home/oyoungquist/Research/IsaacLab
TERM=xterm conda run -n lr_lab /home/oyoungquist/Research/IsaacLab/isaaclab.sh --install none
conda run -n lr_lab pip install --no-deps -e .
conda run -n lr_lab pip install qpth "git+https://github.com/YueWang996/bard-pytorch-dynamics.git@d272de7"
```

On the current Ubuntu 20.04 host, Isaac Sim 5.1.0 cannot be installed: the host
provides glibc 2.31 while NVIDIA's wheel is tagged `manylinux_2_35`. Isaac Lab,
CUDA PyTorch, qpth, BARD, and this workspace are installed in `lr_lab`, and the
registration/trimesh/CUDA checks pass, but simulator smoke remains blocked until
the host is upgraded to a compatible OS/glibc or Isaac Sim 5.1.0 is supplied
through a compatible container/runtime.

## Launch commands

The launch script activates the correct Conda environment, exports `SIMULATOR`,
and rejects CPU execution. It accepts `backend`, `task`, then extra training
arguments:

```bash
GPU=cuda:0 bash legged_gym/scripts/go2_hard_pact.sh genesis go2_hard_pact
GPU=cuda:0 bash legged_gym/scripts/go2_hard_pact.sh isaaclab go2_hard_pact
```

Neutral position-policy pretraining:

```bash
GPU=cuda:0 bash legged_gym/scripts/go2_hard_pact_pos.sh genesis
GPU=cuda:0 bash legged_gym/scripts/go2_hard_pact_pos.sh isaaclab
```

The eight thin ablation registrations are:

```text
go2_hard_pact_baseline
go2_hard_pact_soft_only
go2_hard_pact_hard_only
go2_hard_pact_full
go2_hard_pact_stop_gradient_qp
go2_hard_pact_soft_penalty
go2_hard_pact_inverse_only
go2_hard_pact_rollout_only
```

Launch each with either selected backend, for example:

```bash
GPU=cuda:0 bash legged_gym/scripts/go2_hard_pact.sh genesis go2_hard_pact_baseline
GPU=cuda:0 bash legged_gym/scripts/go2_hard_pact.sh isaaclab go2_hard_pact_baseline
```

Substitute any of the eight task names above; task logic and configuration stay
in the approach folder while the backend remains a launch-time choice.

## Checkpoint transfer

Load a HardPACTPos checkpoint into HardPACT through `Go2HardPACTRunner.load`.
Migration requires every shared tensor to match, allows only initialization of
`std[12:24]`, and finishes with `load_state_dict(..., strict=True)`. Position
optimizer moments are deliberately not migrated because their action standard
deviation has a different shape.

## Verification

```bash
SIMULATOR=genesis NUMBA_DISABLE_JIT=1 XDG_CACHE_HOME=/tmp/hard_pact_unit_cache MPLCONFIGDIR=/tmp/hard_pact_mpl_cache conda run -n genesis_lr python -m unittest discover -s tests -p 'test_go2_hard_pact_*.py'
SIMULATOR=isaaclab conda run -n lr_lab python -m unittest discover -s tests -p 'test_go2_hard_pact_*.py'
SIMULATOR=genesis NUMBA_DISABLE_JIT=1 HARD_PACT_SMOKE_PHYSICS=1 XDG_CACHE_HOME=/tmp/hard_pact_genesis_cache MPLCONFIGDIR=/tmp/hard_pact_mpl_cache conda run -n genesis_lr python tests/smoke_go2_hard_pact_backend.py --task go2_hard_pact --num_envs 2 --headless --gpu=cuda:0
SIMULATOR=isaaclab HARD_PACT_SMOKE_PHYSICS=1 conda run -n lr_lab python tests/smoke_go2_hard_pact_backend.py --task go2_hard_pact --num_envs 2 --headless --gpu=cuda:0
```
