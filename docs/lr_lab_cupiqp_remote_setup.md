# Recreating lr_lab_cupiqp on a remote training server

Inspected locally on 2026-09-22. This guide reproduces the installed environment;
it is not a claim that its dependencies are mutually compatible or that the
remote simulator has been validated. No environment packages were changed.

## 1. Choose the right server

Use Linux x86_64, preferably Ubuntu 22.04 to match the source host, with glibc
2.35 or newer. Isaac Sim 5.1 requires Python 3.11 and glibc 2.35+ for its Linux
wheels. See [NVIDIA's versioned installation instructions](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html).

Choose an RTX-capable GPU. NVIDIA explicitly lists A100/H100 GPUs without RT
cores as unsupported for this simulator; headless operation does not establish
support for them. The 5.1 requirements list Linux driver 580.65.06 as a tested
version. Ask the server administrator to provision a compatible NVIDIA driver
and Vulkan support, including graphics support in containers. See
[Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).

For large B1Z1 runs, provision ample VRAM, host RAM, and local SSD storage; measure
the actual task before choosing environment count. Reserve space for the packed
archive, extracted environment, source/assets, compilation caches, and logs.
The environment archive does not contain the host NVIDIA driver.

On the server, check:

```bash
uname -m
ldd --version
nvidia-smi
vulkaninfo --summary
df -h "$HOME"
```

Have the administrator install `vulkaninfo` if unavailable. A successful
`nvidia-smi` alone does not verify that Isaac Sim can initialize Vulkan.

## 2. Installed reference stack

| Component | Observed version/revision |
| --- | --- |
| Python | 3.11.16 |
| Isaac Sim and extension-cache packages | 5.1.0.0 |
| IsaacLab checkout | `37ddf626871758333d6ed89cf64ad702aef127d0` |
| isaaclab Python package | 0.54.2 |
| PyTorch / torchvision / torchaudio | 2.7.0+cu128 / 0.22.0+cu128 / 2.7.0+cu128 |
| NumPy / SciPy | 1.26.0 / 1.15.3 |
| Warp / CuPy | 1.17.0 / cupy-cuda12x 13.6.0 |
| nvmath-python / cuda-toolkit | 0.7.0 / 12.9.2.0 |
| pin / libpinocchio | 3.9.0 / 3.9.0 |
| BARD git revision | `d272de725500a4fe8ea815cba33614f5ba06759a` |
| cupiqp git revision | `9b601101cf6c933db40188f6f60a7f34f9cefc93` |
| socu git revision | `42318d3b737994537754ad43ad87cd24c76ce013` |

IsaacLab's six source packages and this HCR repository are editable installs.
Their source directories are outside the conda environment and must be
transferred separately. The local IsaacLab checkout was clean; HCR includes
uncommitted additions, so cloning only its HEAD will omit those additions.

Do NOT use this repository's `environment.yml`: it describes `genesis_lr`, Python
3.10, Genesis, and a different PyTorch/CUDA stack. Do not install a PyPI `rsl-rl`
over the repository's custom `rsl_rl` implementation.

## 3. Important dependency caveat

The repository now includes `environment.lr_lab_cupiqp.yml`, a separate
`conda env export --no-builds` reference with the local prefix removed,
NVIDIA/PyTorch wheel indexes added, and git-installed packages pinned to source
revisions. HCR itself must be installed from the transferred checkout afterward.
This does not replace the Genesis `environment.yml`. It has been YAML-validated,
not dependency-solver or installation validated. Because the source contains
conflicts and pip-overwritten conda packages, this export is not guaranteed to
recreate a runnable environment with a single `conda env create` command.

To try a fresh recreation in an isolated staging environment (not an existing
training environment):

```bash
conda env create -n lr_lab_cupiqp_staging -f environment.lr_lab_cupiqp.yml
```

If solving fails, retain the error and use the archive procedure below for
source-state migration; do not silently remove pins to make the solve pass.

The inspected environment does NOT pass `pip check`. Examples:

- cupiqp requires NumPy >=2.0, but the environment has 1.26.0.
- cmeel-boost 1.89 requires NumPy >=2.3 on this Python version.
- Torch expects nvidia-cublas-cu12 12.8.3.14, but 12.9.2.10 is installed.
- Several Isaac Sim pinned dependencies differ from installed versions.
- hpp-fcl requirements conflict with the installed cmeel packages.

Preserving these versions reproduces the source state, not a clean dependency
resolution. Do not blindly upgrade NumPy or CUDA libraries to fix one package:
that can break another compiled dependency. Validate the intended B1Z1 task and,
if needed, resolve dependencies in a separate staging environment. The most
recent tiny local IsaacLab check failed GPU-device initialization before stepping;
successful CPU tests are not simulator validation.

## 4. Export on the source workstation

Run these commands on the original workstation, not the remote server. Use a new
export directory and ensure sufficient space. They do not modify the training env.

```bash
export SRC_ENV="$HOME/.conda/envs/lr_lab_cupiqp"
export HCR="$HOME/Research/Genesis_Development/HCR_Genesis_PACT_Development"
export LAB="$HOME/Research/IsaacLab"
export BUNDLE="$HOME/lr_lab_cupiqp_transfer"
mkdir -p "$BUNDLE"

conda list -p "$SRC_ENV" --explicit > "$BUNDLE/conda-explicit-linux-64.txt"
conda env export -p "$SRC_ENV" > "$BUNDLE/environment-reference.yml"
"$SRC_ENV/bin/python" -m pip freeze --all > "$BUNDLE/pip-freeze-reference.txt"
"$SRC_ENV/bin/python" -m pip list --format=json > "$BUNDLE/pip-installed.json"
"$SRC_ENV/bin/python" -m pip list --editable > "$BUNDLE/editable-installs.txt"
"$SRC_ENV/bin/python" -m pip check > "$BUNDLE/pip-check-source.txt" 2>&1 || true
git -C "$LAB" rev-parse HEAD > "$BUNDLE/isaaclab-commit.txt"
git -C "$HCR" rev-parse HEAD > "$BUNDLE/hcr-commit.txt"
git -C "$HCR" status --short > "$BUNDLE/hcr-worktree-status.txt"
```

The raw freeze contains conda build-machine `file:///...` paths and editable
paths. It is an audit record, NOT a portable `pip install -r` file.

Install the packaging tool in its own environment, not the active training env:

```bash
conda create -n lr_env_pack -c conda-forge conda-pack -y
conda run -n lr_env_pack conda-pack -p "$SRC_ENV" \
    --ignore-editable-packages -o "$BUNDLE/lr_lab_cupiqp.tar.gz"
```

`--ignore-editable-packages` bypasses the editable-install check; it does not
bundle their source trees. If packing reports missing/overwritten conda files,
stop and inspect the report. Do not automatically add `--ignore-missing-files`.
Relocation needs a compatible OS; choose the final destination before running
`conda-unpack`. See [conda-pack usage and caveats](https://conda.github.io/conda-pack/)
and [its CLI options](https://conda.github.io/conda-pack/cli.html).

Archive both source trees, including local edits and robot resources:

```bash
tar -czf "$BUNDLE/IsaacLab-source.tar.gz" \
    --exclude='IsaacLab/logs' --exclude='IsaacLab/_isaac_sim' \
    -C "$(dirname "$LAB")" "$(basename "$LAB")"
tar -czf "$BUNDLE/HCR-source.tar.gz" \
    --exclude='HCR_Genesis_PACT_Development/logs' \
    --exclude='HCR_Genesis_PACT_Development/.cache' \
    -C "$(dirname "$HCR")" "$(basename "$HCR")"
cd "$BUNDLE"
sha256sum *.tar.gz > SHA256SUMS
```

Inspect archives before transfer for credentials, private data, and symlinks to
assets outside these directories. Transfer any required external assets too.
Check NVIDIA's applicable licenses before redistributing the packed runtime;
keep these archives private. Checkpoints excluded under `logs` can be transferred
separately when resuming a run.

```bash
# Replace this destination with your actual SSH account/server.
rsync -avP "$BUNDLE/" user@server:~/lr_lab_cupiqp_transfer/
```

## 5. Restore on the server

Install conda/Miniforge for Linux x86_64 first if needed. Do not extract over an
existing environment. The following paths are examples; choose persistent paths
before unpacking, then keep them fixed.

```bash
export BUNDLE="$HOME/lr_lab_cupiqp_transfer"
export ENV="$HOME/.conda/envs/lr_lab_cupiqp"
export WORK="$HOME/Research"
cd "$BUNDLE"
sha256sum -c SHA256SUMS
test ! -e "$ENV"
mkdir -p "$ENV" "$WORK"
tar -xzf lr_lab_cupiqp.tar.gz -C "$ENV"
tar -xzf IsaacLab-source.tar.gz -C "$WORK"
tar -xzf HCR-source.tar.gz -C "$WORK"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
conda-unpack
export LAB="$WORK/IsaacLab"
export HCR="$WORK/HCR_Genesis_PACT_Development"
```

Rebind editable packages to the NEW paths, without resolving/upgrading dependencies:

```bash
for pkg in isaaclab isaaclab_assets isaaclab_contrib isaaclab_mimic isaaclab_rl isaaclab_tasks; do
    python -m pip install --no-deps --no-build-isolation -e "$LAB/source/$pkg"
done
python -m pip install --no-deps --no-build-isolation -e "$HCR"
python -m pip list --editable
python -m pip check > "$BUNDLE/pip-check-remote.txt" 2>&1 || true
diff -u "$BUNDLE/pip-check-source.txt" "$BUNDLE/pip-check-remote.txt" || true
```

Minor ordering/path differences in `pip check` are expected; investigate new
dependency failures. Do not run `isaaclab.sh --install` or an unpinned
`pip install -e '.[isaaclab]'` over this restored environment: these can change
the versions you just reproduced. Rebinding local editables is intentional.

## 6. Validate before large-scale training

Run inside an allocated GPU job, not a cluster login node. Select a GPU assigned
to you; do not clear a scheduler's `CUDA_VISIBLE_DEVICES`. `cuda:0` means the first
visible CUDA device, which need not be physical GPU 0. With unrestricted device
visibility, use the actual assigned index. Confirm Kit/Vulkan selects that same
GPU; remapping can differ from CUDA. Do not reuse workstation scripts that unset
GPU visibility or hardcode `/home/oyoungquist/...` paths.

```bash
export SIMULATOR=isaaclab_b1z1_unifp
export TRAIN_DEVICE=cuda:0  # Adjust to the assigned device as explained above.
python - <<'PY'
import os, sys, torch, importlib.metadata as m
print(sys.executable)
for package in ('isaacsim', 'isaaclab', 'torch', 'numpy', 'warp-lang', 'bard', 'cupiqp'):
    print(package, m.version(package))
device = torch.device(os.environ['TRAIN_DEVICE'])
assert torch.cuda.is_available()
x = torch.ones(4, device=device)
assert (x + x).sum().item() == 8
print('CUDA device:', torch.cuda.get_device_name(device))
import legged_gym, rsl_rl, isaaclab
print('Project:', legged_gym.__file__)
print('RSL-RL:', rsl_rl.__file__)
print('IsaacLab:', isaaclab.__file__)
PY
```

After reviewing and accepting the NVIDIA EULA, set
`OMNI_KIT_ACCEPT_EULA=YES` for noninteractive startup. Do not import simulation
modules that require Kit before the application's normal AppLauncher path.

```bash
cd "$HCR"
python -m pytest -q -p no:cacheprovider tests/test_unifp_original_baselines.py
python tests/smoke_unifp_original_baseline.py \
    --task=b1z1_unifp_original --gpu="$TRAIN_DEVICE" --headless
python tests/smoke_unifp_original_baseline.py \
    --task=b1z1_unifp_reject --gpu="$TRAIN_DEVICE" --headless
```

Each smoke script uses one environment and two policy decisions, without learning.
If Kit reports `No device could be created`, diagnose driver/Vulkan/device mapping
before increasing environment count. Startup may initially compile kernels or
fetch assets/extensions; ensure outbound access or pre-stage the required assets.

Then run a SMALL training check before a full job:

```bash
cd "$HCR/legged_gym/scripts"
python -u train.py --task=b1z1_unifp_original --gpu="$TRAIN_DEVICE" \
    --headless --num_envs=16 --max_iterations=2
```

For other variants, select matching task/backend pairs:

| Task | SIMULATOR |
| --- | --- |
| b1z1_unifp, b1z1_unifp_original, b1z1_unifp_reject | isaaclab_b1z1_unifp |
| b1z1_pact | isaaclab_b1z1_pact |
| b1z1_pact_pos | isaaclab_b1z1_pact_pos |

PACT also requires a tiny forward/backward check of its configured BARD or
Pinocchio path. Import/version checks alone do not establish that compiled
dynamics dependencies work. The new UniFP tests do not exercise PACT's PINN.

## 7. Scale up and retain reproducibility

- Increase environment count gradually after finite rollout and optimizer updates.
- Run one process per GPU initially. Multiple GPUs do not imply automatic DDP or
  a shared policy across processes in these custom runners.
- Keep shader/Warp/Torch caches and high-volume logs on suitable local storage.
  Benchmark after warmup, separating collection time from optimizer time.
- Use the scheduler or tmux for long runs. Keep checkpoints on persistent storage.
- Record source commits AND uncommitted changes, resolved config, seed, device,
  driver, package inventory, and archive checksum with every experiment.
- Preserve this environment after validation; test upgrades in a separate env.

## 8. If you require a fresh install instead of an archive

A guaranteed clean solver recipe cannot be derived from this inconsistent source
environment. Start in a separate staging environment with Python 3.11 and the
pinned source revisions above. The version-specific runtime bootstrap is:

```bash
conda create -n lr_lab_cupiqp_staging python=3.11 pip -y
conda activate lr_lab_cupiqp_staging
python -m pip install 'isaacsim[all,extscache]==5.1.0.0' \
    --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

This is only a bootstrap, NOT the complete environment. Restore the pinned
IsaacLab/HCR sources, resolve their requirements, then add the BARD, Pinocchio,
and QP dependencies needed for the chosen tasks. Resolve the NumPy/cmeel/cupiqp
conflicts deliberately; do not use `--no-deps` as a general compatibility fix.
The QP packages are part of the requested full environment but are not automatically
needed for every UniFP task. A cleaned task-specific environment is a different
artifact and needs its own smoke tests and locked manifest.

Isaac Sim 5.1 is now marked unsupported in NVIDIA's documentation. Keep it pinned
for this reproduction; treat an upgrade as a separate tested migration.
