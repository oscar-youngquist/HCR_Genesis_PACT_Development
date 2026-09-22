# IsaacLab on Unity with Apptainer

**For routine OnDemand submission, use [the short Job Composer guide](JOB_COMPOSER.md)
and `job_composer.sbatch`.** It loads the module and runs the chosen Unity `.sh`
launcher. The sections below cover one-time image building and lower-level checks.

One runtime image, separate mounted HCR checkouts for B1Z1 and Go2 HardPACT.
No training implementation or branch is modified by these launchers.
This is an initial build recipe, not a tested SIF: Apptainer is not installed on
the development workstation and Unity access was not available for validation.

## GPU and software selection

On the [Unity GPU list](https://docs.unity.rc.umass.edu/documentation/cluster_specs/gpu_summary/),
look for **NVIDIA Ada Lovelace L40S**, 48 GB, in `gpu`, `gpu-preempt`, or
`gpupod-l40s`. The supplied Unity listing confirms feature `l40s` and GRES
`gpu:l40s:4` on nodes `gpu025` through `gpu033` in partition `gpu`.
Prefer non-preemptible `gpu` initially; don't assume a specialized partition is
available to your account. RTX/driver compatibility must still pass a smoke test.
[Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
exclude GPUs without RT cores such as A100/H100. Use x86_64, not ARM/POWER nodes.

Unity documents [Apptainer builds and Docker-image conversion](https://docs.unity.rc.umass.edu/documentation/software/).
Load the module confirmed by the supplied `module spider` output:

```bash
module load apptainer/latest
sinfo -p gpu -N -o '%N %G %f'
```

The batch template now defaults to `--partition=gpu --constraint=l40s --gpus=1`.
Account access, current GPU availability, and driver compatibility are not proven
by the node listing. Do not launch simulation or a large build on a login node.

## Build inputs

The exported YAML contains conflicting dependencies, so this initial recipe
packages the installed `lr_lab_cupiqp` instead of attempting a new pip solve.
It preserves those conflicts, not fixes them. For details see
[the environment migration guide](../../docs/lr_lab_cupiqp_remote_setup.md).
Do not treat an image build as proof of numerical/runtime compatibility.

On the source workstation, create a new build-context directory and populate it:

```bash
export HCR="$HOME/Research/Genesis_Development/HCR_Genesis_PACT_Development"
export CONTEXT="$HOME/isaaclab-apptainer-build"
mkdir -p "$CONTEXT"
cp "$HCR/containers/isaaclab/isaaclab.def" "$CONTEXT/"
cp "$HCR/containers/isaaclab/preflight.py" "$CONTEXT/"
conda create -n lr_env_pack -c conda-forge conda-pack -y  # Once only.
conda run -n lr_env_pack conda-pack -n lr_lab_cupiqp \
    --ignore-editable-packages -o "$CONTEXT/lr_lab_cupiqp.tar.gz"
tar -czf "$CONTEXT/IsaacLab-source.tar.gz" \
    --exclude='IsaacLab/.git' --exclude='IsaacLab/logs' \
    --exclude='IsaacLab/_isaac_sim' \
    -C "$HOME/Research" IsaacLab
cd "$CONTEXT"
sha256sum *.tar.gz > SHA256SUMS
```

Stop if conda-pack reports missing/overwritten package files; do not blindly
ignore that error. Inspect sources/archives for credentials and external symlinks.
Respect NVIDIA's redistribution/license terms; keep the image private.
The packed environment excludes editable source content, so the separate
IsaacLab archive is required. HCR is intentionally not installed in the image:
the selected checkout supplies `legged_gym` and its custom `rsl_rl` via PYTHONPATH.
Both branches must use compatible versions of this runtime.

Transfer the context to a permitted build machine with adequate disk/temp space.
Use Unity's documented build form when supported by the installed Apptainer:

```bash
cd /path/to/isaaclab-apptainer-build
sha256sum -c SHA256SUMS
export APPTAINER_CACHEDIR=/path/to/large/writable/cache
export APPTAINER_TMPDIR=/path/to/large/writable/build-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
apptainer build --ignore-fakeroot-command isaaclab-hcr.sif isaaclab.def
sha256sum isaaclab-hcr.sif > isaaclab-hcr.sif.sha256
```

Build/fakeroot policy varies by host. If this fails for privilege/user-namespace
reasons, ask Unity support or build with administrator support on a compatible
Linux workstation and transfer the SIF. Do not bypass cluster policy.
The base Ubuntu tag is not digest-pinned; record the finished SIF checksum.
Build-time tests only inspect CPU package metadata, not a GPU or simulator.

## Keep branches separate

Use two ordinary clones on Unity, or existing worktrees whose git metadata is
also available on the host. Do not change a branch while a job is running.

```bash
git clone --branch legged_manip_gym \
    git@github.com:oscar-youngquist/HCR_Genesis_PACT_Development.git hcr-b1z1
git clone --branch aligned_iclr_2027_qp_pinn \
    git@github.com:oscar-youngquist/HCR_Genesis_PACT_Development.git hcr-hardpact
```

The inspected aligned branch name is `aligned_iclr_2027_qp_pinn`, not literally
`aligned_iclr`. Transfer/commit required uncommitted files too; cloning HEAD alone
will not include them. Include all robot resources and external asset dependencies.
Retain the new launcher files from the B1Z1 checkout; they can mount either clone.

## Launch configuration

Run in an allocated job. Set absolute paths on the Unity filesystem:

```bash
export IMAGE=/path/to/isaaclab-hcr.sif
export REPO=/path/to/hcr-b1z1
export LAUNCHER=/path/to/hcr-b1z1/containers/isaaclab/run.sh
export RUN_DIR=/path/to/persistent/runs/b1z1-smoke-001
export TASK=b1z1_unifp_original
export MODE=smoke
# Set only after reviewing/accepting NVIDIA's EULA:
export OMNI_KIT_ACCEPT_EULA=YES
```

`run.sh` preserves Slurm's CUDA_VISIBLE_DEVICES and requires exactly one visible
GPU. It never unsets scheduler visibility. `cuda:0` inside training is the first
visible CUDA device, not a hardcoded physical GPU. It isolates the container
home, imports, logs, and writable Kit/cache directories. Online W&B is disabled
by default. Use a unique RUN_DIR for concurrent jobs; RUNTIME_DIR may instead
point to unique node-local scratch for caches.

`--nv` does not by itself enforce a physical GPU security boundary, and Vulkan
enumeration can differ from CUDA. Slurm/device cgroups must enforce allocation;
verify Kit's selected GPU in startup output. See
[Apptainer GPU guidance](https://apptainer.org/docs/user/main/gpu.html).
If the NVIDIA Vulkan ICD JSON is missing inside the image, set NVIDIA_ICD_FILE
to the host's NVIDIA JSON path. Do not bind a Mesa ICD as a substitute. If graphics
driver injection is still missing, ask Unity support about their supported
Apptainer GPU configuration before changing device visibility.

Load the module before submitting with your actual account:

```bash
module load apptainer/latest
sbatch --account=YOUR_ACCOUNT --partition=gpu --constraint=l40s \
    /path/to/hcr-b1z1/containers/isaaclab/unity.sbatch
```

Replace `YOUR_ACCOUNT`; the module and GPU constraint are now confirmed.
The batch template requests 1 GPU, 8 CPUs, 64 GB RAM, and 1 hour. Adjust these
within your allocation. It uses `srun` and does not load a workstation conda env.

## Validation sequence

1. `MODE=check`: tiny Torch CUDA/autograd check, package versions, task registration.
2. `MODE=smoke`: two environments, one learning iteration, headless. This IS a
   real short PPO update, unlike the earlier two-decision no-learning test.
3. Repeat smoke for each intended task, including the full HardPACT cuPIQP path.
4. Inspect finite losses, successful checkpoint creation, decoder updates, and
   the actual GPU selected by Kit. Only then increase environment count.

Modes and task/backend mappings:

| TASK | Checkout | Backend / entrypoint |
| --- | --- | --- |
| b1z1_unifp, b1z1_unifp_original, b1z1_unifp_reject | legged_manip_gym | isaaclab_b1z1_unifp / train.py |
| b1z1_pact | legged_manip_gym | isaaclab_b1z1_pact / train.py |
| b1z1_pact_pos | legged_manip_gym | isaaclab_b1z1_pact_pos / train.py |
| go2_hard_pact_full_isaaclab | aligned_iclr_2027_qp_pinn | isaaclab / train_hard_pact.py |
| Other registered go2_hard_pact*_isaaclab ablations | aligned_iclr_2027_qp_pinn | isaaclab / train_hard_pact.py |

For HardPACT change REPO, TASK and RUN_DIR, then submit with `--qp_solver cupiqp`
after the batch-script path. The `train_hard_pact.py` wrapper retains the branch's
pre-AppLauncher Warp import and compatibility aliases. Do not replace it with
plain train.py for cuPIQP. This environment does not promise support for Moreau.

Example after allocating a GPU interactively:

```bash
bash "$LAUNCHER" check "$TASK"
bash "$LAUNCHER" smoke "$TASK" --qp_solver cupiqp  # HardPACT only.
# After successful checks, for B1Z1 or HardPACT as configured:
bash "$LAUNCHER" train "$TASK" --num_envs 4096 --max_iterations 50000 --seed 1
```

The launcher's check currently probes cuPIQP imports for all HardPACT tasks;
it does not claim to solve a QP. Smoke the actual solver selection to validate it.
Logs/checkpoints are redirected to RUN_DIR/logs; resolved launch metadata records
the host checkout commit and dirty status. Training assets may write converted
USD caches into the bound checkout, which is writable. Avoid simultaneous first
conversion of the same asset by warming up each checkout once.

## Limitations and acceptance

No runtime dependencies were repaired or silently upgraded. Source conflicts
(notably NumPy versus cupiqp/cmeel and Torch versus cuBLAS) remain an acceptance
gate. A failed cuPIQP/BARD/Pinocchio test requires a separately validated runtime,
not simply suppressing import warnings. This recipe does not use the unsolved YAML.
Do not assume successful UniFP training validates HardPACT or B1Z1 PINN losses.

No local SIF build, Unity submission, or GPU test has been performed. Local checks
cover launcher syntax and argument forwarding only. Before production, require
successful image build, CUDA/Vulkan initialization on the assigned L40S, and one
finite PPO update for each selected variant. For scaling, measure collection and
optimizer time separately after kernel warmup. This launcher is single-GPU;
multiple independent jobs do not automatically implement distributed PPO.
