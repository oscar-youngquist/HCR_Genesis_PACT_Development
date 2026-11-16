# Portable BR and MULE training (Apptainer)

Reuse the existing **isaaclab-hcr-v2.sif** rather than rebuilding a different
Python/CUDA environment. The SIF is a runtime, not a copy of training code:
mount this `rescaled_qp_exps` checkout or the source bundle below. BR and MULE
use their existing production entrypoints, rewards, curricula and checkpoints.
No QP/PINN component is enabled by the container. Existing Unity `run.sh script`
and Job Composer usage remains valid.

Start here: [host setup](#host-setup) → [transfer](#transfer-to-another-machine)
→ [check/smoke/train](#check-smoke-train) → [outputs and monitoring](#outputs-and-monitoring)
→ [resume](#resume). For cluster submission see [Unity](#unity).

## Requirements

* Linux x86-64, Apptainer (`--nv`), compatible host NVIDIA driver and an RTX GPU.
  Isaac Sim 5.1 does not support A100/H100 GPUs without RT cores; see
  [NVIDIA requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).
* This image contains Torch 2.7.0+cu128, Isaac Sim 5.1.0.0, Isaac Lab 0.54.2.
  The **host driver is not inside the image**. Verify each target machine.
* Review NVIDIA's license and explicitly set `OMNI_KIT_ACCEPT_EULA=YES`.
  Keep the image private; do not publish it to Docker Hub or a public registry.
* Allow space for the ~9.7 GiB image, source, writable caches and run outputs.
  First launch may need network access for simulator extensions/assets. This is
  not a certified offline bundle. No host Conda installation is needed to train.
* Docker cannot directly `docker run` this SIF. Apptainer is the supported reuse
  path here; this does not claim a new Docker/OCI image was built.

## Host setup

Install Apptainer using the [official installation instructions](https://apptainer.org/docs/admin/latest/installation.html)
for your Linux distribution, or ask your cluster administrator for the supported
module. The local validation used Apptainer 1.5.4. Do not install the workstation
Conda environment or Isaac Lab separately on the destination; they are inside
the SIF. The host still needs a working NVIDIA driver and permission to run
Apptainer containers with GPU access.

Verify these on the destination before launching:

```bash
uname -m                 # expected: x86_64
apptainer --version
nvidia-smi
df -h /path/to/image-storage /path/to/run-storage
```

Obtain the private `isaaclab-hcr-v2.sif` from the source workstation below or
your project's shared storage; it is not included in Git. Keep the provided
checksum with it. You also need either a matching repository checkout or the
source bundle, including robot assets. A SIF alone does not contain this
project's training implementation. Existing local checkouts can skip the
transfer/extraction steps and set `IMAGE` to the existing SIF path.

## Transfer to another machine

On the source machine (also includes uncommitted source changes; inspect them):

```bash
cd /path/to/HCR_HardPact
python3 containers/isaaclab/bundle_comparisons.py \
  --image /home/oyoungquist/isaaclab-apptainer-build/isaaclab-hcr-v2.sif \
  --output /path/to/new-transfer-directory
rsync -avP /path/to/new-transfer-directory/ user@remote:/data/hardpact/
rsync -avP /home/oyoungquist/isaaclab-apptainer-build/isaaclab-hcr-v2.sif \
  user@remote:/data/hardpact/
```

The bundle includes code, configs and tracked robot assets, plus hashes/Git
provenance; it excludes logs, checkpoints, .git, environments and SIF contents.
It rejects symlinks rather than silently packaging machine-local dependencies.
Review its file manifest before transferring. Checkpoints must be copied
separately when resuming. The output directory must not already exist.

On the target (or locally, with paths adjusted):

```bash
cd /data/hardpact
sha256sum -c SHA256SUMS
tar -xzf hardpact-comparisons-source.tar.gz
cd hardpact-comparisons
export IMAGE=/data/hardpact/isaaclab-hcr-v2.sif
export CUDA_VISIBLE_DEVICES=0       # physical GPU on a standalone workstation
export OMNI_KIT_ACCEPT_EULA=YES     # only after accepting NVIDIA terms
```

On Slurm do **not** override the scheduler's CUDA_VISIBLE_DEVICES. Each process
uses logical `cuda:0` inside its assigned visibility. Run one independent job
per GPU/machine; this launcher does not implement distributed PPO. Apptainer
`--nv` exposes driver devices and CUDA visibility does not enforce security or
necessarily Vulkan enumeration: inspect Kit's GPU selection on multi-GPU hosts.
Set `NVIDIA_ICD_FILE` to the host NVIDIA Vulkan ICD JSON only if required.

## Check, smoke, train

Use a separate persistent RUN_DIR for each concurrent run. Logs/checkpoints
live under RUN_DIR/logs; runtime cache/home/temp data live under RUN_DIR/runtime.
The mounted checkout is writable for USD asset conversion: use separate copies
for concurrent initial conversions, or prewarm it first.

```bash
RUN_DIR=/data/runs/br-check bash containers/isaaclab/comparisons.sh br check
RUN_DIR=/data/runs/mule-check bash containers/isaaclab/comparisons.sh mule check
RUN_DIR=/data/runs/br-smoke bash containers/isaaclab/comparisons.sh br smoke --seed 1
RUN_DIR=/data/runs/mule-smoke bash containers/isaaclab/comparisons.sh mule smoke --seed 1

RUN_DIR=/data/runs/br-seed1 bash containers/isaaclab/comparisons.sh br train \
  --seed 1 --num-envs 4096 --max-iterations 20000
RUN_DIR=/data/runs/mule-seed1 bash containers/isaaclab/comparisons.sh mule train \
  --seed 1 --num-envs 4096 --max-iterations 20000
```

`check` executes CUDA/autograd and real CLI configuration validation without
creating a simulator. `smoke` runs two environments and one production-reward
update with test-local small rollout/optimizer sizing. `train` retains the
resolved production defaults except supplied CLI overrides. Both are headless.
`--dry-run` on any command prints routing only, writes nothing and does not
validate the simulator. Use `check` for configuration/import validation.
Training arguments remain those of train_beyond_robustness.py/train_mule.py;
the wrapper rejects task/device overrides to avoid losing the GPU assignment.
`--max-iterations` is a **total** update budget, not an additional budget.

### RTX 3090: local launch sequence

The same SIF can be tried on an RTX 3090; no image rebuild is prescribed.
**This image has been tested on an RTX 4090, not a 3090.** The 3090 has
24 GB VRAM and RT cores, but NVIDIA's Isaac Sim 5.1 requirements list an RTX
4080 as the minimum GPU specification. Compatibility, throughput and capacity
on a 3090 remain unverified; do not assume 4,096 environments will fit.

From the checkout or extracted source directory, set the actual image path
and select the physical GPU. These workstation examples use GPU 0:

```bash
cd /path/to/hardpact-comparisons  # or your HCR_HardPact checkout
export IMAGE=/path/to/isaaclab-hcr-v2.sif
export CUDA_VISIBLE_DEVICES=0   # use 1 for the second GPU; not --device cuda:1
export OMNI_KIT_ACCEPT_EULA=YES # only after reviewing/accepting NVIDIA's terms
nvidia-smi

# Validate each approach separately before training.
RUN_DIR="$HOME/hardpact-runs/br-check" \
  bash containers/isaaclab/comparisons.sh br check
RUN_DIR="$HOME/hardpact-runs/br-smoke" \
  bash containers/isaaclab/comparisons.sh br smoke --seed 1
RUN_DIR="$HOME/hardpact-runs/mule-check" \
  bash containers/isaaclab/comparisons.sh mule check
RUN_DIR="$HOME/hardpact-runs/mule-smoke" \
  bash containers/isaaclab/comparisons.sh mule smoke --seed 1
```

After successful checks, choose **one** of these training commands. Start with
1,024 environments as a cautious trial, not a validated memory guarantee:

```bash
# Beyond Robustness
RUN_DIR="$HOME/hardpact-runs/br-seed1-1024" \
  bash containers/isaaclab/comparisons.sh br train \
    --seed 1 --num-envs 1024 --max-iterations 20000

# MULE (run separately, or on a different allocated GPU)
RUN_DIR="$HOME/hardpact-runs/mule-seed1-1024" \
  bash containers/isaaclab/comparisons.sh mule train \
    --seed 1 --num-envs 1024 --max-iterations 20000
```

For a bounded initial training trial, use `--max-iterations 10` and a separate
RUN_DIR instead of starting the full budget. Monitor the GPU from another
terminal with `watch -n 2 nvidia-smi`. If memory headroom and iteration times
permit, try 2,048 and then 4,096 environments in new runs. Changing environment
count changes rollout batch size, so it is an experimental setting, not merely
a runtime optimization. Do not change it silently when resuming a checkpoint.

The examples are headless and use the existing production phase/curriculum
defaults. Checkpoints and TensorBoard logs stay under the chosen host RUN_DIR
as described below. Use fresh run directories when repeating trials. On Unity,
retain scheduler GPU visibility instead of exporting a physical GPU index.

## Outputs and monitoring

Checkpoints and TensorBoard event files are written directly to the host,
not trapped inside the SIF. The launcher binds host `$RUN_DIR/logs` onto
`/workspace/repo/logs` inside the container. Default host paths are:

```text
$RUN_DIR/logs/beyond_robustness/<timestamp>_br/model_<iteration>.pt
$RUN_DIR/logs/mule/<timestamp>_mule/model_<iteration>.pt
```

These directories also contain resolved configuration/source archives and run
manifests. They survive container exit. The entrypoints save periodic checkpoints
and a final checkpoint at normal completion; forced termination or a crash does
not guarantee a new final checkpoint. Preserve the last completed checkpoint.
Keep persistent runs outside `/tmp` and node-local scratch that is deleted after
a cluster job. `RUNTIME_DIR` may be placed on separate writable scratch for
disposable simulator caches without moving the checkpoints.

To retain terminal output as well, launch with a host-side log:

```bash
export RUN_DIR="$HOME/hardpact-runs/br-seed1"
mkdir -p "$RUN_DIR"
set -o pipefail
bash containers/isaaclab/comparisons.sh br train --seed 1 \
  2>&1 | tee "$RUN_DIR/console.log"
# From another terminal:
tail -f "$HOME/hardpact-runs/br-seed1/console.log"
```

If TensorBoard is installed on the host, point it at the host logs:

```bash
tensorboard --logdir "$HOME/hardpact-runs/br-seed1/logs" \
  --host 127.0.0.1 --port 6006
```

For a remote host, run TensorBoard there and forward its loopback port from
your workstation with `ssh -L 6006:127.0.0.1:6006 user@remote`, then open
`http://127.0.0.1:6006` locally. Do not expose TensorBoard publicly by default.
Choose different output directories (and ports, when needed) for parallel runs.

## Troubleshooting

* **CUDA unavailable / NVIDIA driver errors:** first check host `nvidia-smi`,
  your allocated GPU, and Apptainer `--nv` support. Do not install a kernel
  driver inside the image. `comparisons.sh ... check` verifies CUDA explicitly.
* **Missing files or checkpoint:** use absolute host paths for `IMAGE`,
  `RUN_DIR`, and `CHECKPOINT_DIR`; use mounted container paths for CLI arguments
  such as `--checkpoint /checkpoints/...`. The source bundle excludes checkpoints.
* **Permission errors:** run as your normal user and ensure output/cache
  locations are writable. Ask the administrator about user-namespace or bind
  restrictions rather than launching the training process as root.
* **Out of memory:** stop the failed run and retry with fewer `--num-envs` and
  a fresh output directory. Small smoke memory numbers exclude PhysX/Kit and
  cannot predict production VRAM. Inspect other processes with `nvidia-smi`.
* **Kit change-watch `errno=28`:** check disk space first. If storage is not
  full, host inotify limits may be exhausted by concurrent Kit processes; ask
  the administrator to inspect limits. Our local smokes completed despite these
  warnings; they are not automatically proof of failed training.

## Resume

Copy the entire source checkpoint run when practical. Mount its parent:

```bash
export CHECKPOINT_DIR=/data/old-runs
RUN_DIR=/data/runs/br-resumed bash containers/isaaclab/comparisons.sh br train \
  --checkpoint /checkpoints/br/model_20000.pt --additional-iterations 2000
RUN_DIR=/data/runs/mule-resumed bash containers/isaaclab/comparisons.sh mule train \
  --checkpoint /checkpoints/mule/model_20000.pt --additional-iterations 2000
```

Use compatible saved source/config contracts. Normal resume retains the saved
phase, optimizers and curricula, but restarts physical episodes; no exact PhysX
continuation claim. Never point outputs at the source run. Phase/stage and
weights-only options are forwarded unchanged and validated by the method CLI.
All container paths, including explicit `--log-dir`, must refer to mounted
locations (`/output`, `/checkpoints`, `/workspace/repo`), not host paths.

## Unity

Continue using the existing submission template, with site-specific account,
partition and GPU choices:

```bash
export REPO=/data/hardpact/hardpact-comparisons RUN_ROOT=/data/runs
export TRAIN_SCRIPT=go2_beyond_robustness_unity.sh  # or go2_mule_unity.sh
sbatch --export=ALL --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  containers/isaaclab/job_composer.sbatch --seed 1
```

No Unity submission or remote-machine training is implied by local validation.
Container reuse preserves installed dependency conflicts; a successful import
is not a simulator/learning smoke. Results from actual checks are recorded
separately in `COMPARISONS_VALIDATION.md`.
