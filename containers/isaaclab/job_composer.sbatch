#!/bin/bash
#SBATCH --job-name=b1z1-isaaclab
#SBATCH --partition=gpu
#SBATCH --constraint=l40s
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-10:00:00
#SBATCH --output=isaaclab-%j.out
set -euo pipefail
module load apptainer/latest

# Edit these paths once in Job Composer. The image must already exist on Unity.
PROJECT=/project/pi_hzhang2_umass_edu/oyoungquist_umass_edu
export IMAGE="$PROJECT/containers/isaaclab-hcr.sif"
export REPO="$PROJECT/KITE/HCR_Genesis_PACT_Development"
TOOLS="$PROJECT/KITE/HCR_Genesis_PACT_Development/containers/isaaclab"

# Select a Unity launcher from legged_gym/scripts in the chosen checkout.
TRAIN_SCRIPT=b1z1_unifp_unity.sh
TRAIN_ARGS=(--seed=1 --num_envs=2 --max_iterations=1) # First job: small smoke test.
export RUN_DIR="$PROJECT/training_runs/${SLURM_JOB_ID}"
export OMNI_KIT_ACCEPT_EULA=YES # Submit only after reviewing/accepting NVIDIA's EULA.

# All mounts, caches, and scheduler GPU visibility are managed by the helper.
exec srun bash "$TOOLS/run.sh" script "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
