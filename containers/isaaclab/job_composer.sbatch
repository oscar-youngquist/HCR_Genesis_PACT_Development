#!/usr/bin/env bash
#SBATCH --job-name=hardpact-rss
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=hardpact-%j.out
# Configure partition/account/GPU type via sbatch flags or Job Composer.
set -euo pipefail
: "${REPO:?Set checkout path}"
: "${IMAGE:?Set Isaac Lab SIF path}"
: "${RUN_ROOT:?Set persistent output parent}"
: "${TRAIN_SCRIPT:=go2_hard_pact_rss_full_unity.sh}"
export RUN_DIR="${RUN_DIR:-$RUN_ROOT/${SLURM_JOB_ID:?}}"
export REPO IMAGE RUN_DIR
export SLURM_EXPORT_ENV=ALL
# Load the site's Apptainer module before submission, if required. EULA
# acceptance is explicit in the submitting environment, never silently set.
exec srun /bin/bash "$REPO/containers/isaaclab/run.sh" script "$TRAIN_SCRIPT" "$@"
