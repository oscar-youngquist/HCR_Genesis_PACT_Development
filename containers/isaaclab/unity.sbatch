#!/usr/bin/env bash
# Supply your account with sbatch; partition/constraint match the supplied Unity listing.
#SBATCH --job-name=isaaclab
#SBATCH --partition=gpu
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --output=isaaclab-%j.out
set -euo pipefail
: "${LAUNCHER:?Absolute path to containers/isaaclab/run.sh}"
: "${TASK:?Set the registered training task}"
command -v apptainer >/dev/null || {
    echo "Load Unity's available Apptainer module before submission." >&2; exit 1;
}
# srun sets the step's GPU visibility. Never unset CUDA_VISIBLE_DEVICES here.
exec srun bash "$LAUNCHER" "${MODE:-smoke}" "$TASK" "$@"
