#!/usr/bin/env sh
set -e

# Historical filename retained; playback now uses IsaacLab.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh
conda activate /home/oyoungquist/.conda/envs/lr_lab_cupiqp

unset CUDA_VISIBLE_DEVICES
export SIMULATOR=isaaclab_b1z1_pact

cd "$SCRIPT_DIR"
python play_b1z1_pact.py --task=b1z1_pact --seed=1 --gpu=cuda:1 "$@"
