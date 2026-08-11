#!/usr/bin/env sh

# Play a trained B1/Z1 UniFP policy with the Isaac Gym backend.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh
conda activate /home/oyoungquist/.conda/envs/lr_gym

export SIMULATOR=isaacgym_b1z1_unifp

cd "$SCRIPT_DIR"
python play_exp_unifp.py --task=b1z1_unifp --seed=1 --gpu=cuda:0 "$@"
