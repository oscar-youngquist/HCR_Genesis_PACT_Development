#!/usr/bin/env sh

# Play a trained coupled position/torque B1/Z1 PACT policy with Isaac Gym.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh
conda activate /home/oyoungquist/.conda/envs/lr_gym

export SIMULATOR=isaacgym_b1z1_pact

cd "$SCRIPT_DIR"
python play_exp.py --task=b1z1_pact --seed=1 --gpu=cuda:0 "$@"
