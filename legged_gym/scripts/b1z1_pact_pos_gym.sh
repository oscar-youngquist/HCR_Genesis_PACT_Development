#!/usr/bin/env sh

LEGGED_GYM_ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/lr_gym

export SIMULATOR=isaacgym_b1z1_pact_pos

python train.py --task=b1z1_pact_pos --seed=1 --headless --gpu=cuda:0 "$@"
