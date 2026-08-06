#!/usr/bin/env sh

LEGGED_GYM_ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_b1z1_pact_pos

python train.py --task=b1z1_pact_pos --seed=1 --gpu=cuda:0 "$@"
