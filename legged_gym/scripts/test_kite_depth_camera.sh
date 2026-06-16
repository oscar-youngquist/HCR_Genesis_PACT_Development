#!/bin/bash
set -e

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh
conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_kite_depth

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python test_kite_depth_camera.py \
    --task=go2_kite \
    --terrain=rough \
    --roughness=0.12 \
    --normal-length=0.12 \
    --normal-refresh-steps=5 \
    --viz-height-offset=0.02 \
    --num_envs=1 \
    --gpu=cuda:0 \
    --seed=1
