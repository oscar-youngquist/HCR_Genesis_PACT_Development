#!/usr/bin/env sh
set -e

LEGGED_GYM_ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# The Docker image already contains the packed lr_gym environment
# at /opt/lr_gym, so do NOT source/activate the host Anaconda env.

export PATH="/opt/lr_gym/bin:$PATH"
export LD_LIBRARY_PATH="/opt/lr_gym/lib:${LD_LIBRARY_PATH:-}"

# Required by the project's simulator-selection logic.
export SIMULATOR=isaacgym_b1z1_pact_pos

python train.py \
    --task=b1z1_pact_pos \
    --seed=1 \
    --headless \
    --gpu=cuda:0 \
    "$@"