#!/bin/bash
set -euo pipefail

# The five collection settings. Every selected platform runs every sampled round.
# Supported now: go1-dact, go2-dact. Reserved: go2-hardpact (adapter not implemented).
# CUDA device index (within CUDA_VISIBLE_DEVICES if it is set).
GPU=1

PLATFORMS=(go1-dact go2-dact)
NUM_ENVS=2
EPISODES_PER_ENV=5
ROUNDS=500
EPISODE_SECONDS=10

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
exec "$HOME/.conda/envs/genesis_lr/bin/python" -m liquid_collection.launch \
    --gpu "$GPU" \
    --platforms "${PLATFORMS[@]}" \
    --num-envs "$NUM_ENVS" \
    --episodes-per-env "$EPISODES_PER_ENV" \
    --rounds "$ROUNDS" \
    --episode-seconds "$EPISODE_SECONDS" \
    "$@"
