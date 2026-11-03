#!/bin/bash
set -euo pipefail

# The five collection settings. Every selected platform runs every sampled round.
# Supported now: go1-dact, go2-dact. Reserved: go2-hardpact (adapter not implemented).
PLATFORMS=(go1-dact go2-dact)
NUM_ENVS=2
EPISODES_PER_ENV=2
ROUNDS=1
EPISODE_SECONDS=5

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
exec "$HOME/.conda/envs/genesis_lr/bin/python" -m liquid_collection.launch \
    --platforms "${PLATFORMS[@]}" \
    --num-envs "$NUM_ENVS" \
    --episodes-per-env "$EPISODES_PER_ENV" \
    --rounds "$ROUNDS" \
    --episode-seconds "$EPISODE_SECONDS" \
    "$@"
