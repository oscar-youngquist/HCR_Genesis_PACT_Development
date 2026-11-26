#!/usr/bin/env bash
# Explicit legacy-to-current contract migration; normal B1Z1PACT environment.
set -euo pipefail
cd "$(dirname "$0")/../.."
export SIMULATOR=isaaclab_b1z1_pact
python -u legged_gym/scripts/train_pilot_continuation.py \
    --production-interactions --gpu cuda:0 "$@"
