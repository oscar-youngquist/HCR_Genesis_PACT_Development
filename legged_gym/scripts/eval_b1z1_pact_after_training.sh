#!/usr/bin/env bash
set -euo pipefail
# Auto-detect the active local PACT run, then run the unchanged three-seed suite.
# Preview: bash legged_gym/scripts/eval_b1z1_pact_after_training.sh --dry-run
# Detached: nohup bash legged_gym/scripts/eval_b1z1_pact_after_training.sh \
#   > pact_after_training.log 2>&1 &
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec python -u "$HERE/eval_b1z1_pact_after_training.py" "$@"
