#!/usr/bin/env bash
set -euo pipefail
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
# Freshly trained checkpoints only: RUN_DIRECTORY CHECKPOINT_NUMBER [OUTPUT_BASE].
# bash legged_gym/scripts/eval_b1z1_pact_ab13_workspace_three_seeds.sh /path/to/run 70000
exec bash "$HERE/eval_b1z1_pact_new_ablations_three_seeds.sh" b1z1_pact_ab13_workspace "$@"
