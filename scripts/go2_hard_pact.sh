#!/usr/bin/env bash
# One backend-selecting launcher for HardPACT, HardPACTPos, and all ablations.
set -eu

usage() {
    echo "Usage: $0 --task go2_hard_pact_<variant>_<genesis|isaaclab> [train arguments] [--smoke]" >&2
    echo "       $0 --task go2_hard_pact_pos_<genesis|isaaclab> [train arguments] [--smoke]" >&2
}

task=""
smoke=0
previous=""
forward=()
for argument in "$@"; do
    if [ "$argument" = "--smoke" ]; then smoke=1; continue; fi
    forward+=("$argument")
    case "$argument" in --task=*) task="${argument#--task=}" ;; esac
    if [ "$previous" = "--task" ]; then task="$argument"; fi
    previous="$argument"
done

case "$task" in
    go2_hard_pact_pos_genesis) export SIMULATOR=genesis_pact_pos ;;
    go2_hard_pact_genesis|go2_hard_pact_baseline_genesis|go2_hard_pact_soft_genesis|go2_hard_pact_hard_genesis|go2_hard_pact_full_genesis|go2_hard_pact_stopgrad_genesis|go2_hard_pact_soft_penalty_genesis|go2_hard_pact_inverse_genesis|go2_hard_pact_rollout_genesis)
        export SIMULATOR=genesis_pact ;;
    go2_hard_pact_pos_isaaclab|go2_hard_pact_isaaclab|go2_hard_pact_baseline_isaaclab|go2_hard_pact_soft_isaaclab|go2_hard_pact_hard_isaaclab|go2_hard_pact_full_isaaclab|go2_hard_pact_stopgrad_isaaclab|go2_hard_pact_soft_penalty_isaaclab|go2_hard_pact_inverse_isaaclab|go2_hard_pact_rollout_isaaclab)
        export SIMULATOR=isaaclab ;;
    *) usage; exit 2 ;;
esac

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
repo_dir="$(dirname -- "$script_dir")"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
# Some packaged Genesis builds cannot locate their source-tree cache path.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_hard_pact}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_hard_pact}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/hard_pact_cache}"
if [ "$smoke" -eq 1 ]; then
    # One iteration contains 32 real control steps and a differentiable PPO
    # update. User-provided values later in argv retain normal argparse order.
    exec python -u "$repo_dir/legged_gym/scripts/train.py" \
        --num_envs 8 --max_iterations 1 "${forward[@]}"
fi
exec python -u "$repo_dir/legged_gym/scripts/train.py" "${forward[@]}"
