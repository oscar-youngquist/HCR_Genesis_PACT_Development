#!/usr/bin/env bash
# Backend-neutral HardPACT launcher. The caller's active Python environment,
# GPU selection, seed, output path, and all train.py arguments are untouched.
set -eu

usage() {
    echo "Usage: $0 --task go2_hard_pact_<variant>_<backend> [train arguments]" >&2
    echo "Available HardPACT tasks:" >&2
    for variant in baseline soft hard full stopgrad soft_penalty inverse rollout; do
        echo "  go2_hard_pact_${variant}_{genesis,isaacgym,isaaclab}" >&2
    done
    echo "  go2_hard_pact_{genesis,isaacgym,isaaclab} (full aliases)" >&2
}

task=""
for argument in "$@"; do
    case "$argument" in
        --task=*) task="${argument#--task=}" ;;
    esac
done
if [ -z "$task" ]; then
    previous=""
    for argument in "$@"; do
        if [ "$previous" = "--task" ]; then task="$argument"; break; fi
        previous="$argument"
    done
fi

case "$task" in
    go2_hard_pact_baseline_genesis|go2_hard_pact_soft_genesis|go2_hard_pact_hard_genesis|go2_hard_pact_full_genesis|go2_hard_pact_stopgrad_genesis|go2_hard_pact_soft_penalty_genesis|go2_hard_pact_inverse_genesis|go2_hard_pact_rollout_genesis|go2_hard_pact_genesis)
        export SIMULATOR=genesis_pact ;;
    go2_hard_pact_baseline_isaacgym|go2_hard_pact_soft_isaacgym|go2_hard_pact_hard_isaacgym|go2_hard_pact_full_isaacgym|go2_hard_pact_stopgrad_isaacgym|go2_hard_pact_soft_penalty_isaacgym|go2_hard_pact_inverse_isaacgym|go2_hard_pact_rollout_isaacgym|go2_hard_pact_isaacgym)
        export SIMULATOR=isaacgym ;;
    go2_hard_pact_baseline_isaaclab|go2_hard_pact_soft_isaaclab|go2_hard_pact_hard_isaaclab|go2_hard_pact_full_isaaclab|go2_hard_pact_stopgrad_isaaclab|go2_hard_pact_soft_penalty_isaaclab|go2_hard_pact_inverse_isaaclab|go2_hard_pact_rollout_isaaclab|go2_hard_pact_isaaclab)
        export SIMULATOR=isaaclab ;;
    *) usage; exit 2 ;;
esac

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
repo_dir="$(dirname -- "$script_dir")"
exec python -u "$repo_dir/legged_gym/scripts/train.py" "$@"
