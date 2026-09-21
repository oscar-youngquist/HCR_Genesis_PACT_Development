#!/usr/bin/env bash
# Backend-selecting visual checkpoint player for Go2 HardPACTPos.
set -eu

usage() {
    echo "Usage: $0 --task go2_hard_pact_pos_<genesis|isaaclab> --load_run RUN [--ckpt N] [other play arguments]" >&2
}

task=""
previous=""
for argument in "$@"; do
    case "$argument" in
        --task=*) task="${argument#--task=}" ;;
        --help|-h) usage; exit 0 ;;
    esac
    if [ "$previous" = "--task" ]; then task="$argument"; fi
    previous="$argument"
done

case "$task" in
    go2_hard_pact_pos_genesis)
        export SIMULATOR=genesis_pact_pos
        target_env="${HARD_PACT_GENESIS_ENV:-genesis_lr}"
        ;;
    go2_hard_pact_pos_isaaclab)
        export SIMULATOR=isaaclab
        target_env="${HARD_PACT_ISAACLAB_ENV:-lr_lab}"
        ;;
    *) usage; exit 2 ;;
esac

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
repo_dir="$(dirname -- "$(dirname -- "$script_dir")")"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_hard_pact}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_hard_pact}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/hard_pact_cache}"

target_env="${HARD_PACT_CONDA_ENV:-$target_env}"
if [ "${HARD_PACT_SKIP_CONDA_ACTIVATE:-0}" != 1 ]; then
    # shellcheck disable=SC1090
    . "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$target_env"
fi
echo "HardPACTPos play environment: ${CONDA_DEFAULT_ENV:-$target_env}; simulator: $SIMULATOR" >&2
exec python -u "$script_dir/play_go2_hard_pact_pos.py" "$@"
