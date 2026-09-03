#!/usr/bin/env bash
# One backend-selecting launcher for HardPACT, HardPACTPos, and all ablations.
set -eu

usage() {
    echo "Usage: $0 --task go2_hard_pact_<variant>_<genesis|isaacgym|isaaclab> [train arguments] [--smoke]" >&2
    echo "       $0 --task go2_hard_pact_pos_<genesis|isaaclab> [train arguments] [--smoke]" >&2
    echo "Available HardPACT tasks: variants baseline soft hard full stopgrad soft_penalty inverse rollout" >&2
}

task=""
smoke=0
previous=""
forward=()
requested_solvers=()
for argument in "$@"; do
    if [ "$argument" = "--smoke" ]; then smoke=1; continue; fi
    forward+=("$argument")
    case "$argument" in --task=*) task="${argument#--task=}" ;; esac
    if [ "$previous" = "--task" ]; then task="$argument"; fi
    case "$argument" in
        --qp_solver=*|--rollout_qp_solver=*|--ppo_qp_solver=*)
            requested_solvers+=("${argument#*=}") ;;
    esac
    case "$previous" in
        --qp_solver|--rollout_qp_solver|--ppo_qp_solver)
            requested_solvers+=("$argument") ;;
    esac
    previous="$argument"
done

case "$task" in
    go2_hard_pact_pos_genesis) export SIMULATOR=genesis_pact_pos; backend=genesis ;;
    go2_hard_pact_genesis|go2_hard_pact_baseline_genesis|go2_hard_pact_soft_genesis|go2_hard_pact_hard_genesis|go2_hard_pact_full_genesis|go2_hard_pact_stopgrad_genesis|go2_hard_pact_soft_penalty_genesis|go2_hard_pact_inverse_genesis|go2_hard_pact_rollout_genesis)
        export SIMULATOR=genesis_pact; backend=genesis ;;
    go2_hard_pact_isaacgym|go2_hard_pact_baseline_isaacgym|go2_hard_pact_soft_isaacgym|go2_hard_pact_hard_isaacgym|go2_hard_pact_full_isaacgym|go2_hard_pact_stopgrad_isaacgym|go2_hard_pact_soft_penalty_isaacgym|go2_hard_pact_inverse_isaacgym|go2_hard_pact_rollout_isaacgym)
        export SIMULATOR=isaacgym; backend=isaacgym ;;
    go2_hard_pact_pos_isaaclab|go2_hard_pact_isaaclab|go2_hard_pact_baseline_isaaclab|go2_hard_pact_soft_isaaclab|go2_hard_pact_hard_isaaclab|go2_hard_pact_full_isaaclab|go2_hard_pact_stopgrad_isaaclab|go2_hard_pact_soft_penalty_isaaclab|go2_hard_pact_inverse_isaaclab|go2_hard_pact_rollout_isaaclab)
        export SIMULATOR=isaaclab; backend=isaaclab ;;
    *) usage; exit 2 ;;
esac

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
repo_dir="$(dirname -- "$(dirname -- "$script_dir")")"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
# Some packaged Genesis builds cannot locate their source-tree cache path.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_hard_pact}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_hard_pact}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/hard_pact_cache}"

# Select the same simulator environments used by the legacy launchers. Solver
# clones are selected only when that solver is requested. Every name can be
# overridden without editing this repository.
case "$backend" in
    genesis) target_env="${HARD_PACT_GENESIS_ENV:-genesis_lr}" ;;
    isaacgym) target_env="${HARD_PACT_ISAACGYM_ENV:-lr_gym}" ;;
    isaaclab) target_env="${HARD_PACT_ISAACLAB_ENV:-lr_lab}" ;;
esac
for solver in "${requested_solvers[@]}"; do
    case "$solver" in
        cupiqp)
            if [ "$backend" = isaaclab ]; then
                target_env="${HARD_PACT_ISAACLAB_CUPIQP_ENV:-lr_lab_cupiqp}"
            fi ;;
        moreau)
            if [ "$backend" = isaaclab ]; then
                target_env="${HARD_PACT_ISAACLAB_MOREAU_ENV:-lr_lab_moreau}"
            fi ;;
    esac
done
target_env="${HARD_PACT_CONDA_ENV:-$target_env}"
if [ "${HARD_PACT_SKIP_CONDA_ACTIVATE:-0}" != 1 ]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "Conda is unavailable; cannot activate HardPACT environment '$target_env'." >&2
        exit 1
    fi
    conda_base="$(conda info --base)"
    # shellcheck disable=SC1090
    . "$conda_base/etc/profile.d/conda.sh"
    conda activate "$target_env"
fi
echo "HardPACT environment: ${CONDA_DEFAULT_ENV:-$target_env}; simulator: $SIMULATOR" >&2
if [ "$smoke" -eq 1 ]; then
    # One iteration contains 32 real control steps and a differentiable PPO
    # update. User-provided values later in argv retain normal argparse order.
    exec python -u "$script_dir/train.py" \
        --num_envs 8 --max_iterations 1 "${forward[@]}"
fi
exec python -u "$script_dir/train.py" "${forward[@]}"
