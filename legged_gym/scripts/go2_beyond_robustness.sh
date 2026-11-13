#!/usr/bin/env bash
# Thin runtime selection; all config/learning/stage/resume logic stays in Python.
set -euo pipefail
mode=$1; shift
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)
export SIMULATOR=isaaclab
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
python_bin=${BR_PYTHON:-python}
target_env=${BR_CONDA_ENV:-${HARD_PACT_CONDA_ENV:-${HARD_PACT_ISAACLAB_ENV:-lr_lab}}}
forward=(); dry_run=${HARD_PACT_DRY_RUN:-0}
for argument in "$@"; do
    if [[ $argument == --dry-run ]]; then dry_run=1; continue; fi
    if [[ $mode == unity ]]; then
        case "$argument" in --gpu|--gpu=*|--device|--device=*|--cpu)
            echo 'Unity preserves assigned GPU visibility and uses logical cuda:0.' >&2; exit 2;; esac
    fi
    forward+=("$argument")
done
case "$mode" in
    lab) ;;
    unity)
        : "${CUDA_VISIBLE_DEVICES:?Use the scheduler GPU assignment}"
        python_bin=${BR_CONTAINER_PYTHON:-/opt/env/bin/python}
        forward=(--device cuda:0 --headless "${forward[@]}") ;;
    *) exit 2;;
esac
if [[ $dry_run == 1 ]]; then
    printf 'mode=%s conda=%s CUDA_VISIBLE_DEVICES=%s\n' "$mode" "$target_env" "${CUDA_VISIBLE_DEVICES:-unset}"
    printf '%q ' "$python_bin" -u "$script_dir/train_beyond_robustness.py" "${forward[@]}"; printf '\n'
    exit 0
fi
if [[ $mode == lab && ${BR_SKIP_CONDA_ACTIVATE:-0} != 1 ]]; then
    command -v conda >/dev/null || { echo "Conda unavailable; activate $target_env or set BR_SKIP_CONDA_ACTIVATE=1 and BR_PYTHON." >&2; exit 1; }
    conda_base=$(conda info --base)
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate "$target_env"
fi
cd "$repo_dir"
exec "$python_bin" -u "$script_dir/train_beyond_robustness.py" "${forward[@]}"
