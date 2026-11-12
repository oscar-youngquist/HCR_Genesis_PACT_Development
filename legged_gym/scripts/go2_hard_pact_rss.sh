#!/usr/bin/env bash
# Shared local/container dispatch; tiny variant wrappers contain no training logic.
set -euo pipefail
location=${1:?expected lab or unity}; variant=${2:?expected RSS variant}; shift 2
case "$variant" in pos_ppo|coupled_ppo|guided_no_qp|qp_no_post|qp_imitation|full|no_fk|no_ff|no_channels|no_qp_wrench) ;;
    *) echo "Unknown RSS variant: $variant" >&2; exit 2;; esac
for arg in "$@"; do
    case "$arg" in --task|--task=*) echo "Task is selected by the variant wrapper." >&2; exit 2;; esac
done
extra=()
case "$location" in
    lab) ;;
    unity)
        : "${CUDA_VISIBLE_DEVICES:?Preserve the scheduler-assigned GPU visibility}"
        for arg in "$@"; do
            case "$arg" in --gpu|--gpu=*|--cpu) echo "Unity uses logical cuda:0." >&2; exit 2;; esac
        done
        export HARD_PACT_SKIP_CONDA_ACTIVATE=1
        export HARD_PACT_PYTHON=${HARD_PACT_PYTHON:-/opt/env/bin/python}
        extra=(--gpu cuda:0 --headless)
        ;;
    *) echo "Expected lab or unity" >&2; exit 2;;
esac
exec bash "$(dirname "$0")/go2_hard_pact.sh"     --task "go2_hard_pact_rss_${variant}_isaaclab" "${extra[@]}" "$@"
