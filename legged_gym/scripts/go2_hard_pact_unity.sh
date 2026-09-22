#!/usr/bin/env bash
# Copy this wrapper into the aligned checkout; retain its existing Warp/QP setup.
set -euo pipefail
cd "$(dirname "$0")"
export HARD_PACT_SKIP_CONDA_ACTIVATE=1
# Default full IsaacLab task; callers may select a registered IsaacLab ablation.
task=go2_hard_pact_full_isaaclab
previous=""
for argument in "$@"; do
    case "$argument" in --task=*) task=${argument#--task=} ;; esac
    [[ "$previous" != --task ]] || task=$argument
    previous=$argument
done
case "$task" in
    go2_hard_pact*_isaaclab) ;;
    *) echo "Unity container supports only IsaacLab HardPACT tasks." >&2; exit 2 ;;
esac
exec bash go2_hard_pact.sh --task=go2_hard_pact_full_isaaclab \
    "$@" --gpu=cuda:0 --headless
