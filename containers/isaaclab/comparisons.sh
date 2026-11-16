#!/usr/bin/env bash
# Portable local/remote entrypoint; same runtime and Python training as Unity.
set -euo pipefail
[[ $# -ge 2 ]] || { echo 'Usage: comparisons.sh {br|mule} {check|smoke|train} [training arguments]' >&2; exit 2; }
method=$1; mode=$2; shift 2
case "$method" in
    br) task=go2_beyond_robustness_isaaclab ;;
    mule) task=go2_mule_isaaclab ;;
    *) echo 'Choose br or mule.' >&2; exit 2 ;;
esac
case "$mode" in check|smoke|train) ;; *) echo 'Choose check, smoke, or train.' >&2; exit 2;; esac
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export REPO=${REPO:-$(cd "$here/../.." && pwd)}
# On a workstation choose physical GPU 0 unless explicitly set. Never replace
# a scheduler-provided assignment; training sees its logical cuda:0.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0}
if [[ -f "$REPO/CONTAINER_SOURCE.json" ]]; then
    export TRAINING_GIT_SHA=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["git_sha"])' "$REPO/CONTAINER_SOURCE.json")
    export TRAINING_GIT_STATUS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["git_status"])' "$REPO/CONTAINER_SOURCE.json")
fi
exec bash "$here/run.sh" "$mode" "$task" "$@"
