#!/usr/bin/env bash
# One assigned GPU, one checkout, one process. No conda activation on the cluster.
set -euo pipefail
usage() {
    echo "Usage: bash run.sh {check|smoke|train} TASK [training arguments]" >&2
    echo "       bash run.sh script TRAINING_UNITY_SCRIPT [training arguments]" >&2
    echo "Required: IMAGE REPO RUN_DIR CUDA_VISIBLE_DEVICES OMNI_KIT_ACCEPT_EULA=YES" >&2
}
[[ $# -ge 2 ]] || { usage; exit 2; }
mode=$1; task=$2; shift 2
case "$mode" in check|smoke|train|script) ;; *) usage; exit 2 ;; esac
: "${IMAGE:?Set IMAGE to the SIF path}"
: "${REPO:?Set REPO to the desired branch checkout}"
: "${RUN_DIR:?Set RUN_DIR to a unique persistent output directory}"
: "${CUDA_VISIBLE_DEVICES:?Use the scheduler-assigned CUDA_VISIBLE_DEVICES}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* && "$CUDA_VISIBLE_DEVICES" != -1 ]] || {
    echo "Request exactly one GPU for this launcher." >&2; exit 2;
}
[[ ${OMNI_KIT_ACCEPT_EULA:-} == YES ]] || {
    echo "Review NVIDIA's EULA, then set OMNI_KIT_ACCEPT_EULA=YES." >&2; exit 2;
}
if [[ "$mode" == script ]]; then
    entry="$task"
    case "$entry" in
        b1z1_unifp_unity.sh|b1z1_unifp_original_unity.sh|b1z1_unifp_reject_unity.sh)
            simulator=isaaclab_b1z1_unifp ;;
        b1z1_pact_unity.sh) simulator=isaaclab_b1z1_pact ;;
        b1z1_pact_pos_unity.sh) simulator=isaaclab_b1z1_pact_pos ;;
        go2_hard_pact_unity.sh) simulator=isaaclab ;;
        *) echo "Unsupported Unity launcher: $entry" >&2; exit 2 ;;
    esac
else
case "$task" in
    b1z1_unifp|b1z1_unifp_original|b1z1_unifp_reject)
        simulator=isaaclab_b1z1_unifp; entry=train.py ;;
    b1z1_pact) simulator=isaaclab_b1z1_pact; entry=train.py ;;
    b1z1_pact_pos) simulator=isaaclab_b1z1_pact_pos; entry=train.py ;;
    go2_hard_pact*_isaaclab) simulator=isaaclab; entry=train_hard_pact.py ;;
    *) echo "Unsupported IsaacLab task: $task" >&2; exit 2 ;;
esac
fi
for argument in "$@"; do
    case "$argument" in
        --task|--task=*)
            [[ "$mode" == script && "$entry" == go2_hard_pact_unity.sh ]] || {
                echo "Task is managed by this launcher." >&2; exit 2;
            } ;;
        --gpu|--gpu=*|--cpu)
            echo "Task/device are managed by this launcher, not extra arguments." >&2; exit 2 ;;
        --num_envs|--num_envs=*|--max_iterations|--max_iterations=*)
            [[ "$mode" != smoke ]] || { echo "Smoke size cannot be overridden." >&2; exit 2; } ;;
    esac
done
IMAGE=$(realpath "$IMAGE")
REPO=$(realpath "$REPO")
[[ -f "$IMAGE" && -f "$REPO/legged_gym/scripts/$entry" ]] || {
    echo "Missing image or task entrypoint in selected checkout." >&2; exit 2;
}
mkdir -p "$RUN_DIR" "$REPO/logs"
RUN_DIR=$(realpath "$RUN_DIR")
runtime=${RUNTIME_DIR:-$RUN_DIR/runtime}
mkdir -p "$runtime" "$RUN_DIR/logs"
runtime=$(realpath "$runtime")
mkdir -p "$runtime"/{home,cache,config,data,matplotlib,numba,torch_extensions,kit-cache,kit-data,kit-logs}
# Bind syntax uses colon/comma separators; fail rather than mount the wrong path.
for path in "$IMAGE" "$REPO" "$RUN_DIR" "$runtime"; do
    [[ "$path" != *:* && "$path" != *,* ]] || { echo "Unsupported bind path: $path" >&2; exit 2; }
done
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES
export APPTAINERENV_SIMULATOR="$simulator"
export APPTAINERENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export APPTAINERENV_WANDB_MODE="${WANDB_MODE:-offline}"
kit=/opt/env/lib/python3.11/site-packages/isaacsim/kit
options=(exec --nv --cleanenv --containall
    --home "$runtime/home:/home/container"
    --bind "$REPO:/workspace/repo" --bind "$runtime:/runtime"
    --bind "$RUN_DIR:/output" --bind "$RUN_DIR/logs:/workspace/repo/logs"
    --bind "$runtime/kit-cache:$kit/cache" --bind "$runtime/kit-data:$kit/data"
    --bind "$runtime/kit-logs:$kit/logs" --pwd /workspace/repo/legged_gym/scripts)
# --nv supplies driver libraries; some installations also need the host ICD JSON.
if [[ -n ${NVIDIA_ICD_FILE:-} ]]; then
    [[ -f "$NVIDIA_ICD_FILE" ]] || { echo "Missing NVIDIA_ICD_FILE" >&2; exit 2; }
    options+=(--bind "$NVIDIA_ICD_FILE:/etc/vulkan/icd.d/nvidia_icd.json:ro")
fi
{
    printf 'task=%s\nmode=%s\nrepo=%s\nimage=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
        "$task" "$mode" "$REPO" "$IMAGE" "$CUDA_VISIBLE_DEVICES"
    git -C "$REPO" rev-parse HEAD || true
    git -C "$REPO" status --short || true
} > "$RUN_DIR/launch-${SLURM_JOB_ID:-local}-$$.txt"
if [[ "$mode" == check ]]; then
    exec apptainer "${options[@]}" "$IMAGE" /opt/env/bin/python /opt/hcr-container/preflight.py "$task"
fi
if [[ "$mode" == script ]]; then
    exec apptainer "${options[@]}" "$IMAGE" /bin/bash "$entry" "$@"
fi
limits=()
[[ "$mode" != smoke ]] || limits=(--num_envs=2 --max_iterations=1)
exec apptainer "${options[@]}" "$IMAGE" /opt/env/bin/python -u "$entry" \
    --task="$task" --gpu=cuda:0 --headless "${limits[@]}" "$@"
