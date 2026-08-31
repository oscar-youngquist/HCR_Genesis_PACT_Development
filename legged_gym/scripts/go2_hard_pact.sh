#!/bin/bash

set -euo pipefail

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

backend="${1:-genesis}"
task="${2:-go2_hard_pact}"
gpu="${GPU:-cuda:0}"
if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi

case "$backend" in
    genesis)
        conda activate /home/oyoungquist/.conda/envs/genesis_lr
        ;;
    isaaclab)
        conda activate /home/oyoungquist/.conda/envs/lr_lab
        ;;
    *)
        echo "Unsupported HardPACT backend '$backend'; use genesis or isaaclab." >&2
        exit 2
        ;;
esac

if [[ "$gpu" == "cpu" ]]; then
    echo "Go2 HardPACT launchers require a CUDA device; set GPU=cuda:N." >&2
    exit 2
fi
for argument in "$@"; do
    if [[ "$argument" == "--cpu" ]]; then
        echo "Go2 HardPACT launchers do not accept --cpu." >&2
        exit 2
    fi
done

export SIMULATOR="$backend"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$script_dir/train.py" \
    --task="$task" \
    --headless \
    --seed="${SEED:-1}" \
    --gpu="$gpu" \
    "$@"
