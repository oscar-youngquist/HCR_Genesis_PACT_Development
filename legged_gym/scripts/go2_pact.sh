#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_pact

# Keep the legacy launch unchanged by default, while allowing a HardPACT smoke
# run to use this same entry point without maintaining a second launcher.
TASK="${TASK:-go2_pact}"
GPU="${GPU:-cuda:1}"
SEED="${SEED:-1}"

set -- train.py --task="$TASK" --headless --seed="$SEED" --gpu="$GPU"
if [ -n "$NUM_ENVS" ]; then
    set -- "$@" --num_envs="$NUM_ENVS"
fi
if [ -n "$MAX_ITERATIONS" ]; then
    set -- "$@" --max_iterations="$MAX_ITERATIONS"
fi
if [ -n "$PINN_LOSS_WEIGHT" ]; then
    set -- "$@" --pinn_loss_weight="$PINN_LOSS_WEIGHT"
fi

python -u "$@"
