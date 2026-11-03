#!/bin/bash
set -e
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ! command -v conda >/dev/null 2>&1; then
    . /home/oyoungquist/anaconda3/etc/profile.d/conda.sh
else
    . "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "${GENESIS_PACT_ENV:-genesis_lr}"
export SIMULATOR=genesis_pact_water
export PYTHONPATH="$(cd "$script_dir/../.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
exec python "$script_dir/play_pact_water.py" "$@"
