#!/bin/bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$(cd "$script_dir/../.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
exec "$HOME/.conda/envs/genesis_lr/bin/python" -m liquid_collection.visualize "$@"
