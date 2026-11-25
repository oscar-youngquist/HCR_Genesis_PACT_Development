#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SIMULATOR=isaaclab_b1z1_pact
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python legged_gym/scripts/eval_b1z1_pilot_compliance.py "$@"
