#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec python -u legged_gym/scripts/eval_pilot_figure1.py "$@"
