#!/usr/bin/env bash
# Keep one shared player for HardPACT and HardPACTPos.
set -euo pipefail
exec bash "$(dirname "$0")/play_go2_hard_pact_pos.sh" "$@"
