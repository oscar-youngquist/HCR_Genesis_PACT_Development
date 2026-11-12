#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/go2_hard_pact_rss.sh" lab pos_ppo "$@"
