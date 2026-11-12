#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/go2_hard_pact_rss.sh" lab guided_no_qp "$@"
