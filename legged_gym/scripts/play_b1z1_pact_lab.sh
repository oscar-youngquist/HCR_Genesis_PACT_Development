#!/usr/bin/env sh
# Clear backend-specific alias; keep existing invocations working too.
exec sh "$(dirname "$0")/play_b1z1_pact_gym.sh" "$@"
