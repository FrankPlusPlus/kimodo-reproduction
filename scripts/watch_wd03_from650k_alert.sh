#!/usr/bin/env bash
# Historical alias. Live watcher is lastwd1-from750k.
set -euo pipefail
exec bash "$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)/watch_lastwd1_from750k_alert.sh" "$@"
