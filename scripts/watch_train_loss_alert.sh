#!/usr/bin/env bash
# Live train-loss watcher. Follows the 750k last-layer wd=1 fork.
set -euo pipefail
exec bash "$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)/watch_lastwd1_from750k_alert.sh" "$@"
