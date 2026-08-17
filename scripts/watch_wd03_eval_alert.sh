#!/usr/bin/env bash
# Historical alias. Live eval watcher is lastwd1-from750k (760k every 20k).
set -euo pipefail
exec bash "$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)/watch_lastwd1_from750k_eval_alert.sh" "$@"
