#!/usr/bin/env bash
set -euo pipefail
root=/home/share/yezitao-kimodo-reproduction/eval-results
shopt -s nullglob
for d in "$root"/v2-1m-hostnet-proxy128 "$root"/v2-1m-hostnet-proxy128.trashed-*; do
  echo "removing $d"
  rm -rf "$d"
done
echo done
ls -la "$root" | egrep "proxy128|stratified" || true
