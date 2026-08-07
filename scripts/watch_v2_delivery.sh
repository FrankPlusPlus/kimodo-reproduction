#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
builder_pid="${KIMODO_V2_BUILDER_PID:-}"
final_root="${KIMODO_V2_FINAL_ROOT:-/pvc/benchmark-v2-soma30-v2.2}"
delivery_dir="${KIMODO_V2_DELIVERY_DIR:-${final_root}.delivery}"
poll_seconds="${KIMODO_V2_DELIVERY_POLL_SECONDS:-30}"
log_file="${KIMODO_V2_DELIVERY_LOG:-${final_root}.delivery-watcher.log}"

[[ "${poll_seconds}" =~ ^[1-9][0-9]*$ ]] || {
  echo "KIMODO_V2_DELIVERY_POLL_SECONDS must be a positive integer" >&2
  exit 2
}
if [[ -n "${builder_pid}" && ! "${builder_pid}" =~ ^[1-9][0-9]*$ ]]; then
  echo "KIMODO_V2_BUILDER_PID must be a positive integer" >&2
  exit 2
fi

mkdir -p -- "$(dirname -- "${log_file}")"
timestamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
stage() { echo "[$(timestamp)] V2 delivery watcher: $*" | tee -a "${log_file}"; }

delivery_is_complete() {
  local metadata
  metadata="$(find "${delivery_dir}" -maxdepth 1 -type f \
    -name '*.tar.zst.metadata.json' -print -quit 2>/dev/null || true)"
  [[ -n "${metadata}" ]] || return 1
  "${KIMODO_PYTHON:-${script_dir}/../.venv/bin/python}" - \
    "${delivery_dir}" "${metadata}" "${final_root}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
metadata = Path(sys.argv[2]).resolve()
bundle = Path(sys.argv[3]).resolve()
record = json.loads(metadata.read_text(encoding="utf-8"))
if record.get("schema_version") != 2 or record.get("status") != "v2_delivery_archive_verified_after_relocation":
    raise SystemExit(1)
archive_name = record.get("archive", {}).get("path")
checksum_name = record.get("checksum_file")
if not all(
    isinstance(value, str) and value and Path(value).name == value and value not in {".", ".."}
    for value in (archive_name, checksum_name)
):
    raise SystemExit(1)
archive = root / archive_name
checksum = root / checksum_name
if not archive.is_file() or not checksum.is_file():
    raise SystemExit(1)
parts = checksum.read_text(encoding="utf-8").split()
if len(parts) != 2 or parts[1].lstrip("*") != archive_name:
    raise SystemExit(1)
expected = parts[0]
digest = hashlib.sha256()
with archive.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
if (
    digest.hexdigest() != expected
    or record["archive"]["sha256"] != expected
    or record["archive"].get("size") != archive.stat().st_size
    or record.get("bundle_name") != bundle.name
):
    raise SystemExit(1)
state = bundle / "resource-state.json"
if not state.is_file() or hashlib.sha256(state.read_bytes()).hexdigest() != record.get(
    "bundle_resource_state_sha256"
):
    raise SystemExit(1)
expected_files = {archive_name, checksum_name, metadata.name}
if {path.name for path in root.iterdir() if path.is_file()} != expected_files:
    raise SystemExit(1)
PY
}

if delivery_is_complete; then
  stage "verified delivery already exists at ${delivery_dir}"
  exit 0
fi

while [[ ! -f "${final_root}/resource-state.json" ]]; do
  if [[ -n "${builder_pid}" && ! -d "/proc/${builder_pid}" ]]; then
    stage "builder PID ${builder_pid} exited before publishing a train-ready bundle"
    exit 3
  fi
  stage "waiting for train-ready bundle at ${final_root}"
  sleep "${poll_seconds}"
done

stage "train-ready bundle detected; starting portable archive construction"
"${script_dir}/package_v2_bundle.sh" 2>&1 | tee -a "${log_file}"
delivery_is_complete
stage "delivery archive and SHA-256 verified at ${delivery_dir}"
