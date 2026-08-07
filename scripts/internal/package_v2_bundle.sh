#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Internal stage implementation; use scripts/v2_pipeline.sh.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
python_bin="${KIMODO_PYTHON:-${repo_root}/.venv/bin/python}"
storage_root="${KIMODO_STORAGE_ROOT:-/mnt/kimodo}"
bundle="${KIMODO_V2_FINAL_ROOT:-${storage_root}/data/benchmark-v2-soma30-v2.2}"
compression_level="${KIMODO_ZSTD_LEVEL:-6}"
compression_threads="${KIMODO_ZSTD_THREADS:-32}"

[[ "${compression_level}" =~ ^([1-9]|1[0-9])$ ]] || {
  echo "KIMODO_ZSTD_LEVEL must be an integer from 1 through 19" >&2
  exit 2
}
[[ "${compression_threads}" =~ ^[1-9][0-9]*$ ]] || {
  echo "KIMODO_ZSTD_THREADS must be a positive integer" >&2
  exit 2
}
command -v zstd >/dev/null || { echo "zstd is required" >&2; exit 2; }

bundle="$(realpath -e -- "${bundle}")"
bundle_parent="$(dirname -- "${bundle}")"
bundle_name="$(basename -- "${bundle}")"
delivery="${KIMODO_V2_DELIVERY_DIR:-${bundle}.delivery}"
delivery_parent="$(dirname -- "${delivery}")"
mkdir -p -- "${delivery_parent}"
delivery_parent="$(realpath -e -- "${delivery_parent}")"
delivery="${delivery_parent}/$(basename -- "${delivery}")"
exec 7>"${delivery}.lock"
flock 7
archive_name="${bundle_name}.tar.zst"
checksum_name="${archive_name}.sha256"
metadata_name="${archive_name}.metadata.json"

case "${delivery}/" in
  "${bundle}/"*) echo "delivery directory must not be inside the V2 bundle" >&2; exit 2 ;;
esac
[[ ! -e "${delivery}" ]] || {
  echo "refusing to overwrite an existing V2 delivery directory: ${delivery}" >&2
  exit 2
}
[[ -f "${bundle}/resource-state.json" ]] || {
  echo "V2 bundle is not marked train-ready: ${bundle}" >&2
  exit 2
}

# Serialize with the builder even when this script is launched by a separate watcher.
exec 8>"${bundle}/.bundle-build.lock"
flock -s 8

"${python_bin}" -m kimodo.training.v2_resource_state_cli verify \
  --root "${bundle}" >/dev/null

"${python_bin}" - "${bundle}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path, PurePosixPath

root = Path(sys.argv[1]).resolve()
receipt = json.loads((root / "resource-state.json").read_text(encoding="utf-8"))
if receipt.get("status") != "v2_train_ready":
    raise SystemExit("resource-state.json does not mark a V2 train-ready bundle")
paths = {
    "cached_manifest_sha256": root / "train.cached.jsonl",
    "cached_manifest_metadata_sha256": root / "train.cached.jsonl.metadata.json",
    "inventory_sha256": root / "train.cached.references.jsonl",
    "inventory_metadata_sha256": root / "train.cached.references.jsonl.metadata.json",
    "stats_metadata_sha256": root / "stats/repro-soma30-30fps/stats.metadata.json",
    "paths_yaml_sha256": root / "repro.paths.yaml",
}
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for field, path in paths.items():
    if not path.is_file() or receipt.get("outputs", {}).get(field) != sha256(path):
        raise SystemExit(f"resource-state hash mismatch or missing file: {path}")
for path in root.rglob("*"):
    if path.is_symlink():
        raise SystemExit(f"delivery bundle must not contain symlinks: {path}")
    if not path.is_dir() and not path.is_file():
        raise SystemExit(f"delivery bundle contains a non-regular entry: {path}")
    if path.is_file() and not os.access(path, os.R_OK):
        raise SystemExit(f"delivery file is not readable: {path}")

def safe_relative(value, label):
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{label} is empty")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise SystemExit(f"{label} is not a safe relative POSIX path: {value!r}")

with (root / "train.cached.jsonl").open(encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for field in ("motion", "text_embedding", "text_embedding_metadata"):
            safe_relative(row.get(field), f"manifest line {line_number} {field}")
with (root / "train.cached.references.jsonl").open(encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if line.strip():
            safe_relative(json.loads(line).get("path"), f"inventory line {line_number} path")
PY

echo "[package] fully verifying the source bundle reference inventory"
"${python_bin}" -m kimodo.training.reference_inventory_cli verify \
  --manifest "${bundle}/train.cached.jsonl" \
  --inventory "${bundle}/train.cached.references.jsonl"

echo "[package] rerunning a real batch preflight from the final source path"
KIMODO_DATA_ROOT="${bundle}" KIMODO_RUN_ROOT="${bundle_parent}/runs" \
  "${python_bin}" -m kimodo.training.cli \
  --config "${repo_root}/configs/training/kimodo_soma_seed_v2_30k.yaml" \
  --paths "${bundle}/repro.paths.yaml" --preflight >/dev/null

staging="$(mktemp -d --tmpdir="${delivery_parent}" ".${bundle_name}.delivery.building.XXXXXX")"
verify_root="$(mktemp -d --tmpdir="${bundle_parent}" ".${bundle_name}.package-verify.XXXXXX")"
cleanup() {
  [[ -n "${staging:-}" && "${staging}" == "${delivery_parent}/.${bundle_name}.delivery.building."* ]] \
    && rm -rf -- "${staging}"
  [[ -n "${verify_root:-}" && "${verify_root}" == "${bundle_parent}/.${bundle_name}.package-verify."* ]] \
    && rm -rf -- "${verify_root}"
}
trap cleanup EXIT
archive="${staging}/${archive_name}"
checksum="${staging}/${checksum_name}"
package_metadata="${staging}/${metadata_name}"

echo "[package] creating portable tar.zst (level ${compression_level}, ${compression_threads} threads)"
tar --format=posix --hard-dereference --sort=name \
  --pax-option=delete=atime,delete=ctime \
  --owner=0 --group=0 --numeric-owner --mode='u+rwX,go+rX,go-w' \
  -cf - -C "${bundle_parent}" -- "${bundle_name}" \
  | zstd -q -T"${compression_threads}" "-${compression_level}" --check -o "${archive}"

echo "[package] testing the complete zstd frame and member paths"
zstd -q -t -- "${archive}"
members="${staging}/archive.members"
zstd -q -dc -- "${archive}" | tar -tf - >"${members}"
"${python_bin}" - "${bundle_name}" "${members}" <<'PY'
import sys
from pathlib import PurePosixPath
root = sys.argv[1]
seen = set()
with open(sys.argv[2], encoding="utf-8") as handle:
    for line_number, raw in enumerate(handle, 1):
        value = raw.rstrip("\n")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != root:
            raise SystemExit(f"unsafe archive member at line {line_number}: {value!r}")
        if value in seen:
            raise SystemExit(f"duplicate archive member at line {line_number}: {value!r}")
        seen.add(value)
if not seen:
    raise SystemExit("archive has no members")
PY
rm -f -- "${members}"

echo "[package] extracting into an isolated relocation path"
zstd -q -dc -- "${archive}" | tar --no-same-owner -xf - -C "${verify_root}"
extracted="${verify_root}/${bundle_name}"
[[ -d "${extracted}" ]] || { echo "archive did not extract the expected bundle root" >&2; exit 3; }

echo "[package] fully verifying inventory after relocation"
"${python_bin}" -m kimodo.training.reference_inventory_cli verify \
  --manifest "${extracted}/train.cached.jsonl" \
  --inventory "${extracted}/train.cached.references.jsonl"

echo "[package] verifying every resource-state output after relocation"
"${python_bin}" -m kimodo.training.v2_resource_state_cli verify \
  --root "${extracted}" >/dev/null

echo "[package] running a real batch from the relocated copy"
KIMODO_DATA_ROOT="${extracted}" KIMODO_RUN_ROOT="${verify_root}/runs" \
  "${python_bin}" -m kimodo.training.cli \
  --config "${repo_root}/configs/training/kimodo_soma_seed_v2_30k.yaml" \
  --paths "${extracted}/repro.paths.yaml" --preflight >/dev/null

(
  cd -- "${staging}"
  sha256sum -- "${archive_name}" >"${checksum_name}"
)
"${python_bin}" - "${bundle}" "${archive}" "${checksum}" "${package_metadata}" \
  "${extracted}" "${compression_level}" "${compression_threads}" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

bundle, archive, checksum, output, extracted = map(Path, sys.argv[1:6])
source_state_sha256 = hashlib.sha256(
    (bundle / "resource-state.json").read_bytes()
).hexdigest()
extracted_state_sha256 = hashlib.sha256(
    (extracted / "resource-state.json").read_bytes()
).hexdigest()
if extracted_state_sha256 != source_state_sha256:
    raise SystemExit("relocated resource-state.json differs from the source bundle")
record = {
    "schema_version": 2,
    "status": "v2_delivery_archive_verified_after_relocation",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "bundle_name": bundle.name,
    "bundle_resource_state_sha256": source_state_sha256,
    "archive": {
        "path": archive.name,
        "size": archive.stat().st_size,
        "sha256": checksum.read_text(encoding="utf-8").split()[0],
        "format": "posix-tar+zstd",
        "zstd_level": int(sys.argv[6]),
        "zstd_threads": int(sys.argv[7]),
        "hardlinks": "dereferenced",
        "normalized_owner": "0:0",
        "normalized_mode": "u+rwX,go+rX,go-w",
        "zstd_test": "passed",
        "isolated_extraction": "passed",
        "relocated_full_inventory": "passed",
        "relocated_data_preflight": "passed",
    },
    "checksum_file": checksum.name,
}
with output.open("w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

rm -rf -- "${verify_root}"
verify_root=""
mv -T -- "${staging}" "${delivery}"
staging=""
trap - EXIT

echo "[package] atomic delivery directory ready: ${delivery}"
echo "[package] archive: ${delivery}/${archive_name}"
echo "[package] checksum: ${delivery}/${checksum_name}"
echo "[package] metadata: ${delivery}/${metadata_name}"
