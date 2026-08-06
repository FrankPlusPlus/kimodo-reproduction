#!/usr/bin/env bash
set -euo pipefail

# Container entrypoint for the data-preparation Job. It is intentionally
# separate from GPU training so retries/restarts do not rebuild the dataset.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
python_bin="${KIMODO_PYTHON:-python}"
storage_root="${KIMODO_STORAGE_ROOT:-/mnt/kimodo}"
resource_paths="${KIMODO_RESOURCE_PATHS:-${storage_root}/config/resources.paths.yaml}"
training_paths="${KIMODO_PATHS_CONFIG:-${storage_root}/config/repro.paths.yaml}"
prepared_root="${KIMODO_PREPARED_ROOT:-}"
catalog="${KIMODO_RESOURCE_CATALOG:-${project_root}/resources/catalog.public.yaml}"

mkdir -p "${storage_root}/config" "${storage_root}/runs"

if [[ -n "${prepared_root}" ]]; then
  exec "${python_bin}" -m kimodo.resources.cli \
    --catalog "${catalog}" bind-prepared \
    --prepared-root "${prepared_root}" \
    --run-root "${storage_root}/runs" \
    --output "${training_paths}"
fi

if [[ ! -f "${resource_paths}" ]]; then
  "${python_bin}" -m kimodo.resources.cli \
    --catalog "${catalog}" init \
    --output "${resource_paths}" \
    --storage-root "${storage_root}" \
    --asset-mode "${KIMODO_ASSET_MODE:-hardlink}"
fi

KIMODO_PYTHON="${python_bin}" exec "${project_root}/scripts/resources/resources.sh" \
  --paths "${resource_paths}" all
