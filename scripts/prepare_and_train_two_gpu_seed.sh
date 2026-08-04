#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper around the portable resource and training entrypoints.
# New deployments should call scripts/resources/resources.sh and
# scripts/train_two_gpu_seed.sh explicitly.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
resource_paths="${KIMODO_RESOURCE_PATHS:-${project_root}/resources/paths.local.yaml}"
training_paths="${KIMODO_PATHS_CONFIG:-}"

if [[ ! -f "${resource_paths}" ]]; then
  echo "Resource paths YAML is missing: ${resource_paths}" >&2
  echo "Copy resources/paths.example.yaml, edit it, and set KIMODO_RESOURCE_PATHS." >&2
  exit 2
fi

"${project_root}/scripts/resources/resources.sh" \
  --paths "${resource_paths}" prepare

if [[ "${KIMODO_PREPARE_ONLY:-0}" == 1 ]]; then
  exit 0
fi
if [[ -z "${training_paths}" || ! -f "${training_paths}" ]]; then
  echo "Set KIMODO_PATHS_CONFIG to pipeline.repro_paths_yaml from the resource YAML." >&2
  exit 2
fi

exec "${script_dir}/train_two_gpu_seed.sh" "$@"
