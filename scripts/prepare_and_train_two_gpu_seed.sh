#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper around the portable resource and training entrypoints.
# New deployments should call scripts/resources/resources.sh and
# scripts/train_two_gpu_seed.sh explicitly.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
resource_paths="${KIMODO_RESOURCE_PATHS:-${project_root}/resources/paths.local.yaml}"
training_paths="${KIMODO_PATHS_CONFIG:-}"
python_bin="${KIMODO_PYTHON:-${project_root}/.venv/bin/python}"

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
if [[ -z "${training_paths}" ]]; then
  training_paths="$(${python_bin} - "${resource_paths}" "${project_root}" <<'PY'
import sys
from pathlib import Path
from kimodo.resources.config import load_catalog, load_paths

root = Path(sys.argv[2])
catalog = load_catalog(root / "resources/catalog.public.yaml")
paths = load_paths(sys.argv[1], catalog)
if paths.pipeline is None:
    raise SystemExit("resource paths YAML has no pipeline section")
print(paths.pipeline.repro_paths_yaml)
PY
)"
fi
if [[ ! -f "${training_paths}" ]]; then
  echo "Generated training paths YAML is missing: ${training_paths}" >&2
  exit 2
fi

KIMODO_PATHS_CONFIG="${training_paths}" exec "${script_dir}/train_two_gpu_seed.sh" "$@"
