#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
python_bin="${KIMODO_PYTHON:-${project_root}/.venv/bin/python}"
catalog="${KIMODO_RESOURCE_CATALOG:-${project_root}/resources/catalog.public.yaml}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python environment is missing: ${python_bin}" >&2
  echo "Run scripts/resources/setup_env.sh first or set KIMODO_PYTHON." >&2
  exit 2
fi

cd "${project_root}"
exec "${python_bin}" -m kimodo.resources.cli --catalog "${catalog}" "$@"
