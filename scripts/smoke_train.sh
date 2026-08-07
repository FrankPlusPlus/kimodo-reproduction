#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
python_bin="${KIMODO_PYTHON:-${project_root}/.venv/bin/python}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python environment is missing: ${python_bin}" >&2
  exit 2
fi

if [[ -n "${KIMODO_SMOKE_ROOT:-}" ]]; then
  smoke_root="$(realpath -m -- "${KIMODO_SMOKE_ROOT}")"
  mkdir -p -- "${smoke_root}"
else
  smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/kimodo-smoke.XXXXXX")"
fi
fixture_root="${smoke_root}/fixture"
run_root="${smoke_root}/run"

"${python_bin}" -m kimodo.devtools.smoke_fixture_cli --output "${fixture_root}"

common_args=(
  --config "${project_root}/configs/training/kimodo_tiny_smoke.yaml"
  --set "data.manifest=${fixture_root}/manifest.jsonl"
  --set "model.stats_path=${fixture_root}/stats"
  --set "runtime.output_dir=${run_root}"
)

"${python_bin}" -m kimodo.training.cli "${common_args[@]}" --preflight
"${python_bin}" -m kimodo.training.cli "${common_args[@]}"

test -f "${run_root}/checkpoints/step-000000002.pt"
test -f "${run_root}/exports/step-000000002/model.pt"
echo "Kimodo two-step smoke passed: ${smoke_root}"
