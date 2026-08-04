#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
storage_root=""
paths_file=""
legacy_root=""
conversion_inventory=""
prepared_root=""
asset_mode="hardlink"
run_training=false
setup_system_site=false
gpu_ids=""
hf_login=false

usage() {
  echo "Usage: $0 --storage-root /shared/kimodo [--legacy-root /shared/old-bundle]"
  echo "          [--prepared-root /shared/copied-train-ready-bundle]"
  echo "          [--conversion-inventory FILE] [--asset-mode hardlink|copy]"
  echo "          [--hf-login] [--gpus 0,1] [--train] [--system-site-packages]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --storage-root) storage_root="$2"; shift 2 ;;
    --paths) paths_file="$2"; shift 2 ;;
    --legacy-root) legacy_root="$2"; shift 2 ;;
    --prepared-root) prepared_root="$2"; shift 2 ;;
    --conversion-inventory) conversion_inventory="$2"; shift 2 ;;
    --asset-mode) asset_mode="$2"; shift 2 ;;
    --gpus) gpu_ids="$2"; shift 2 ;;
    --hf-login) hf_login=true; shift ;;
    --train) run_training=true; shift ;;
    --system-site-packages) setup_system_site=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -z "${storage_root}" ]]; then
  usage >&2
  exit 2
fi
if [[ -z "${paths_file}" ]]; then
  paths_file="${storage_root}/config/resources.paths.yaml"
fi
if [[ -n "${legacy_root}" && -n "${prepared_root}" ]]; then
  echo "--legacy-root and --prepared-root are mutually exclusive" >&2
  exit 2
fi

setup_args=()
if [[ "${setup_system_site}" == true ]]; then
  setup_args+=(--system-site-packages)
fi
if [[ -n "${legacy_root}" || -n "${prepared_root}" ]]; then
  setup_args+=(--skip-flowmatching)
fi
"${project_root}/scripts/resources/setup_env.sh" "${setup_args[@]}"
python_bin="${KIMODO_VENV:-${project_root}/.venv}/bin/python"
if [[ "${hf_login}" == true ]]; then
  "$(dirname -- "${python_bin}")/hf" auth login
fi
training_paths="${storage_root}/config/repro.paths.yaml"
if [[ -n "${prepared_root}" ]]; then
  "${python_bin}" -m kimodo.resources.cli \
    --catalog "${project_root}/resources/catalog.public.yaml" bind-prepared \
    --prepared-root "${prepared_root}" \
    --run-root "${storage_root}/runs" \
    --output "${training_paths}"
else
  init_args=(--catalog "${project_root}/resources/catalog.public.yaml" init
    --output "${paths_file}" --storage-root "${storage_root}" --asset-mode "${asset_mode}")
  if [[ -n "${legacy_root}" ]]; then
    init_args+=(--legacy-root "${legacy_root}")
  fi
  if [[ -n "${conversion_inventory}" ]]; then
    init_args+=(--conversion-inventory "${conversion_inventory}")
  fi
  "${python_bin}" -m kimodo.resources.cli "${init_args[@]}"
fi

if [[ -n "${prepared_root}" ]]; then
  :
elif [[ -n "${legacy_root}" ]]; then
  "${project_root}/scripts/resources/resources.sh" --paths "${paths_file}" adopt-legacy
else
  "${project_root}/scripts/resources/resources.sh" --paths "${paths_file}" all
fi

echo "Training resources are ready. Paths: ${training_paths}"
if [[ "${run_training}" != true ]]; then
  echo "Start two-GPU training with:"
  echo "KIMODO_PATHS_CONFIG=${training_paths} CUDA_VISIBLE_DEVICES=<gpu0,gpu1> scripts/train_two_gpu_seed.sh"
  exit 0
fi
if [[ -n "${gpu_ids}" ]]; then
  export CUDA_VISIBLE_DEVICES="${gpu_ids}"
fi
KIMODO_PATHS_CONFIG="${training_paths}" exec "${project_root}/scripts/train_two_gpu_seed.sh"
