#!/usr/bin/env bash
set -euo pipefail

# Hardware-neutral image dispatcher.
#
# Company contract: the image provides the runtime environment and this
# launcher. Training/eval code is read from the shared PVC checkout pointed to
# by KIMODO_CODE_ROOT. An explicit Docker/Kubernetes command bypasses this
# script through Docker's normal CMD override semantics. When no command is
# supplied, KIMODO_CONTAINER_MODE selects one reviewed workflow.

image_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image_root="$(cd -- "${image_script_dir}/.." && pwd)"

DEFAULT_CODE_ROOT="/home/share/yzt/kimodo-reproduction"
DEFAULT_STORAGE_ROOT="/home/share/yezitao-kimodo-reproduction"

resolve_code_root() {
  local candidate="${KIMODO_CODE_ROOT:-${DEFAULT_CODE_ROOT}}"
  if [[ -d "${candidate}/scripts" && -d "${candidate}/kimodo" ]]; then
    printf '%s\n' "${candidate}"
    return 0
  fi
  # Image-local fallback keeps idle/debug Pods and unit tests usable without
  # the company share PVC mounted.
  if [[ -d "${image_root}/scripts" && -d "${image_root}/kimodo" ]]; then
    printf '%s\n' "${image_root}"
    return 0
  fi
  echo "Kimodo launcher: code root not found." >&2
  echo "Kimodo launcher: set KIMODO_CODE_ROOT to the PVC checkout (default ${DEFAULT_CODE_ROOT})" >&2
  echo "Kimodo launcher: and mount the share PVC at /home/share." >&2
  return 1
}

project_root="$(resolve_code_root)"
script_dir="${project_root}/scripts"
export KIMODO_CODE_ROOT="${project_root}"
export KIMODO_STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-${DEFAULT_STORAGE_ROOT}}"
export PYTHONPATH="${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
# Load WANDB / MiMo keys and other secrets from PVC .env if present.
# shellcheck disable=SC1091
source "${script_dir}/load_kimodo_env.sh"
kimodo_load_env_files
mode="${KIMODO_CONTAINER_MODE:-idle}"

if [[ "${project_root}" != "${image_root}" ]]; then
  echo "Kimodo launcher: using PVC code at ${project_root}" >&2
else
  echo "Kimodo launcher: using image-local code at ${project_root}" >&2
fi

usage() {
  cat <<EOF
Kimodo container modes:
  idle           keep a generic/debug Pod alive (default)
  train-company  launch company training (topology via env; default 2x8 / 16 ranks)
  train-local    launch the explicit two-GPU laboratory contract
  prepare        prepare or bind data on mounted shared storage
  preflight      load and validate one real training batch without training
  eval-watch     evaluate immutable exports produced by a training run
  eval-official  build the fixed official-model benchmark baseline
  help           print this message and exit

Company defaults:
  KIMODO_CODE_ROOT=${DEFAULT_CODE_ROOT}
  KIMODO_STORAGE_ROOT=${DEFAULT_STORAGE_ROOT}

Mount the share PVC at /home/share. The image runtime stays in the container;
reviewed modes exec scripts from KIMODO_CODE_ROOT. An explicit Kubernetes
command may invoke any \$KIMODO_CODE_ROOT/scripts/*.sh entry directly.
EOF
}

case "${mode}" in
  idle)
    echo "Kimodo image is ready in idle mode; set KIMODO_CONTAINER_MODE or override the Pod command." >&2
    exec sleep infinity
    ;;
  train-company)
    exec "${script_dir}/train_company.sh" "$@"
    ;;
  train-local)
    exec "${script_dir}/train_two_gpu_seed.sh" "$@"
    ;;
  prepare)
    exec "${script_dir}/prepare_container.sh" "$@"
    ;;
  preflight)
    python_bin="${KIMODO_PYTHON:-python}"
    config_path="${KIMODO_TRAINING_CONFIG:-${project_root}/configs/training/kimodo_soma_seed_v2_1m_16h200.yaml}"
    storage_root="${KIMODO_STORAGE_ROOT}"
    export KIMODO_DATA_ROOT="${KIMODO_DATA_ROOT:-${storage_root}/benchmark-v2-soma30-v2.2}"
    export KIMODO_RUN_ROOT="${KIMODO_RUN_ROOT:-${storage_root}/runs}"
    paths_path="${KIMODO_PATHS_CONFIG:-${KIMODO_DATA_ROOT}/repro.paths.yaml}"
    exec "${python_bin}" -m kimodo.training.cli \
      --config "${config_path}" --paths "${paths_path}" --preflight "$@"
    ;;
  eval-watch)
    exec "${script_dir}/eval_company_watcher.sh" "$@"
    ;;
  eval-official)
    exec "${script_dir}/eval_official_baseline.sh" "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "unknown KIMODO_CONTAINER_MODE: ${mode}" >&2
    usage >&2
    exit 2
    ;;
esac
