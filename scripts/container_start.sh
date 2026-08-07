#!/usr/bin/env bash
set -euo pipefail

# Hardware-neutral image dispatcher.  An explicit Docker/Kubernetes command
# bypasses this script through Docker's normal CMD override semantics.  When no
# command is supplied, KIMODO_CONTAINER_MODE selects one reviewed workflow.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
mode="${KIMODO_CONTAINER_MODE:-idle}"

usage() {
  cat <<'EOF'
Kimodo container modes:
  idle           keep a generic/debug Pod alive (default)
  train-company  launch the 16-rank company training contract
  train-local    launch the explicit two-GPU laboratory contract
  prepare        prepare or bind data on mounted shared storage
  preflight      load and validate one real training batch without training
  eval-watch     evaluate immutable exports produced by a training run
  eval-official  build the fixed official-model benchmark baseline
  help           print this message and exit

An explicit Kubernetes command may invoke any /workspace/scripts/*.sh entry
directly and does not need KIMODO_CONTAINER_MODE.
EOF
}

case "${mode}" in
  idle)
    echo "Kimodo image is ready in idle mode; set KIMODO_CONTAINER_MODE or override the Pod command." >&2
    exec sleep infinity
    ;;
  train-company)
    exec "${script_dir}/train_company_16h200.sh" "$@"
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
    storage_root="${KIMODO_STORAGE_ROOT:-/mnt/kimodo}"
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
