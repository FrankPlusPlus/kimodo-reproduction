#!/usr/bin/env bash
set -euo pipefail

# Production entrypoint for 16 H200 DDP ranks. Kubernetes may expose them as
# 2 pods x 8 GPUs, 4 pods x 4 GPUs, or 16 pods x 1 GPU; a pod cannot span
# physical nodes. KIMODO_NNODES is the number of launcher pods, not the number
# of physical machines.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"

default_training_config="${project_root}/configs/training/kimodo_soma_seed_v2_1m_16h200.yaml"
export KIMODO_TRAINING_CONFIG="${KIMODO_TRAINING_CONFIG:-${default_training_config}}"
export KIMODO_TRAINING_OVERLAY=""
export KIMODO_PATHS_CONFIG="${KIMODO_PATHS_CONFIG:-/mnt/kimodo/config/repro.paths.yaml}"
export KIMODO_NNODES="${KIMODO_NNODES:-${NNODES:-2}}"
export KIMODO_NPROC_PER_NODE="${KIMODO_NPROC_PER_NODE:-${NPROC_PER_NODE:-8}}"
export KIMODO_EXPECTED_GPU_PATTERN="${KIMODO_EXPECTED_GPU_PATTERN:-H200}"
export KIMODO_REQUIRE_RDMA="${KIMODO_REQUIRE_RDMA:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ ! "${KIMODO_NNODES}" =~ ^[1-9][0-9]*$ || ! "${KIMODO_NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "KIMODO_NNODES and KIMODO_NPROC_PER_NODE must be positive integers" >&2
  exit 2
fi
if (( KIMODO_NNODES * KIMODO_NPROC_PER_NODE != 16 )); then
  echo "company launcher requires exactly 16 total ranks; got ${KIMODO_NNODES}x${KIMODO_NPROC_PER_NODE}" >&2
  exit 2
fi

if [[ "${KIMODO_TRAINING_CONFIG}" != "${default_training_config}" && -z "${KIMODO_RUN_DIR:-}" ]]; then
  echo "KIMODO_RUN_DIR is required when selecting a non-default V1/V2 training config" >&2
  exit 2
fi
run_dir="${KIMODO_RUN_DIR:-/mnt/kimodo/runs/v2-1m-production}"
resume_args=()
data_args=()
if [[ -n "${KIMODO_DATA_WORKERS:-}" ]]; then
  if [[ ! "${KIMODO_DATA_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "KIMODO_DATA_WORKERS must be a positive integer" >&2
    exit 2
  fi
  data_args+=(--set "data.num_workers=${KIMODO_DATA_WORKERS}")
fi
if [[ "${KIMODO_AUTO_RESUME:-1}" == 1 ]]; then
  latest_pointer="${run_dir}/checkpoints/latest.txt"
  if [[ -f "${latest_pointer}" ]]; then
    checkpoint_name="$(<"${latest_pointer}")"
    if [[ ! "${checkpoint_name}" =~ ^step-[0-9]{9}\.pt$ ]]; then
      echo "invalid checkpoint pointer in ${latest_pointer}: ${checkpoint_name}" >&2
      exit 2
    fi
    if [[ "${checkpoint_name}" == "step-001000000.pt" ]]; then
      final_bundle="${run_dir}/exports/step-001000000"
      if [[ -f "${final_bundle}/model.pt" && -f "${final_bundle}/config.yaml" && -d "${final_bundle}/stats" ]]; then
        echo "company training is already complete: ${run_dir}/checkpoints/${checkpoint_name}" >&2
        exit 0
      fi
    fi
    checkpoint_path="${run_dir}/checkpoints/${checkpoint_name}"
    [[ -f "${checkpoint_path}" ]] || {
      echo "checkpoint pointer target is missing: ${checkpoint_path}" >&2
      exit 2
    }
    resume_args+=(--set "runtime.resume=${checkpoint_path}")
  fi
fi

exec "${script_dir}/train_distributed.sh" \
  --set "runtime.output_dir=${run_dir}" \
  "${data_args[@]}" \
  "${resume_args[@]}" \
  "$@"
