#!/usr/bin/env bash
set -euo pipefail

# Single company training entry. Topology, data, and runtime contracts are
# controlled by environment variables. Defaults match the formal 2x8 / 16-rank
# V2 production run on Hanhai.
#
# Examples:
#   # Production 2x8
#   KIMODO_NNODES=2 KIMODO_NPROC_PER_NODE=8 \
#     bash scripts/train_company.sh
#
#   # Connectivity smoke 2x3
#   KIMODO_NNODES=2 KIMODO_NPROC_PER_NODE=3 \
#   KIMODO_EXPECTED_WORLD_SIZE=6 \
#   KIMODO_BATCH_SIZE=8 KIMODO_MAX_STEPS=5 \
#   KIMODO_RUN_DIR=/home/share/yezitao-kimodo-reproduction/runs/v2-2x3-smoke \
#     bash scripts/train_company.sh
#
# Key env:
#   KIMODO_NNODES / KIMODO_NPROC_PER_NODE   (PET_* / NNODES / NPROC_PER_NODE fallbacks)
#   KIMODO_EXPECTED_WORLD_SIZE             default 16; set empty to disable the gate
#   KIMODO_TRAINING_CONFIG / OVERLAY / RUN_DIR / DATA_ROOT / PATHS_CONFIG
#   KIMODO_BATCH_SIZE / KIMODO_GRAD_ACCUM / KIMODO_MAX_STEPS / KIMODO_DATA_WORKERS
#   KIMODO_AUTO_RESUME / KIMODO_REQUIRE_RDMA / KIMODO_EXPECTED_GPU_PATTERN

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
export KIMODO_CODE_ROOT="${KIMODO_CODE_ROOT:-${project_root}}"
storage_root="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
export KIMODO_STORAGE_ROOT="${storage_root}"
export PYTHONPATH="${KIMODO_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# shellcheck disable=SC1091
source "${script_dir}/load_kimodo_env.sh"
kimodo_load_env_files
# shellcheck disable=SC1091
source "${script_dir}/nccl_rdma_env.sh"
kimodo_nccl_rdma_env

default_training_config="${project_root}/configs/training/kimodo_soma_seed_v2_1m_16h200.yaml"
export KIMODO_TRAINING_CONFIG="${KIMODO_TRAINING_CONFIG:-${default_training_config}}"
export KIMODO_TRAINING_OVERLAY="${KIMODO_TRAINING_OVERLAY:-}"
export KIMODO_DATA_ROOT="${KIMODO_DATA_ROOT:-${storage_root}/benchmark-v2-soma30-v2.2}"
export KIMODO_RUN_ROOT="${KIMODO_RUN_ROOT:-${storage_root}/runs}"
export KIMODO_PATHS_CONFIG="${KIMODO_PATHS_CONFIG:-${KIMODO_DATA_ROOT}/repro.paths.yaml}"
export KIMODO_NNODES="${KIMODO_NNODES:-${NNODES:-${PET_NNODES:-2}}}"
export KIMODO_NPROC_PER_NODE="${KIMODO_NPROC_PER_NODE:-${NPROC_PER_NODE:-${PET_NPROC_PER_NODE:-8}}}"
export KIMODO_NODE_RANK="${KIMODO_NODE_RANK:-${NODE_RANK:-${PET_NODE_RANK:-${JOB_COMPLETION_INDEX:-}}}}"
export MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-}}"
export MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29500}}"
if [[ "${KIMODO_NPROC_PER_NODE}" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    KIMODO_NPROC_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  else
    KIMODO_NPROC_PER_NODE=8
  fi
  export KIMODO_NPROC_PER_NODE
fi
export KIMODO_EXPECTED_GPU_PATTERN="${KIMODO_EXPECTED_GPU_PATTERN:-H200}"
export KIMODO_REQUIRE_RDMA="${KIMODO_REQUIRE_RDMA:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ ! "${KIMODO_NNODES}" =~ ^[1-9][0-9]*$ || ! "${KIMODO_NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "KIMODO_NNODES and KIMODO_NPROC_PER_NODE must be positive integers" >&2
  exit 2
fi

world_size=$((KIMODO_NNODES * KIMODO_NPROC_PER_NODE))
# Default 16 when unset. Set KIMODO_EXPECTED_WORLD_SIZE= (empty) to disable the gate.
if [[ "${KIMODO_EXPECTED_WORLD_SIZE+x}" == "x" ]]; then
  expected_world_size="${KIMODO_EXPECTED_WORLD_SIZE}"
else
  expected_world_size="16"
fi
if [[ -n "${expected_world_size}" ]]; then
  if [[ ! "${expected_world_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "KIMODO_EXPECTED_WORLD_SIZE must be a positive integer or empty; got ${expected_world_size}" >&2
    exit 2
  fi
  if (( world_size != expected_world_size )); then
    echo "company launcher expected ${expected_world_size} total ranks; got ${KIMODO_NNODES}x${KIMODO_NPROC_PER_NODE}=${world_size}" >&2
    exit 2
  fi
fi

if [[ -n "${KIMODO_TRAINING_OVERLAY}" && ! -f "${KIMODO_TRAINING_OVERLAY}" ]]; then
  echo "training overlay is missing: ${KIMODO_TRAINING_OVERLAY}" >&2
  exit 2
fi

if [[ "${KIMODO_TRAINING_CONFIG}" != "${default_training_config}" && -z "${KIMODO_RUN_DIR:-}" ]]; then
  echo "KIMODO_RUN_DIR is required when selecting a non-default V1/V2 training config" >&2
  exit 2
fi

if [[ -n "${KIMODO_RUN_DIR:-}" ]]; then
  run_dir="${KIMODO_RUN_DIR}"
elif (( world_size == 16 )) && [[ -z "${KIMODO_TRAINING_OVERLAY}" && -z "${KIMODO_MAX_STEPS:-}" && -z "${KIMODO_BATCH_SIZE:-}" ]]; then
  run_dir="${KIMODO_RUN_ROOT}/v2-1m-production"
else
  run_dir="${KIMODO_RUN_ROOT}/v2-${KIMODO_NNODES}x${KIMODO_NPROC_PER_NODE}"
fi

runtime_args=(--set "runtime.output_dir=${run_dir}")
runtime_args+=(--set "runtime.expected_world_size=${world_size}")

if [[ -n "${KIMODO_BATCH_SIZE:-}" ]]; then
  if [[ ! "${KIMODO_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "KIMODO_BATCH_SIZE must be a positive integer" >&2
    exit 2
  fi
  grad_accum="${KIMODO_GRAD_ACCUM:-1}"
  if [[ ! "${grad_accum}" =~ ^[1-9][0-9]*$ ]]; then
    echo "KIMODO_GRAD_ACCUM must be a positive integer" >&2
    exit 2
  fi
  global_batch=$((KIMODO_BATCH_SIZE * world_size * grad_accum))
  runtime_args+=(--set "runtime.batch_size=${KIMODO_BATCH_SIZE}")
  runtime_args+=(--set "runtime.gradient_accumulation_steps=${grad_accum}")
  runtime_args+=(--set "runtime.expected_global_batch=${global_batch}")
  runtime_args+=(--set "runtime.enforce_paper_scale=false")
elif (( world_size == 16 )); then
  : # keep production yaml contract (batch 128, global 2048)
elif [[ -n "${KIMODO_TRAINING_OVERLAY}" ]]; then
  : # overlay owns deployment-scale fields
else
  echo "non-16-rank launch requires KIMODO_BATCH_SIZE or KIMODO_TRAINING_OVERLAY (config defaults assume 16x128=2048)" >&2
  exit 2
fi

if [[ -n "${KIMODO_MAX_STEPS:-}" ]]; then
  if [[ ! "${KIMODO_MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "KIMODO_MAX_STEPS must be a positive integer" >&2
    exit 2
  fi
  runtime_args+=(--set "runtime.max_steps_override=${KIMODO_MAX_STEPS}")
  runtime_args+=(--set "runtime.enforce_paper_scale=false")
fi

data_args=()
if [[ -n "${KIMODO_DATA_WORKERS:-}" ]]; then
  if [[ ! "${KIMODO_DATA_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "KIMODO_DATA_WORKERS must be a positive integer" >&2
    exit 2
  fi
  data_args+=(--set "data.num_workers=${KIMODO_DATA_WORKERS}")
fi

resume_args=()
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

echo "Kimodo company train: ${KIMODO_NNODES}x${KIMODO_NPROC_PER_NODE}=${world_size} -> ${run_dir}" >&2

exec "${script_dir}/train_distributed.sh" \
  "${runtime_args[@]}" \
  ${data_args[@]+"${data_args[@]}"} \
  ${resume_args[@]+"${resume_args[@]}"} \
  "$@"
