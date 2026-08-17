#!/usr/bin/env bash
# Read-only 2x2 probe on the stopped lr3e6 rescue:
#   weights 795k (healthy) vs 800k (takeoff onset)
#   constraint clocks 795000 vs 800000
# No optimizer step, no weight write.
#
# Prefer 1 instance x 1 GPU if 2x8 is queued. Same experiment, 4 samples.
# 2x8 is optional (16 ranks x 4 samples). Do not Restart a queued job.
# Do not use kimodo-dev. Do not attach to the 795 eval pod.
# Start box is sh:
#   exec bash /home/share/yzt/kimodo-reproduction/scripts/company_run_body_jacobian_probe_wd03_795_800.sh
set -euo pipefail

export KIMODO_CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
export KIMODO_STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
export PYTHONPATH="${KIMODO_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export KIMODO_SKIP_MANIFEST_PATH_STAT="${KIMODO_SKIP_MANIFEST_PATH_STAT:-1}"
export KIMODO_FEATURE_CACHE_INDEX_MODE="${KIMODO_FEATURE_CACHE_INDEX_MODE:-deterministic}"
export KIMODO_FEATURE_CACHE_DIR="${KIMODO_FEATURE_CACHE_DIR:-${KIMODO_STORAGE_ROOT}/feature-cache/v1}"
export KIMODO_NNODES="${KIMODO_NNODES:-${PET_NNODES:-${NNODES:-2}}}"
export KIMODO_NPROC_PER_NODE="${KIMODO_NPROC_PER_NODE:-${PET_NPROC_PER_NODE:-${NPROC_PER_NODE:-8}}}"
resolved_node_rank="${KIMODO_NODE_RANK:-${PET_NODE_RANK:-${NODE_RANK:-${JOB_COMPLETION_INDEX:-}}}}"
export KIMODO_NODE_RANK="${resolved_node_rank}"
export MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-}}"
export MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29500}}"
export KIMODO_NCCL_ENV_MODE="${KIMODO_NCCL_ENV_MODE:-respect}"
export KIMODO_NCCL_PROBE="${KIMODO_NCCL_PROBE:-0}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
unset NCCL_IB_DISABLE

if [[ -f "${KIMODO_CODE_ROOT}/scripts/load_kimodo_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${KIMODO_CODE_ROOT}/scripts/load_kimodo_env.sh"
  kimodo_load_env_files
fi
if [[ -f "${KIMODO_CODE_ROOT}/scripts/nccl_rdma_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${KIMODO_CODE_ROOT}/scripts/nccl_rdma_env.sh"
  kimodo_nccl_rdma_env
fi

run_dir="${KIMODO_RESCUE_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-wd03-from780k-lr3e6}"
config="${KIMODO_PROBE_CONFIG:-${run_dir}/config.resolved.yaml}"
out_dir="${KIMODO_PROBE_OUT_DIR:-${KIMODO_STORAGE_ROOT}/diagnostics/body-jacobian-wd03-795-800}"
preserve_dir="${KIMODO_PRESERVE_DIR:-${KIMODO_STORAGE_ROOT}/preserved-pre-collapse/v2-1m-hostnet-wd03-from780k-lr3e6}"
ckpt_795="${KIMODO_CKPT_795:-${run_dir}/checkpoints/step-000795000.pt}"
ckpt_800="${KIMODO_CKPT_800:-${run_dir}/checkpoints/step-000800000.pt}"
if [[ ! -r "${ckpt_795}" && -r "${preserve_dir}/step-000795000.pt" ]]; then
  ckpt_795="${preserve_dir}/step-000795000.pt"
fi
if [[ ! -r "${ckpt_800}" && -r "${preserve_dir}/step-000800000.pt" ]]; then
  ckpt_800="${preserve_dir}/step-000800000.pt"
fi

cd "${KIMODO_CODE_ROOT}"
python_bin="${KIMODO_PYTHON:-python3}"
if ! grep -q "summarize_takeoff_grid" "${KIMODO_CODE_ROOT}/kimodo/training/body_jacobian_probe.py"; then
  echo "body_jacobian_probe.py is missing summarize_takeoff_grid; PVC code is stale" >&2
  exit 2
fi
if ! grep -q "constraint-steps" "${KIMODO_CODE_ROOT}/scripts/diagnose_body_jacobian.py"; then
  echo "diagnose_body_jacobian.py is missing --constraint-steps; PVC code is stale" >&2
  exit 2
fi

visible_gpus=0
if command -v nvidia-smi >/dev/null 2>&1; then
  visible_gpus="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
fi
if [[ "${KIMODO_NPROC_PER_NODE}" == "auto" ]]; then
  if [[ "${visible_gpus}" =~ ^[1-9][0-9]*$ ]]; then
    KIMODO_NPROC_PER_NODE="${visible_gpus}"
  else
    KIMODO_NPROC_PER_NODE=8
  fi
  export KIMODO_NPROC_PER_NODE
fi
if [[ "${visible_gpus}" == "1" ]]; then
  export KIMODO_NNODES=1
  export KIMODO_NPROC_PER_NODE=1
  export KIMODO_PROBE_SKIP_TORCHRUN="${KIMODO_PROBE_SKIP_TORCHRUN:-1}"
fi
if [[ "${KIMODO_NNODES}" == "1" && "${KIMODO_NPROC_PER_NODE}" == "1" ]]; then
  export KIMODO_PROBE_SKIP_TORCHRUN="${KIMODO_PROBE_SKIP_TORCHRUN:-1}"
fi

if [[ -z "${KIMODO_PROBE_SKIP_TORCHRUN:-}" ]]; then
  if [[ -z "${MASTER_ADDR}" ]]; then
    echo "MASTER_ADDR/PET_MASTER_ADDR is empty; worker cannot join rank0. Create a new 2-node job, do not Restart." >&2
    env | sort | grep -E '^(PET_|MASTER_|NNODES|NODE_RANK|RANK|WORLD|JOB_)' >&2 || true
    exit 2
  fi
  if [[ -z "${KIMODO_NODE_RANK}" ]]; then
    echo "No node rank was injected; expected KIMODO_NODE_RANK, PET_NODE_RANK, NODE_RANK, or JOB_COMPLETION_INDEX." >&2
    exit 2
  fi
  case "${KIMODO_NODE_RANK}" in
    ''|*[!0-9]*)
      echo "node rank must be a non-negative integer; got ${KIMODO_NODE_RANK}" >&2
      exit 2
      ;;
  esac
  case "${KIMODO_NNODES}" in
    ''|*[!0-9]*)
      echo "node count must be a positive integer; got ${KIMODO_NNODES}" >&2
      exit 2
      ;;
  esac
  if [ "${KIMODO_NNODES}" -lt 1 ] || [ "${KIMODO_NODE_RANK}" -ge "${KIMODO_NNODES}" ]; then
    echo "invalid topology: node_rank=${KIMODO_NODE_RANK}, nnodes=${KIMODO_NNODES}" >&2
    exit 2
  fi
fi

mkdir -p "${out_dir}"
for path in "${config}" "${ckpt_795}" "${ckpt_800}"; do
  if [[ ! -r "${path}" ]]; then
    echo "missing readable path: ${path}" >&2
    exit 2
  fi
done

if [[ "${KIMODO_NODE_RANK:-0}" == "0" ]]; then
  mkdir -p "${preserve_dir}"
  for ckpt in "${ckpt_795}" "${ckpt_800}"; do
    dest="${preserve_dir}/$(basename "${ckpt}")"
    if [[ ! -f "${dest}" ]]; then
      cp -n "${ckpt}" "${dest}" || true
    fi
  done
fi

echo "wd03 795/800 2x2 probe: python=${python_bin}"
echo "config=${config}"
echo "ckpts=${ckpt_795} ${ckpt_800}"
echo "out=${out_dir}"
echo "topology nnodes=${KIMODO_NNODES} nproc=${KIMODO_NPROC_PER_NODE} node_rank=${KIMODO_NODE_RANK:-skip} master=${MASTER_ADDR:-UNSET}:${MASTER_PORT}"

diagnose_args=(
  "${KIMODO_CODE_ROOT}/scripts/diagnose_body_jacobian.py"
  --config "${config}"
  --checkpoints "${ckpt_795}" "${ckpt_800}"
  --samples "${KIMODO_PROBE_SAMPLES:-4}"
  --pair-samples "${KIMODO_PROBE_PAIR_SAMPLES:-4}"
  --seed "${KIMODO_PROBE_SEED:-20260816}"
  --constraint-steps 795000 800000
  --device "${KIMODO_PROBE_DEVICE:-auto}"
  --output-dir "${out_dir}"
)

if [[ -n "${KIMODO_PROBE_SKIP_TORCHRUN:-}" ]]; then
  exec "${python_bin}" "${diagnose_args[@]}"
fi

exec "${python_bin}" -m torch.distributed.run \
  --nnodes="${KIMODO_NNODES}" \
  --nproc-per-node="${KIMODO_NPROC_PER_NODE}" \
  --node-rank="${KIMODO_NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  "${diagnose_args[@]}"
