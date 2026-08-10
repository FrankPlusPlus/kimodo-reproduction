#!/usr/bin/env bash
# Short throughput sweep for KIMODO_DATA_WORKERS on a free 2x8 hostnet slot.
# Usage: KIMODO_DATA_WORKERS=4 bash scripts/company_sweep_data_workers.sh
# Compare steady system/optimizer_steps_per_second across runs/v2-workers-w{4,8,12}.
set -euo pipefail

workers="${KIMODO_DATA_WORKERS:?set KIMODO_DATA_WORKERS to 4, 8, or 12}"
export KIMODO_CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
export KIMODO_STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
export PYTHONPATH="${KIMODO_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export KIMODO_NNODES="${PET_NNODES:-${NNODES:-2}}"
export KIMODO_NPROC_PER_NODE="${PET_NPROC_PER_NODE:-${NPROC_PER_NODE:-8}}"
export KIMODO_NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-${JOB_COMPLETION_INDEX:-0}}}"
export MASTER_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR:-}}"
export MASTER_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-29500}}"
export KIMODO_AUTO_RESUME=0
export KIMODO_MAX_STEPS="${KIMODO_MAX_STEPS:-200}"
export KIMODO_DATA_WORKERS="${workers}"
export KIMODO_NCCL_ENV_MODE=respect
unset NCCL_IB_DISABLE
export KIMODO_RUN_DIR="${KIMODO_STORAGE_ROOT}/runs/v2-workers-w${workers}"

cd "${KIMODO_CODE_ROOT}"
echo "worker sweep: DATA_WORKERS=${workers} MAX_STEPS=${KIMODO_MAX_STEPS} RUN_DIR=${KIMODO_RUN_DIR}"
exec bash "${KIMODO_CODE_ROOT}/scripts/train_company.sh"
