#!/bin/sh
# Paste variants for the platform "训练启动脚本" box (/bin/sh — no pipefail).
# Image stays :v8 / :pvc-train; only NCCL env mode changes.
#
# Trial A (recommended first): override platform GID with sysfs RoCEv2 guess
# Trial B: pin GID index N (try 0..7 if A still errno 19)
# Trial C: respect platform (old behavior)
#
# Do NOT set NCCL_IB_DISABLE=1 for these trials.

export KIMODO_CODE_ROOT=/home/share/yzt/kimodo-reproduction
export KIMODO_STORAGE_ROOT=/home/share/yezitao-kimodo-reproduction
export PYTHONPATH="${KIMODO_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export KIMODO_NNODES="${PET_NNODES:-${NNODES:-2}}"
export KIMODO_NPROC_PER_NODE="${PET_NPROC_PER_NODE:-${NPROC_PER_NODE:-8}}"
export KIMODO_NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-${JOB_COMPLETION_INDEX:-0}}}"
export MASTER_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR:-}}"
export MASTER_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-29500}}"

# --- pick ONE mode (NCCL 2.28: prefer force-dyn = unset GID) ---
export KIMODO_NCCL_ENV_MODE=force-dyn
# export KIMODO_NCCL_ENV_MODE=force-single
# export KIMODO_NCCL_ENV_MODE=force-gid=5
# export KIMODO_NCCL_ENV_MODE=respect

export KIMODO_NCCL_DEBUG=1
export KIMODO_NCCL_PROBE=0
export NCCL_IB_ROCE_VERSION_NUM=2
export NCCL_CROSS_NIC=0
unset NCCL_IB_DISABLE
unset NCCL_IB_GID_INDEX
unset NCCL_ASYNC_ERROR_HANDLING

cd "${KIMODO_CODE_ROOT}" || exit 2
# Optional one-shot dump before train (also runs inside nccl_rdma_env):
# bash "${KIMODO_CODE_ROOT}/scripts/probe_rdma_gids.sh" || true
exec bash "${KIMODO_CODE_ROOT}/scripts/train_company.sh"
