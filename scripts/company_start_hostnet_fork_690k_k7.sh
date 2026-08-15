#!/usr/bin/env bash
# Fork the kf-smooth V2 1M run from 690k into a new directory. Freezes sampled
# Kmax at 7 (does not rescale the 1→20 ramp). 690k is still healthy (gnorm
# ~0.52) and leaves ~6.6k steps of K≤7 before the 696.6k climb; 695k only
# leaves 1.6k. Do not paste the 695k/800 launchers.
#
# Stop every previous v2-1m GPU job first, then CREATE a new 2x8 task.
# Do not click Restart on a hung job — PET_NODE_RANK / MASTER_ADDR often
# are not re-injected, and both nodes become rank 0 (NCCL hangs 30 min).
#
# UI: enable_host_network=True, RDMA/IB x8, 2 instances x 8 GPU.
# Same as the old resume launcher: the start box is sh, so invoke bash explicitly:
#   exec bash /home/share/yzt/kimodo-reproduction/scripts/company_start_hostnet_fork_690k_k7.sh
set -euo pipefail

export KIMODO_CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
export KIMODO_STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
export PYTHONPATH="${KIMODO_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export KIMODO_NNODES="${KIMODO_NNODES:-${PET_NNODES:-${NNODES:-2}}}"
export KIMODO_NPROC_PER_NODE="${KIMODO_NPROC_PER_NODE:-${PET_NPROC_PER_NODE:-${NPROC_PER_NODE:-8}}}"
resolved_node_rank="${KIMODO_NODE_RANK:-${PET_NODE_RANK:-${NODE_RANK:-${JOB_COMPLETION_INDEX:-}}}}"
export KIMODO_NODE_RANK="${resolved_node_rank}"
export MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-}}"
export MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29500}}"

export KIMODO_RUN_DIR="${KIMODO_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-k7-from690k}"
export KIMODO_TRAINING_OVERLAY="${KIMODO_TRAINING_OVERLAY:-${KIMODO_CODE_ROOT}/configs/overlays/v2_1m_k7_from695k.yaml}"
export KIMODO_AUTO_RESUME=0
export KIMODO_NCCL_ENV_MODE=respect
export KIMODO_NCCL_PROBE=0
export KIMODO_RESUME_ALLOW_CODE_MISMATCH=1
export KIMODO_DATA_WORKERS="${KIMODO_DATA_WORKERS:-20}"
export KIMODO_FEATURE_CACHE_DIR="${KIMODO_FEATURE_CACHE_DIR:-${KIMODO_STORAGE_ROOT}/feature-cache/v1}"
export KIMODO_STARTUP_CACHE_ROOT="${KIMODO_STARTUP_CACHE_ROOT:-/dev/shm/kimodo-startup-cache}"
# Cold nodes hang for hours if 16 ranks stat 1.4M relative motion/embedding paths.
export KIMODO_SKIP_MANIFEST_PATH_STAT="${KIMODO_SKIP_MANIFEST_PATH_STAT:-1}"
# Schema-1 cache paths are SHA256(sample_id); skip parsing the 252M index.jsonl.
export KIMODO_FEATURE_CACHE_INDEX_MODE="${KIMODO_FEATURE_CACHE_INDEX_MODE:-deterministic}"
export PYTHONUNBUFFERED=1
unset NCCL_IB_DISABLE

PARENT_CHECKPOINT="${KIMODO_FORK_PARENT_CHECKPOINT:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-kf-smooth-lr1e5/checkpoints/step-000690000.pt}"
PARENT_LOCK="${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-kf-smooth-lr1e5/.kimodo-active-run.lock"

cd "${KIMODO_CODE_ROOT}"
STATUS_DIR="${KIMODO_RUN_DIR}.launch-status"
mkdir -p "${STATUS_DIR}"
STATUS_LOG="${STATUS_DIR}/node-${KIMODO_NODE_RANK:-unknown}.log"
log() {
  echo "$@"
  echo "$@" >> "${STATUS_LOG}"
}

log "Kimodo 690k K7 freeze fork: code=${KIMODO_CODE_ROOT} run=${KIMODO_RUN_DIR}"
log "hostname=$(hostname) overlay=${KIMODO_TRAINING_OVERLAY}"
log "parent=${PARENT_CHECKPOINT}"
log "topology nnodes=${KIMODO_NNODES} nproc=${KIMODO_NPROC_PER_NODE} node_rank=${KIMODO_NODE_RANK} master=${MASTER_ADDR:-UNSET}:${MASTER_PORT}"
log "env PET_NNODES=${PET_NNODES:-} PET_NPROC_PER_NODE=${PET_NPROC_PER_NODE:-} PET_NODE_RANK=${PET_NODE_RANK:-} PET_MASTER_ADDR=${PET_MASTER_ADDR:-}"
log "env NNODES=${NNODES:-} NODE_RANK=${NODE_RANK:-} JOB_COMPLETION_INDEX=${JOB_COMPLETION_INDEX:-} MASTER_ADDR=${MASTER_ADDR:-}"

if [[ -z "${MASTER_ADDR}" ]]; then
  echo "MASTER_ADDR/PET_MASTER_ADDR is empty; worker cannot join rank0. Create a new 2-node job, do not Restart." >&2
  env | sort | grep -E '^(PET_|MASTER_|NNODES|NODE_RANK|RANK|WORLD|JOB_)' >&2 || true
  exit 2
fi
if [[ -z "${KIMODO_NODE_RANK}" ]]; then
  echo "No node rank was injected; expected KIMODO_NODE_RANK, PET_NODE_RANK, NODE_RANK, or JOB_COMPLETION_INDEX." >&2
  env | sort | grep -E '^(PET_|MASTER_|NNODES|NODE_RANK|RANK|WORLD|JOB_)' >&2 || true
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

if [ ! -f "${KIMODO_TRAINING_OVERLAY}" ]; then
  echo "training overlay is missing: ${KIMODO_TRAINING_OVERLAY}" >&2
  exit 2
fi
if [ ! -f "${KIMODO_CODE_ROOT}/kimodo/training/config.py" ]; then
  echo "config.py is missing under ${KIMODO_CODE_ROOT}" >&2
  exit 2
fi
if ! grep -q "sparse_keyframes_hard_cap" "${KIMODO_CODE_ROOT}/kimodo/training/config.py"; then
  echo "config.py does not define sparse_keyframes_hard_cap; PVC code is stale" >&2
  exit 2
fi
if ! grep -q "sparse_keyframes_hard_cap:" "${KIMODO_TRAINING_OVERLAY}"; then
  echo "overlay is missing sparse_keyframes_hard_cap: ${KIMODO_TRAINING_OVERLAY}" >&2
  exit 2
fi
if grep -qE "sparse_keyframes_max:[[:space:]]*7[[:space:]]*$" "${KIMODO_TRAINING_OVERLAY}"; then
  echo "overlay must not set sparse_keyframes_max: 7 (that rescales the 1→20 ramp)" >&2
  exit 2
fi
if ! grep -q "_scheduled_for_sampling" "${KIMODO_CODE_ROOT}/kimodo/training/constraints.py"; then
  echo "constraints.py is missing hard-cap sampling; PVC code is stale" >&2
  exit 2
fi
if [ ! -f "${KIMODO_CODE_ROOT}/scripts/train_company.sh" ]; then
  echo "train_company.sh is missing: ${KIMODO_CODE_ROOT}/scripts/train_company.sh" >&2
  exit 2
fi

if [ -f "${PARENT_LOCK}" ]; then
  log "warning: parent run lock exists at ${PARENT_LOCK}; confirm the old GPU job is stopped"
fi

if [ -n "${KIMODO_FEATURE_CACHE_DIR}" ]; then
  if [ ! -f "${KIMODO_FEATURE_CACHE_DIR}/meta.json" ] || [ ! -f "${KIMODO_FEATURE_CACHE_DIR}/index.jsonl" ]; then
    echo "feature cache not ready: need ${KIMODO_FEATURE_CACHE_DIR}/{meta.json,index.jsonl}" >&2
    exit 2
  fi
fi

if [ ! -f "${PARENT_CHECKPOINT}" ]; then
  echo "690k parent checkpoint is missing: ${PARENT_CHECKPOINT}" >&2
  exit 2
fi

if [ -f "${KIMODO_RUN_DIR}/checkpoints/latest.txt" ]; then
  log "child run already has checkpoints; in-place AUTO_RESUME in ${KIMODO_RUN_DIR}"
  export KIMODO_AUTO_RESUME=1
  log "downstream exec"
  exec /bin/bash "${KIMODO_CODE_ROOT}/scripts/train_company.sh"
fi

if [ -d "${KIMODO_RUN_DIR}" ] && [ -n "$(find "${KIMODO_RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  echo "child run exists but has no checkpoint and is not empty: ${KIMODO_RUN_DIR}" >&2
  echo "choose a new KIMODO_RUN_DIR for a clean fork; refusing to delete or overwrite partial artifacts" >&2
  exit 2
fi

MANIFEST_PATH="${KIMODO_STORAGE_ROOT}/benchmark-v2-soma30-v2.2/train.cached.jsonl"
STAGE_HELPER="${KIMODO_CODE_ROOT}/scripts/stage_startup_file.py"
if [ ! -f "${MANIFEST_PATH}" ] || [ ! -f "${STAGE_HELPER}" ]; then
  echo "startup staging input is missing: manifest=${MANIFEST_PATH} helper=${STAGE_HELPER}" >&2
  exit 2
fi
log "staging manifest to node-local cache ${KIMODO_STARTUP_CACHE_ROOT}"
local_cache_fallback="/tmp/kimodo-startup-cache"
python_bin="${KIMODO_PYTHON:-}"
if [ -z "${python_bin}" ]; then
  python_bin="$(command -v python3 || command -v python || true)"
fi
if [ -z "${python_bin}" ]; then
  echo "no python3/python available to stage the startup manifest" >&2
  exit 2
fi
local_manifest_read="$("${python_bin}" "${STAGE_HELPER}" "${MANIFEST_PATH}" "${KIMODO_STARTUP_CACHE_ROOT}" --fallback-root "${local_cache_fallback}")"
export KIMODO_LOCAL_MANIFEST_READ_PATH="${local_manifest_read}"
log "staged manifest ${local_manifest_read}"

mkdir -p "${KIMODO_RUN_DIR}"
if [ -n "$(find "${KIMODO_RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  echo "fresh fork directory became non-empty during launch: ${KIMODO_RUN_DIR}" >&2
  exit 2
fi
log "fork directory ready and empty: ${KIMODO_RUN_DIR}"

log "exec train_company resume=${PARENT_CHECKPOINT} mode=fork"
log "downstream exec"
exec /bin/bash "${KIMODO_CODE_ROOT}/scripts/train_company.sh" \
  --set "runtime.resume=${PARENT_CHECKPOINT}" \
  --set "runtime.resume_mode=fork"
