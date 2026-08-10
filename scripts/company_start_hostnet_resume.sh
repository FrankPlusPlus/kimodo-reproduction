#!/usr/bin/env bash
# Paste into Hanhai start command after step-000010000.pt exists.
# UI: enable_host_network=True, RDMA/IB x8, 2 instances x 8 GPU.
# Uses PVC code with persistent_workers + async metrics + prefetch=4.
set -euo pipefail

export KIMODO_CODE_ROOT=/home/share/yzt/kimodo-reproduction
export KIMODO_STORAGE_ROOT=/home/share/yezitao-kimodo-reproduction
export PYTHONPATH="${KIMODO_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export KIMODO_NNODES="${PET_NNODES:-${NNODES:-2}}"
export KIMODO_NPROC_PER_NODE="${PET_NPROC_PER_NODE:-${NPROC_PER_NODE:-8}}"
export KIMODO_NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-${JOB_COMPLETION_INDEX:-0}}}"
export MASTER_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR:-}}"
export MASTER_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-29500}}"

export KIMODO_RUN_DIR=/home/share/yezitao-kimodo-reproduction/runs/v2-1m-hostnet
export KIMODO_AUTO_RESUME=1
export KIMODO_NCCL_ENV_MODE=respect
export KIMODO_NCCL_PROBE=0
# Checkpoint was written before PVC throughput hotfixes; allow code_snapshot drift.
export KIMODO_RESUME_ALLOW_CODE_MISMATCH=1
# 8 workers regressed vs prior ~1.3 steps/s at 12; try 16 on 2x8 hostnet.
export KIMODO_DATA_WORKERS="${KIMODO_DATA_WORKERS:-16}"
# After offline build + ≥20k ckpt, enable mmap features:
#   export KIMODO_FEATURE_CACHE_DIR=/home/share/yezitao-kimodo-reproduction/feature-cache/v1
unset NCCL_IB_DISABLE

cd "${KIMODO_CODE_ROOT}"
echo "Kimodo resume hostnet: code=${KIMODO_CODE_ROOT} run=${KIMODO_RUN_DIR}"
echo "persistent_workers/prefetch from training yaml; AUTO_RESUME from checkpoints/latest.txt"
echo "KIMODO_RESUME_ALLOW_CODE_MISMATCH=${KIMODO_RESUME_ALLOW_CODE_MISMATCH} KIMODO_DATA_WORKERS=${KIMODO_DATA_WORKERS}"

# Only the master launcher clears a leftover cross-host lock (workers must not race).
LOCK_PATH="${KIMODO_RUN_DIR}/.kimodo-active-run.lock"
LOCK_TOKEN="${KIMODO_CLEAR_RUN_LOCK_TOKEN:-}"
if [[ "${KIMODO_NODE_RANK}" == "0" && -f "${LOCK_PATH}" ]]; then
  echo "Kimodo resume: clearing prior run lock at ${LOCK_PATH}"
  python3 - <<PY
from pathlib import Path
import json
import os
import sys

path = Path(${LOCK_PATH@Q})
expected = ${LOCK_TOKEN@Q}
raw = path.read_text(encoding="utf-8")
record = json.loads(raw)
token = record.get("token") if isinstance(record, dict) else None
if expected and token != expected and expected != "FORCE":
    print(f"refusing to clear lock: token mismatch have={token!r} want={expected!r}", file=sys.stderr)
    sys.exit(2)
os.unlink(path)
print(f"cleared lock token={token} owner={record.get('hostname') if isinstance(record, dict) else None}")
PY
fi

exec bash "${KIMODO_CODE_ROOT}/scripts/train_company.sh"
