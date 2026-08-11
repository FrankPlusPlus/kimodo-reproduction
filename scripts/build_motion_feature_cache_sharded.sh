#!/usr/bin/env bash
# High-throughput sharded feature-cache build for the company PVC.
# Avoids one giant ProcessPool over the full 1.4M manifest.
set -euo pipefail

CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
BUNDLE_ROOT="${KIMODO_BUNDLE_ROOT:-${STORAGE_ROOT}/benchmark-v2-soma30-v2.2}"
MANIFEST="${KIMODO_FEATURE_CACHE_MANIFEST:-${BUNDLE_ROOT}/train.cached.jsonl}"
STATS_PATH="${KIMODO_FEATURE_CACHE_STATS:-${BUNDLE_ROOT}/stats/repro-soma30-30fps}"
OUTPUT="${KIMODO_FEATURE_CACHE_DIR:-${STORAGE_ROOT}/feature-cache/v1}"
NUM_SHARDS="${KIMODO_FEATURE_CACHE_SHARDS:-48}"
LOG_DIR="${STORAGE_ROOT}/feature-cache"
LOG="${LOG_DIR}/build-v1-sharded.nohup.out"
PID_DIR="${LOG_DIR}/shard-pids"
# Leave unset to write straight to JuiceFS OUTPUT (cross-device /tmp staging
# doubles I/O). Set KIMODO_FEATURE_CACHE_WORK_ROOT only for same-filesystem staging.
WORK_ROOT="${KIMODO_FEATURE_CACHE_WORK_ROOT:-}"

export PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

PYTHON_BIN="${CODE_ROOT}/.venv-feature-cache/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

SCRIPT="${CODE_ROOT}/scripts/build_motion_feature_cache_sharded.py"
if [[ ! -f "${SCRIPT}" ]]; then
  echo "missing ${SCRIPT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT}/features" "${OUTPUT}/shards" "${LOG_DIR}" "${PID_DIR}"
if [[ -n "${WORK_ROOT}" ]]; then
  mkdir -p "${WORK_ROOT}"
fi

# Pre-split once so each worker reads ~1/N of the JSONL (not the full 1.4G).
SHARD_DIR="${KIMODO_FEATURE_CACHE_SHARD_DIR:-/tmp/kimodo-fc-shards-${NUM_SHARDS}}"
PATH_BASE="$(cd "$(dirname "${MANIFEST}")" && pwd)"
rm -rf "${SHARD_DIR}"
mkdir -p "${SHARD_DIR}"
echo "splitting manifest into ${NUM_SHARDS} shards under ${SHARD_DIR}"
split -n "l/${NUM_SHARDS}" -d -a 4 --additional-suffix=".jsonl" \
  "${MANIFEST}" "${SHARD_DIR}/shard_"
mapfile -t SHARD_FILES < <(ls -1 "${SHARD_DIR}"/shard_*.jsonl | sort)
if [[ "${#SHARD_FILES[@]}" -ne "${NUM_SHARDS}" ]]; then
  echo "expected ${NUM_SHARDS} shard files, got ${#SHARD_FILES[@]}" >&2
  exit 1
fi

exec >>"${LOG}" 2>&1

echo "Kimodo sharded feature cache build $(date -Is)"
echo "  python=${PYTHON_BIN}"
echo "  manifest=${MANIFEST}"
echo "  path_base=${PATH_BASE}"
echo "  stats=${STATS_PATH}"
echo "  output=${OUTPUT}"
echo "  shards=${NUM_SHARDS}"
echo "  shard_dir=${SHARD_DIR}"
echo "  work_root=${WORK_ROOT:-<direct output>}"

rm -f "${PID_DIR}"/shard-*.pid
pids=()
for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_log="${LOG_DIR}/shard-${shard_id}.log"
  shard_file="${SHARD_FILES[$shard_id]}"
  extra=()
  if [[ -n "${WORK_ROOT}" ]]; then
    extra+=(--work-root "${WORK_ROOT}")
  fi
  "${PYTHON_BIN}" "${SCRIPT}" \
    --manifest "${shard_file}" \
    --path-base "${PATH_BASE}" \
    --output "${OUTPUT}" \
    --stats-path "${STATS_PATH}" \
    --shard-id "${shard_id}" \
    --num-shards "${NUM_SHARDS}" \
    "${extra[@]}" \
    >"${shard_log}" 2>&1 &
  pid=$!
  pids+=("${pid}")
  echo "${pid}" >"${PID_DIR}/shard-${shard_id}.pid"
  echo "started shard ${shard_id} pid=${pid} file=${shard_file}"
done

echo "all shards launched $(date -Is); waiting..."
fail=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  if wait "${pid}"; then
    echo "shard ${idx} OK pid=${pid}"
  else
    echo "shard ${idx} FAILED pid=${pid}"
    fail=1
  fi
done

if [[ "${fail}" -ne 0 ]]; then
  echo "One or more shards failed; skip merge $(date -Is)"
  exit 1
fi

echo "merging $(date -Is)"
"${PYTHON_BIN}" "${SCRIPT}" \
  --manifest "${MANIFEST}" \
  --output "${OUTPUT}" \
  --stats-path "${STATS_PATH}" \
  --shard-id -1 \
  --num-shards "${NUM_SHARDS}"
echo "ALL_DONE $(date -Is)"
