#!/usr/bin/env bash
# Offline motion-feature cache builder for company share storage.
# Safe to run while the 1M training job continues (CPU-only, separate pod).
#
# Default layout (data PVC):
#   /home/share/yezitao-kimodo-reproduction/feature-cache/v1
set -euo pipefail

CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
BUNDLE_ROOT="${KIMODO_BUNDLE_ROOT:-${STORAGE_ROOT}/benchmark-v2-soma30-v2.2}"

# Company V2 bundle keeps the cached train manifest at bundle root.
MANIFEST="${KIMODO_FEATURE_CACHE_MANIFEST:-${BUNDLE_ROOT}/train.cached.jsonl}"
STATS_PATH="${KIMODO_FEATURE_CACHE_STATS:-${BUNDLE_ROOT}/stats/repro-soma30-30fps}"
OUTPUT="${KIMODO_FEATURE_CACHE_DIR:-${STORAGE_ROOT}/feature-cache/v1}"
NUM_WORKERS="${KIMODO_FEATURE_CACHE_WORKERS:-$(( ${KIMODO_BUILD_CPUS:-$(nproc)} ))}"
VERIFY_SAMPLE="${KIMODO_FEATURE_CACHE_VERIFY:-32}"
SPLIT="${KIMODO_FEATURE_CACHE_SPLIT:-train}"
FPS="${KIMODO_FEATURE_CACHE_FPS:-30}"
SKELETON_JOINTS="${KIMODO_FEATURE_CACHE_JOINTS:-30}"
MIN_FRAMES="${KIMODO_FEATURE_CACHE_MIN_FRAMES:-2}"

export PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${CODE_ROOT}"

# Prefer the PVC CPU venv if present (dev / CPU-only build pods).
if [[ -x "${CODE_ROOT}/.venv-feature-cache/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "${CODE_ROOT}/.venv-feature-cache/bin/activate"
  PYTHON_BIN="${CODE_ROOT}/.venv-feature-cache/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

echo "Kimodo feature cache build"
echo "  python=${PYTHON_BIN}"
echo "  manifest=${MANIFEST}"
echo "  stats=${STATS_PATH}"
echo "  output=${OUTPUT}"
echo "  workers=${NUM_WORKERS} verify_sample=${VERIFY_SAMPLE}"

EXTRA=()
if [[ "${KIMODO_FEATURE_CACHE_OVERWRITE:-0}" == "1" ]]; then
  EXTRA+=(--overwrite)
fi

exec "${PYTHON_BIN}" -m kimodo.training.feature_cache_cli \
  --manifest "${MANIFEST}" \
  --output "${OUTPUT}" \
  --stats-path "${STATS_PATH}" \
  --split "${SPLIT}" \
  --fps "${FPS}" \
  --skeleton-joints "${SKELETON_JOINTS}" \
  --min-frames "${MIN_FRAMES}" \
  --num-workers "${NUM_WORKERS}" \
  --verify-sample "${VERIFY_SAMPLE}" \
  "${EXTRA[@]}"
