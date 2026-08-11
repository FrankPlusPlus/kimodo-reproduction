#!/usr/bin/env bash
# Dynamic proxy-128 benchmark monitor for v2-1m-hostnet on a free 1xH200 pod.
# Consumes exports/ under KIMODO_EVAL_EXPORT_RUN_DIR (offline-exported or milestone).
set -euo pipefail

storage_root="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
code_root="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
asset_root="${KIMODO_EVAL_ASSET_ROOT:-${storage_root}/yezitao-kimodo-eval-v1}"
model_root="${KIMODO_MODEL_ROOT:-${storage_root}/models}"

export KIMODO_STORAGE_ROOT="${storage_root}"
export KIMODO_CODE_ROOT="${code_root}"
export KIMODO_EVAL_ASSET_ROOT="${asset_root}"
export KIMODO_MODEL_ROOT="${model_root}"
export HF_HOME="${HF_HOME:-${storage_root}/hf-cache}"
export PYTHONPATH="${code_root}${PYTHONPATH:+:${PYTHONPATH}}"

# Training run (read-only). Offline exports go to a writable shadow run dir.
export KIMODO_TRAIN_RUN_DIR="${KIMODO_TRAIN_RUN_DIR:-${storage_root}/runs/v2-1m-hostnet}"
export KIMODO_EVAL_EXPORT_RUN_DIR="${KIMODO_EVAL_EXPORT_RUN_DIR:-${storage_root}/eval-exports/v2-1m-hostnet}"
export KIMODO_RUN_DIR="${KIMODO_RUN_DIR:-${KIMODO_EVAL_EXPORT_RUN_DIR}}"
export KIMODO_EVAL_ROOT="${KIMODO_EVAL_ROOT:-${storage_root}/eval-results/v2-1m-hostnet-proxy128}"
export KIMODO_BENCHMARK_ROOT="${KIMODO_BENCHMARK_ROOT:-${asset_root}/benchmark/proxy-128}"
export KIMODO_OFFICIAL_BASELINE_SUMMARY="${KIMODO_OFFICIAL_BASELINE_SUMMARY:-${asset_root}/baselines/official-seed-v1.1/summary_rows.json}"

export TEXT_ENCODER_MODE="${TEXT_ENCODER_MODE:-local}"
export KIMODO_LLM2VEC_FOUNDATION="${KIMODO_LLM2VEC_FOUNDATION:-${model_root}/llm2vec/foundation}"
export KIMODO_LLM2VEC_MNTP="${KIMODO_LLM2VEC_MNTP:-${model_root}/llm2vec/mntp-adapter}"
export KIMODO_LLM2VEC_SUPERVISED="${KIMODO_LLM2VEC_SUPERVISED:-${model_root}/llm2vec/supervised-adapter}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${model_root}/checkpoints}"
export LOCAL_CACHE="${LOCAL_CACHE:-True}"

export KIMODO_EVAL_DIFFUSION_STEPS="${KIMODO_EVAL_DIFFUSION_STEPS:-100}"
export KIMODO_EVAL_BATCH_SIZE="${KIMODO_EVAL_BATCH_SIZE:-1}"
export KIMODO_EVAL_WORKERS="${KIMODO_EVAL_WORKERS:-4}"
export KIMODO_EVAL_PAPER_PROTOCOL="${KIMODO_EVAL_PAPER_PROTOCOL:-1}"
export KIMODO_EVAL_TEXT_ENCODER_FP32="${KIMODO_EVAL_TEXT_ENCODER_FP32:-0}"
export KIMODO_EVAL_POLL_SECONDS="${KIMODO_EVAL_POLL_SECONDS:-60}"

mkdir -p "${KIMODO_EVAL_EXPORT_RUN_DIR}" "${KIMODO_EVAL_ROOT}"

if ! python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  echo "CUDA torch unavailable in this pod image." >&2
  echo "Recreate/attach this 1xH200 with hub.inner.ai.kingsoft.com/hh-678395/kimodo-reproduction:v8" >&2
  exit 2
fi

if [[ ! -d "${KIMODO_BENCHMARK_ROOT}/content" ]]; then
  echo "missing proxy-128 benchmark at ${KIMODO_BENCHMARK_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${KIMODO_OFFICIAL_BASELINE_SUMMARY}" ]]; then
  echo "missing official baseline summary: ${KIMODO_OFFICIAL_BASELINE_SUMMARY}" >&2
  exit 2
fi

cd "${code_root}"
echo "proxy128 watcher: train=${KIMODO_TRAIN_RUN_DIR}"
echo "proxy128 watcher: exports=${KIMODO_RUN_DIR}/exports"
echo "proxy128 watcher: output=${KIMODO_EVAL_ROOT}"
echo "proxy128 watcher: benchmark=${KIMODO_BENCHMARK_ROOT}"
exec bash "${code_root}/scripts/eval_company_watcher.sh" "$@"
