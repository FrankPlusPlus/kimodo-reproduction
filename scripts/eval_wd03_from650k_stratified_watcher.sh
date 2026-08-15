#!/usr/bin/env bash
# 1xH200 stratified-10pct watcher for the 650k wd=0.3 stable fork.
# 50k cadence from 700k (first quality gate after the 20k jsonl look).
# Sidecar exports EMA from trainer 5k checkpoints into a jovyan-writable dir.
#
# UI: 1 instance x 1 GPU. Do not attach this to the 16-GPU training job.
# Start:
#   exec bash /home/share/yzt/kimodo-reproduction/scripts/eval_wd03_from650k_stratified_watcher.sh
set -euo pipefail

export KIMODO_CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
export KIMODO_STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
export PYTHONPATH="${KIMODO_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export KIMODO_EVAL_ASSET_ROOT="${KIMODO_EVAL_ASSET_ROOT:-${KIMODO_STORAGE_ROOT}/yezitao-kimodo-eval-v2}"
export KIMODO_MODEL_ROOT="${KIMODO_MODEL_ROOT:-${KIMODO_STORAGE_ROOT}/models}"
export HF_HOME="${HF_HOME:-${KIMODO_STORAGE_ROOT}/hf-cache}"

export KIMODO_TRAIN_RUN_DIR="${KIMODO_TRAIN_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-wd03-from650k}"
export KIMODO_EVAL_EXPORT_RUN_DIR="${KIMODO_EVAL_EXPORT_RUN_DIR:-${KIMODO_STORAGE_ROOT}/eval-exports/v2-1m-hostnet-wd03-from650k}"
export KIMODO_RUN_DIR="${KIMODO_RUN_DIR:-${KIMODO_EVAL_EXPORT_RUN_DIR}}"
export KIMODO_EVAL_ROOT="${KIMODO_EVAL_ROOT:-${KIMODO_STORAGE_ROOT}/eval-results/v2-1m-hostnet-wd03-from650k-stratified10pct}"
export KIMODO_BENCHMARK_ROOT="${KIMODO_BENCHMARK_ROOT:-${KIMODO_EVAL_ASSET_ROOT}/benchmark/stratified-10pct}"
export KIMODO_OFFICIAL_BASELINE_SUMMARY="${KIMODO_OFFICIAL_BASELINE_SUMMARY:-${KIMODO_EVAL_ASSET_ROOT}/baselines/official-seed-v1.1/summary_rows.json}"
export KIMODO_EXPORT_MIN_STEP="${KIMODO_EXPORT_MIN_STEP:-700000}"
export KIMODO_EXPORT_STEP_EVERY="${KIMODO_EXPORT_STEP_EVERY:-50000}"
export KIMODO_EVAL_MINIMUM_STEP="${KIMODO_EVAL_MINIMUM_STEP:-700000}"

export TEXT_ENCODER_MODE="${TEXT_ENCODER_MODE:-local}"
export KIMODO_LLM2VEC_FOUNDATION="${KIMODO_LLM2VEC_FOUNDATION:-${KIMODO_MODEL_ROOT}/llm2vec/foundation}"
export KIMODO_LLM2VEC_MNTP="${KIMODO_LLM2VEC_MNTP:-${KIMODO_MODEL_ROOT}/llm2vec/mntp-adapter}"
export KIMODO_LLM2VEC_SUPERVISED="${KIMODO_LLM2VEC_SUPERVISED:-${KIMODO_MODEL_ROOT}/llm2vec/supervised-adapter}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${KIMODO_MODEL_ROOT}/checkpoints}"
export LOCAL_CACHE="${LOCAL_CACHE:-True}"

export KIMODO_EVAL_DIFFUSION_STEPS="${KIMODO_EVAL_DIFFUSION_STEPS:-100}"
export KIMODO_EVAL_BATCH_SIZE="${KIMODO_EVAL_BATCH_SIZE:-1}"
export KIMODO_EVAL_WORKERS="${KIMODO_EVAL_WORKERS:-4}"
export KIMODO_EVAL_PAPER_PROTOCOL="${KIMODO_EVAL_PAPER_PROTOCOL:-1}"
export KIMODO_EVAL_TEXT_ENCODER_FP32="${KIMODO_EVAL_TEXT_ENCODER_FP32:-1}"
export KIMODO_EVAL_POLL_SECONDS="${KIMODO_EVAL_POLL_SECONDS:-60}"
export KIMODO_EXPORT_PYTHON="${KIMODO_EXPORT_PYTHON:-python}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

mkdir -p "${KIMODO_EVAL_EXPORT_RUN_DIR}/exports" "${KIMODO_EVAL_ROOT}"

if ! python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  echo "CUDA torch unavailable in this pod image." >&2
  echo "Recreate/attach this 1xH200 with hub.inner.ai.kingsoft.com/hh-678395/kimodo-reproduction:v8" >&2
  exit 2
fi
if [[ ! -d "${KIMODO_BENCHMARK_ROOT}/content" ]]; then
  echo "missing stratified benchmark: ${KIMODO_BENCHMARK_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${KIMODO_OFFICIAL_BASELINE_SUMMARY}" ]]; then
  echo "missing official baseline summary: ${KIMODO_OFFICIAL_BASELINE_SUMMARY}" >&2
  exit 2
fi

extra=(--minimum-step "${KIMODO_EVAL_MINIMUM_STEP}")

if [[ "${KIMODO_EVAL_EXPORT_SIDECAR:-1}" == 1 ]]; then
  nohup bash "${KIMODO_CODE_ROOT}/scripts/eval_export_hostnet_checkpoints.sh" \
    >>"${KIMODO_EVAL_EXPORT_RUN_DIR}/export-sidecar.log" 2>&1 &
  echo "wd03-from650k eval: export sidecar pid=$!"
fi

cd "${KIMODO_CODE_ROOT}"
echo "wd03-from650k eval: train=${KIMODO_TRAIN_RUN_DIR}"
echo "wd03-from650k eval: exports=${KIMODO_RUN_DIR}/exports"
echo "wd03-from650k eval: output=${KIMODO_EVAL_ROOT}"
echo "wd03-from650k eval: benchmark=${KIMODO_BENCHMARK_ROOT}"
echo "wd03-from650k eval: minimum_step=${KIMODO_EVAL_MINIMUM_STEP} every=${KIMODO_EXPORT_STEP_EVERY}"
exec bash "${KIMODO_CODE_ROOT}/scripts/eval_company_watcher.sh" "${extra[@]}" "$@"
