#!/usr/bin/env bash
# Export and evaluate the healthy 695k EMA checkpoint once on the exact
# stratified benchmark used for the official SEED-v1 baseline.
set -euo pipefail

export KIMODO_CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
export KIMODO_STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
export PYTHONPATH="${KIMODO_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export KIMODO_EVAL_ASSET_ROOT="${KIMODO_EVAL_ASSET_ROOT:-${KIMODO_STORAGE_ROOT}/yezitao-kimodo-eval-v2}"
export KIMODO_MODEL_ROOT="${KIMODO_MODEL_ROOT:-${KIMODO_STORAGE_ROOT}/models}"
export KIMODO_TRAIN_RUN_DIR="${KIMODO_TRAIN_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-kf-smooth-lr1e5}"
export KIMODO_EVAL_EXPORT_RUN_DIR="${KIMODO_EVAL_EXPORT_RUN_DIR:-${KIMODO_STORAGE_ROOT}/eval-exports/v2-1m-hostnet-kf-smooth-lr1e5-step695k}"
export KIMODO_RUN_DIR="${KIMODO_RUN_DIR:-${KIMODO_EVAL_EXPORT_RUN_DIR}}"
export KIMODO_EVAL_ROOT="${KIMODO_EVAL_ROOT:-${KIMODO_STORAGE_ROOT}/eval-results/v2-1m-hostnet-kf-smooth-lr1e5-step695k-stratified10pct}"
export KIMODO_BENCHMARK_ROOT="${KIMODO_BENCHMARK_ROOT:-${KIMODO_EVAL_ASSET_ROOT}/benchmark/stratified-10pct}"
export KIMODO_OFFICIAL_BASELINE_SUMMARY="${KIMODO_OFFICIAL_BASELINE_SUMMARY:-${KIMODO_STORAGE_ROOT}/eval-results/official-seed-v1-stratified10pct/summary_rows.json}"

export TEXT_ENCODER_MODE="${TEXT_ENCODER_MODE:-local}"
export KIMODO_LLM2VEC_FOUNDATION="${KIMODO_LLM2VEC_FOUNDATION:-${KIMODO_MODEL_ROOT}/llm2vec/foundation}"
export KIMODO_LLM2VEC_MNTP="${KIMODO_LLM2VEC_MNTP:-${KIMODO_MODEL_ROOT}/llm2vec/mntp-adapter}"
export KIMODO_LLM2VEC_SUPERVISED="${KIMODO_LLM2VEC_SUPERVISED:-${KIMODO_MODEL_ROOT}/llm2vec/supervised-adapter}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${KIMODO_MODEL_ROOT}/checkpoints}"
export HF_HOME="${HF_HOME:-${KIMODO_STORAGE_ROOT}/hf-cache}"
export LOCAL_CACHE="${LOCAL_CACHE:-True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export KIMODO_EVAL_DIFFUSION_STEPS="${KIMODO_EVAL_DIFFUSION_STEPS:-100}"
export KIMODO_EVAL_BATCH_SIZE="${KIMODO_EVAL_BATCH_SIZE:-1}"
export KIMODO_EVAL_WORKERS="${KIMODO_EVAL_WORKERS:-4}"
export KIMODO_EVAL_PAPER_PROTOCOL="${KIMODO_EVAL_PAPER_PROTOCOL:-1}"
export KIMODO_EVAL_TEXT_ENCODER_FP32="${KIMODO_EVAL_TEXT_ENCODER_FP32:-1}"

export KIMODO_EXPORT_MIN_STEP=695000
export KIMODO_EXPORT_STEP_EVERY=695000
export KIMODO_EXPORT_ONCE=1
export KIMODO_EXPORT_PYTHON="${KIMODO_EXPORT_PYTHON:-python3}"

checkpoint="${KIMODO_TRAIN_RUN_DIR}/checkpoints/step-000695000.pt"
resolved_config="${KIMODO_TRAIN_RUN_DIR}/config.resolved.yaml"
if [[ ! -r "${checkpoint}" ]]; then
  echo "Missing readable 695k checkpoint: ${checkpoint}" >&2
  exit 2
fi
if [[ ! -r "${resolved_config}" ]]; then
  echo "Missing resolved training config: ${resolved_config}" >&2
  exit 2
fi
if [[ ! -d "${KIMODO_BENCHMARK_ROOT}/content" ]]; then
  echo "Missing stratified benchmark: ${KIMODO_BENCHMARK_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${KIMODO_OFFICIAL_BASELINE_SUMMARY}" ]]; then
  echo "Official SEED-v1 baseline is still running; evaluating 695k without inline deltas."
  unset KIMODO_OFFICIAL_BASELINE_SUMMARY
fi

echo "695k eval: checkpoint=${checkpoint}"
echo "695k eval: export=${KIMODO_EVAL_EXPORT_RUN_DIR}"
echo "695k eval: output=${KIMODO_EVAL_ROOT}"
echo "695k eval: baseline=${KIMODO_OFFICIAL_BASELINE_SUMMARY:-pending}"

bash "${KIMODO_CODE_ROOT}/scripts/eval_export_hostnet_checkpoints.sh"
exec bash "${KIMODO_CODE_ROOT}/scripts/eval_company_watcher.sh" \
  --minimum-step 695000 \
  --once
