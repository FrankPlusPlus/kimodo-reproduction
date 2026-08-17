#!/usr/bin/env bash
# Export and evaluate rescue 795k EMA once on the stratified-10pct proxy.
# This is the last healthy checkpoint of from780k-lr3e6 (800k already took off).
#
# UI: 1 instance x 1 GPU. Recreate/attach after the container restart.
# Do not attach this to a 16-GPU job. Do not use the demo kimodo-dev GPU.
# Start box is sh:
#   exec bash /home/share/yzt/kimodo-reproduction/scripts/eval_wd03_795_once.sh
set -euo pipefail

export KIMODO_CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
export KIMODO_STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
export PYTHONPATH="${KIMODO_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export KIMODO_EVAL_ASSET_ROOT="${KIMODO_EVAL_ASSET_ROOT:-${KIMODO_STORAGE_ROOT}/yezitao-kimodo-eval-v2}"
export KIMODO_MODEL_ROOT="${KIMODO_MODEL_ROOT:-${KIMODO_STORAGE_ROOT}/models}"
export HF_HOME="${HF_HOME:-${KIMODO_STORAGE_ROOT}/hf-cache}"

export KIMODO_TRAIN_RUN_DIR="${KIMODO_TRAIN_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-wd03-from780k-lr3e6}"
export KIMODO_EVAL_EXPORT_RUN_DIR="${KIMODO_EVAL_EXPORT_RUN_DIR:-${KIMODO_STORAGE_ROOT}/eval-exports/v2-1m-hostnet-wd03-from780k-lr3e6-step795k}"
export KIMODO_RUN_DIR="${KIMODO_RUN_DIR:-${KIMODO_EVAL_EXPORT_RUN_DIR}}"
export KIMODO_EVAL_ROOT="${KIMODO_EVAL_ROOT:-${KIMODO_STORAGE_ROOT}/eval-results/v2-1m-hostnet-wd03-from780k-lr3e6-step795k-stratified10pct}"
export KIMODO_BENCHMARK_ROOT="${KIMODO_BENCHMARK_ROOT:-${KIMODO_EVAL_ASSET_ROOT}/benchmark/stratified-10pct}"
export KIMODO_OFFICIAL_BASELINE_SUMMARY="${KIMODO_OFFICIAL_BASELINE_SUMMARY:-${KIMODO_EVAL_ASSET_ROOT}/baselines/official-seed-v1.1/summary_rows.json}"
export KIMODO_RESOLVED_CONFIG="${KIMODO_RESOLVED_CONFIG:-${KIMODO_TRAIN_RUN_DIR}/config.resolved.yaml}"

export TEXT_ENCODER_MODE="${TEXT_ENCODER_MODE:-local}"
export KIMODO_LLM2VEC_FOUNDATION="${KIMODO_LLM2VEC_FOUNDATION:-${KIMODO_MODEL_ROOT}/llm2vec/foundation}"
export KIMODO_LLM2VEC_MNTP="${KIMODO_LLM2VEC_MNTP:-${KIMODO_MODEL_ROOT}/llm2vec/mntp-adapter}"
export KIMODO_LLM2VEC_SUPERVISED="${KIMODO_LLM2VEC_SUPERVISED:-${KIMODO_MODEL_ROOT}/llm2vec/supervised-adapter}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${KIMODO_MODEL_ROOT}/checkpoints}"
export LOCAL_CACHE="${LOCAL_CACHE:-True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export KIMODO_EVAL_DIFFUSION_STEPS="${KIMODO_EVAL_DIFFUSION_STEPS:-100}"
export KIMODO_EVAL_BATCH_SIZE="${KIMODO_EVAL_BATCH_SIZE:-1}"
export KIMODO_EVAL_WORKERS="${KIMODO_EVAL_WORKERS:-4}"
export KIMODO_EVAL_PAPER_PROTOCOL="${KIMODO_EVAL_PAPER_PROTOCOL:-1}"
export KIMODO_EVAL_TEXT_ENCODER_FP32="${KIMODO_EVAL_TEXT_ENCODER_FP32:-1}"
export KIMODO_EXPORT_PYTHON="${KIMODO_EXPORT_PYTHON:-python3}"

checkpoint="${KIMODO_CKPT_795:-${KIMODO_TRAIN_RUN_DIR}/checkpoints/step-000795000.pt}"
preserved="${KIMODO_STORAGE_ROOT}/preserved-pre-collapse/v2-1m-hostnet-wd03-from780k-lr3e6/step-000795000.pt"
if [[ ! -r "${checkpoint}" && -r "${preserved}" ]]; then
  checkpoint="${preserved}"
fi
if [[ ! -r "${checkpoint}" ]]; then
  echo "Missing readable 795k checkpoint: ${checkpoint}" >&2
  exit 2
fi
if [[ ! -r "${KIMODO_RESOLVED_CONFIG}" ]]; then
  echo "Missing resolved training config: ${KIMODO_RESOLVED_CONFIG}" >&2
  exit 2
fi
if [[ ! -d "${KIMODO_BENCHMARK_ROOT}/content" ]]; then
  echo "Missing stratified benchmark: ${KIMODO_BENCHMARK_ROOT}" >&2
  exit 2
fi
if [[ "${KIMODO_EVAL_SKIP_CUDA_CHECK:-0}" != "1" ]]; then
  if ! python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    echo "CUDA torch unavailable in this pod image." >&2
    echo "Recreate/attach this 1xH200 with hub.inner.ai.kingsoft.com/hh-678395/kimodo-reproduction:v8" >&2
    exit 2
  fi
fi
if [[ ! -f "${KIMODO_OFFICIAL_BASELINE_SUMMARY}" ]]; then
  echo "official baseline summary missing; evaluating 795k without inline deltas."
  unset KIMODO_OFFICIAL_BASELINE_SUMMARY
fi

parent_750_summary="${KIMODO_STORAGE_ROOT}/eval-results/v2-1m-hostnet-wd03-from650k-stratified10pct/step-000750000/summary_rows.json"
echo "795k eval: checkpoint=${checkpoint}"
echo "795k eval: export=${KIMODO_EVAL_EXPORT_RUN_DIR}"
echo "795k eval: output=${KIMODO_EVAL_ROOT}"
echo "795k eval: baseline=${KIMODO_OFFICIAL_BASELINE_SUMMARY:-pending}"
if [[ -f "${parent_750_summary}" ]]; then
  echo "795k eval: compare later to parent 750k ${parent_750_summary}"
fi

mkdir -p "${KIMODO_EVAL_EXPORT_RUN_DIR}/exports" "${KIMODO_EVAL_ROOT}"
python_bin="${KIMODO_EXPORT_PYTHON}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin="$(command -v python3 || command -v python)"
fi
"${python_bin}" "${KIMODO_CODE_ROOT}/scripts/export_trainer_checkpoint_bundle.py" \
  --checkpoint "${checkpoint}" \
  --resolved-config "${KIMODO_RESOLVED_CONFIG}" \
  --output-run-dir "${KIMODO_EVAL_EXPORT_RUN_DIR}" \
  --step 795000

exec bash "${KIMODO_CODE_ROOT}/scripts/eval_company_watcher.sh" \
  --minimum-step 795000 \
  --once
