#!/usr/bin/env bash
set -euo pipefail

# Run this in a separate one-GPU eval Pod. It reads immutable EMA exports from
# the training PV and never imports or mutates the live DDP trainer.
storage_root="${KIMODO_STORAGE_ROOT:-/mnt/kimodo}"
run_dir="${KIMODO_RUN_DIR:-${storage_root}/runs/v2-1m-production}"
benchmark_root="${KIMODO_BENCHMARK_ROOT:-${storage_root}/eval/benchmark/proxy}"
output_root="${KIMODO_EVAL_ROOT:-${storage_root}/eval/v2-1m}"

args=(
  --run-dir "${run_dir}"
  --benchmark "${benchmark_root}"
  --output-root "${output_root}"
  --batch-size "${KIMODO_EVAL_BATCH_SIZE:-1}"
  --num-workers "${KIMODO_EVAL_WORKERS:-4}"
  --diffusion-steps "${KIMODO_EVAL_DIFFUSION_STEPS:-100}"
  --poll-seconds "${KIMODO_EVAL_POLL_SECONDS:-60}"
)
if [[ -n "${KIMODO_OFFICIAL_BASELINE_SUMMARY:-}" ]]; then
  args+=(--baseline-summary "${KIMODO_OFFICIAL_BASELINE_SUMMARY}")
fi
if [[ "${KIMODO_EVAL_PAPER_PROTOCOL:-0}" == 1 ]]; then
  args+=(--paper-protocol)
fi
if [[ "${KIMODO_EVAL_TEXT_ENCODER_FP32:-0}" == 1 ]]; then
  args+=(--text-encoder-fp32)
fi

asset_root="${KIMODO_EVAL_ASSET_ROOT:-${storage_root}/yezitao-kimodo-eval-v2}"
if [[ -z "${KIMODO_EVAL_TEXT_GALLERY:-}" && "${benchmark_root}" == *stratified-10pct* ]]; then
  frozen="${asset_root}/galleries/tmr-text-wd03-750k"
  parent="${storage_root}/eval-results/v2-1m-hostnet-wd03-from650k-stratified10pct/step-000750000/generated"
  if [[ -d "${frozen}" ]]; then
    export KIMODO_EVAL_TEXT_GALLERY="${frozen}"
  elif [[ -d "${parent}" ]]; then
    export KIMODO_EVAL_TEXT_GALLERY="${parent}"
  fi
fi
if [[ "${benchmark_root}" == *stratified-10pct* ]]; then
  if [[ -z "${KIMODO_EVAL_TEXT_GALLERY:-}" || ! -d "${KIMODO_EVAL_TEXT_GALLERY}" ]]; then
    echo "stratified-10pct eval needs the frozen 750k TMR text gallery." >&2
    echo "Copy parent 750k text_embedding.npy to ${asset_root}/galleries/tmr-text-wd03-750k" >&2
    echo "or set KIMODO_EVAL_TEXT_GALLERY." >&2
    exit 2
  fi
  args+=(--text-gallery "${KIMODO_EVAL_TEXT_GALLERY}")
  echo "eval: text_gallery=${KIMODO_EVAL_TEXT_GALLERY}"
fi

exec python -m kimodo.evaluation.eval_monitor_cli "${args[@]}" "$@"
