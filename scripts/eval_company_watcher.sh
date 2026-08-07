#!/usr/bin/env bash
set -euo pipefail

# Run this in a separate one-GPU eval Pod. It reads immutable EMA exports from
# the training PV and never imports or mutates the live DDP trainer.
storage_root="${KIMODO_STORAGE_ROOT:-/mnt/kimodo}"
run_dir="${KIMODO_RUN_DIR:-${storage_root}/runs/v2-1m-production}"
benchmark_root="${KIMODO_BENCHMARK_ROOT:?set KIMODO_BENCHMARK_ROOT to the fixed public proxy}"
output_root="${KIMODO_EVAL_ROOT:?set KIMODO_EVAL_ROOT to a separate eval volume}"

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

exec python -m kimodo.evaluation.eval_monitor_cli "${args[@]}" "$@"
