#!/usr/bin/env bash
set -euo pipefail

# Run once on the exact proxy consumed by eval_company_watcher.sh. This creates
# the released SEED-v1.1 reference; it never touches a training checkpoint.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="${KIMODO_BENCHMARK_ROOT:?set KIMODO_BENCHMARK_ROOT to the fixed public proxy}"
output_root="${KIMODO_OFFICIAL_EVAL_ROOT:?set KIMODO_OFFICIAL_EVAL_ROOT for baseline outputs}"
official_model="${KIMODO_OFFICIAL_MODEL:-Kimodo-SOMA-SEED-v1.1}"
device="${KIMODO_EVAL_DEVICE:-cuda}"

cd "${project_root}"
python_bin="${project_root}/.venv/bin/python"
export PYTHONPATH="${project_root}:${PYTHONPATH:-}"
text_encoder_args=()
if [[ "${KIMODO_EVAL_TEXT_ENCODER_FP32:-0}" == 1 ]]; then
  text_encoder_args+=(--text_encoder_fp32)
fi
"${python_bin}" benchmark/generate_eval.py \
  --benchmark "${benchmark_root}" \
  --output "${output_root}" \
  --model "${official_model}" \
  --batch_size "${KIMODO_EVAL_BATCH_SIZE:-1}" \
  --num_workers "${KIMODO_EVAL_WORKERS:-4}" \
  --diffusion_steps "${KIMODO_EVAL_DIFFUSION_STEPS:-100}" \
  "${text_encoder_args[@]}"
"${python_bin}" benchmark/embed_folder.py "${output_root}" --device "${device}" "${text_encoder_args[@]}"
evaluate_args=("${output_root}" --device "${device}")
if [[ "${KIMODO_EVAL_PAPER_PROTOCOL:-0}" == 1 ]]; then
  evaluate_args+=(--paper-protocol)
fi
"${python_bin}" benchmark/evaluate_folder.py "${evaluate_args[@]}"
"${python_bin}" benchmark/parse_folder.py \
  "${output_root}" \
  --output "${output_root}/summary_rows.json"

echo "official baseline: ${output_root}/summary_rows.json"
