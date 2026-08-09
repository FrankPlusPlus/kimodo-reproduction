#!/usr/bin/env bash
set -euo pipefail

# Re-run Official + Core10 A1 40k checkpoints on the stratified 10% benchmark subset.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="${KIMODO_BENCHMARK_STRATIFIED_ROOT:-/home/yezitao/PublicWorkspace/yzt/kimodo-benchmark-stratified-10pct}"
result_root="${KIMODO_STRATIFIED_RESULT_ROOT:-/home/yezitao/PublicWorkspace/yzt/kimodo-benchmark-results/core10-loss-domain-stratified-10pct}"
physical_run="${KIMODO_PHYSICAL_RUN:-/home/yezitao/PublicWorkspace/yzt/kimodo-validation-runs/core10-from-scratch-20k-10k}"
normalized_run="${KIMODO_NORMALIZED_RUN:-/home/yezitao/PublicWorkspace/yzt/kimodo-validation-runs/core10-normalized-20k-10k}"
target_step="${KIMODO_BENCHMARK_STEP:-40000}"
device="${KIMODO_EVAL_DEVICE:-cuda}"
gpu="${CUDA_VISIBLE_DEVICES:-1}"
python_bin="${project_root}/.venv/bin/python"
log_path="${result_root}/stratified_benchmark_pipeline.log"
status_path="${result_root}/stratified_benchmark_status.json"

mkdir -p "${result_root}"
exec >>"${log_path}" 2>&1

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/storage/data/metaiot_data/yzt/kimodo-repro/models}"
export LOCAL_CACHE=True
export HF_HUB_OFFLINE=1
export TEXT_ENCODER_MODE=local
export KIMODO_LLM2VEC_FOUNDATION="${KIMODO_LLM2VEC_FOUNDATION:-${CHECKPOINT_DIR}/llm2vec/foundation}"
export KIMODO_LLM2VEC_MNTP="${KIMODO_LLM2VEC_MNTP:-${CHECKPOINT_DIR}/llm2vec/mntp-adapter}"
export KIMODO_LLM2VEC_SUPERVISED="${KIMODO_LLM2VEC_SUPERVISED:-${CHECKPOINT_DIR}/llm2vec/supervised-adapter}"
export PYTHONPATH="${project_root}:${PYTHONPATH:-}"

write_status() {
  local state=$1 detail=$2 step=${3:-0}
  jq -n \
    --arg state "${state}" \
    --arg detail "${detail}" \
    --arg updated_at "$(date --iso-8601=seconds)" \
    --argjson latest_step "${step}" \
    '{state:$state,detail:$detail,latest_step:$latest_step,updated_at:$updated_at}' \
    >"${status_path}"
}

run_eval() {
  local name=$1 run_dir=$2 out_dir=$3 baseline=${4:-}
  local args=(
    --run-dir "${run_dir}"
    --benchmark "${benchmark_root}"
    --output-root "${out_dir}"
    --minimum-step "${target_step}"
    --once
    --python "${python_bin}"
    --device "${device}"
    --batch-size 1
    --num-workers 4
    --diffusion-steps 100
    --text-encoder-fp32
    --paper-protocol
  )
  if [[ -n "${baseline}" ]]; then
    args+=(--baseline-summary "${baseline}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m kimodo.evaluation.eval_monitor_cli "${args[@]}"
}

write_status starting "validating stratified benchmark asset" "${target_step}"
if [[ ! -s "${benchmark_root}/proxy_manifest.json" ]]; then
  echo "missing ${benchmark_root}/proxy_manifest.json; run scripts/build_benchmark_stratified_proxy.sh first" >&2
  exit 1
fi

official_root="${result_root}/official-seed-v1.1"
write_status official "running Official SEED-v1.1 on stratified subset" "${target_step}"
if [[ ! -s "${official_root}/summary_rows.json" ]]; then
  CUDA_VISIBLE_DEVICES="${gpu}" \
  KIMODO_BENCHMARK_ROOT="${benchmark_root}" \
  KIMODO_OFFICIAL_EVAL_ROOT="${official_root}" \
  KIMODO_EVAL_PAPER_PROTOCOL=1 \
  KIMODO_EVAL_TEXT_ENCODER_FP32=1 \
    scripts/eval_official_baseline.sh
fi

physical_out="${result_root}/old-physical-direct-40k"
write_status physical "running physical@40k on stratified subset" "${target_step}"
run_eval physical "${physical_run}" "${physical_out}" "${official_root}/summary_rows.json"

normalized_out="${result_root}/new-normalized-direct-40k"
write_status normalized "running normalized@40k on stratified subset" "${target_step}"
run_eval normalized "${normalized_run}" "${normalized_out}" "${official_root}/summary_rows.json"

physical_complete="${physical_out}/step-$(printf '%09d' "${target_step}")/complete.json"
normalized_complete="${normalized_out}/step-$(printf '%09d' "${target_step}")/complete.json"

case_count="$(jq -r '.selected_case_count' "${benchmark_root}/proxy_manifest.json")"
jq -n \
  --slurpfile official "${official_root}/summary_rows.json" \
  --slurpfile old "${physical_complete}" \
  --slurpfile new "${normalized_complete}" \
  --arg benchmark "${benchmark_root}" \
  --argjson cases "${case_count}" \
  --argjson step "${target_step}" \
  '{schema_version:3,protocol:{benchmark:"official public testsuite stratified-10pct",benchmark_root:$benchmark,cases:$cases,diffusion_steps:100,generation_batch_size:1,text_encoder_precision:"fp32",postprocess:false,paper_protocol:true,training_steps:$step,phase_schedule:"20k_phase1+20k_phase2_via_max_steps_override",subset_manifest:"proxy_manifest.json"},official_seed_v1_1:$official[0],old_physical_direct_40k:$old[0].summary,new_normalized_direct_40k:$new[0].summary,physical_vs_official:$old[0].official_baseline.deltas,normalized_vs_official:$new[0].official_baseline.deltas}' \
  >"${result_root}/comparison-40k-stratified.json"

write_status complete "stratified benchmark comparisons written to comparison-40k-stratified.json" "${target_step}"
