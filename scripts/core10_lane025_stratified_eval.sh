#!/usr/bin/env bash
# Eval Norm+lane=0.25 @40k on stratified 10% (eval-v2), then refresh comparisons.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark_root="${KIMODO_BENCHMARK_STRATIFIED_ROOT:-/home/yezitao/PublicWorkspace/yzt/kimodo-benchmark-stratified-10pct}"
result_root="${KIMODO_STRATIFIED_RESULT_ROOT:-/home/yezitao/PublicWorkspace/yzt/kimodo-benchmark-results/core10-loss-domain-stratified-10pct}"
lane_run="${KIMODO_LANE025_RUN:-/home/yezitao/PublicWorkspace/yzt/kimodo-validation-runs/core10-normalized-benchmark-lane-40k}"
physical_run_out="${result_root}/old-physical-direct-40k"
normalized_run_out="${result_root}/new-normalized-direct-40k"
official_root="${result_root}/official-seed-v1.1"
target_step="${KIMODO_BENCHMARK_STEP:-40000}"
device="${KIMODO_EVAL_DEVICE:-cuda}"
gpu="${CUDA_VISIBLE_DEVICES:-1}"
python_bin="${project_root}/.venv/bin/python"
log_path="${result_root}/lane025_stratified_eval.log"
status_path="${result_root}/lane025_stratified_eval_status.json"
step_tag="$(printf 'step-%09d' "${target_step}")"

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

on_error() {
  write_status failed "lane0.25 stratified eval failed at line ${1}; see ${log_path}" "${target_step}"
}
trap 'on_error $LINENO' ERR

write_status starting "validating inputs for lane0.25 stratified eval" "${target_step}"

[[ -s "${benchmark_root}/proxy_manifest.json" ]] || {
  echo "missing stratified benchmark" >&2
  exit 1
}
[[ -s "${official_root}/summary_rows.json" ]] || {
  echo "missing official stratified summary; run core10_stratified_benchmark_pipeline.sh first" >&2
  exit 1
}
[[ -s "${normalized_run_out}/${step_tag}/complete.json" ]] || {
  echo "missing normalized lane0 stratified complete.json" >&2
  exit 1
}
[[ -s "${lane_run}/exports/${step_tag}/model.pt" ]] || {
  echo "missing lane0.25 export ${lane_run}/exports/${step_tag}/model.pt" >&2
  exit 1
}

lane_out="${result_root}/new-normalized-benchmark-lane-40k"
write_status benchmarking "running normalized+lane0.25@40k on stratified eval-v2" "${target_step}"
CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m kimodo.evaluation.eval_monitor_cli \
  --run-dir "${lane_run}" \
  --benchmark "${benchmark_root}" \
  --output-root "${lane_out}" \
  --baseline-summary "${official_root}/summary_rows.json" \
  --minimum-step "${target_step}" \
  --once \
  --python "${python_bin}" \
  --device "${device}" \
  --batch-size 1 \
  --num-workers 4 \
  --diffusion-steps 100 \
  --text-encoder-fp32 \
  --paper-protocol

lane_complete="${lane_out}/${step_tag}/complete.json"
physical_complete="${physical_run_out}/${step_tag}/complete.json"
normalized_complete="${normalized_run_out}/${step_tag}/complete.json"
case_count="$(jq -r '.selected_case_count' "${benchmark_root}/proxy_manifest.json")"

write_status comparing "writing stratified comparison JSONs" "${target_step}"

# Keep A1-style comparison file, and write a 4-arm eval-v2 scorecard.
jq -n \
  --slurpfile official "${official_root}/summary_rows.json" \
  --slurpfile old "${physical_complete}" \
  --slurpfile new "${normalized_complete}" \
  --slurpfile lane "${lane_complete}" \
  --arg benchmark "${benchmark_root}" \
  --argjson cases "${case_count}" \
  --argjson step "${target_step}" \
  '{
     schema_version:4,
     protocol:{
       benchmark:"official public testsuite stratified-10pct",
       benchmark_root:$benchmark,
       cases:$cases,
       diffusion_steps:100,
       generation_batch_size:1,
       text_encoder_precision:"fp32",
       postprocess:false,
       paper_protocol:true,
       training_steps:$step,
       phase_schedule:"20k_phase1+20k_phase2_via_max_steps_override",
       subset_manifest:"proxy_manifest.json",
       alias:"eval-v2"
     },
     official_seed_v1_1:$official[0],
     old_physical_direct_40k:$old[0].summary,
     new_normalized_direct_40k:$new[0].summary,
     new_normalized_benchmark_lane_40k:$lane[0].summary,
     physical_vs_official:$old[0].official_baseline.deltas,
     normalized_vs_official:$new[0].official_baseline.deltas,
     lane025_vs_official:$lane[0].official_baseline.deltas
   }' \
  >"${result_root}/comparison-40k-stratified.json"

jq -n \
  --slurpfile official "${official_root}/summary_rows.json" \
  --slurpfile baseline "${normalized_complete}" \
  --slurpfile lane "${lane_complete}" \
  --arg benchmark "${benchmark_root}" \
  --argjson cases "${case_count}" \
  --argjson step "${target_step}" \
  '{
     schema_version:2,
     protocol:{
       benchmark:"official public testsuite stratified-10pct",
       alias:"eval-v2",
       benchmark_root:$benchmark,
       cases:$cases,
       diffusion_steps:100,
       generation_batch_size:1,
       text_encoder_precision:"fp32",
       postprocess:false,
       paper_protocol:true,
       training_steps:$step,
       loss_domain:"normalized",
       benchmark_coverage_probability:0.25,
       benchmark_overlay:"configs/overlays/benchmark_v2_constraints.yaml"
     },
     official_seed_v1_1:$official[0],
     normalized_lane0_40k:$baseline[0].summary,
     normalized_lane025_40k:$lane[0].summary,
     lane025_vs_official:$lane[0].official_baseline.deltas
   }' \
  >"${result_root}/comparison-benchmark-lane-40k-stratified.json"

write_status complete "lane0.25 stratified eval done; comparisons refreshed" "${target_step}"
echo "[$(date --iso-8601=seconds)] lane0.25 stratified eval complete"
