#!/usr/bin/env bash
# A2: Core10 normalized + benchmark_v2_constraints lane, from scratch 40k, then 128-case benchmark.
set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${project_root}/.venv/bin/python"
benchmark_root="/home/yezitao/PublicWorkspace/yzt/kimodo-benchmark-proxy-128"
result_root="/home/yezitao/PublicWorkspace/yzt/kimodo-benchmark-results/core10-loss-domain-128"
run_dir="/home/yezitao/PublicWorkspace/yzt/kimodo-validation-runs/core10-normalized-benchmark-lane-40k"
baseline_run="/home/yezitao/PublicWorkspace/yzt/kimodo-validation-runs/core10-normalized-20k-10k"
log_path="${result_root}/core10_benchmark_lane_40k_pipeline.log"
status_path="${result_root}/core10_benchmark_lane_40k_status.json"
target_step=40000

mkdir -p "${result_root}"
exec >>"${log_path}" 2>&1

write_status() {
  local state=$1
  local detail=$2
  local step=${3:-0}
  local temporary="${status_path}.tmp"
  jq -n \
    --arg state "${state}" \
    --arg detail "${detail}" \
    --arg updated_at "$(date --iso-8601=seconds)" \
    --argjson latest_step "${step}" \
    --argjson target_step "${target_step}" \
    '{state:$state,detail:$detail,latest_step:$latest_step,target_step:$target_step,updated_at:$updated_at}' \
    >"${temporary}"
  mv "${temporary}" "${status_path}"
}

on_error() {
  write_status failed "benchmark lane 40k pipeline failed at line ${1}; inspect ${log_path}" "${latest_step:-0}"
}
trap 'on_error $LINENO' ERR

write_status started "normalized + benchmark lane 40k from scratch" 0

export CUDA_VISIBLE_DEVICES=0,2
export KIMODO_PATHS_CONFIG="${project_root}/configs/paths/core10_normalized_benchmark_lane_40k.local.yaml"
export KIMODO_TRAINING_OVERLAY="${project_root}/configs/overlays/two_h200_gb512.yaml"
write_status training "training normalized + benchmark_v2_constraints on CUDA 0,2" 0
"${project_root}/scripts/train_two_gpu_seed.sh" \
  --config "${project_root}/configs/training/kimodo_soma_seed_public.yaml" \
  --paths "${project_root}/configs/paths/core10_normalized_benchmark_lane_40k.local.yaml" \
  --overlay "${project_root}/configs/overlays/two_h200_gb512.yaml" \
  --overlay "${project_root}/configs/experiments/validation_core10_from_scratch.yaml" \
  --overlay "${project_root}/configs/overlays/benchmark_v2_constraints.yaml" \
  --set "loss.direct_feature_domain=normalized" \
  --set "runtime.max_steps_override=${target_step}"

latest=0
if [[ -s "${run_dir}/train.jsonl" ]]; then
  latest=$(tail -n 1 "${run_dir}/train.jsonl" | jq -r '.global_step // 0')
fi
if [[ "${latest}" -lt "${target_step}" ]]; then
  echo "training finished below target step ${target_step}: latest=${latest}" >&2
  exit 1
fi
if [[ ! -s "${run_dir}/exports/step-$(printf '%09d' "${target_step}")/model.pt" ]]; then
  echo "missing step-${target_step} export" >&2
  exit 1
fi
write_status training_complete "normalized + benchmark lane reached step ${target_step}" "${target_step}"

eval_output="${result_root}/new-normalized-benchmark-lane-40k"
write_status benchmarking "running 128-case proxy at step ${target_step}" "${target_step}"
export CHECKPOINT_DIR=/storage/data/metaiot_data/yzt/kimodo-repro/models
export LOCAL_CACHE=True
export HF_HUB_OFFLINE=1
export TEXT_ENCODER_MODE=local
export KIMODO_LLM2VEC_FOUNDATION=/storage/data/metaiot_data/yzt/kimodo-repro/models/llm2vec/foundation
export KIMODO_LLM2VEC_MNTP=/storage/data/metaiot_data/yzt/kimodo-repro/models/llm2vec/mntp-adapter
export KIMODO_LLM2VEC_SUPERVISED=/storage/data/metaiot_data/yzt/kimodo-repro/models/llm2vec/supervised-adapter
export PATH="${project_root}/.venv/bin:${PATH}"
cd "${project_root}"
CUDA_VISIBLE_DEVICES=0 "${python_bin}" -m kimodo.evaluation.eval_monitor_cli \
  --run-dir "${run_dir}" \
  --benchmark "${benchmark_root}" \
  --output-root "${eval_output}" \
  --baseline-summary "${result_root}/new-normalized-direct-40k/step-000040000/summary_rows.json" \
  --minimum-step "${target_step}" \
  --once \
  --python "${python_bin}" \
  --device cuda \
  --batch-size 1 \
  --num-workers 4 \
  --diffusion-steps 100 \
  --text-encoder-fp32 \
  --paper-protocol

lane_complete="${eval_output}/step-$(printf '%09d' "${target_step}")/complete.json"
baseline_complete="${result_root}/new-normalized-direct-40k/step-000040000/complete.json"
jq -n \
  --slurpfile official "${result_root}/official-seed-v1.1/summary_rows.json" \
  --slurpfile baseline "${baseline_complete}" \
  --slurpfile lane "${lane_complete}" \
  '{schema_version:1,protocol:{benchmark:"official public testsuite proxy",cases:128,diffusion_steps:100,generation_batch_size:1,text_encoder_precision:"fp32",postprocess:false,paper_protocol:true,training_steps:40000,phase_schedule:"20k_phase1+20k_phase2_via_max_steps_override",loss_domain:"normalized",benchmark_coverage_probability:0.25,benchmark_overlay:"configs/overlays/benchmark_v2_constraints.yaml"},official_seed_v1_1:$official[0],normalized_lane0_40k:$baseline[0].summary,normalized_lane025_40k:$lane[0].summary,lane025_vs_lane0:$lane[0].official_baseline.deltas}' \
  >"${result_root}/comparison-benchmark-lane-40k.json"

"${python_bin}" benchmark/parse_folder.py \
  "${eval_output}/step-$(printf '%09d' "${target_step}")/generated" \
  --format md \
  --output "${eval_output}/step-$(printf '%09d' "${target_step}")/summary_rows.json" \
  >"${result_root}/new-normalized-benchmark-lane-40k.tables.md"

write_status complete "benchmark lane 40k training and evaluation finished" "${target_step}"
echo "[$(date --iso-8601=seconds)] pipeline complete"
