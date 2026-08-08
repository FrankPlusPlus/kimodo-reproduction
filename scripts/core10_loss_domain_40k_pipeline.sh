#!/usr/bin/env bash
# Resume core10 physical/normalized DDPM runs from 30k to 40k, then run 128-case benchmarks.
set -Eeuo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${project_root}/.venv/bin/python"
benchmark_root="/home/yezitao/PublicWorkspace/yzt/kimodo-benchmark-proxy-128"
result_root="/home/yezitao/PublicWorkspace/yzt/kimodo-benchmark-results/core10-loss-domain-128"
physical_run="/home/yezitao/PublicWorkspace/yzt/kimodo-validation-runs/core10-from-scratch-20k-10k"
normalized_run="/home/yezitao/PublicWorkspace/yzt/kimodo-validation-runs/core10-normalized-20k-10k"
log_path="${result_root}/core10_loss_domain_40k_pipeline.log"
status_path="${result_root}/core10_loss_domain_40k_status.json"
target_step=40000

mkdir -p "${result_root}"
exec >>"${log_path}" 2>&1

write_status() {
  local state=$1
  local detail=$2
  local step=${3:-0}
  local arm=${4:-""}
  local temporary="${status_path}.tmp"
  jq -n \
    --arg state "${state}" \
    --arg detail "${detail}" \
    --arg arm "${arm}" \
    --arg updated_at "$(date --iso-8601=seconds)" \
    --argjson latest_step "${step}" \
    --argjson target_step "${target_step}" \
    '{state:$state,detail:$detail,arm:$arm,latest_step:$latest_step,target_step:$target_step,updated_at:$updated_at}' \
    >"${temporary}"
  mv "${temporary}" "${status_path}"
}

patch_provenance() {
  local paths_yaml=$1
  local loss_domain=$2
  local ckpt=$3
  (
    cd "${project_root}"
    "${python_bin}" scripts/patch_checkpoint_provenance.py \
      --paths "${paths_yaml}" \
      --overlay configs/overlays/two_h200_gb512.yaml \
      --overlay configs/experiments/validation_core10_from_scratch.yaml \
      --set "loss.direct_feature_domain=${loss_domain}" \
      --set "runtime.max_steps_override=${target_step}" \
      --checkpoint "${ckpt}"
  )
}

resume_train() {
  local name=$1
  local paths_yaml=$2
  local loss_domain=$3
  local run_dir=$4
  local ckpt="${run_dir}/checkpoints/step-000030000.pt"

  write_status training "${name}: patching checkpoint provenance" 30000 "${name}"
  patch_provenance "${paths_yaml}" "${loss_domain}" "${ckpt}"

  write_status training "${name}: resuming 30k -> 40k on CUDA 0,2" 30000 "${name}"
  export CUDA_VISIBLE_DEVICES=0,2
  export KIMODO_PATHS_CONFIG="${paths_yaml}"
  export KIMODO_TRAINING_OVERLAY="${project_root}/configs/overlays/two_h200_gb512.yaml"
  "${project_root}/scripts/train_two_gpu_seed.sh" \
    --config "${project_root}/configs/training/kimodo_soma_seed_public.yaml" \
    --paths "${paths_yaml}" \
    --overlay "${project_root}/configs/overlays/two_h200_gb512.yaml" \
    --overlay "${project_root}/configs/experiments/validation_core10_from_scratch.yaml" \
    --set "loss.direct_feature_domain=${loss_domain}" \
    --set "runtime.max_steps_override=${target_step}"

  local latest=0
  if [[ -s "${run_dir}/train.jsonl" ]]; then
    latest=$(tail -n 1 "${run_dir}/train.jsonl" | jq -r '.global_step // 0')
  fi
  if [[ "${latest}" -lt "${target_step}" ]]; then
    echo "${name} finished below target step ${target_step}: latest=${latest}" >&2
    exit 1
  fi
  if [[ ! -s "${run_dir}/exports/step-$(printf '%09d' "${target_step}")/model.pt" ]]; then
    echo "${name} missing step-${target_step} export" >&2
    exit 1
  fi
  write_status training_complete "${name} reached step ${target_step}" "${target_step}" "${name}"
}

run_benchmark() {
  local name=$1
  local run_dir=$2
  local output_root=$3
  write_status benchmarking "${name}: running 128-case proxy at step ${target_step}" "${target_step}" "${name}"
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
    --output-root "${output_root}" \
    --baseline-summary "${result_root}/official-seed-v1.1/summary_rows.json" \
    --minimum-step "${target_step}" \
    --once \
    --python "${python_bin}" \
    --device cuda \
    --batch-size 1 \
    --num-workers 4 \
    --diffusion-steps 100 \
    --text-encoder-fp32 \
    --paper-protocol
  write_status benchmark_complete "${name} benchmark finished" "${target_step}" "${name}"
}

write_status started "core10 loss-domain 40k pipeline started" 30000 ""

resume_train \
  physical \
  "${project_root}/configs/paths/core10_physical_resume.local.yaml" \
  physical \
  "${physical_run}"

resume_train \
  normalized \
  "${project_root}/configs/paths/core10_normalized_resume.local.yaml" \
  normalized \
  "${normalized_run}"

physical_eval="${result_root}/old-physical-direct-40k"
normalized_eval="${result_root}/new-normalized-direct-40k"
run_benchmark physical "${physical_run}" "${physical_eval}"
run_benchmark normalized "${normalized_run}" "${normalized_eval}"

physical_complete="${physical_eval}/step-$(printf '%09d' "${target_step}")/complete.json"
normalized_complete="${normalized_eval}/step-$(printf '%09d' "${target_step}")/complete.json"
jq -n \
  --slurpfile official "${result_root}/official-seed-v1.1/summary_rows.json" \
  --slurpfile old "${physical_complete}" \
  --slurpfile new "${normalized_complete}" \
  '{schema_version:2,protocol:{benchmark:"official public testsuite proxy",cases:128,diffusion_steps:100,generation_batch_size:1,text_encoder_precision:"fp32",postprocess:false,paper_protocol:true,training_steps:40000,phase_schedule:"20k_phase1+20k_phase2_via_max_steps_override"},official_seed_v1_1:$official[0],old_physical_direct_40k:$old[0].summary,new_normalized_direct_40k:$new[0].summary,physical_vs_official:$old[0].official_baseline.deltas,normalized_vs_official:$new[0].official_baseline.deltas}' \
  >"${result_root}/comparison-40k.json"

"${python_bin}" benchmark/parse_folder.py "${physical_eval}/step-$(printf '%09d' "${target_step}")/generated" --format md \
  --output "${physical_eval}/step-$(printf '%09d' "${target_step}")/summary_rows.json" \
  >"${result_root}/old-physical-direct-40k.tables.md"
"${python_bin}" benchmark/parse_folder.py "${normalized_eval}/step-$(printf '%09d' "${target_step}")/generated" --format md \
  --output "${normalized_eval}/step-$(printf '%09d' "${target_step}")/summary_rows.json" \
  >"${result_root}/new-normalized-direct-40k.tables.md"

write_status complete "40k training and benchmark comparison finished" "${target_step}" "both"
echo "[$(date --iso-8601=seconds)] pipeline complete"
