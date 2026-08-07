#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

stage="${1:-}"
if [[ -z "${stage}" ]]; then
  echo "usage: $0 {pilot|generate|audit-pilot|audit|manifest}" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${KIMODO_PYTHON:-${repo_root}/.venv/bin/python}"
v2_root="${KIMODO_V2_ROOT:-/pvc/v2.building}"
v1_root="${KIMODO_V1_ROOT:-/pvc/v1}"
provenance_root="${v2_root}/provenance"
requests="${KIMODO_LLM_REQUESTS:-${provenance_root}/qwen.requests.v2.2.jsonl}"
response_selection="${KIMODO_LLM_RESPONSE_SELECTION:-${provenance_root}/mimo.responses.selected.v2.2.json}"
raw_responses="${provenance_root}/mimo.responses.v2.2.jsonl"
case "${stage}" in
  audit|manifest)
    [[ -f "${response_selection}" ]] || {
      echo "${stage} requires a finalized response selection: ${response_selection}" >&2
      exit 3
    }
    selected_responses="$("${python_bin}" -m kimodo.training.response_selection_cli resolve \
      --selection "${response_selection}" --requests "${requests}")"
    if [[ -n "${KIMODO_LLM_RESPONSES:-}" ]] \
      && [[ "$(realpath -e -- "${KIMODO_LLM_RESPONSES}")" != "${selected_responses}" ]]; then
      echo "KIMODO_LLM_RESPONSES may not bypass the finalized response selection" >&2
      exit 3
    fi
    responses="${selected_responses}"
    ;;
  *)
    responses="${KIMODO_LLM_RESPONSES:-${raw_responses}}"
    ;;
esac
plan="${KIMODO_V2_PLAN:-${provenance_root}/timeline.selected.v2.2.jsonl}"
quality_report="${KIMODO_LLM_QUALITY_REPORT:-${provenance_root}/mimo.quality.v2.2.json}"
review_sample="${KIMODO_LLM_REVIEW_SAMPLE:-${provenance_root}/mimo.review.v2.2.jsonl}"
train_split="${KIMODO_TRAIN_SPLIT:-${repo_root}/artifacts/benchmark-metadata/splits/train_split_paths.txt}"
source_manifest="${KIMODO_V1_RAW_MANIFEST:-${v1_root}/train.raw.jsonl}"
model="${PRODUCT_GRAPH_LLM_MODEL:-mimo-v2.5-pro}"
base_url="${PRODUCT_GRAPH_LLM_BASE_URL:-https://api.xiaomimimo.com/v1}"

mkdir -p "${provenance_root}"

require_api_key() {
  if [[ -z "${PRODUCT_GRAPH_LLM_API_KEY:-}" ]]; then
    echo "PRODUCT_GRAPH_LLM_API_KEY must be injected through the environment or a secret." >&2
    exit 2
  fi
}

case "${stage}" in
  pilot)
    require_api_key
    "${python_bin}" -m kimodo.training.llm_api_augmentation_cli \
      --requests "${requests}" \
      --base-url "${base_url}" \
      --model "${model}" --judge-model "${model}" \
      --batch-size "${KIMODO_LLM_BATCH_SIZE:-8}" \
      --concurrency "${KIMODO_LLM_CONCURRENCY:-32}" \
      --requests-per-minute "${KIMODO_LLM_RPM:-90}" \
      --max-requests "${KIMODO_LLM_PILOT_SIZE:-64}" \
      --output "${provenance_root}/mimo.responses.pilot.jsonl"
    ;;
  generate)
    require_api_key
    "${python_bin}" -m kimodo.training.llm_api_augmentation_cli \
      --requests "${requests}" \
      --base-url "${base_url}" \
      --model "${model}" --judge-model "${model}" \
      --batch-size "${KIMODO_LLM_BATCH_SIZE:-8}" \
      --concurrency "${KIMODO_LLM_CONCURRENCY:-32}" \
      --requests-per-minute "${KIMODO_LLM_RPM:-90}" \
      --output "${provenance_root}/mimo.responses.v2.2.jsonl"
    ;;
  audit-pilot)
    "${python_bin}" -m kimodo.training.llm_quality_cli \
      --requests "${requests}" \
      --responses "${provenance_root}/mimo.responses.pilot.jsonl" \
      --allow-partial --report-only \
      --report "${provenance_root}/mimo.quality.pilot.json" \
      --review-sample "${provenance_root}/mimo.review.pilot.jsonl"
    ;;
  audit)
    "${python_bin}" -m kimodo.training.llm_quality_cli \
      --requests "${requests}" \
      --responses "${responses}" \
      --plan "${plan}" \
      --sample-per-event-count 100 --max-risk-samples 400 --max-high-reuse-samples 400 \
      --report "${quality_report}" \
      --review-sample "${review_sample}"
    ;;
  manifest)
    "${python_bin}" -m kimodo.training.v2_manifest_cli \
      --source-manifest "${source_manifest}" \
      --plan "${plan}" \
      --responses "${responses}" \
      --train-split "${train_split}" \
      --expected-model "${model}" --expected-revision provider-managed \
      --output "${v2_root}/train.raw.jsonl"
    ;;
  *)
    echo "unknown stage: ${stage}" >&2
    exit 2
    ;;
esac
