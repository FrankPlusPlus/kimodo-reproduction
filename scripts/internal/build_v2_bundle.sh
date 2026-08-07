#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Internal stage implementation; use scripts/v2_pipeline.sh.
set -euo pipefail

command_name="${1:-all}"
if [[ "${command_name}" != "all" && "${command_name}" != "status" && "${command_name}" != "plan" ]]; then
  echo "usage: $0 {all|status|plan}" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
python_bin="${KIMODO_PYTHON:-${repo_root}/.venv/bin/python}"
storage_root="${KIMODO_STORAGE_ROOT:-/mnt/kimodo}"
v2_root="${KIMODO_V2_ROOT:-${storage_root}/data/benchmark-v2-soma30-v2.2.building}"
if [[ "${v2_root}" != *.building ]]; then
  echo "KIMODO_V2_ROOT must end in .building: ${v2_root}" >&2
  exit 2
fi
v2_final="${KIMODO_V2_FINAL_ROOT:-${v2_root%.building}}"
v1_root="${KIMODO_V1_ROOT:-${storage_root}/data/adopted-legacy-soma30-v1}"
provenance_root="${v2_root}/provenance"
requests="${KIMODO_LLM_REQUESTS:-${provenance_root}/qwen.requests.v2.2.jsonl}"
response_selection="${KIMODO_LLM_RESPONSE_SELECTION:-${provenance_root}/mimo.responses.selected.v2.2.json}"
raw_responses="${provenance_root}/mimo.responses.v2.2.jsonl"
if [[ "${command_name}" == "all" ]]; then
  [[ -f "${response_selection}" ]] || {
    echo "V2 construction requires a finalized response selection: ${response_selection}" >&2
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
elif [[ -f "${response_selection}" ]]; then
  responses="$("${python_bin}" -m kimodo.training.response_selection_cli resolve \
    --selection "${response_selection}" --requests "${requests}")"
else
  responses="${KIMODO_LLM_RESPONSES:-${raw_responses}}"
fi
quality_report="${KIMODO_LLM_QUALITY_REPORT:-${provenance_root}/mimo.quality.v2.2.json}"
review_sample="${KIMODO_LLM_REVIEW_SAMPLE:-${provenance_root}/mimo.review.v2.2.jsonl}"
expert_review="${KIMODO_EXPERT_REVIEW:-${provenance_root}/mimo.expert-review.v2.2.json}"
expert_verdicts="${KIMODO_EXPERT_VERDICTS:-${provenance_root}/mimo.expert-verdicts.v2.2.jsonl}"
embedding_canary="${provenance_root}/llm2vec-v1-numerical-canary.v2.2.json"
raw_manifest="${v2_root}/train.raw.jsonl"
llm_raw_manifest="${v2_root}/train.llm.raw.jsonl"
llm_cached_manifest="${v2_root}/train.llm.cached.jsonl"
cached_manifest="${v2_root}/train.cached.jsonl"
inventory="${v2_root}/train.cached.references.jsonl"
stats_root="${v2_root}/stats/repro-soma30-30fps"
preflight_paths="${provenance_root}/v2-preflight.paths.yaml"
preflight_report="${provenance_root}/v2-preflight.json"
expected_entries="${KIMODO_V2_EXPECTED_ENTRIES:-1440741}"
poll_seconds="${KIMODO_V2_POLL_SECONDS:-30}"
log_file="${KIMODO_V2_LOG:-${v2_root}.pipeline.log}"

foundation_model="${KIMODO_LLM2VEC_FOUNDATION:-${storage_root}/models/llm2vec/foundation}"
mntp_model="${KIMODO_LLM2VEC_MNTP:-${storage_root}/models/llm2vec/mntp-adapter}"
supervised_model="${KIMODO_LLM2VEC_SUPERVISED:-${storage_root}/models/llm2vec/supervised-adapter}"
expert_model="${KIMODO_EXPERT_MODEL:-${storage_root}/models/Qwen3-32B}"
foundation_revision="${KIMODO_LLM2VEC_FOUNDATION_REVISION:-53346005fb0ef11d3b6a83b12c895cca40156b6c}"
mntp_revision="${KIMODO_LLM2VEC_MNTP_REVISION:-31474e395ada192e8ed1586db6be79fb3b70c9c0}"
supervised_revision="${KIMODO_LLM2VEC_SUPERVISED_REVISION:-baa8ebf04a1c2500e61288e7dad65e8ae42601a7}"
text_device="${KIMODO_TEXT_DEVICE:-cuda:0}"
stats_workers="${KIMODO_STATS_WORKERS:-16}"
gpu_idle_memory_mib="${KIMODO_GPU_IDLE_MEMORY_MIB:-10000}"
gpu_idle_utilization="${KIMODO_GPU_IDLE_UTILIZATION:-10}"
gpu_min_total_memory_mib="${KIMODO_TEXT_GPU_MIN_TOTAL_MEMORY_MIB:-70000}"

timestamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
stage() { echo "[$(timestamp)] V2 stage: $*" | tee -a "${log_file}"; }

wait_for_text_gpu() {
  if [[ "${text_device}" != cuda:* ]]; then
    return
  fi
  local gpu_index="${text_device#cuda:}"
  [[ "${gpu_index}" =~ ^[0-9]+$ ]] || {
    echo "KIMODO_TEXT_DEVICE must be cpu or cuda:<integer>: ${text_device}" >&2
    exit 2
  }
  [[ "${gpu_idle_memory_mib}" =~ ^[0-9]+$ \
    && "${gpu_idle_utilization}" =~ ^[0-9]+$ \
    && "${gpu_min_total_memory_mib}" =~ ^[0-9]+$ ]] || {
    echo "GPU idle thresholds must be non-negative integers" >&2
    exit 2
  }
  command -v nvidia-smi >/dev/null || {
    echo "nvidia-smi is required for the text GPU idle gate" >&2
    exit 2
  }
  # CUDA's logical device order is not guaranteed to match the numeric order
  # printed by nvidia-smi. Resolve the requested torch device to its physical
  # UUID first, then use that UUID for the NVML idle check. This keeps the gate
  # and the subsequent PyTorch allocation on the same physical GPU.
  local gpu_uuid
  gpu_uuid="$(${python_bin} - "${gpu_index}" <<'PY'
import sys
import torch

index = int(sys.argv[1])
if index < 0 or index >= torch.cuda.device_count():
    raise SystemExit(f"CUDA device index is unavailable: {index}")
value = str(torch.cuda.get_device_properties(index).uuid)
if value.startswith(("GPU-", "MIG-")):
    print(value)
else:
    print(f"GPU-{value}")
PY
  )"
  [[ "${gpu_uuid}" == GPU-* || "${gpu_uuid}" == MIG-* ]] || {
    echo "failed to resolve physical GPU UUID for ${text_device}" >&2
    exit 2
  }
  stage "resolved ${text_device} to physical GPU ${gpu_uuid}"

  while true; do
    local observation memory utilization total_memory
    observation="$(nvidia-smi --query-gpu=memory.used,utilization.gpu,memory.total \
      --format=csv,noheader,nounits -i "${gpu_uuid}" | head -n 1)"
    IFS=',' read -r memory utilization total_memory <<<"${observation}"
    memory="${memory//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    total_memory="${total_memory//[[:space:]]/}"
    if [[ ! "${total_memory}" =~ ^[0-9]+$ ]] || (( total_memory < gpu_min_total_memory_mib )); then
      echo "text GPU ${gpu_uuid} has insufficient or unknown total memory: ${total_memory:-unknown} MiB < ${gpu_min_total_memory_mib} MiB" >&2
      exit 2
    fi
    if [[ "${memory}" =~ ^[0-9]+$ && "${utilization}" =~ ^[0-9]+$ ]] \
      && (( memory <= gpu_idle_memory_mib && utilization <= gpu_idle_utilization )); then
      stage "text GPU ${gpu_index} is idle (${memory} MiB, ${utilization}% utilization)"
      return
    fi
    stage "waiting for text GPU ${gpu_index} (${memory:-unknown} MiB, ${utilization:-unknown}% utilization)"
    sleep "${poll_seconds}"
  done
}

verify_pair() {
  "${python_bin}" - "$1" "${2:-}" <<'PY'
import hashlib, json, sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
expected = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
sidecar = path.with_suffix(path.suffix + ".metadata.json")
if not path.is_file() or not sidecar.is_file():
    raise SystemExit(f"incomplete output pair: {path}, {sidecar}")
metadata = json.loads(sidecar.read_text(encoding="utf-8"))
digest = hashlib.sha256(path.read_bytes()).hexdigest()
output = metadata.get("output", {})
if output.get("sha256") != digest:
    raise SystemExit(f"output hash disagrees with sidecar: {path}")
rows = sum(bool(line.strip()) for line in path.open("rb"))
if output.get("entries") != rows:
    raise SystemExit(f"output row count disagrees with sidecar: {path}")
if expected is not None and rows != expected:
    raise SystemExit(f"unexpected row count for {path}: {rows} != {expected}")
PY
}

verify_quality_gate() {
  "${python_bin}" - "$1" "$2" "$3" <<'PY'
import hashlib, json, sys
from pathlib import Path

quality, responses, sample = map(lambda value: Path(value).resolve(), sys.argv[1:])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
report = json.loads(quality.read_text(encoding="utf-8"))
if report.get("quality_gate", {}).get("eligible") is not True:
    raise SystemExit("deterministic LLM quality gate is not eligible")
response_sha = sha(responses)
response_metadata_path = responses.with_suffix(responses.suffix + ".metadata.json")
response_metadata = json.loads(response_metadata_path.read_text(encoding="utf-8"))
sources = report.get("sources", {}).get("responses", [])
request_sha = response_metadata.get("requests", {}).get("sha256")
if not any(
    isinstance(row, dict)
    and row.get("sha256") == response_sha
    and row.get("metadata_sha256") == sha(response_metadata_path)
    and row.get("producer_identity_sha256")
    == response_metadata.get("producer_identity_sha256")
    and row.get("requests_sha256") == request_sha
    for row in sources
):
    raise SystemExit("LLM quality report is stale or bound to different responses")
if report.get("sources", {}).get("requests", {}).get("sha256") != request_sha:
    raise SystemExit("LLM quality report is bound to different requests")
bound_sample = report.get("review_sample", {})
if bound_sample.get("sha256") != sha(sample) or bound_sample.get("entries") != 1200:
    raise SystemExit("LLM review sample is incomplete or disagrees with its quality report")
PY
}

verify_expert_gate() {
  "${python_bin}" - "$1" "$2" "$3" "$4" "$5" <<'PY'
import hashlib, json, sys
from pathlib import Path

expert, verdicts, responses, quality, sample = map(
    lambda value: Path(value).resolve(), sys.argv[1:]
)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
report = json.loads(expert.read_text(encoding="utf-8"))
if report.get("status") != "approved":
    raise SystemExit("independent expert review has not approved V2")
bindings = report.get("bindings", {})
expected = {
    "responses_sha256": sha(responses),
    "quality_report_sha256": sha(quality),
    "review_sample_sha256": sha(sample),
    "verdicts_sha256": sha(verdicts),
}
if any(bindings.get(key) != value for key, value in expected.items()):
    raise SystemExit("independent expert review is stale or has a broken binding")
review = report.get("review", {})
if review.get("reviewed_unique_requests") != 1200:
    raise SystemExit("independent expert review does not cover exactly 1,200 requests")
if review.get("unresolved_critical_errors") != 0 or review.get("major_semantic_errors") != 0:
    raise SystemExit("independent expert review contains unresolved major errors")
PY
}

verify_stats() {
  "${python_bin}" - "${stats_root}" "${cached_manifest}" <<'PY'
import hashlib, json, sys
from pathlib import Path
import numpy as np

root, manifest = map(lambda value: Path(value).resolve(), sys.argv[1:])
metadata = json.loads((root / "stats.metadata.json").read_text(encoding="utf-8"))
if metadata.get("manifest_sha256") != hashlib.sha256(manifest.read_bytes()).hexdigest():
    raise SystemExit("stats are not bound to the V2 cached manifest")
for group, width in (("global_root", 5), ("local_root", 4), ("body", 364)):
    for name in ("mean.npy", "std.npy"):
        path = root / group / name
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (width,) or not np.isfinite(array).all():
            raise SystemExit(f"invalid stats array: {path}")
PY
}

status() {
  local completed=0 total=0 state="not-started"
  [[ -f "${requests}" ]] && total="$(wc -l < "${requests}")"
  if [[ -f "${responses}" ]]; then
    completed="$(wc -l < "${responses}")"
    state="llm-complete"
  elif [[ -f "${responses}.partial" ]]; then
    completed="$(wc -l < "${responses}.partial")"
    state="llm-partial"
  fi
  if [[ -f "${v2_final}/resource-state.json" ]]; then state="train-ready"; fi
  printf 'state=%s completed=%s total=%s building=%s final=%s\n' \
    "${state}" "${completed}" "${total}" "${v2_root}" "${v2_final}"
}

print_plan() {
  cat <<EOF
require-selected-final-response
audit-quality-gate
independent-expert-review-gate
build-raw-manifest
extract-llm-lane
wait-for-text-gpu
verify-v1-v2-embedding-canary
cache-llm2vec
compose-cached-manifest
fit-v2-stats
build-and-verify-reference-inventory
real-batch-preflight
validate-and-atomic-publish
EOF
}

if [[ "${command_name}" == "status" ]]; then status; exit 0; fi
if [[ "${command_name}" == "plan" ]]; then print_plan; exit 0; fi

mkdir -p "$(dirname -- "${log_file}")"
if [[ -f "${v2_final}/resource-state.json" ]]; then
  stage "already train-ready at ${v2_final}"
  exit 0
fi
mkdir -p "${v2_root}" "${provenance_root}"
exec 9>"${v2_root}/.bundle-build.lock"
if ! flock -n 9; then
  echo "another V2 bundle orchestrator already holds ${v2_root}/.bundle-build.lock" >&2
  exit 3
fi

bound_metadata="${responses}.metadata.json"
if [[ ! -f "${bound_metadata}" ]]; then
  bound_metadata="${responses}.partial.metadata.json"
fi
bound_model="mimo-v2.5-pro"
bound_base_url="https://api.xiaomimimo.com/v1"
if [[ -f "${bound_metadata}" ]]; then
  bound_model="$("${python_bin}" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m.get("model") or m["producer_identity"]["model"])' "${bound_metadata}")"
  bound_base_url="$("${python_bin}" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m.get("base_url") or m["producer_identity"]["base_url"])' "${bound_metadata}")"
fi
export PRODUCT_GRAPH_LLM_MODEL="${bound_model}"
export PRODUCT_GRAPH_LLM_BASE_URL="${bound_base_url}"
export KIMODO_V2_ROOT="${v2_root}"
export KIMODO_V1_ROOT="${v1_root}"
export KIMODO_LLM_BATCH_SIZE="${KIMODO_LLM_BATCH_SIZE:-8}"
export KIMODO_LLM_CONCURRENCY="${KIMODO_LLM_CONCURRENCY:-32}"
export KIMODO_LLM_RPM="${KIMODO_LLM_RPM:-90}"

while [[ ! -f "${responses}" ]]; do
  if pgrep -f "kimodo\.training\.llm_api_augmentation_cli.*--output ${responses}" >/dev/null; then
    stage "waiting for active LLM generator ($(wc -l < "${responses}.partial")/$(wc -l < "${requests}"))"
    sleep "${poll_seconds}"
    continue
  fi
  if [[ -z "${PRODUCT_GRAPH_LLM_API_KEY:-}" ]]; then
    echo "LLM generation is incomplete and PRODUCT_GRAPH_LLM_API_KEY is unavailable" >&2
    exit 4
  fi
  stage "resuming LLM generation from its validated partial"
  "${script_dir}/build_v2_llm.sh" generate 2>&1 | tee -a "${log_file}"
done
verify_pair "${responses}" "$(wc -l < "${requests}")"

if [[ ! -f "${quality_report}" ]]; then
  stage "auditing complete LLM output"
  "${script_dir}/build_v2_llm.sh" audit 2>&1 | tee -a "${log_file}"
fi
verify_quality_gate "${quality_report}" "${responses}" "${review_sample}"

if [[ ! -f "${expert_review}" ]]; then
  [[ -d "${expert_model}" ]] || { echo "independent expert model is missing: ${expert_model}" >&2; exit 5; }
  wait_for_text_gpu
  stage "independently reviewing 1,200 weighted/risk-stratified samples with local Qwen3-32B"
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 "${python_bin}" -m kimodo.training.expert_review_cli \
    --review-sample "${review_sample}" \
    --responses "${responses}" --quality-report "${quality_report}" \
    --model "${expert_model}" --device "${text_device}" \
    --batch-size "${KIMODO_EXPERT_BATCH_SIZE:-8}" \
    --verdicts "${expert_verdicts}" --output "${expert_review}" \
    2>&1 | tee -a "${log_file}"
fi
verify_expert_gate "${expert_review}" "${expert_verdicts}" "${responses}" \
  "${quality_report}" "${review_sample}"

if [[ ! -f "${raw_manifest}" ]]; then
  stage "building V2 raw manifest"
  "${script_dir}/build_v2_llm.sh" manifest 2>&1 | tee -a "${log_file}"
fi
verify_pair "${raw_manifest}" "${expected_entries}"
"${python_bin}" -m kimodo.training.v2_lineage_cli \
  --building-root "${v2_root}" --responses "${responses}" --through raw >/dev/null

if [[ ! -f "${llm_raw_manifest}" ]]; then
  stage "extracting the V2 LLM lane"
  "${python_bin}" -m kimodo.training.v2_cached_manifest_cli extract \
    --v2-raw-manifest "${raw_manifest}" --output "${llm_raw_manifest}" 2>&1 | tee -a "${log_file}"
fi
verify_pair "${llm_raw_manifest}"
"${python_bin}" -m kimodo.training.v2_lineage_cli \
  --building-root "${v2_root}" --responses "${responses}" --through llm_raw >/dev/null

if [[ ! -f "${llm_cached_manifest}" ]]; then
  for model_path in "${foundation_model}" "${mntp_model}" "${supervised_model}"; do
    [[ -d "${model_path}" ]] || { echo "LLM2Vec model directory is missing: ${model_path}" >&2; exit 5; }
  done
  wait_for_text_gpu
  stage "encoding new V2 texts with pinned local LLM2Vec on ${text_device}"
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 "${python_bin}" -m kimodo.training.text_cache_cli \
    --manifest "${llm_raw_manifest}" \
    --output-manifest "${llm_cached_manifest}" \
    --cache-dir "${v2_root}/text-cache-v2-llm" \
    --device "${text_device}" --encode-call-size "${KIMODO_TEXT_ENCODE_CALL_SIZE:-64}" \
    --canary-manifest "${v1_root}/train.cached.jsonl" \
    --canary-report "${embedding_canary}" --canary-count 16 \
    --foundation-model "${foundation_model}" --foundation-revision "${foundation_revision}" \
    --mntp-model "${mntp_model}" --mntp-revision "${mntp_revision}" \
    --supervised-model "${supervised_model}" --supervised-revision "${supervised_revision}" \
    2>&1 | tee -a "${log_file}"
fi
verify_pair "${llm_cached_manifest}" "$(wc -l < "${llm_raw_manifest}")"
"${python_bin}" -m kimodo.training.v2_lineage_cli \
  --building-root "${v2_root}" --responses "${responses}" --through llm_cached >/dev/null
[[ -f "${embedding_canary}" ]] || {
  echo "V1/V2 embedding numerical canary is missing: ${embedding_canary}" >&2
  exit 5
}

if [[ ! -f "${cached_manifest}" ]]; then
  stage "composing V1 cache and V2 LLM cache"
  "${python_bin}" -m kimodo.training.v2_cached_manifest_cli compose \
    --v2-raw-manifest "${raw_manifest}" \
    --v1-cached-manifest "${v1_root}/train.cached.jsonl" \
    --llm-cached-manifest "${llm_cached_manifest}" \
    --base-cache-dir text-cache-v1 --llm-cache-dir text-cache-v2-llm \
    --output "${cached_manifest}" 2>&1 | tee -a "${log_file}"
fi
verify_pair "${cached_manifest}" "${expected_entries}"
"${python_bin}" -m kimodo.training.v2_lineage_cli \
  --building-root "${v2_root}" --responses "${responses}" --through cached >/dev/null

if [[ ! -d "${stats_root}" ]]; then
  stats_building="${stats_root}.building"
  if [[ -d "${stats_building}" ]]; then
    echo "incomplete stats staging exists; inspect before retrying: ${stats_building}" >&2
    exit 6
  fi
  stage "fitting V2 normalization statistics with ${stats_workers} workers"
  "${python_bin}" -m kimodo.training.stats_cli \
    --manifest "${cached_manifest}" --output "${stats_building}" --split train \
    --skeleton-joints 30 --fps 30 --max-seconds 10 --min-frames 2 --seed 1234 \
    --num-workers "${stats_workers}" 2>&1 | tee -a "${log_file}"
  mv "${stats_building}" "${stats_root}"
fi
verify_stats

if [[ ! -f "${inventory}" ]]; then
  stage "building the full reference inventory"
  "${python_bin}" -m kimodo.training.reference_inventory_cli build \
    --manifest "${cached_manifest}" --output "${inventory}" 2>&1 | tee -a "${log_file}"
fi

if [[ ! -f "${preflight_report}" ]]; then
  stage "running one real CPU batch preflight"
  "${python_bin}" - "${preflight_paths}" "${cached_manifest}" "${inventory}" "${stats_root}" "${v2_root}" <<'PY'
import sys, yaml
from pathlib import Path
path, manifest, inventory, stats, root = map(Path, sys.argv[1:])
payload = {
    "schema_version": 1,
    "data": {"manifest": str(manifest), "reference_inventory": str(inventory)},
    "model": {"stats_path": str(stats), "checkpoint_dir": None, "checkpoint_weights": None},
    "runtime": {"output_dir": str(root / "preflight-run"), "resume": None},
}
path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY
  preflight_tmp="${preflight_report}.tmp.$$"
  "${python_bin}" -m kimodo.training.cli \
    --config "${repo_root}/configs/training/kimodo_soma_seed_v2_30k.yaml" \
    --paths "${preflight_paths}" --preflight >"${preflight_tmp}"
  "${python_bin}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["event"] == "kimodo_full_data_preflight_passed"' "${preflight_tmp}"
  mv "${preflight_tmp}" "${preflight_report}"
fi

rm -f -- "${preflight_paths}"
stage "verifying every inventory reference and atomically publishing"
"${python_bin}" -m kimodo.training.v2_bundle_publish_cli \
  --building-root "${v2_root}" --final-root "${v2_final}" \
  --manifest "${cached_manifest}" --inventory "${inventory}" --stats "${stats_root}" \
  --quality-report "${quality_report}" --responses "${responses}" \
  --response-selection "${response_selection}" \
  --expert-review "${expert_review}" --expert-verdicts "${expert_verdicts}" \
  --embedding-canary "${embedding_canary}" \
  --preflight-report "${preflight_report}" \
  --expected-entries "${expected_entries}" 2>&1 | tee -a "${log_file}"
stage "V2 train-ready bundle published at ${v2_final}"
