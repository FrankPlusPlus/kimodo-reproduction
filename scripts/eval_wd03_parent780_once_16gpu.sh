#!/usr/bin/env bash
# Parent wd03 780k stratified-10pct. Generation already wrote 2269 motions;
# this run skips them and finishes embed/evaluate on rank 0.
#
# CREATE 1 instance x 1 GPU. Do not Restart the dead 2x8 job.
# Image: hub.inner.ai.kingsoft.com/hh-678395/kimodo-reproduction:v8
# Start box is sh:
#   exec bash /home/share/yzt/kimodo-reproduction/scripts/eval_wd03_parent780_once_16gpu.sh
set -euo pipefail

export KIMODO_CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
export KIMODO_STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
export PYTHONPATH="${KIMODO_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

export KIMODO_NNODES="${KIMODO_NNODES:-${PET_NNODES:-${NNODES:-1}}}"
export KIMODO_NPROC_PER_NODE="${KIMODO_NPROC_PER_NODE:-${PET_NPROC_PER_NODE:-${NPROC_PER_NODE:-auto}}}"
resolved_node_rank="${KIMODO_NODE_RANK:-${PET_NODE_RANK:-${NODE_RANK:-${JOB_COMPLETION_INDEX:-}}}}"
export KIMODO_NODE_RANK="${resolved_node_rank}"
export MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-}}"
export MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29500}}"

export KIMODO_EVAL_ASSET_ROOT="${KIMODO_EVAL_ASSET_ROOT:-${KIMODO_STORAGE_ROOT}/yezitao-kimodo-eval-v2}"
export KIMODO_MODEL_ROOT="${KIMODO_MODEL_ROOT:-${KIMODO_STORAGE_ROOT}/models}"
export HF_HOME="${HF_HOME:-${KIMODO_STORAGE_ROOT}/hf-cache}"

export KIMODO_TRAIN_RUN_DIR="${KIMODO_TRAIN_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-wd03-from650k}"
export KIMODO_EVAL_EXPORT_RUN_DIR="${KIMODO_EVAL_EXPORT_RUN_DIR:-${KIMODO_STORAGE_ROOT}/eval-exports/v2-1m-hostnet-wd03-from650k-step780k}"
export KIMODO_EVAL_ROOT="${KIMODO_EVAL_ROOT:-${KIMODO_STORAGE_ROOT}/eval-results/v2-1m-hostnet-wd03-from650k-parent780k-stratified10pct}"
export KIMODO_BENCHMARK_ROOT="${KIMODO_BENCHMARK_ROOT:-${KIMODO_EVAL_ASSET_ROOT}/benchmark/stratified-10pct}"
export KIMODO_OFFICIAL_BASELINE_SUMMARY="${KIMODO_OFFICIAL_BASELINE_SUMMARY:-${KIMODO_EVAL_ASSET_ROOT}/baselines/official-seed-v1.1/summary_rows.json}"
export KIMODO_PARENT_750_SUMMARY="${KIMODO_PARENT_750_SUMMARY:-${KIMODO_STORAGE_ROOT}/eval-results/v2-1m-hostnet-wd03-from650k-stratified10pct/step-000750000/summary_rows.json}"
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

STEP=780000
STEP_NAME="$(printf 'step-%09d' "${STEP}")"
checkpoint="${KIMODO_CKPT_780:-${KIMODO_TRAIN_RUN_DIR}/checkpoints/${STEP_NAME}.pt}"
bundle_dir="${KIMODO_EVAL_EXPORT_RUN_DIR}/exports/${STEP_NAME}"
final_dir="${KIMODO_EVAL_ROOT}/${STEP_NAME}"
building_dir="${KIMODO_EVAL_ROOT}/.${STEP_NAME}.building"
generated_dir="${building_dir}/generated"
summary_path="${building_dir}/summary_rows.json"

cd "${KIMODO_CODE_ROOT}"
python_bin="${KIMODO_EXPORT_PYTHON}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin="$(command -v python3 || command -v python)"
fi

if [[ "${KIMODO_NPROC_PER_NODE}" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    KIMODO_NPROC_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  else
    KIMODO_NPROC_PER_NODE=1
  fi
  export KIMODO_NPROC_PER_NODE
fi
for value_name in KIMODO_NNODES KIMODO_NPROC_PER_NODE MASTER_PORT; do
  value="${!value_name}"
  [[ "${value}" =~ ^[0-9]+$ ]] || { echo "${value_name} must be a non-negative integer; got ${value}" >&2; exit 2; }
done
if [[ "${KIMODO_NNODES}" == "1" ]]; then
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export KIMODO_NODE_RANK="${KIMODO_NODE_RANK:-0}"
fi
if [[ -z "${MASTER_ADDR}" ]]; then
  echo "MASTER_ADDR/PET_MASTER_ADDR is empty; worker cannot join rank0. Create a new job, do not Restart." >&2
  env | sort | grep -E '^(PET_|MASTER_|NNODES|NODE_RANK|RANK|WORLD|JOB_)' >&2 || true
  exit 2
fi
if [[ -z "${KIMODO_NODE_RANK}" ]]; then
  echo "No node rank was injected; expected KIMODO_NODE_RANK, PET_NODE_RANK, NODE_RANK, or JOB_COMPLETION_INDEX." >&2
  exit 2
fi
case "${KIMODO_NODE_RANK}" in
  ''|*[!0-9]*)
    echo "node rank must be a non-negative integer; got ${KIMODO_NODE_RANK}" >&2
    exit 2
    ;;
esac
if [ "${KIMODO_NNODES}" -lt 1 ] || [ "${KIMODO_NODE_RANK}" -ge "${KIMODO_NNODES}" ]; then
  echo "invalid topology: node_rank=${KIMODO_NODE_RANK}, nnodes=${KIMODO_NNODES}" >&2
  exit 2
fi
if ! grep -q "select_example_shard" "${KIMODO_CODE_ROOT}/benchmark/generate_eval.py"; then
  echo "generate_eval.py is missing sharded generation; PVC code is stale" >&2
  exit 2
fi
if ! grep -q "def select_example_shard" "${KIMODO_CODE_ROOT}/kimodo/evaluation/generate_shards.py"; then
  echo "generate_shards.py is missing; PVC code is stale" >&2
  exit 2
fi
if ! grep -q "def pin_local_cuda_device" "${KIMODO_CODE_ROOT}/kimodo/evaluation/rank_cuda.py"; then
  echo "rank_cuda.py is missing pin_local_cuda_device; PVC code is stale" >&2
  exit 2
fi
if ! grep -q "pin_local_cuda_device" "${KIMODO_CODE_ROOT}/scripts/generate_eval_rank.py"; then
  echo "generate_eval_rank.py is missing GPU pinning; PVC code is stale" >&2
  exit 2
fi
if [[ ! -r "${checkpoint}" ]]; then
  echo "Missing readable 780k parent checkpoint: ${checkpoint}" >&2
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
  if ! "${python_bin}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    echo "CUDA torch unavailable in this pod image." >&2
    echo "Recreate this 1xH200 with hub.inner.ai.kingsoft.com/hh-678395/kimodo-reproduction:v8" >&2
    exit 2
  fi
fi

echo "parent780 16gpu eval: hostname=$(hostname) node_rank=${KIMODO_NODE_RANK} nnodes=${KIMODO_NNODES} nproc=${KIMODO_NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "parent780 16gpu eval: checkpoint=${checkpoint}"
echo "parent780 16gpu eval: export=${KIMODO_EVAL_EXPORT_RUN_DIR}"
echo "parent780 16gpu eval: output=${KIMODO_EVAL_ROOT}"

if [[ -f "${final_dir}/complete.json" ]]; then
  echo "parent780 16gpu eval: already complete at ${final_dir}"
  exit 0
fi

mkdir -p "${KIMODO_EVAL_EXPORT_RUN_DIR}/exports" "${KIMODO_EVAL_ROOT}" "${generated_dir}"

if [[ "${KIMODO_NODE_RANK}" == "0" ]]; then
  if [[ -f "${bundle_dir}/model.pt" && -f "${bundle_dir}/config.yaml" && -d "${bundle_dir}/stats" ]]; then
    echo "parent780 16gpu eval: reuse existing export ${bundle_dir}"
  else
    "${python_bin}" "${KIMODO_CODE_ROOT}/scripts/export_trainer_checkpoint_bundle.py" \
      --checkpoint "${checkpoint}" \
      --resolved-config "${KIMODO_RESOLVED_CONFIG}" \
      --output-run-dir "${KIMODO_EVAL_EXPORT_RUN_DIR}" \
      --step "${STEP}"
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "${bundle_dir}/.export-ready"
  fi
fi

export_deadline=$((SECONDS + 900))
until [[ -f "${bundle_dir}/model.pt" && -f "${bundle_dir}/config.yaml" && -d "${bundle_dir}/stats" ]]; do
  if (( SECONDS >= export_deadline )); then
    echo "timed out waiting for 780k export bundle at ${bundle_dir}" >&2
    exit 2
  fi
  sleep 5
done
echo "parent780 16gpu eval: export bundle ready ${bundle_dir}"

generate=(
  -m torch.distributed.run
  --nnodes="${KIMODO_NNODES}"
  --nproc-per-node="${KIMODO_NPROC_PER_NODE}"
  --node-rank="${KIMODO_NODE_RANK}"
  --master-addr="${MASTER_ADDR}"
  --master-port="${MASTER_PORT}"
  "${KIMODO_CODE_ROOT}/scripts/generate_eval_rank.py"
  --benchmark "${KIMODO_BENCHMARK_ROOT}"
  --output "${generated_dir}"
  --checkpoint-bundle "${bundle_dir}"
  --batch_size "${KIMODO_EVAL_BATCH_SIZE}"
  --num_workers "${KIMODO_EVAL_WORKERS}"
  --diffusion_steps "${KIMODO_EVAL_DIFFUSION_STEPS}"
)
if [[ "${KIMODO_EVAL_TEXT_ENCODER_FP32}" == "1" ]]; then
  generate+=(--text_encoder_fp32)
fi
echo "parent780 16gpu eval: sharded generate world=$((KIMODO_NNODES * KIMODO_NPROC_PER_NODE))"
"${python_bin}" "${generate[@]}"

if [[ "${KIMODO_NODE_RANK}" != "0" ]]; then
  score_wait="${KIMODO_EVAL_SCORE_WAIT_SECONDS:-7200}"
  echo "parent780 16gpu eval: node ${KIMODO_NODE_RANK} finished generate; waiting for rank0 to score"
  if [[ "${score_wait}" -le 0 ]]; then
    exit 0
  fi
  score_deadline=$((SECONDS + score_wait))
  until [[ -f "${final_dir}/complete.json" ]]; do
    if (( SECONDS >= score_deadline )); then
      echo "timed out waiting for rank0 summary at ${final_dir}/complete.json" >&2
      exit 2
    fi
    sleep 15
  done
  echo "parent780 16gpu eval: rank0 complete, worker exit"
  exit 0
fi

embed=(
  benchmark/embed_folder.py
  "${generated_dir}"
  --device cuda
)
if [[ "${KIMODO_EVAL_TEXT_ENCODER_FP32}" == "1" ]]; then
  embed+=(--text_encoder_fp32)
fi
"${python_bin}" "${embed[@]}"

evaluate=(
  benchmark/evaluate_folder.py
  "${generated_dir}"
  --device cuda
)
if [[ "${KIMODO_EVAL_PAPER_PROTOCOL}" == "1" ]]; then
  evaluate+=(--paper-protocol)
fi
"${python_bin}" "${evaluate[@]}"

"${python_bin}" benchmark/parse_folder.py "${generated_dir}" --output "${summary_path}"
printf '{"step":780000,"event":"parent780_16gpu"}\n' > "${building_dir}/complete.json"
if [[ -d "${final_dir}" ]]; then
  echo "refusing to replace existing ${final_dir}" >&2
  exit 2
fi
mv "${building_dir}" "${final_dir}"
echo "parent780 16gpu eval: wrote ${final_dir}/summary_rows.json"

"${python_bin}" - "${final_dir}/summary_rows.json" "${KIMODO_PARENT_750_SUMMARY}" "${KIMODO_OFFICIAL_BASELINE_SUMMARY:-}" <<'PY'
import json
import sys
from pathlib import Path

KEYS = (
    "Full-Body Pos (gen, cm)",
    "End-Effector Pos (gen, cm)",
    "2D Root Pos (gen, cm)",
    "Skate (gen, cm/s)",
    "FID gen-GT",
    "R@3 (gen)",
)

def walk(obj, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}/{key}" if prefix else str(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                yield path, float(value)
            else:
                yield from walk(value, path)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from walk(value, f"{prefix}[{index}]")

def pick(summary):
    tables = (summary or {}).get("tables") or {}
    block = tables.get("content") if isinstance(tables, dict) else None
    highlights = {}
    if not isinstance(block, dict):
        return highlights
    preferred = []
    fallback = []
    for path, value in walk(block, "content"):
        for key in KEYS:
            if key not in path:
                continue
            bucket = preferred if ("text_following[0]" in path or "constraints[0]" in path) else fallback
            bucket.append((key, value))
    for key, value in preferred + fallback:
        highlights.setdefault(key, value)
    return highlights

def load(path):
    p = Path(path)
    if not path or not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

print("parent 780k content:", pick(load(sys.argv[1])))
if len(sys.argv) > 2:
    parent = load(sys.argv[2])
    if parent:
        print("parent 750k content:", pick(parent))
if len(sys.argv) > 3:
    official = load(sys.argv[3])
    if official:
        print("official v1.1 content:", pick(official))
PY
