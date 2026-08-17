#!/usr/bin/env bash
# Re-score parent 780k with the same TMR text gallery as parent 750k.
# Motions stay; only text_embedding.npy is replaced, then evaluate/parse.
# GT motion embeddings already match 750k; mismatched text made R@3/FID unusable.
#
# CREATE 1 instance x 1 GPU. Do not Restart the dead 2x8 job.
# Image: hub.inner.ai.kingsoft.com/hh-678395/kimodo-reproduction:v8
# Start box is sh:
#   exec bash /home/share/yzt/kimodo-reproduction/scripts/eval_wd03_parent780_rescore_text.sh
set -euo pipefail

export KIMODO_CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
export KIMODO_STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
export PYTHONPATH="${KIMODO_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

export KIMODO_EVAL_ASSET_ROOT="${KIMODO_EVAL_ASSET_ROOT:-${KIMODO_STORAGE_ROOT}/yezitao-kimodo-eval-v2}"
export KIMODO_MODEL_ROOT="${KIMODO_MODEL_ROOT:-${KIMODO_STORAGE_ROOT}/models}"
export HF_HOME="${HF_HOME:-${KIMODO_STORAGE_ROOT}/hf-cache}"
export TEXT_ENCODER_MODE="${TEXT_ENCODER_MODE:-local}"
export KIMODO_LLM2VEC_FOUNDATION="${KIMODO_LLM2VEC_FOUNDATION:-${KIMODO_MODEL_ROOT}/llm2vec/foundation}"
export KIMODO_LLM2VEC_MNTP="${KIMODO_LLM2VEC_MNTP:-${KIMODO_MODEL_ROOT}/llm2vec/mntp-adapter}"
export KIMODO_LLM2VEC_SUPERVISED="${KIMODO_LLM2VEC_SUPERVISED:-${KIMODO_MODEL_ROOT}/llm2vec/supervised-adapter}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${KIMODO_MODEL_ROOT}/checkpoints}"
export LOCAL_CACHE="${LOCAL_CACHE:-True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
unset TEXT_ENCODER_DEVICE

export KIMODO_EVAL_PAPER_PROTOCOL="${KIMODO_EVAL_PAPER_PROTOCOL:-1}"
export KIMODO_EVAL_TEXT_ENCODER_FP32="${KIMODO_EVAL_TEXT_ENCODER_FP32:-1}"
export KIMODO_EXPORT_PYTHON="${KIMODO_EXPORT_PYTHON:-python3}"

generated="${KIMODO_780_GENERATED:-${KIMODO_STORAGE_ROOT}/eval-results/v2-1m-hostnet-wd03-from650k-parent780k-stratified10pct/step-000780000/generated}"
source_text="${KIMODO_750_GENERATED:-${KIMODO_STORAGE_ROOT}/eval-results/v2-1m-hostnet-wd03-from650k-stratified10pct/step-000750000/generated}"
final_dir="$(dirname "${generated}")"
summary_path="${final_dir}/summary_rows.json"
parent_750_summary="${KIMODO_PARENT_750_SUMMARY:-${KIMODO_STORAGE_ROOT}/eval-results/v2-1m-hostnet-wd03-from650k-stratified10pct/step-000750000/summary_rows.json}"
official_summary="${KIMODO_OFFICIAL_BASELINE_SUMMARY:-${KIMODO_EVAL_ASSET_ROOT}/baselines/official-seed-v1.1/summary_rows.json}"

if [[ ! -d "${generated}" ]]; then
  echo "Missing 780k generated tree: ${generated}" >&2
  exit 2
fi
if [[ ! -d "${source_text}" ]]; then
  echo "Missing 750k generated tree to copy text embeddings from: ${source_text}" >&2
  exit 2
fi
motion_count="$(find "${generated}" -name motion.npz | wc -l | tr -d ' ')"
if [[ "${motion_count}" -lt 2269 ]]; then
  echo "780k generated tree is incomplete: motion.npz=${motion_count}" >&2
  exit 2
fi

python_bin="${KIMODO_EXPORT_PYTHON}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin="$(command -v python3 || command -v python)"
fi
if [[ "${KIMODO_EVAL_SKIP_CUDA_CHECK:-0}" != "1" ]]; then
  if ! "${python_bin}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    echo "CUDA torch unavailable in this pod image." >&2
    echo "Recreate this 1xH200 with hub.inner.ai.kingsoft.com/hh-678395/kimodo-reproduction:v8" >&2
    exit 2
  fi
fi

echo "parent780 rescore: generated=${generated}"
echo "parent780 rescore: text source=${source_text}"
echo "parent780 rescore: motions=${motion_count}"

copied="$("${python_bin}" - "${source_text}" "${generated}" <<'PY'
import hashlib
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1]).resolve()
dst = Path(sys.argv[2]).resolve()
copied = 0
missing = []
for src_path in sorted(src.rglob("text_embedding.npy")):
    rel = src_path.relative_to(src)
    dst_path = dst / rel
    if not dst_path.parent.is_dir():
        missing.append(str(rel))
        continue
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    copied += 1
if missing:
    raise SystemExit(f"780k tree missing {len(missing)} sample dirs, e.g. {missing[0]}")
# Confirm a retrieval-critical subset now matches the 750k gallery.
overview_src = sorted((src / "content/text2motion/overview").rglob("text_embedding.npy"))
same = 0
for path in overview_src:
    rel = path.relative_to(src)
    left = hashlib.sha256(path.read_bytes()).hexdigest()
    right = hashlib.sha256((dst / rel).read_bytes()).hexdigest()
    if left == right:
        same += 1
if not overview_src or same != len(overview_src):
    raise SystemExit(f"overview text copy mismatch: {same}/{len(overview_src)}")
print(copied)
PY
)"
echo "parent780 rescore: copied ${copied} text embeddings from 750k"

if [[ -f "${summary_path}" ]]; then
  cp "${summary_path}" "${final_dir}/summary_rows.pre-rescore.json"
fi

cd "${KIMODO_CODE_ROOT}"
evaluate=(
  benchmark/evaluate_folder.py
  "${generated}"
  --device cuda
)
if [[ "${KIMODO_EVAL_PAPER_PROTOCOL}" == "1" ]]; then
  evaluate+=(--paper-protocol)
fi
"${python_bin}" "${evaluate[@]}"
"${python_bin}" benchmark/parse_folder.py "${generated}" --output "${summary_path}"
printf '{"step":780000,"event":"parent780_rescore_text"}\n' > "${final_dir}/complete.json"
echo "parent780 rescore: wrote ${summary_path}"

"${python_bin}" - "${summary_path}" "${parent_750_summary}" "${official_summary:-}" <<'PY'
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
    retrieval = ((summary or {}).get("paper_protocol") or {}).get("retrieval") or []
    for row in retrieval:
        if row.get("split") == "content" and row.get("category") == "overview":
            cols = row.get("paper_reported_columns") or {}
            highlights["R@3 GT"] = cols.get("R@3_ground_truth_percent")
            break
    return highlights

def load(path):
    p = Path(path)
    if not path or not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

print("parent 780k rescore content:", pick(load(sys.argv[1])))
if len(sys.argv) > 2:
    parent = load(sys.argv[2])
    if parent:
        print("parent 750k content:", pick(parent))
if len(sys.argv) > 3:
    official = load(sys.argv[3])
    if official:
        print("official v1.1 content:", pick(official))
PY
