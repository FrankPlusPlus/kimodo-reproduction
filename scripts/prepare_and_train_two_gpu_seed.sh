#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
python_bin="${KIMODO_PYTHON:-${project_root}/.venv/bin/python}"
training_data="${KIMODO_TRAINING_DATA:-/home/yezitao/PublicWorkspace/yzt/kimodo-training-data}"
model_root="${KIMODO_MODEL_ROOT:-/home/yezitao/data/yzt/kimodo-repro/models}"
raw_manifest="${training_data}/train.raw.jsonl"
cached_manifest="${training_data}/train.cached.jsonl"
cache_dir="${training_data}/text-cache"
stats_dir="${training_data}/stats/repro-soma30-30fps"
inventory="${training_data}/train.cached.references.jsonl"
model_lock="${project_root}/configs/models.server.lock.json"
text_device="${KIMODO_TEXT_DEVICE:-cuda:0}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python environment is missing or not executable: ${python_bin}" >&2
  exit 2
fi
if [[ ! "${CUDA_VISIBLE_DEVICES:-}" =~ ^[^,[:space:]]+,[^,[:space:]]+$ ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to exactly the two devices allocated to this tenant." >&2
  exit 2
fi

required_files=(
  "${raw_manifest}"
  "${raw_manifest}.metadata.json"
  "${model_lock}"
  "${model_root}/llm2vec/foundation/model-00001-of-00004.safetensors"
  "${model_root}/llm2vec/foundation/model-00002-of-00004.safetensors"
  "${model_root}/llm2vec/foundation/model-00003-of-00004.safetensors"
  "${model_root}/llm2vec/foundation/model-00004-of-00004.safetensors"
  "${model_root}/llm2vec/mntp-adapter/adapter_model.safetensors"
  "${model_root}/llm2vec/supervised-adapter/adapter_model.safetensors"
)
for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required training asset is missing: ${path}" >&2
    exit 2
  fi
done

"${python_bin}" - <<'PY'
import torch

count = torch.cuda.device_count() if torch.cuda.is_available() else 0
if count != 2:
    raise SystemExit(f"Preparation requires exactly two visible allocated GPUs; torch sees {count}")
print(f"GPU preflight passed: {count} visible devices", flush=True)
PY

mkdir -p "${training_data}" "${cache_dir}" "$(dirname -- "${stats_dir}")"

cached_sidecar="${cached_manifest}.metadata.json"
if [[ ! -e "${cached_manifest}" && ! -e "${cached_sidecar}" ]]; then
  echo "[1/3] Building deterministic LLM2Vec text cache on ${text_device}"
  "${python_bin}" -m kimodo.training.text_cache_cli \
    --manifest "${raw_manifest}" \
    --output-manifest "${cached_manifest}" \
    --cache-dir "${cache_dir}" \
    --provider local \
    --device "${text_device}" \
    --model-lock "${model_lock}" \
    --foundation-model "${model_root}/llm2vec/foundation" \
    --foundation-revision 53346005fb0ef11d3b6a83b12c895cca40156b6c \
    --mntp-model "${model_root}/llm2vec/mntp-adapter" \
    --mntp-revision 31474e395ada192e8ed1586db6be79fb3b70c9c0 \
    --supervised-model "${model_root}/llm2vec/supervised-adapter" \
    --supervised-revision baa8ebf04a1c2500e61288e7dad65e8ae42601a7
elif [[ ! -f "${cached_manifest}" || ! -f "${cached_sidecar}" ]]; then
  echo "Text-cache manifest/sidecar is orphaned; inspect before retrying: ${cached_manifest}" >&2
  exit 2
else
  echo "[1/3] Reusing and validating existing text-cache manifest"
  "${python_bin}" - "${raw_manifest}" "${cached_manifest}" "${cached_sidecar}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

raw, cached, sidecar_path = map(Path, sys.argv[1:])
sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
if sidecar.get("schema_version") != 3:
    raise SystemExit("Unsupported text-cache sidecar schema")
if sidecar.get("source_manifest_sha256") != sha256(raw):
    raise SystemExit("Raw manifest differs from the text-cache provenance")
output = sidecar.get("output", {})
if Path(output.get("path", "")).resolve() != cached.resolve():
    raise SystemExit("Text-cache sidecar points to a different output path")
if output.get("sha256") != sha256(cached):
    raise SystemExit("Text-cache manifest differs from its sidecar")
print("Existing text-cache manifest provenance passed", flush=True)
PY
fi

stats_metadata="${stats_dir}/stats.metadata.json"
stats_files=(
  "${stats_dir}/global_root/mean.npy" "${stats_dir}/global_root/std.npy"
  "${stats_dir}/local_root/mean.npy" "${stats_dir}/local_root/std.npy"
  "${stats_dir}/body/mean.npy" "${stats_dir}/body/std.npy"
  "${stats_metadata}"
)
stats_complete=true
for path in "${stats_files[@]}"; do
  [[ -f "${path}" ]] || stats_complete=false
done
if [[ ! -e "${stats_dir}" ]]; then
  echo "[2/3] Computing deterministic normalization statistics"
  "${python_bin}" -m kimodo.training.stats_cli \
    --manifest "${cached_manifest}" \
    --output "${stats_dir}" \
    --split train \
    --skeleton-joints 30 \
    --fps 30
elif [[ "${stats_complete}" != true ]]; then
  echo "Normalization stats directory is incomplete; inspect before retrying: ${stats_dir}" >&2
  exit 2
else
  echo "[2/3] Reusing and validating existing normalization statistics"
  "${python_bin}" - "${cached_manifest}" "${stats_dir}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

manifest, root = map(Path, sys.argv[1:])
digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
metadata = json.loads((root / "stats.metadata.json").read_text(encoding="utf-8"))
if metadata.get("manifest_sha256") != digest:
    raise SystemExit("Stats were fitted from a different training manifest")
for group, width in (("global_root", 5), ("local_root", 4), ("body", 364)):
    for name in ("mean.npy", "std.npy"):
        array = np.load(root / group / name, allow_pickle=False)
        if array.shape != (width,) or not np.isfinite(array).all():
            raise SystemExit(f"Invalid stats array: {group}/{name}")
print("Existing normalization statistics passed", flush=True)
PY
fi

inventory_metadata="${inventory}.metadata.json"
if [[ ! -e "${inventory}" && ! -e "${inventory_metadata}" ]]; then
  echo "[3/3] Building and fully verifying reference inventory"
  "${python_bin}" -m kimodo.training.reference_inventory_cli build \
    --manifest "${cached_manifest}" \
    --output "${inventory}"
  "${python_bin}" -m kimodo.training.reference_inventory_cli verify \
    --manifest "${cached_manifest}" \
    --inventory "${inventory}"
elif [[ ! -f "${inventory}" || ! -f "${inventory_metadata}" ]]; then
  echo "Reference inventory/metadata is orphaned; inspect before retrying: ${inventory}" >&2
  exit 2
else
  echo "[3/3] Reusing and validating existing reference inventory identity"
  "${python_bin}" - "${cached_manifest}" "${inventory}" <<'PY'
import sys
from kimodo.training.reference_inventory import load_inventory_summary

summary = load_inventory_summary(sys.argv[1], sys.argv[2])
print(f"Reference inventory identity passed: {summary['reference_count']} files", flush=True)
PY
fi

echo "Preparation complete: public BONES-SEED engineering reproduction is trainable."
if [[ "${KIMODO_PREPARE_ONLY:-0}" == 1 ]]; then
  exit 0
fi

export KIMODO_TRAINING_DATA="${training_data}"
export KIMODO_LOCAL_ROOT="${project_root}"
exec "${script_dir}/train_two_gpu_seed.sh" "$@"
