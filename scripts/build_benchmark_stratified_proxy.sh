#!/usr/bin/env bash
set -euo pipefail

# Build a deterministic 10% stratified official benchmark subset with gt_motion.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_testsuite="${KIMODO_BENCHMARK_METADATA:?set KIMODO_BENCHMARK_METADATA to testsuite root}"
output_root="${KIMODO_BENCHMARK_STRATIFIED_ROOT:-/home/yezitao/PublicWorkspace/yzt/kimodo-benchmark-stratified-10pct}"
seed_dataset="${KIMODO_SEED_DATASET:-/storage/data/metaiot_data/yzt/seed/soma_uniform}"
legacy_proxy="${KIMODO_BENCHMARK_PROXY128:-/home/yezitao/PublicWorkspace/yzt/kimodo-benchmark-proxy-128}"
rate="${KIMODO_BENCHMARK_SUBSET_RATE:-0.10}"
seed="${KIMODO_BENCHMARK_SUBSET_SEED:-20260809}"
workers="${KIMODO_BENCHMARK_CREATE_WORKERS:-8}"

cd "${project_root}"
python_bin="${project_root}/.venv/bin/python"
export PYTHONPATH="${project_root}:${PYTHONPATH:-}"

"${python_bin}" -m kimodo.devtools.benchmark_subset_cli \
  --source-testsuite "${source_testsuite}" \
  --output "${output_root}" \
  --manifest "${output_root}/proxy_manifest.json" \
  --name stratified-10pct-v1 \
  --rate "${rate}" \
  --seed "${seed}" \
  --min-constraint 10 \
  --min-text2motion 40 \
  --gt-source-root "${legacy_proxy}" \
  --overwrite

"${python_bin}" benchmark/create_benchmark.py \
  "${output_root}" \
  --dataset "${seed_dataset}" \
  --workers "${workers}"

missing_gt="$(
  find "${output_root}" -name meta.json | while read -r meta; do
    dir="$(dirname "${meta}")"
    if [[ ! -s "${dir}/gt_motion.npz" ]]; then
      echo "${dir#${output_root}/}"
    fi
  done
)"
if [[ -n "${missing_gt}" ]]; then
  echo "subset build incomplete; missing gt_motion for:" >&2
  echo "${missing_gt}" | head >&2
  exit 1
fi

"${python_bin}" - <<'PY' "${output_root}"
import json, sys
from pathlib import Path
from kimodo.evaluation.eval_monitor_cli import benchmark_inventory_sha256
root = Path(sys.argv[1])
print(json.dumps({"benchmark_inventory_sha256": benchmark_inventory_sha256(root)}, indent=2))
PY

echo "stratified benchmark ready: ${output_root}/proxy_manifest.json"
