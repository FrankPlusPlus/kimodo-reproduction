#!/usr/bin/env bash
# Read-only pre-crash probe on wd03-from650k: 750k (healthy) vs 780k
# (L15 just ticking) vs 790k (takeoff). Same frozen batch. Two constraint
# clocks (750k K≈10 and 790k K≈12) to split "weights already sharp" from
# "this step's curriculum is harder".
#
# 1 instance x 1 GPU. Do not attach to the 16-GPU trainer. Do not use
# kimodo-dev (demo owns that H200). Start box is sh:
#   exec bash /home/share/yzt/kimodo-reproduction/scripts/company_run_body_jacobian_probe_wd03_precrash.sh
set -euo pipefail

export KIMODO_CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
export KIMODO_STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
export PYTHONPATH="${KIMODO_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export KIMODO_SKIP_MANIFEST_PATH_STAT="${KIMODO_SKIP_MANIFEST_PATH_STAT:-1}"
export KIMODO_FEATURE_CACHE_INDEX_MODE="${KIMODO_FEATURE_CACHE_INDEX_MODE:-deterministic}"
export KIMODO_FEATURE_CACHE_DIR="${KIMODO_FEATURE_CACHE_DIR:-${KIMODO_STORAGE_ROOT}/feature-cache/v1}"

node_rank="${KIMODO_NODE_RANK:-${PET_NODE_RANK:-${NODE_RANK:-${JOB_COMPLETION_INDEX:-0}}}}"
if [[ "${node_rank}" != "0" ]]; then
  echo "body jacobian probe is 1-process; skipping node rank ${node_rank}"
  exit 0
fi

run_dir="${KIMODO_WD03_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-wd03-from650k}"
config="${KIMODO_PROBE_CONFIG:-${run_dir}/config.resolved.yaml}"
out_dir="${KIMODO_PROBE_OUT_DIR:-${KIMODO_STORAGE_ROOT}/diagnostics/body-jacobian-wd03-750-780-790}"
ckpt_750="${KIMODO_CKPT_750:-${run_dir}/checkpoints/step-000750000.pt}"
ckpt_780="${KIMODO_CKPT_780:-${run_dir}/checkpoints/step-000780000.pt}"
ckpt_790="${KIMODO_CKPT_790:-${run_dir}/checkpoints/step-000790000.pt}"

cd "${KIMODO_CODE_ROOT}"
mkdir -p "${out_dir}"
for path in "${config}" "${ckpt_750}" "${ckpt_780}" "${ckpt_790}"; do
  if [[ ! -r "${path}" ]]; then
    echo "missing readable path: ${path}" >&2
    exit 2
  fi
done

python_bin="${KIMODO_PYTHON:-python3}"
echo "wd03 precrash probe: python=${python_bin} device=${KIMODO_PROBE_DEVICE:-auto}"
echo "config=${config}"
echo "ckpts=${ckpt_750} ${ckpt_780} ${ckpt_790}"

for constraint_step in 750000 790000; do
  out="${out_dir}/probe-c${constraint_step}.json"
  echo "constraint_step=${constraint_step} out=${out}"
  "${python_bin}" "${KIMODO_CODE_ROOT}/scripts/diagnose_body_jacobian.py" \
    --config "${config}" \
    --checkpoints "${ckpt_750}" "${ckpt_780}" "${ckpt_790}" \
    --samples "${KIMODO_PROBE_SAMPLES:-4}" \
    --pair-samples "${KIMODO_PROBE_PAIR_SAMPLES:-4}" \
    --seed "${KIMODO_PROBE_SEED:-20260816}" \
    --constraint-step "${constraint_step}" \
    --device "${KIMODO_PROBE_DEVICE:-auto}" \
    --output "${out}"
done
echo "wd03 precrash probe done: ${out_dir}"
