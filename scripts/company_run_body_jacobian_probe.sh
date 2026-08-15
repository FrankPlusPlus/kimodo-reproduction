#!/usr/bin/env bash
# 1-process read-only body Jacobian probe. Do not submit this as the 2x8
# training job: it does not use torchrun, NCCL, or MASTER_ADDR.
#
# UI: 1 instance x 1 GPU, host network optional, start box is sh:
#   exec bash /home/share/yzt/kimodo-reproduction/scripts/company_run_body_jacobian_probe.sh
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

kf_run="${KIMODO_KF_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-kf-smooth-lr1e5}"
k7_run="${KIMODO_K7_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-k7-from690k}"
config="${KIMODO_PROBE_CONFIG:-${kf_run}/config.resolved.yaml}"
out_dir="${KIMODO_PROBE_OUT_DIR:-${KIMODO_STORAGE_ROOT}/diagnostics/body-jacobian-650-690-696}"
ckpt_650="${KIMODO_CKPT_650:-${kf_run}/checkpoints/step-000650000.pt}"
ckpt_690="${KIMODO_CKPT_690:-${kf_run}/checkpoints/step-000690000.pt}"
ckpt_696="${KIMODO_CKPT_696:-${k7_run}/checkpoints/step-000696000.pt}"

cd "${KIMODO_CODE_ROOT}"
mkdir -p "${out_dir}"
for path in "${config}" "${ckpt_650}" "${ckpt_690}" "${ckpt_696}"; do
  if [[ ! -r "${path}" ]]; then
    echo "missing readable path: ${path}" >&2
    exit 2
  fi
done

python_bin="${KIMODO_PYTHON:-python3}"
echo "body jacobian probe: python=${python_bin} device=${KIMODO_PROBE_DEVICE:-auto}"
echo "config=${config}"
echo "ckpts=${ckpt_650} ${ckpt_690} ${ckpt_696}"
echo "out=${out_dir}/probe.json"

exec "${python_bin}" "${KIMODO_CODE_ROOT}/scripts/diagnose_body_jacobian.py" \
  --config "${config}" \
  --checkpoints "${ckpt_650}" "${ckpt_690}" "${ckpt_696}" \
  --samples "${KIMODO_PROBE_SAMPLES:-4}" \
  --pair-samples "${KIMODO_PROBE_PAIR_SAMPLES:-4}" \
  --seed "${KIMODO_PROBE_SEED:-20260815}" \
  --constraint-step "${KIMODO_PROBE_CONSTRAINT_STEP:-690000}" \
  --device "${KIMODO_PROBE_DEVICE:-auto}" \
  --output "${out_dir}/probe.json"
