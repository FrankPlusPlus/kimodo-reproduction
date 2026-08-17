#!/usr/bin/env bash
# Experiment S variant: per-term dL/dα on 500k fork checkpoints (slopes only).
#   exec bash /home/share/yzt/kimodo-reproduction/scripts/company_run_body_rootcause_S_wd03_from500k.sh
set -euo pipefail

export KIMODO_CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
export KIMODO_STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
export PYTHONPATH="${KIMODO_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export KIMODO_SKIP_MANIFEST_PATH_STAT="${KIMODO_SKIP_MANIFEST_PATH_STAT:-1}"
export KIMODO_FEATURE_CACHE_INDEX_MODE="${KIMODO_FEATURE_CACHE_INDEX_MODE:-deterministic}"
export KIMODO_FEATURE_CACHE_DIR="${KIMODO_FEATURE_CACHE_DIR:-${KIMODO_STORAGE_ROOT}/feature-cache/v1}"
export KIMODO_PROBE_SKIP_TORCHRUN="${KIMODO_PROBE_SKIP_TORCHRUN:-1}"

if [[ -f "${KIMODO_CODE_ROOT}/scripts/load_kimodo_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${KIMODO_CODE_ROOT}/scripts/load_kimodo_env.sh"
  kimodo_load_env_files
fi

fork_run="${KIMODO_FORK_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-wd03-from500k}"
hostnet_run="${KIMODO_HOSTNET_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet}"
config="${KIMODO_PROBE_CONFIG:-${fork_run}/config.resolved.yaml}"
out_dir="${KIMODO_PROBE_OUT_DIR:-${KIMODO_STORAGE_ROOT}/diagnostics/body-rootcause-S-wd03-from500k}"

ckpt_500="${KIMODO_CKPT_500:-${hostnet_run}/checkpoints/step-000500000.pt}"
ckpt_520="${KIMODO_CKPT_520:-${fork_run}/checkpoints/step-000520000.pt}"
ckpt_540="${KIMODO_CKPT_540:-${fork_run}/checkpoints/step-000540000.pt}"
ckpt_550="${KIMODO_CKPT_550:-${fork_run}/checkpoints/step-000550000.pt}"

cd "${KIMODO_CODE_ROOT}"
python_bin="${KIMODO_PYTHON:-${KIMODO_CODE_ROOT}/.venv-feature-cache/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="${KIMODO_PYTHON:-python3}"
fi

mkdir -p "${out_dir}"
echo "{\"status\":\"launcher_mkdir\"}" > "${out_dir}/running.json"
for path in "${config}" "${ckpt_500}" "${ckpt_520}" "${ckpt_540}" "${ckpt_550}"; do
  if [[ ! -r "${path}" ]]; then
    echo "missing readable path: ${path}" >&2
    exit 2
  fi
done

echo "rootcause S wd03-from500k: out=${out_dir}"
exec "${python_bin}" "${KIMODO_CODE_ROOT}/scripts/diagnose_body_onset_path.py" \
  --config "${config}" \
  --checkpoints "${ckpt_500}" "${ckpt_520}" "${ckpt_540}" "${ckpt_550}" \
  --samples "${KIMODO_PROBE_SAMPLES:-4}" \
  --seed "${KIMODO_PROBE_SEED:-20260816}" \
  --constraint-step "${KIMODO_PROBE_CONSTRAINT_STEP:-750000}" \
  --healthy-step 500000 \
  --preflip-step 520000 \
  --flipped-step 540000 \
  --slopes-only \
  --device "${KIMODO_PROBE_DEVICE:-cuda:0}" \
  --output-dir "${out_dir}"
