#!/usr/bin/env bash
# Experiment R (read-only): full-stack attn/ffn cosine+σ for L0–L15.
# Checkpoints: hostnet 400/500k, kf-smooth 650/690/695k, wd03 700/750/780/790k.
# Prefer 1x GPU. Start:
#   exec bash /home/share/yzt/kimodo-reproduction/scripts/company_run_body_rootcause_R_fullstack.sh
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

parent_run="${KIMODO_WD03_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-wd03-from650k}"
kf_run="${KIMODO_KF_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-kf-smooth-lr1e5}"
hostnet_run="${KIMODO_HOSTNET_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet}"
preserve_dir="${KIMODO_PRESERVE_DIR:-${KIMODO_STORAGE_ROOT}/preserved-pre-collapse/v2-1m-hostnet-wd03-from650k}"
config="${KIMODO_PROBE_CONFIG:-${parent_run}/config.resolved.yaml}"
out_dir="${KIMODO_PROBE_OUT_DIR:-${KIMODO_STORAGE_ROOT}/diagnostics/body-rootcause-R-fullstack}"

_resolve_ckpt() {
  local primary="$1"
  local fallback="$2"
  if [[ -r "${primary}" ]]; then
    echo "${primary}"
  else
    echo "${fallback}"
  fi
}

ckpt_400="${KIMODO_CKPT_400:-${hostnet_run}/checkpoints/step-000400000.pt}"
ckpt_500="${KIMODO_CKPT_500:-${hostnet_run}/checkpoints/step-000500000.pt}"
ckpt_650="${KIMODO_CKPT_650:-${kf_run}/checkpoints/step-000650000.pt}"
ckpt_690="${KIMODO_CKPT_690:-${kf_run}/checkpoints/step-000690000.pt}"
ckpt_695="${KIMODO_CKPT_695:-${kf_run}/checkpoints/step-000695000.pt}"
ckpt_700="$(_resolve_ckpt "${parent_run}/checkpoints/step-000700000.pt" "${preserve_dir}/step-000700000.pt")"
ckpt_750="$(_resolve_ckpt "${parent_run}/checkpoints/step-000750000.pt" "${preserve_dir}/step-000750000.pt")"
ckpt_780="$(_resolve_ckpt "${parent_run}/checkpoints/step-000780000.pt" "${preserve_dir}/step-000780000.pt")"
ckpt_790="$(_resolve_ckpt "${parent_run}/checkpoints/step-000790000.pt" "${preserve_dir}/step-000790000.pt")"

cd "${KIMODO_CODE_ROOT}"
python_bin="${KIMODO_PYTHON:-python3}"
if ! grep -q "full_stack_timeline" "${KIMODO_CODE_ROOT}/kimodo/training/body_residual_cancel_probe.py"; then
  echo "stale PVC code: missing full_stack_timeline" >&2
  exit 2
fi

mkdir -p "${out_dir}"
echo "{\"status\":\"launcher_mkdir\"}" > "${out_dir}/running.json"
for path in "${config}" "${ckpt_400}" "${ckpt_500}" "${ckpt_650}" "${ckpt_690}" "${ckpt_695}" "${ckpt_700}" "${ckpt_750}" "${ckpt_780}" "${ckpt_790}"; do
  if [[ ! -r "${path}" ]]; then
    echo "missing readable path: ${path}" >&2
    exit 2
  fi
done

echo "rootcause R fullstack: out=${out_dir}"
exec "${python_bin}" "${KIMODO_CODE_ROOT}/scripts/diagnose_body_residual_cancel.py" \
  --config "${config}" \
  --checkpoints "${ckpt_400}" "${ckpt_500}" "${ckpt_650}" "${ckpt_690}" "${ckpt_695}" \
    "${ckpt_700}" "${ckpt_750}" "${ckpt_780}" "${ckpt_790}" \
  --samples "${KIMODO_PROBE_SAMPLES:-4}" \
  --seed "${KIMODO_PROBE_SEED:-20260816}" \
  --constraint-steps "${KIMODO_PROBE_CONSTRAINT_STEP:-750000}" \
  --all-layers \
  --device "${KIMODO_PROBE_DEVICE:-auto}" \
  --output-dir "${out_dir}"
