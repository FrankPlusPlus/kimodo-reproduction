#!/usr/bin/env bash
# Causal σ cause: 750k vs 790k, scale vs direction. kimodo-dev GPU OK.
#   exec bash /home/share/yzt/kimodo-reproduction/scripts/company_run_body_sigma_cause.sh
set -euo pipefail

export KIMODO_CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
export KIMODO_STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
export PYTHONPATH="${KIMODO_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export KIMODO_SKIP_MANIFEST_PATH_STAT="${KIMODO_SKIP_MANIFEST_PATH_STAT:-1}"
export KIMODO_FEATURE_CACHE_INDEX_MODE="${KIMODO_FEATURE_CACHE_INDEX_MODE:-deterministic}"
export KIMODO_FEATURE_CACHE_DIR="${KIMODO_FEATURE_CACHE_DIR:-${KIMODO_STORAGE_ROOT}/feature-cache/v1}"

if [[ -f "${KIMODO_CODE_ROOT}/scripts/load_kimodo_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${KIMODO_CODE_ROOT}/scripts/load_kimodo_env.sh"
  kimodo_load_env_files
fi

wd03="${KIMODO_WD03_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-wd03-from650k}"
preserve="${KIMODO_PRESERVE_DIR:-${KIMODO_STORAGE_ROOT}/preserved-pre-collapse/v2-1m-hostnet-wd03-from650k}"
config="${KIMODO_PROBE_CONFIG:-${wd03}/config.resolved.yaml}"
out_dir="${KIMODO_PROBE_OUT_DIR:-${KIMODO_STORAGE_ROOT}/diagnostics/body-sigma-cause}"

_resolve() { if [[ -r "$1" ]]; then echo "$1"; else echo "$2"; fi; }
ckpt_750="$(_resolve "${wd03}/checkpoints/step-000750000.pt" "${preserve}/step-000750000.pt")"
ckpt_790="$(_resolve "${wd03}/checkpoints/step-000790000.pt" "${preserve}/step-000790000.pt")"

cd "${KIMODO_CODE_ROOT}"
python_bin="${KIMODO_PYTHON:-/home/jovyan/.venv-kimodo-demo/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="${KIMODO_CODE_ROOT}/.venv-feature-cache/bin/python"
fi

mkdir -p "${out_dir}"
echo "{\"status\":\"launcher_mkdir\"}" > "${out_dir}/running.json"
for path in "${config}" "${ckpt_750}" "${ckpt_790}"; do
  if [[ ! -r "${path}" ]]; then
    echo "missing readable path: ${path}" >&2
    exit 2
  fi
done

echo "sigma-cause: out=${out_dir}"
exec "${python_bin}" "${KIMODO_CODE_ROOT}/scripts/diagnose_body_sigma_cause.py" \
  --config "${config}" \
  --healthy-checkpoint "${ckpt_750}" \
  --crashed-checkpoint "${ckpt_790}" \
  --samples 4 \
  --seed 20260816 \
  --constraint-step 750000 \
  --device "${KIMODO_PROBE_DEVICE:-cuda:0}" \
  --output-dir "${out_dir}"
