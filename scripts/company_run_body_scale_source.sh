#!/usr/bin/env bash
# Scale-source probe: weights of all 16 body layers + existing residual-cancel JSON.
# GPU optional (weights are CPU). kimodo-dev is allowed.
#   exec bash /home/share/yzt/kimodo-reproduction/scripts/company_run_body_scale_source.sh
set -euo pipefail

export KIMODO_CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
export KIMODO_STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
export PYTHONPATH="${KIMODO_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

if [[ -f "${KIMODO_CODE_ROOT}/scripts/load_kimodo_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${KIMODO_CODE_ROOT}/scripts/load_kimodo_env.sh"
  kimodo_load_env_files
fi

hostnet="${KIMODO_HOSTNET_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet}"
fork500="${KIMODO_FORK_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-wd03-from500k}"
kf="${KIMODO_KF_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-kf-smooth-lr1e5}"
wd03="${KIMODO_WD03_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-wd03-from650k}"
preserve="${KIMODO_PRESERVE_DIR:-${KIMODO_STORAGE_ROOT}/preserved-pre-collapse/v2-1m-hostnet-wd03-from650k}"
out_dir="${KIMODO_PROBE_OUT_DIR:-${KIMODO_STORAGE_ROOT}/diagnostics/body-scale-source}"

_resolve() {
  if [[ -r "$1" ]]; then
    echo "$1"
  else
    echo "$2"
  fi
}

ckpt_400="${hostnet}/checkpoints/step-000400000.pt"
ckpt_500="${hostnet}/checkpoints/step-000500000.pt"
ckpt_550="${fork500}/checkpoints/step-000550000.pt"
ckpt_650="${kf}/checkpoints/step-000650000.pt"
ckpt_690="${kf}/checkpoints/step-000690000.pt"
ckpt_695="${kf}/checkpoints/step-000695000.pt"
ckpt_700="$(_resolve "${wd03}/checkpoints/step-000700000.pt" "${preserve}/step-000700000.pt")"
ckpt_750="$(_resolve "${wd03}/checkpoints/step-000750000.pt" "${preserve}/step-000750000.pt")"
ckpt_780="$(_resolve "${wd03}/checkpoints/step-000780000.pt" "${preserve}/step-000780000.pt")"
ckpt_790="$(_resolve "${wd03}/checkpoints/step-000790000.pt" "${preserve}/step-000790000.pt")"

cd "${KIMODO_CODE_ROOT}"
python_bin="${KIMODO_PYTHON:-/home/jovyan/.venv-kimodo-demo/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="${KIMODO_CODE_ROOT}/.venv-feature-cache/bin/python"
fi

mkdir -p "${out_dir}"
echo "{\"status\":\"launcher_mkdir\"}" > "${out_dir}/running.json"
for path in "${ckpt_400}" "${ckpt_500}" "${ckpt_550}" "${ckpt_650}" "${ckpt_690}" "${ckpt_695}" "${ckpt_700}" "${ckpt_750}" "${ckpt_780}" "${ckpt_790}"; do
  if [[ ! -r "${path}" ]]; then
    echo "missing readable path: ${path}" >&2
    exit 2
  fi
done

echo "scale-source: out=${out_dir} python=${python_bin}"
exec "${python_bin}" "${KIMODO_CODE_ROOT}/scripts/diagnose_body_scale_source.py" \
  --all-layers \
  --checkpoints \
    "${ckpt_400}" "${ckpt_500}" "${ckpt_550}" \
    "${ckpt_650}" "${ckpt_690}" "${ckpt_695}" \
    "${ckpt_700}" "${ckpt_750}" "${ckpt_780}" "${ckpt_790}" \
  --activation-dirs \
    "${KIMODO_STORAGE_ROOT}/diagnostics/body-rootcause-R-fullstack" \
    "${KIMODO_STORAGE_ROOT}/diagnostics/body-rootcause-R-wd03-from500k" \
  --windows \
    500000:550000 \
    400000:500000 \
    650000:700000 \
    700000:750000 \
    750000:780000 \
    780000:790000 \
    690000:695000 \
  --output-dir "${out_dir}"
