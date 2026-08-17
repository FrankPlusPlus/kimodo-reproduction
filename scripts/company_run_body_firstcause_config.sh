#!/usr/bin/env bash
# First-cause config probe on kimodo-dev (GPU OK). Official SEED vs our ckpts,
# then multi-batch 1-step drift at kf-smooth 690k under filled-in knobs.
#   exec bash /home/share/yzt/kimodo-reproduction/scripts/company_run_body_firstcause_config.sh
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

kf="${KIMODO_KF_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-kf-smooth-lr1e5}"
wd03="${KIMODO_WD03_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-wd03-from650k}"
preserve="${KIMODO_PRESERVE_DIR:-${KIMODO_STORAGE_ROOT}/preserved-pre-collapse/v2-1m-hostnet-wd03-from650k}"
official="${KIMODO_OFFICIAL_DIR:-${KIMODO_STORAGE_ROOT}/models/checkpoints/Kimodo-SOMA-SEED-v1.1}"
config="${KIMODO_PROBE_CONFIG:-${kf}/config.resolved.yaml}"
out_dir="${KIMODO_PROBE_OUT_DIR:-${KIMODO_STORAGE_ROOT}/diagnostics/body-firstcause-config}"

_resolve() {
  if [[ -r "$1" ]]; then echo "$1"; else echo "$2"; fi
}

ckpt_650="${kf}/checkpoints/step-000650000.pt"
ckpt_690="${kf}/checkpoints/step-000690000.pt"
ckpt_695="${kf}/checkpoints/step-000695000.pt"
ckpt_750="$(_resolve "${wd03}/checkpoints/step-000750000.pt" "${preserve}/step-000750000.pt")"
ckpt_790="$(_resolve "${wd03}/checkpoints/step-000790000.pt" "${preserve}/step-000790000.pt")"

cd "${KIMODO_CODE_ROOT}"
python_bin="${KIMODO_PYTHON:-/home/jovyan/.venv-kimodo-demo/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="${KIMODO_CODE_ROOT}/.venv-feature-cache/bin/python"
fi

mkdir -p "${out_dir}"
echo "{\"status\":\"launcher_mkdir\"}" > "${out_dir}/running.json"
for path in "${config}" "${official}/config.yaml" "${ckpt_650}" "${ckpt_690}" "${ckpt_695}" "${ckpt_750}" "${ckpt_790}"; do
  if [[ ! -r "${path}" ]]; then
    echo "missing readable path: ${path}" >&2
    exit 2
  fi
done

echo "firstcause-config: out=${out_dir} python=${python_bin}"
exec "${python_bin}" "${KIMODO_CODE_ROOT}/scripts/diagnose_body_firstcause_config.py" \
  --config "${config}" \
  --official-dir "${official}" \
  --checkpoints "${ckpt_650}" "${ckpt_690}" "${ckpt_695}" "${ckpt_750}" "${ckpt_790}" \
  --labels kf650 kf690 kf695 wd750 wd790 \
  --preflip-checkpoint "${ckpt_690}" \
  --samples "${KIMODO_PROBE_SAMPLES:-16}" \
  --chunk-size 4 \
  --seed 20260816 \
  --constraint-step 750000 \
  --device "${KIMODO_PROBE_DEVICE:-cuda:0}" \
  --precisions fp32 bf16 \
  --variants atan2 adam atan2_wd03 atan2_lambda1 atan2_noclip \
  --clip-norm 1.0 \
  --output-dir "${out_dir}"
