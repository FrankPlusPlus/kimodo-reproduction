#!/usr/bin/env bash
# Read-only tail-gain probe on the stopped lr3e6 rescue:
#   795k vs 800k weight spectra (L15 QKV/O, L14 FFN)
#   plus frozen-batch activation effective rank and incoming grads
# No optimizer step, no weight write.
#
# Prefer 1 instance x 1 GPU. Do not Restart. Do not use kimodo-dev.
# Do not attach to the 795 eval pod.
# Start box is sh:
#   exec bash /home/share/yzt/kimodo-reproduction/scripts/company_run_body_tail_gain_probe_wd03_795_800.sh
set -euo pipefail

export KIMODO_CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
export KIMODO_STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
export PYTHONPATH="${KIMODO_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export KIMODO_SKIP_MANIFEST_PATH_STAT="${KIMODO_SKIP_MANIFEST_PATH_STAT:-1}"
export KIMODO_FEATURE_CACHE_INDEX_MODE="${KIMODO_FEATURE_CACHE_INDEX_MODE:-deterministic}"
export KIMODO_FEATURE_CACHE_DIR="${KIMODO_FEATURE_CACHE_DIR:-${KIMODO_STORAGE_ROOT}/feature-cache/v1}"
export KIMODO_NNODES="${KIMODO_NNODES:-${PET_NNODES:-${NNODES:-1}}}"
export KIMODO_NPROC_PER_NODE="${KIMODO_NPROC_PER_NODE:-${PET_NPROC_PER_NODE:-${NPROC_PER_NODE:-1}}}"
resolved_node_rank="${KIMODO_NODE_RANK:-${PET_NODE_RANK:-${NODE_RANK:-${JOB_COMPLETION_INDEX:-}}}}"
export KIMODO_NODE_RANK="${resolved_node_rank}"
export MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-}}"
export MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29500}}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

if [[ -f "${KIMODO_CODE_ROOT}/scripts/load_kimodo_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${KIMODO_CODE_ROOT}/scripts/load_kimodo_env.sh"
  kimodo_load_env_files
fi

run_dir="${KIMODO_RESCUE_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-wd03-from780k-lr3e6}"
config="${KIMODO_PROBE_CONFIG:-${run_dir}/config.resolved.yaml}"
out_dir="${KIMODO_PROBE_OUT_DIR:-${KIMODO_STORAGE_ROOT}/diagnostics/body-tail-gain-wd03-795-800}"
preserve_dir="${KIMODO_PRESERVE_DIR:-${KIMODO_STORAGE_ROOT}/preserved-pre-collapse/v2-1m-hostnet-wd03-from780k-lr3e6}"
ckpt_795="${KIMODO_CKPT_795:-${run_dir}/checkpoints/step-000795000.pt}"
ckpt_800="${KIMODO_CKPT_800:-${run_dir}/checkpoints/step-000800000.pt}"
if [[ ! -r "${ckpt_795}" && -r "${preserve_dir}/step-000795000.pt" ]]; then
  ckpt_795="${preserve_dir}/step-000795000.pt"
fi
if [[ ! -r "${ckpt_800}" && -r "${preserve_dir}/step-000800000.pt" ]]; then
  ckpt_800="${preserve_dir}/step-000800000.pt"
fi

cd "${KIMODO_CODE_ROOT}"
python_bin="${KIMODO_PYTHON:-python3}"
if ! grep -q "summarize_tail_gain" "${KIMODO_CODE_ROOT}/kimodo/training/body_tail_gain_probe.py"; then
  echo "body_tail_gain_probe.py is missing summarize_tail_gain; PVC code is stale" >&2
  exit 2
fi
if ! grep -q "constraint-step" "${KIMODO_CODE_ROOT}/scripts/diagnose_body_tail_gain.py"; then
  echo "diagnose_body_tail_gain.py is missing --constraint-step; PVC code is stale" >&2
  exit 2
fi

visible_gpus=0
if command -v nvidia-smi >/dev/null 2>&1; then
  visible_gpus="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
fi
if [[ "${visible_gpus}" == "1" || "${KIMODO_NNODES}" == "1" && "${KIMODO_NPROC_PER_NODE}" == "1" ]]; then
  export KIMODO_NNODES=1
  export KIMODO_NPROC_PER_NODE=1
  export KIMODO_PROBE_SKIP_TORCHRUN="${KIMODO_PROBE_SKIP_TORCHRUN:-1}"
fi

mkdir -p "${out_dir}"
for path in "${config}" "${ckpt_795}" "${ckpt_800}"; do
  if [[ ! -r "${path}" ]]; then
    echo "missing readable path: ${path}" >&2
    exit 2
  fi
done

echo "wd03 795/800 tail-gain probe: python=${python_bin}"
echo "config=${config}"
echo "ckpts=${ckpt_795} ${ckpt_800}"
echo "out=${out_dir}"

diagnose_args=(
  "${KIMODO_CODE_ROOT}/scripts/diagnose_body_tail_gain.py"
  --config "${config}"
  --checkpoints "${ckpt_795}" "${ckpt_800}"
  --samples "${KIMODO_PROBE_SAMPLES:-4}"
  --seed "${KIMODO_PROBE_SEED:-20260816}"
  --constraint-step "${KIMODO_PROBE_CONSTRAINT_STEP:-795000}"
  --device "${KIMODO_PROBE_DEVICE:-auto}"
  --output-dir "${out_dir}"
)

if [[ -n "${KIMODO_PROBE_SKIP_TORCHRUN:-}" ]]; then
  exec "${python_bin}" "${diagnose_args[@]}"
fi

if [[ -z "${MASTER_ADDR}" || -z "${KIMODO_NODE_RANK}" ]]; then
  echo "multi-GPU needs MASTER_ADDR and node rank; prefer 1 instance x 1 GPU" >&2
  exit 2
fi
exec "${python_bin}" -m torch.distributed.run \
  --nnodes="${KIMODO_NNODES}" \
  --nproc-per-node="${KIMODO_NPROC_PER_NODE}" \
  --node-rank="${KIMODO_NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  "${diagnose_args[@]}"
