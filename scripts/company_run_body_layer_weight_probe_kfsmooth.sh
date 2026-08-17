#!/usr/bin/env bash
# Weights-only: last-layer vs inner-layer RMS from 650k to the 696k flip.
# No forward, no data, no optimizer. Prefer 1 instance x 1 GPU (CPU is enough).
# Do not Restart. Do not use kimodo-dev.
# Start box is sh:
#   exec bash /home/share/yzt/kimodo-reproduction/scripts/company_run_body_layer_weight_probe_kfsmooth.sh
set -euo pipefail

export KIMODO_CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
export KIMODO_STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
export PYTHONPATH="${KIMODO_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
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

kf_run="${KIMODO_KF_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-kf-smooth-lr1e5}"
wd03_preserve="${KIMODO_WD03_PRESERVE_DIR:-${KIMODO_STORAGE_ROOT}/preserved-pre-collapse/v2-1m-hostnet-wd03-from650k}"
out_dir="${KIMODO_PROBE_OUT_DIR:-${KIMODO_STORAGE_ROOT}/diagnostics/body-layer-weight-kfsmooth-650-695}"
ckpt_650="${KIMODO_CKPT_650:-${kf_run}/checkpoints/step-000650000.pt}"
ckpt_690="${KIMODO_CKPT_690:-${kf_run}/checkpoints/step-000690000.pt}"
ckpt_695="${KIMODO_CKPT_695:-${kf_run}/checkpoints/step-000695000.pt}"
ckpt_750="${KIMODO_CKPT_750:-${wd03_preserve}/step-000750000.pt}"
ckpt_790="${KIMODO_CKPT_790:-${wd03_preserve}/step-000790000.pt}"

cd "${KIMODO_CODE_ROOT}"
python_bin="${KIMODO_PYTHON:-python3}"
if ! grep -q "summarize_layer_weight_timeline" "${KIMODO_CODE_ROOT}/kimodo/training/body_layer_weight_probe.py"; then
  echo "body_layer_weight_probe.py is missing summarize_layer_weight_timeline; PVC code is stale" >&2
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
echo "{\"status\":\"launcher_mkdir\"}" > "${out_dir}/running.json"
ckpts=("${ckpt_650}" "${ckpt_690}" "${ckpt_695}")
for extra in "${ckpt_750}" "${ckpt_790}"; do
  if [[ -r "${extra}" ]]; then
    ckpts+=("${extra}")
  fi
done
for path in "${ckpts[@]}"; do
  if [[ ! -r "${path}" ]]; then
    echo "missing readable path: ${path}" >&2
    exit 2
  fi
done

echo "layer-weight RMS probe: python=${python_bin}"
echo "ckpts=${ckpts[*]}"
echo "out=${out_dir}"

diagnose_args=(
  "${KIMODO_CODE_ROOT}/scripts/diagnose_body_layer_weight.py"
  --checkpoints "${ckpts[@]}"
  --start-step 650000
  --end-step 695000
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
