#!/usr/bin/env bash
# Read-only residual-cancellation TIMELINE:
#   Phase 1: v2-1m-hostnet 400k / 500k (kf-smooth already dropped Phase 1 ckpts)
#   Phase 2 parent: 650k / 700k / 750k / 780k / 790k
# Two constraint clocks on the same frozen noise:
#   400000 = still Phase 1 (no constraints)
#   750000 = Phase 2 (K around the quality-best era)
# So we can split "weights already cancel" from "Phase 2 inputs make them cancel".
# Forward only. No optimizer step, no weight write.
#
# Prefer 1 instance x 1 GPU. Do not Restart. Do not use kimodo-dev.
# Start box is sh:
#   exec bash /home/share/yzt/kimodo-reproduction/scripts/company_run_body_residual_cancel_probe_wd03_timeline.sh
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

parent_run="${KIMODO_WD03_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-wd03-from650k}"
kf_run="${KIMODO_KF_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-kf-smooth-lr1e5}"
hostnet_run="${KIMODO_HOSTNET_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet}"
config="${KIMODO_PROBE_CONFIG:-${parent_run}/config.resolved.yaml}"
out_dir="${KIMODO_PROBE_OUT_DIR:-${KIMODO_STORAGE_ROOT}/diagnostics/body-residual-cancel-wd03-timeline}"
preserve_dir="${KIMODO_PRESERVE_DIR:-${KIMODO_STORAGE_ROOT}/preserved-pre-collapse/v2-1m-hostnet-wd03-from650k}"
ckpt_400="${KIMODO_CKPT_400:-${hostnet_run}/checkpoints/step-000400000.pt}"
ckpt_500="${KIMODO_CKPT_500:-${hostnet_run}/checkpoints/step-000500000.pt}"
ckpt_650="${KIMODO_CKPT_650:-${kf_run}/checkpoints/step-000650000.pt}"
ckpt_700="${KIMODO_CKPT_700:-${parent_run}/checkpoints/step-000700000.pt}"
ckpt_750="${KIMODO_CKPT_750:-${parent_run}/checkpoints/step-000750000.pt}"
ckpt_780="${KIMODO_CKPT_780:-${parent_run}/checkpoints/step-000780000.pt}"
ckpt_790="${KIMODO_CKPT_790:-${parent_run}/checkpoints/step-000790000.pt}"
if [[ ! -r "${ckpt_750}" && -r "${preserve_dir}/step-000750000.pt" ]]; then
  ckpt_750="${preserve_dir}/step-000750000.pt"
fi
if [[ ! -r "${ckpt_780}" && -r "${preserve_dir}/step-000780000.pt" ]]; then
  ckpt_780="${preserve_dir}/step-000780000.pt"
fi
if [[ ! -r "${ckpt_790}" && -r "${preserve_dir}/step-000790000.pt" ]]; then
  ckpt_790="${preserve_dir}/step-000790000.pt"
fi

cd "${KIMODO_CODE_ROOT}"
python_bin="${KIMODO_PYTHON:-python3}"
if ! grep -q "residual_timeline" "${KIMODO_CODE_ROOT}/kimodo/training/body_residual_cancel_probe.py"; then
  echo "body_residual_cancel_probe.py is missing residual_timeline; PVC code is stale" >&2
  exit 2
fi
if ! grep -q "constraint-steps" "${KIMODO_CODE_ROOT}/scripts/diagnose_body_residual_cancel.py"; then
  echo "diagnose_body_residual_cancel.py is missing --constraint-steps; PVC code is stale" >&2
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
for path in "${config}" "${ckpt_400}" "${ckpt_500}" "${ckpt_650}" "${ckpt_700}" "${ckpt_750}" "${ckpt_780}" "${ckpt_790}"; do
  if [[ ! -r "${path}" ]]; then
    echo "missing readable path: ${path}" >&2
    exit 2
  fi
done

echo "wd03 residual-cancel TIMELINE: python=${python_bin}"
echo "config=${config}"
echo "ckpts=${ckpt_400} ${ckpt_500} ${ckpt_650} ${ckpt_700} ${ckpt_750} ${ckpt_780} ${ckpt_790}"
echo "out=${out_dir}"

diagnose_args=(
  "${KIMODO_CODE_ROOT}/scripts/diagnose_body_residual_cancel.py"
  --config "${config}"
  --checkpoints "${ckpt_400}" "${ckpt_500}" "${ckpt_650}" "${ckpt_700}" "${ckpt_750}" "${ckpt_780}" "${ckpt_790}"
  --samples "${KIMODO_PROBE_SAMPLES:-4}"
  --seed "${KIMODO_PROBE_SEED:-20260816}"
  --constraint-steps 400000 750000
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
