#!/usr/bin/env bash
# Onset-path probe: freeze weights.
#   P. dL/dα toward last-layer anti-alignment
#   Q. 20 in-memory optimizer steps (atan2 / detach-σ / Adam / last-layer wd=1)
# Restore weights. No checkpoint write.
#
# Checkpoints: kf-smooth 650k / 690k (pre-flip) / 695k (just flipped).
# Prefer 1 instance x 1 GPU. Do not Restart. Do not use kimodo-dev.
# Start box is sh:
#   exec bash /home/share/yzt/kimodo-reproduction/scripts/company_run_body_onset_path_probe_kfsmooth.sh
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

kf_run="${KIMODO_KF_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-kf-smooth-lr1e5}"
config="${KIMODO_PROBE_CONFIG:-${kf_run}/config.resolved.yaml}"
out_dir="${KIMODO_PROBE_OUT_DIR:-${KIMODO_STORAGE_ROOT}/diagnostics/body-onset-path-kfsmooth-690-695}"
ckpt_650="${KIMODO_CKPT_650:-${kf_run}/checkpoints/step-000650000.pt}"
ckpt_690="${KIMODO_CKPT_690:-${kf_run}/checkpoints/step-000690000.pt}"
ckpt_695="${KIMODO_CKPT_695:-${kf_run}/checkpoints/step-000695000.pt}"

cd "${KIMODO_CODE_ROOT}"
python_bin="${KIMODO_PYTHON:-python3}"
if ! grep -q "_call_denoiser" "${KIMODO_CODE_ROOT}/kimodo/training/body_onset_path_probe.py"; then
  echo "body_onset_path_probe.py is missing _call_denoiser; PVC code is stale" >&2
  exit 2
fi
if ! grep -q "virtual-steps" "${KIMODO_CODE_ROOT}/scripts/diagnose_body_onset_path.py"; then
  echo "diagnose_body_onset_path.py is missing --virtual-steps; PVC code is stale" >&2
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
for path in "${config}" "${ckpt_650}" "${ckpt_690}" "${ckpt_695}"; do
  if [[ ! -r "${path}" ]]; then
    echo "missing readable path: ${path}" >&2
    exit 2
  fi
done

echo "kf-smooth onset-path probe: python=${python_bin}"
echo "config=${config}"
echo "ckpts=${ckpt_650} ${ckpt_690} ${ckpt_695}"
echo "out=${out_dir}"

diagnose_args=(
  "${KIMODO_CODE_ROOT}/scripts/diagnose_body_onset_path.py"
  --config "${config}"
  --checkpoints "${ckpt_650}" "${ckpt_690}" "${ckpt_695}"
  --samples "${KIMODO_PROBE_SAMPLES:-4}"
  --seed "${KIMODO_PROBE_SEED:-20260816}"
  --constraint-step "${KIMODO_PROBE_CONSTRAINT_STEP:-695000}"
  --healthy-step 650000
  --preflip-step 690000
  --flipped-step 695000
  --virtual-steps "${KIMODO_PROBE_VIRTUAL_STEPS:-20}"
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
