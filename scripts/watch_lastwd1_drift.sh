#!/usr/bin/env bash
# Poll lastwd1-from750k trainer checkpoints every 20k and photograph last-layer
# attention cosine vs the preserved 750k parent. Also snapshots jsonl L15/clip.
# Read-only. Safe on the 1xH200 eval pod next to stratified scoring.
#
# Do not use kimodo-dev. Do not attach to the 16-GPU trainer.
set -euo pipefail

code_root="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
storage_root="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
train_run="${KIMODO_TRAIN_RUN_DIR:-${storage_root}/runs/v2-1m-hostnet-lastwd1-from750k}"
out_root="${KIMODO_DRIFT_OUT_DIR:-${storage_root}/diagnostics/lastwd1-from750k-drift}"
parent_ckpt="${KIMODO_DRIFT_PARENT_CHECKPOINT:-${storage_root}/preserved-pre-collapse/v2-1m-hostnet-wd03-from650k/step-000750000.pt}"
config="${KIMODO_DRIFT_CONFIG:-${train_run}/config.resolved.yaml}"
jsonl="${train_run}/train.jsonl"
python_bin="${KIMODO_DRIFT_PYTHON:-${KIMODO_EXPORT_PYTHON:-python3}}"
min_step="${KIMODO_DRIFT_MIN_STEP:-760000}"
step_every="${KIMODO_DRIFT_STEP_EVERY:-20000}"
poll_seconds="${KIMODO_DRIFT_POLL_SECONDS:-60}"
once="${KIMODO_DRIFT_ONCE:-0}"

export PYTHONPATH="${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${out_root}"

if [[ ! -x "${python_bin}" ]]; then
  python_bin="$(command -v python3 || command -v python || true)"
fi
if [[ -z "${python_bin}" ]]; then
  echo "no python available for lastwd1 drift watcher" >&2
  exit 2
fi

echo "drift watcher: train=${train_run}"
echo "drift watcher: parent=${parent_ckpt}"
echo "drift watcher: out=${out_root} min=${min_step} every=${step_every}"

probe_one() {
  local ckpt="$1"
  local step="$2"
  local dest="${out_root}/step-$(printf '%09d' "${step}")"
  if [[ -f "${dest}/health.json" ]]; then
    echo "skip existing ${dest}/health.json"
    return 0
  fi
  if [[ ! -r "${ckpt}" ]]; then
    echo "UNREADABLE ${ckpt}" >&2
    return 0
  fi
  if [[ ! -r "${parent_ckpt}" ]]; then
    echo "missing parent 750k checkpoint: ${parent_ckpt}" >&2
    return 0
  fi
  if [[ ! -r "${config}" ]]; then
    echo "missing resolved config: ${config}" >&2
    return 0
  fi
  mkdir -p "${dest}"
  echo "probing cosine step=${step} from ${ckpt}"
  "${python_bin}" "${code_root}/scripts/diagnose_body_residual_cancel.py" \
    --config "${config}" \
    --checkpoints "${parent_ckpt}" "${ckpt}" \
    --samples "${KIMODO_PROBE_SAMPLES:-4}" \
    --seed "${KIMODO_PROBE_SEED:-20260816}" \
    --constraint-steps 750000 \
    --device "${KIMODO_PROBE_DEVICE:-auto}" \
    --output-dir "${dest}"
  "${python_bin}" "${code_root}/scripts/analyze_lastwd1_drift.py" \
    --step "${step}" \
    --jsonl "${jsonl}" \
    --probe-dir "${dest}" \
    --checkpoint "${ckpt}" \
    --output "${dest}/health.json"
  if [[ -f "${dest}/health.json" ]]; then
    "${python_bin}" -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1], encoding="utf-8")), sort_keys=True))' \
      "${dest}/health.json" >> "${out_root}/health.jsonl"
  fi
}

while true; do
  shopt -s nullglob
  for ckpt in "${train_run}/checkpoints"/step-*.pt; do
    step="$(basename "${ckpt}" | sed -E 's/^step-0*([0-9]+)\.pt$/\1/')"
    if [[ -z "${step}" || "${step}" -lt "${min_step}" ]]; then
      continue
    fi
    if (( step % step_every != 0 )); then
      continue
    fi
    probe_one "${ckpt}" "${step}" || true
  done
  shopt -u nullglob
  if [[ "${once}" == "1" ]]; then
    exit 0
  fi
  sleep "${poll_seconds}"
done
