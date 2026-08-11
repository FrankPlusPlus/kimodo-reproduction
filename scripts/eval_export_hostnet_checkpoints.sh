#!/usr/bin/env bash
# Poll trainer checkpoints and offline-export EMA bundles for proxy monitoring.
# Safe to run on CPU. Needs read access to checkpoints (chmod a+r on training side).
set -euo pipefail

storage_root="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
code_root="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
train_run="${KIMODO_TRAIN_RUN_DIR:-${storage_root}/runs/v2-1m-hostnet}"
export_run="${KIMODO_EVAL_EXPORT_RUN_DIR:-${storage_root}/eval-exports/v2-1m-hostnet}"
resolved="${KIMODO_RESOLVED_CONFIG:-${train_run}/config.resolved.yaml}"
poll_seconds="${KIMODO_EXPORT_POLL_SECONDS:-120}"
python_bin="${KIMODO_EXPORT_PYTHON:-${code_root}/.venv-feature-cache/bin/python}"
min_step="${KIMODO_EXPORT_MIN_STEP:-30000}"
once="${KIMODO_EXPORT_ONCE:-0}"

export PYTHONPATH="${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${export_run}/exports"

if [[ ! -x "${python_bin}" ]]; then
  python_bin="$(command -v python3)"
fi

echo "export watcher: train=${train_run}"
echo "export watcher: out=${export_run}/exports"
echo "export watcher: python=${python_bin} poll=${poll_seconds}s min_step=${min_step}"

export_one() {
  local ckpt="$1"
  local step
  step="$(basename "${ckpt}" | sed -E 's/^step-0*([0-9]+)\.pt$/\1/')"
  if [[ -z "${step}" || "${step}" -lt "${min_step}" ]]; then
    return 0
  fi
  local dest="${export_run}/exports/step-$(printf '%09d' "${step}")"
  if [[ -d "${dest}" && -f "${dest}/model.pt" ]]; then
    echo "skip existing ${dest}"
    return 0
  fi
  if [[ ! -r "${ckpt}" ]]; then
    echo "UNREADABLE ${ckpt} (need training-side: chmod a+r ${train_run}/checkpoints/step-*.pt)" >&2
    return 0
  fi
  echo "exporting step=${step} from ${ckpt}"
  "${python_bin}" "${code_root}/scripts/export_trainer_checkpoint_bundle.py" \
    --checkpoint "${ckpt}" \
    --resolved-config "${resolved}" \
    --output-run-dir "${export_run}" \
    --step "${step}"
}

while true; do
  shopt -s nullglob
  for ckpt in "${train_run}/checkpoints"/step-*.pt; do
    export_one "${ckpt}" || true
  done
  shopt -u nullglob
  if [[ "${once}" == "1" ]]; then
    exit 0
  fi
  sleep "${poll_seconds}"
done
