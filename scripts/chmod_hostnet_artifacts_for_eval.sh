#!/usr/bin/env bash
# Run ONCE as root on the training pod (or any root shell with PVC access).
# Unlocks existing trainer checkpoints so the jovyan eval/dev pod can export+score them.
set -euo pipefail

run_dir="${KIMODO_TRAIN_RUN_DIR:-/home/share/yezitao-kimodo-reproduction/runs/v2-1m-hostnet}"
ckpt_dir="${run_dir}/checkpoints"
exports_dir="${run_dir}/exports"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "must run as root (training pod shell)" >&2
  exit 2
fi

if [[ -d "${ckpt_dir}" ]]; then
  chmod a+rx "${ckpt_dir}" || true
  chmod a+r "${ckpt_dir}"/step-*.pt "${ckpt_dir}"/latest.txt 2>/dev/null || true
  echo "readable checkpoints:"
  ls -l "${ckpt_dir}"/step-*.pt "${ckpt_dir}"/latest.txt 2>/dev/null || true
fi

if [[ -d "${exports_dir}" ]]; then
  chmod -R a+rX "${exports_dir}" || true
  echo "readable exports under ${exports_dir}"
fi

echo "DONE: sidecar eval can now read hostnet artifacts"
