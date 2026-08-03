#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
python_bin="${KIMODO_PYTHON:-${project_root}/.venv/bin/python}"
config_path="${KIMODO_TWO_GPU_CONFIG:-${project_root}/configs/training/kimodo_soma_seed_public_two_gpu.yaml}"

# Make CUDA_VISIBLE_DEVICES use the physical PCI/nvidia-smi ordering on this
# host, whose default CUDA runtime ordering is different.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python environment is missing or not executable: ${python_bin}" >&2
  exit 2
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ ! "${CUDA_VISIBLE_DEVICES}" =~ ^[^,[:space:]]+,[^,[:space:]]+$ ]]; then
    echo "CUDA_VISIBLE_DEVICES must expose exactly two non-empty device IDs; got: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
  fi
fi

"${python_bin}" - <<'PY'
import json
import os
import sys

try:
    import torch
except Exception as error:
    raise SystemExit(f"PyTorch CUDA preflight failed to import: {error}")

count = torch.cuda.device_count() if torch.cuda.is_available() else 0
if count != 2:
    raise SystemExit(
        "Two-GPU launcher requires exactly two scheduler/container-visible CUDA devices; "
        f"torch sees {count}. Set CUDA_VISIBLE_DEVICES to the two devices allocated to this tenant."
    )
devices = []
expected = os.environ.get("KIMODO_EXPECTED_GPU_NAME", "NVIDIA H200 NVL")
for index in range(count):
    props = torch.cuda.get_device_properties(index)
    devices.append(
        {
            "visible_index": index,
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "compute_capability": [int(props.major), int(props.minor)],
        }
    )
names = [device["name"] for device in devices]
if any(name != expected for name in names):
    raise SystemExit(f"Expected two {expected!r} devices, but CUDA exposes: {names}")
print(
    json.dumps(
        {
            "event": "kimodo_two_gpu_preflight",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "devices": devices,
        },
        sort_keys=True,
    ),
    flush=True,
)
PY

cd "${project_root}"
exec "${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node=2 \
  -m kimodo.training.cli \
  --config "${config_path}" \
  "$@"
