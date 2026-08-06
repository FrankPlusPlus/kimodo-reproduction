#!/usr/bin/env bash
set -euo pipefail

# Generic fixed-size torchrun launcher for one or more homogeneous GPU nodes.
# Kubernetes/Slurm is responsible for starting this script once per launcher
# allocation (usually one Pod) and giving each allocation the same rendezvous
# address plus a unique torchrun node rank. Multiple Pods may share one host.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
python_bin="${KIMODO_PYTHON:-python}"
config_path="${KIMODO_TRAINING_CONFIG:-${project_root}/configs/training/kimodo_soma_seed_public.yaml}"
paths_path="${KIMODO_PATHS_CONFIG:-/mnt/kimodo/config/repro.paths.yaml}"
overlay_path="${KIMODO_TRAINING_OVERLAY:-}"

nnodes="${KIMODO_NNODES:-${NNODES:-1}}"
nproc_per_node="${KIMODO_NPROC_PER_NODE:-${NPROC_PER_NODE:-8}}"
node_rank="${KIMODO_NODE_RANK:-${NODE_RANK:-${JOB_COMPLETION_INDEX:-0}}}"
master_addr="${MASTER_ADDR:-}"
master_port="${MASTER_PORT:-29500}"

die() {
  echo "kimodo distributed launcher: $*" >&2
  exit 2
}

for value_name in nnodes nproc_per_node node_rank master_port; do
  value="${!value_name}"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${value_name} must be a non-negative integer; got ${value}"
done
(( nnodes >= 1 )) || die "nnodes must be at least 1"
(( nproc_per_node >= 1 )) || die "nproc_per_node must be at least 1"
(( node_rank < nnodes )) || die "node_rank=${node_rank} must be smaller than nnodes=${nnodes}"
(( master_port >= 1 && master_port <= 65535 )) || die "master_port must be in 1..65535"

if [[ "${nnodes}" == 1 ]]; then
  master_addr="${master_addr:-127.0.0.1}"
elif [[ -z "${master_addr}" ]]; then
  die "MASTER_ADDR is required for multi-node training and must resolve to node rank 0"
fi

[[ -f "${config_path}" ]] || die "training config is missing: ${config_path}"
[[ -f "${paths_path}" ]] || die "paths config is missing: ${paths_path}"
if [[ -n "${overlay_path}" ]]; then
  [[ -f "${overlay_path}" ]] || die "training overlay is missing: ${overlay_path}"
fi
command -v "${python_bin}" >/dev/null 2>&1 || die "Python executable is unavailable: ${python_bin}"

if [[ "${KIMODO_REQUIRE_RDMA:-0}" == 1 ]]; then
  command -v ibv_devices >/dev/null 2>&1 || die "ibv_devices is unavailable in the image"
  [[ -d /sys/class/infiniband ]] || die "/sys/class/infiniband is not mounted"
  compgen -G '/sys/class/infiniband/*' >/dev/null || die "no RDMA device is visible in this container"
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

"${python_bin}" - \
  "${nproc_per_node}" "${nnodes}" "${node_rank}" \
  "${config_path}" "${paths_path}" "${overlay_path}" <<'PY'
import json
import os
import sys

import torch

expected_local = int(sys.argv[1])
nnodes = int(sys.argv[2])
node_rank = int(sys.argv[3])
training_config = sys.argv[4]
paths_config = sys.argv[5]
training_overlay = sys.argv[6] or None
visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
if visible != expected_local:
    raise SystemExit(
        f"expected {expected_local} scheduler-visible GPUs on this node, but PyTorch sees {visible}; "
        "check resources.limits[nvidia.com/gpu] and CUDA_VISIBLE_DEVICES"
    )

devices = []
expected_pattern = os.environ.get("KIMODO_EXPECTED_GPU_PATTERN")
for index in range(visible):
    props = torch.cuda.get_device_properties(index)
    if expected_pattern and expected_pattern not in props.name:
        raise SystemExit(
            f"GPU {index} is {props.name!r}, which does not contain "
            f"KIMODO_EXPECTED_GPU_PATTERN={expected_pattern!r}"
        )
    devices.append(
        {
            "visible_index": index,
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "compute_capability": [int(props.major), int(props.minor)],
        }
    )

print(
    json.dumps(
        {
            "event": "kimodo_distributed_node_preflight",
            "hostname": os.uname().nodename,
            "nnodes": nnodes,
            "node_rank": node_rank,
            "nproc_per_node": expected_local,
            "expected_world_size": nnodes * expected_local,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "training_config": training_config,
            "paths_config": paths_config,
            "training_overlay": training_overlay,
            "devices": devices,
        },
        sort_keys=True,
    ),
    flush=True,
)
PY

cd "${project_root}"
training_args=(
  -m kimodo.training.cli
  --config "${config_path}"
  --paths "${paths_path}"
)
if [[ -n "${overlay_path}" ]]; then
  training_args+=(--overlay "${overlay_path}")
fi

exec "${python_bin}" -m torch.distributed.run \
  --nnodes="${nnodes}" \
  --nproc-per-node="${nproc_per_node}" \
  --node-rank="${node_rank}" \
  --master-addr="${master_addr}" \
  --master-port="${master_port}" \
  "${training_args[@]}" \
  "$@"
