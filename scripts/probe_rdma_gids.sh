#!/usr/bin/env bash
# Hang-safe InfiniBand/RoCE GID probe for company Pods.
# Full sysfs walks can block on broken verbs nodes; every read is timed out.
#
# Usage:
#   bash scripts/probe_rdma_gids.sh
#   source scripts/probe_rdma_gids.sh && kimodo_suggest_nccl_ib_gid
#
# shellcheck shell=bash

kimodo_rdma_read_timeout() {
  # usage: kimodo_rdma_read_timeout <seconds> <path>
  local secs="$1" path="$2"
  if command -v timeout >/dev/null 2>&1; then
    timeout "${secs}" cat "${path}" 2>/dev/null || true
  else
    # Best-effort without coreutils timeout.
    cat "${path}" 2>/dev/null || true
  fi
}

kimodo_probe_rdma_gids() {
  local sysfs="/sys/class/infiniband"
  local max_idx="${KIMODO_NCCL_PROBE_MAX_GID_INDEX:-7}"
  local read_secs="${KIMODO_NCCL_PROBE_READ_TIMEOUT:-1}"

  echo "=== ibv_devices ===" >&2
  if command -v timeout >/dev/null 2>&1; then
    timeout 3 ibv_devices >&2 || echo "(ibv_devices timed out/failed)" >&2
  elif command -v ibv_devices >/dev/null 2>&1; then
    ibv_devices >&2 || true
  else
    echo "(ibv_devices missing)" >&2
  fi

  echo "=== /dev/infiniband (names only) ===" >&2
  if [[ -d /dev/infiniband ]]; then
    if command -v timeout >/dev/null 2>&1; then
      timeout 2 ls /dev/infiniband >&2 || echo "(ls /dev/infiniband timed out)" >&2
    else
      ls /dev/infiniband >&2 || true
    fi
  else
    echo "(no /dev/infiniband)" >&2
  fi

  echo "=== GID table (mlx5_0..7, idx 0..${max_idx}, ${read_secs}s/read) ===" >&2
  if [[ ! -d "${sysfs}" ]]; then
    echo "(no ${sysfs})" >&2
    return 1
  fi

  local dname port_dir pnum idx gid gtype
  # Deterministic order; skip mlx5_12 (non-rail).
  for dname in mlx5_0 mlx5_1 mlx5_2 mlx5_3 mlx5_4 mlx5_5 mlx5_6 mlx5_7; do
    [[ -d "${sysfs}/${dname}" ]] || continue
    for port_dir in "${sysfs}/${dname}/ports"/*; do
      [[ -d "${port_dir}/gids" ]] || continue
      pnum="$(basename "${port_dir}")"
      idx=0
      while (( idx <= max_idx )); do
        if [[ ! -e "${port_dir}/gids/${idx}" ]]; then
          idx=$((idx + 1))
          continue
        fi
        gid="$(kimodo_rdma_read_timeout "${read_secs}" "${port_dir}/gids/${idx}")"
        gtype="$(kimodo_rdma_read_timeout "${read_secs}" "${port_dir}/gid_attrs/types/${idx}")"
        if [[ -z "${gid}" ]]; then
          printf '%s port=%s idx=%s (read empty/timeout)\n' "${dname}" "${pnum}" "${idx}" >&2
        elif [[ "${gid}" == "0000:0000:0000:0000:0000:0000:0000:0000" ]]; then
          : # skip empty
        else
          printf '%s port=%s idx=%s type=%s gid=%s\n' \
            "${dname}" "${pnum}" "${idx}" "${gtype:-?}" "${gid}" >&2
        fi
        idx=$((idx + 1))
      done
      # Only port 1 is needed for NCCL on this cluster.
      break
    done
  done
  echo "=== GID probe done ===" >&2
}

# Print a suggested NCCL_IB_GID_INDEX on stdout (empty if none).
# Preference: RoCE v2 + not IPv4-mapped (::ffff:), else any RoCE v2, else empty.
kimodo_suggest_nccl_ib_gid() {
  local sysfs="/sys/class/infiniband"
  local max_idx="${KIMODO_NCCL_PROBE_MAX_GID_INDEX:-7}"
  local read_secs="${KIMODO_NCCL_PROBE_READ_TIMEOUT:-1}"
  local best_v2_nonmapped="" best_v2=""
  local dname port_dir idx gid gtype

  for dname in mlx5_0 mlx5_1 mlx5_2 mlx5_3 mlx5_4 mlx5_5 mlx5_6 mlx5_7; do
    [[ -d "${sysfs}/${dname}" ]] || continue
    for port_dir in "${sysfs}/${dname}/ports"/*; do
      [[ -d "${port_dir}/gids" ]] || continue
      idx=0
      while (( idx <= max_idx )); do
        [[ -e "${port_dir}/gids/${idx}" ]] || { idx=$((idx + 1)); continue; }
        gid="$(kimodo_rdma_read_timeout "${read_secs}" "${port_dir}/gids/${idx}")"
        gtype="$(kimodo_rdma_read_timeout "${read_secs}" "${port_dir}/gid_attrs/types/${idx}")"
        if [[ -n "${gid}" && "${gid}" != "0000:0000:0000:0000:0000:0000:0000:0000" ]]; then
          if [[ "${gtype}" =~ [Rr][Oo][Cc][Ee].*2 ]]; then
            if [[ "${gid}" != *ffff* && "${gid}" != *FFFF* ]]; then
              best_v2_nonmapped="${idx}"
              printf '%s\n' "${best_v2_nonmapped}"
              return 0
            fi
            if [[ -z "${best_v2}" ]]; then
              best_v2="${idx}"
            fi
          fi
        fi
        idx=$((idx + 1))
      done
      break
    done
  done

  if [[ -n "${best_v2}" ]]; then
    printf '%s\n' "${best_v2}"
    return 0
  fi
  return 1
}

kimodo_default_rail_hca() {
  printf '%s\n' "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  kimodo_probe_rdma_gids || true
  suggested="$(kimodo_suggest_nccl_ib_gid || true)"
  echo "=== suggestion ===" >&2
  if [[ -n "${suggested}" ]]; then
    echo "export NCCL_IB_HCA=$(kimodo_default_rail_hca)" >&2
    echo "export NCCL_IB_GID_INDEX=${suggested}" >&2
    echo "unset NCCL_IB_DISABLE" >&2
    echo "${suggested}"
  else
    echo "(no RoCEv2 GID guessed; dump table above and pick manually)" >&2
    exit 1
  fi
fi
