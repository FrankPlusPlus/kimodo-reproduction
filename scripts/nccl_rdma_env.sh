#!/usr/bin/env bash
# Portable NCCL/RDMA defaults for company multi-node jobs.
#
# Modes (KIMODO_NCCL_ENV_MODE):
#   respect (default)  Keep platform/user NCCL_* if already set; only fill gaps.
#   force-dyn          RECOMMENDED on NCCL>=2.21: rail HCA, UNSET GID index
#                      (dynamic select), RoCEv2, CROSS_NIC=0. Overrides platform
#                      NCCL_IB_GID_INDEX which often causes errno 19 / hangs.
#   force-auto         rail HCA + timed sysfs RoCEv2 GID guess (legacy).
#   force-gid=N        pin GID index N (legacy sweep).
#
# Probe: full GID dump OFF by default. Set KIMODO_NCCL_PROBE=1 only when debugging.
#
# shellcheck shell=bash

kimodo_nccl_rdma_env() {
  local script_dir
  script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  # shellcheck disable=SC1091
  source "${script_dir}/probe_rdma_gids.sh"

  local mode="${KIMODO_NCCL_ENV_MODE:-respect}"
  local rail_hca
  rail_hca="$(kimodo_default_rail_hca)"

  if [[ "${KIMODO_NCCL_PROBE:-0}" == "1" ]]; then
    echo "Kimodo NCCL/RDMA probe (mode=${mode})" >&2
    kimodo_probe_rdma_gids || true
  fi

  case "${mode}" in
    force-dyn)
      # NCCL 2.21+ should pick GID dynamically; platform GID=3 has failed here.
      unset NCCL_IB_DISABLE || true
      unset NCCL_IB_GID_INDEX || true
      export NCCL_IB_HCA="${KIMODO_NCCL_FORCE_HCA:-${rail_hca}}"
      export NCCL_IB_ROCE_VERSION_NUM="${NCCL_IB_ROCE_VERSION_NUM:-2}"
      export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
      export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
      echo "Kimodo force-dyn: HCA=${NCCL_IB_HCA} GID=unset(auto) ROCE=${NCCL_IB_ROCE_VERSION_NUM} CROSS_NIC=${NCCL_CROSS_NIC}" >&2
      ;;
    force-auto)
      unset NCCL_IB_DISABLE || true
      export NCCL_IB_HCA="${KIMODO_NCCL_FORCE_HCA:-${rail_hca}}"
      export NCCL_IB_ROCE_VERSION_NUM="${NCCL_IB_ROCE_VERSION_NUM:-2}"
      export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
      local suggested
      echo "Kimodo force-auto: selecting RoCEv2 GID (timed sysfs reads)..." >&2
      if suggested="$(kimodo_suggest_nccl_ib_gid)"; then
        export NCCL_IB_GID_INDEX="${suggested}"
        echo "Kimodo force-auto: NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}" >&2
      else
        unset NCCL_IB_GID_INDEX || true
        echo "Kimodo force-auto: GID guess failed; leaving unset for NCCL dynamic select" >&2
      fi
      ;;
    force-gid=*)
      unset NCCL_IB_DISABLE || true
      export NCCL_IB_HCA="${KIMODO_NCCL_FORCE_HCA:-${rail_hca}}"
      export NCCL_IB_GID_INDEX="${mode#force-gid=}"
      export NCCL_IB_ROCE_VERSION_NUM="${NCCL_IB_ROCE_VERSION_NUM:-2}"
      export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
      echo "Kimodo force-gid: NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} NCCL_IB_ROCE_VERSION_NUM=${NCCL_IB_ROCE_VERSION_NUM}" >&2
      ;;
    force-single)
      # One rail only — isolates whether multi-HCA rail list is the hang source.
      unset NCCL_IB_DISABLE || true
      unset NCCL_IB_GID_INDEX || true
      export NCCL_IB_HCA="${KIMODO_NCCL_FORCE_HCA:-mlx5_0:1}"
      export NCCL_IB_ROCE_VERSION_NUM="${NCCL_IB_ROCE_VERSION_NUM:-2}"
      export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
      echo "Kimodo force-single: HCA=${NCCL_IB_HCA} GID=unset ROCE=${NCCL_IB_ROCE_VERSION_NUM}" >&2
      ;;
    respect|*)
      if [[ -z "${NCCL_IB_HCA:-}" ]]; then
        export NCCL_IB_HCA="mlx5"
      fi
      ;;
  esac

  if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]] && command -v ip >/dev/null 2>&1; then
    local ifname
    ifname="$(ip -o -4 route show to default 2>/dev/null | awk '{print $5}' | head -1 || true)"
    if [[ -n "${ifname}" ]]; then
      export NCCL_SOCKET_IFNAME="${ifname}"
    fi
  fi

  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  unset NCCL_ASYNC_ERROR_HANDLING 2>/dev/null || true

  if [[ "${KIMODO_NCCL_DEBUG:-0}" == "1" ]]; then
    export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
    export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
  fi

  echo "Kimodo NCCL/RDMA env: mode=${mode} NCCL_IB_HCA=${NCCL_IB_HCA:-} NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-unset} NCCL_IB_ROCE_VERSION_NUM=${NCCL_IB_ROCE_VERSION_NUM:-} NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-auto} NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}" >&2
}
