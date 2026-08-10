#!/bin/sh
# Dump RDMA/GID + netns evidence into Job logs (no training).
export KIMODO_CODE_ROOT=/home/share/yzt/kimodo-reproduction
cd "${KIMODO_CODE_ROOT}" || exit 2

echo "=== kimodo GID dump begin hostname=$(hostname) ==="
echo "=== ibv_devices ==="
ibv_devices 2>&1 || true

echo "=== ip -br a ==="
ip -br a 2>&1 || true

echo "=== ip route ==="
ip route 2>&1 | head -40 || true

echo "=== host netns reachable? ==="
ls -la /host/proc/1/ns/net 2>&1 || true
ls -la /proc/1/ns/net 2>&1 || true
if [ -e /proc/1/ns/net ]; then
  echo "=== ip -br a via nsenter netns (if allowed) ==="
  timeout 5 nsenter --net=/proc/1/ns/net ip -br a 2>&1 | head -40 || echo "(nsenter net failed)"
fi

echo "=== /dev/infiniband names ==="
ls /dev/infiniband 2>&1 || true

echo "=== show_gids ==="
if command -v show_gids >/dev/null 2>&1; then
  timeout 20 show_gids 2>&1 || echo "(show_gids failed/timeout)"
else
  echo "(show_gids not installed)"
fi

echo "=== ibv_devinfo ==="
if command -v ibv_devinfo >/dev/null 2>&1; then
  timeout 30 ibv_devinfo 2>&1 | head -200 || echo "(ibv_devinfo failed/timeout)"
else
  echo "(ibv_devinfo missing)"
fi

echo "=== ibv_devinfo -v mlx5_0 ==="
if command -v ibv_devinfo >/dev/null 2>&1; then
  timeout 20 ibv_devinfo -v -d mlx5_0 2>&1 | head -260 || echo "(ibv_devinfo -v failed/timeout)"
fi

echo "=== sysfs gid mlx5_0 ==="
sh "${KIMODO_CODE_ROOT}/scripts/dump_gid_mlx5_0.sh" mlx5_0 1

echo "=== env NCCL_*/PET_* ==="
env | grep -E '^NCCL_|^PET_' | sort || true

echo "=== kimodo GID dump done ==="
sleep 20
exit 0
