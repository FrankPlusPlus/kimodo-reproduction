#!/bin/sh
# Hang-safe GID dump for one HCA. Args: [device] [port]
# Example: sh dump_gid_mlx5_0.sh mlx5_0 1
DEV=${1:-mlx5_0}
PORT=${2:-1}
BASE=/sys/class/infiniband/$DEV/ports/$PORT
echo "=== $DEV port $PORT ==="
if [ ! -d "$BASE/gids" ]; then
  echo "(no $BASE/gids)"
  exit 0
fi
i=0
while [ "$i" -le 7 ]; do
  if [ -e "$BASE/gids/$i" ]; then
    g=$(timeout 1 cat "$BASE/gids/$i" 2>/dev/null || echo TIMEOUT)
    t=$(timeout 1 cat "$BASE/gid_attrs/types/$i" 2>/dev/null || echo TIMEOUT)
  else
    g=MISSING
    t=MISSING
  fi
  echo "idx=$i type=$t gid=$g"
  i=$((i + 1))
done
