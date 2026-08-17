#!/usr/bin/env bash
# 开发机：500k wd fork 的 stratified 测评结果邮件（与训练 loss 守护分开）。
#
#   nohup bash /home/share/yzt/kimodo-reproduction/scripts/watch_wd03_from500k_eval_alert.sh \
#     >> /home/share/yzt/kimodo-reproduction/watch/v2-1m-hostnet-wd03-from500k/eval-nohup.log 2>&1 &
set -euo pipefail

export KIMODO_CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
export KIMODO_STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
export KIMODO_RUN_DIR="${KIMODO_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-wd03-from500k}"
export KIMODO_EVAL_ROOT="${KIMODO_EVAL_ROOT:-${KIMODO_STORAGE_ROOT}/eval-results/v2-1m-hostnet-wd03-from500k-stratified10pct}"
export KIMODO_PRIOR_EVAL_ROOT="${KIMODO_PRIOR_EVAL_ROOT:-${KIMODO_STORAGE_ROOT}/eval-results/v2-1m-hostnet-stratified10pct}"
export KIMODO_WATCH_DIR="${KIMODO_WATCH_DIR:-${KIMODO_CODE_ROOT}/watch/v2-1m-hostnet-wd03-from500k}"
export KIMODO_EVAL_MILESTONE_START="${KIMODO_EVAL_MILESTONE_START:-520000}"
export KIMODO_EVAL_MILESTONE_EVERY="${KIMODO_EVAL_MILESTONE_EVERY:-20000}"
export KIMODO_FORK_BASELINE_STEP="${KIMODO_FORK_BASELINE_STEP:-750000}"
export KIMODO_HEAD_TO_HEAD="${KIMODO_HEAD_TO_HEAD:-1}"
export KIMODO_TRAIN_WATCH="${KIMODO_TRAIN_WATCH:-0}"
export KIMODO_EVAL_WATCH="${KIMODO_EVAL_WATCH:-1}"
export KIMODO_ALERT_EMAIL="${KIMODO_ALERT_EMAIL:-171024830@qq.com}"
export KIMODO_ALERT_SMTP_HOST="${KIMODO_ALERT_SMTP_HOST:-smtp.qq.com}"
export KIMODO_ALERT_SMTP_PORT="${KIMODO_ALERT_SMTP_PORT:-465}"
export KIMODO_ALERT_SMTP_USER="${KIMODO_ALERT_SMTP_USER:-${KIMODO_ALERT_EMAIL}}"

# shellcheck disable=SC1091
source "${KIMODO_CODE_ROOT}/scripts/load_kimodo_env.sh"
kimodo_load_env_files

WATCH_DIR="${KIMODO_WATCH_DIR}"
mkdir -p "${WATCH_DIR}" "${KIMODO_EVAL_ROOT}"
cd "${KIMODO_CODE_ROOT}"

LOCK="${WATCH_DIR}/eval-watch.lock"
if command -v flock >/dev/null 2>&1; then
  exec 9>"${LOCK}"
  if ! flock -n 9; then
    echo "another eval watcher already holds ${LOCK}" >&2
    exit 2
  fi
fi

echo "wd03-from500k eval daemon: eval=${KIMODO_EVAL_ROOT}"
echo "wd03-from500k eval daemon: prior=${KIMODO_PRIOR_EVAL_ROOT}"
exec python3 "${KIMODO_CODE_ROOT}/scripts/watch_kf_smooth_daemon.py" "$@"
