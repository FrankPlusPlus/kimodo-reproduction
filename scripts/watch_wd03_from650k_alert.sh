#!/usr/bin/env bash
# 开发机常驻：650k wd=0.3 fork 的 loss 异常 + 每 10k 梯度健康 + 700k 起每 50k 测评。
# 与 kf-smooth / Official-vs-695k 守护分开跑，避免抢同一封对照邮件。
#
#   nohup bash /home/share/yzt/kimodo-reproduction/scripts/watch_wd03_from650k_alert.sh \
#     >> /home/share/yzt/kimodo-reproduction/watch/v2-1m-hostnet-wd03-from650k/nohup.log 2>&1 &
set -euo pipefail

export KIMODO_CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
export KIMODO_STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
export KIMODO_RUN_DIR="${KIMODO_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-wd03-from650k}"
export KIMODO_EVAL_ROOT="${KIMODO_EVAL_ROOT:-${KIMODO_STORAGE_ROOT}/eval-results/v2-1m-hostnet-wd03-from650k-stratified10pct}"
export KIMODO_PRIOR_EVAL_ROOT="${KIMODO_PRIOR_EVAL_ROOT:-${KIMODO_STORAGE_ROOT}/eval-results/v2-1m-hostnet-kf-smooth-lr1e5-stratified10pct}"
export KIMODO_WATCH_DIR="${KIMODO_WATCH_DIR:-${KIMODO_CODE_ROOT}/watch/v2-1m-hostnet-wd03-from650k}"
export KIMODO_HEALTH_10K_START="${KIMODO_HEALTH_10K_START:-660000}"
export KIMODO_HEALTH_10K_EVERY="${KIMODO_HEALTH_10K_EVERY:-10000}"
export KIMODO_EVAL_MILESTONE_START="${KIMODO_EVAL_MILESTONE_START:-700000}"
export KIMODO_EVAL_MILESTONE_EVERY="${KIMODO_EVAL_MILESTONE_EVERY:-50000}"
export KIMODO_FORK_BASELINE_STEP="${KIMODO_FORK_BASELINE_STEP:-650000}"
export KIMODO_HEAD_TO_HEAD="${KIMODO_HEAD_TO_HEAD:-0}"
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

if [[ -z "${KIMODO_ALERT_SMTP_PASSWORD:-}" ]]; then
  echo "warning: KIMODO_ALERT_SMTP_PASSWORD empty; daemon will wait and send after you add the QQ SMTP 授权码 to ${KIMODO_CODE_ROOT}/.env" >&2
fi

LOCK="${WATCH_DIR}/watch.lock"
if command -v flock >/dev/null 2>&1; then
  exec 9>"${LOCK}"
  if ! flock -n 9; then
    echo "another loss watcher already holds ${LOCK}" >&2
    exit 2
  fi
fi

echo "wd03-from650k daemon: run=${KIMODO_RUN_DIR}"
echo "wd03-from650k daemon: eval=${KIMODO_EVAL_ROOT}"
echo "wd03-from650k daemon: health=${KIMODO_HEALTH_10K_START}+${KIMODO_HEALTH_10K_EVERY}"
echo "wd03-from650k daemon: watch=${WATCH_DIR}"
echo "wd03-from650k daemon: mail=${KIMODO_ALERT_EMAIL}"
exec python3 "${KIMODO_CODE_ROOT}/scripts/watch_kf_smooth_daemon.py" "$@"
