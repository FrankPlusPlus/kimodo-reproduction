#!/usr/bin/env bash
# 开发机常驻：loss 异常 + 每 10k 梯度健康 + 650k 起每 50k 只发 benchmark 测评。
# 电脑可以带走。SMTP 授权码可稍后写入 PVC .env，进程会自行加载。
#
#   nohup bash /home/share/yzt/kimodo-reproduction/scripts/watch_train_loss_alert.sh \
#     >> /home/share/yzt/kimodo-reproduction/watch/v2-1m-hostnet-kf-smooth-lr1e5/nohup.log 2>&1 &
set -euo pipefail

export KIMODO_CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
export KIMODO_STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
export KIMODO_RUN_DIR="${KIMODO_RUN_DIR:-${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-kf-smooth-lr1e5}"
export KIMODO_EVAL_ROOT="${KIMODO_EVAL_ROOT:-${KIMODO_STORAGE_ROOT}/eval-results/v2-1m-hostnet-kf-smooth-lr1e5-stratified10pct}"
export KIMODO_PRIOR_EVAL_ROOT="${KIMODO_PRIOR_EVAL_ROOT:-${KIMODO_STORAGE_ROOT}/eval-results/v2-1m-hostnet-stratified10pct}"
export KIMODO_WATCH_DIR="${KIMODO_WATCH_DIR:-${KIMODO_CODE_ROOT}/watch/v2-1m-hostnet-kf-smooth-lr1e5}"
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

echo "kf-smooth daemon: run=${KIMODO_RUN_DIR}"
echo "kf-smooth daemon: eval=${KIMODO_EVAL_ROOT}"
echo "kf-smooth daemon: watch=${WATCH_DIR}"
echo "kf-smooth daemon: mail=${KIMODO_ALERT_EMAIL}"
exec python3 "${KIMODO_CODE_ROOT}/scripts/watch_kf_smooth_daemon.py" "$@"
