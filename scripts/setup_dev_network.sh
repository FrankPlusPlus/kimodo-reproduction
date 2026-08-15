#!/usr/bin/env bash
# Install Kimodo notebook proxy helpers into the current user's home.
# Safe to re-run (idempotent markers). Does not store registry passwords.
#
# Usage:
#   bash /home/share/yzt/kimodo-reproduction/scripts/setup_dev_network.sh
#   KIMODO_DEV_PROXY_PORT=7993 bash .../setup_dev_network.sh
#   KIMODO_SETUP_CURSOR_PROXY=0 bash .../setup_dev_network.sh   # skip settings.json
set -euo pipefail

proxy_port="${KIMODO_DEV_PROXY_PORT:-7993}"
proxy_url="http://127.0.0.1:${proxy_port}"
no_proxy_list='localhost,127.0.0.1,::1,.inner.ai.kingsoft.com,hub.inner.ai.kingsoft.com,172.20.0.0/16'
setup_cursor="${KIMODO_SETUP_CURSOR_PROXY:-1}"

profile_marker_begin='# >>> kimodo dev network profile >>>'
profile_marker_end='# <<< kimodo dev network profile <<<'
bashrc_marker_begin='# >>> kimodo local vpn proxy >>>'
bashrc_marker_end='# <<< kimodo local vpn proxy <<<'

profile_block=$(cat <<EOF
${profile_marker_begin}
# always-on proxy via SSH RemoteForward (for IDE extension processes too)
# Docs: docs/dev_notebook_network.zh-CN.md
export HTTP_PROXY=${proxy_url}
export HTTPS_PROXY=${proxy_url}
export http_proxy=${proxy_url}
export https_proxy=${proxy_url}
export NO_PROXY='${no_proxy_list}'
export no_proxy="\${NO_PROXY}"
${profile_marker_end}
EOF
)

bashrc_block=$(cat <<EOF
${bashrc_marker_begin}
# Requires SSH RemoteForward ${proxy_port} -> local Clash/VPN.
# Company intranet must bypass the proxy or Harbor TLS breaks.
# Docs: docs/dev_notebook_network.zh-CN.md
_KIMODO_NO_PROXY='${no_proxy_list}'
_KIMODO_PROXY_URL='${proxy_url}'
proxy_on() {
  export http_proxy="\${_KIMODO_PROXY_URL}"
  export https_proxy="\${_KIMODO_PROXY_URL}"
  export HTTP_PROXY="\${_KIMODO_PROXY_URL}"
  export HTTPS_PROXY="\${_KIMODO_PROXY_URL}"
  export ALL_PROXY="\${_KIMODO_PROXY_URL}"
  export NO_PROXY="\${_KIMODO_NO_PROXY}"
  export no_proxy="\${_KIMODO_NO_PROXY}"
  echo "proxy ON -> \${_KIMODO_PROXY_URL} (NO_PROXY includes company intranet)"
}
proxy_off() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy no_proxy NO_PROXY
  echo "proxy OFF"
}
proxy_status() {
  if [[ -n "\${http_proxy:-}\${https_proxy:-}\${ALL_PROXY:-}" ]]; then
    echo "proxy env: http_proxy=\${http_proxy:-} https_proxy=\${https_proxy:-} ALL_PROXY=\${ALL_PROXY:-}"
    echo "NO_PROXY=\${NO_PROXY:-}"
  else
    echo "proxy env: off"
  fi
  if (echo >/dev/tcp/127.0.0.1/${proxy_port}) >/dev/null 2>&1; then
    echo "tunnel ${proxy_port}: open"
  else
    echo "tunnel ${proxy_port}: closed (reconnect SSH / keep Cursor session)"
    return 0
  fi
  # Distinguish "port listening" from "Clash node actually reaches GitHub".
  local baidu_code github_code
  baidu_code="\$(curl -sS -m 8 -o /dev/null -w '%{http_code}' -x "\${_KIMODO_PROXY_URL}" https://www.baidu.com 2>/dev/null || true)"
  github_code="\$(curl -sS -m 10 -o /dev/null -w '%{http_code}' -x "\${_KIMODO_PROXY_URL}" https://github.com 2>/dev/null || true)"
  [[ -n "\${baidu_code}" ]] || baidu_code=000
  [[ -n "\${github_code}" ]] || github_code=000
  echo "probe baidu=\${baidu_code} github=\${github_code}"
  if [[ "\${github_code}" != "200" ]]; then
    echo "egress: BROKEN — tunnel is up but GitHub TLS fails (fix local Clash node / RemoteForward target port)"
    echo "hint: on laptop, switch Clash node, confirm mixed-port, then reconnect SSH -R ${proxy_port}:127.0.0.1:<clash-port>"
  else
    echo "egress: OK — GitHub reachable via tunnel"
  fi
}
alias codex='HTTP_PROXY=${proxy_url} HTTPS_PROXY=${proxy_url} ALL_PROXY=${proxy_url} NO_PROXY=${no_proxy_list} command codex'
${bashrc_marker_end}
EOF
)

replace_marked_block() {
  local file="$1"
  local begin="$2"
  local end="$3"
  local block="$4"
  mkdir -p "$(dirname -- "${file}")"
  touch "${file}"
  if grep -qF "${begin}" "${file}" 2>/dev/null; then
    local tmp
    tmp="$(mktemp)"
    awk -v begin="${begin}" -v end="${end}" '
      $0 == begin { skip=1; next }
      $0 == end { skip=0; next }
      !skip { print }
    ' "${file}" >"${tmp}"
    mv "${tmp}" "${file}"
  fi
  printf '\n%s\n' "${block}" >>"${file}"
}

replace_marked_block "${HOME}/.profile" "${profile_marker_begin}" "${profile_marker_end}" "${profile_block}"
replace_marked_block "${HOME}/.bashrc" "${bashrc_marker_begin}" "${bashrc_marker_end}" "${bashrc_block}"

if [[ "${setup_cursor}" == "1" ]]; then
  for settings in \
    "${HOME}/.cursor-server/data/Machine/settings.json" \
    "${HOME}/.vscode-server/data/Machine/settings.json"
  do
    mkdir -p "$(dirname -- "${settings}")"
    if [[ -f "${settings}" ]]; then
      python3 - "${settings}" "${proxy_url}" <<'PY' || true
import json, sys
path, proxy = sys.argv[1], sys.argv[2]
try:
    data = json.load(open(path, encoding="utf-8"))
except Exception:
    data = {}
if not isinstance(data, dict):
    data = {}
data["http.proxy"] = proxy
data["http.proxySupport"] = "on"
data["http.proxyStrictSSL"] = False
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)
    f.write("\n")
print(f"updated {path}")
PY
    else
      cat >"${settings}" <<EOF
{
    "http.proxy": "${proxy_url}",
    "http.proxySupport": "on",
    "http.proxyStrictSSL": false
}
EOF
      echo "created ${settings}"
    fi
  done
fi

cat <<EOF
Kimodo dev network installed for ${USER:-user} in ${HOME}
  proxy: ${proxy_url}
  NO_PROXY: ${no_proxy_list}
  shell: ~/.profile ~/.bashrc
  cursor machine json: ~/.cursor-server/data/Machine/settings.json
  template: scripts/resources/cursor-machine-settings.proxy.json
Next (Codex):
  1) Ensure local SSH RemoteForward ${proxy_port} -> your Clash/VPN
  2) Cursor: Kill Server on Host + reconnect (loads ~/.profile into remote host)
  3) Reload Window (picks up Machine settings.json http.proxy)
  4) proxy_status && curl -sS -o /dev/null -w 'google=%{http_code}\\n' https://www.google.com
  5) curl -sk -o /dev/null -w 'harbor=%{http_code}\\n' https://hub.inner.ai.kingsoft.com/v2/
Docs: /home/share/yzt/kimodo-reproduction/docs/dev_notebook_network.zh-CN.md
EOF
