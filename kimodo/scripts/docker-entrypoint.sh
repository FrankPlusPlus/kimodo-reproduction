#!/usr/bin/env bash
set -euo pipefail

HOST_UID="${HOST_UID:-}"
HOST_GID="${HOST_GID:-}"
HOST_USER="${HOST_USER:-user}"

# Hanhai/Kubeflow Notebooks SSH as this account into /home/jovyan.
NB_USER="${NB_USER:-jovyan}"
NB_UID="${NB_UID:-1000}"
NB_GID="${NB_GID:-100}"

normalize_ssh_public_key_line() {
  # YAML folded scalars and UI pastes often inject newlines/spaces into the
  # key body. Collapse horizontal/vertical whitespace to one ssh public-key line.
  printf '%s' "$1" | tr -s '[:space:]' ' ' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

ensure_notebook_user() {
  local home_dir="/home/${NB_USER}"

  if ! getent group "${NB_GID}" >/dev/null 2>&1; then
    groupadd -g "${NB_GID}" "${NB_USER}" 2>/dev/null \
      || groupadd -g "${NB_GID}" "users" 2>/dev/null \
      || true
  fi

  if ! getent passwd "${NB_USER}" >/dev/null 2>&1; then
    if getent passwd "${NB_UID}" >/dev/null 2>&1; then
      echo "Kimodo SSH: uid ${NB_UID} already exists; not recreating ${NB_USER}." >&2
    else
      # /home/jovyan is commonly a mounted Notebook PVC; do not force useradd -m.
      useradd -M -u "${NB_UID}" -g "${NB_GID}" -d "${home_dir}" -s /bin/bash "${NB_USER}" 2>/dev/null \
        || useradd -M -u "${NB_UID}" -d "${home_dir}" -s /bin/bash "${NB_USER}" 2>/dev/null \
        || true
    fi
  fi

  install -d -m 0755 "${home_dir}" 2>/dev/null || true
  if getent passwd "${NB_USER}" >/dev/null 2>&1; then
    chown "${NB_UID}:${NB_GID}" "${home_dir}" 2>/dev/null || true
    # OpenSSH rejects "!" locked accounts even for publickey logins.
    usermod -p '*' "${NB_USER}" 2>/dev/null || true
  fi
}

install_authorized_keys_for_user() {
  local login_user="$1"
  local keys_file="$2"
  local home_dir
  local ssh_dir
  local auth_path

  home_dir="$(getent passwd "${login_user}" | awk -F: '{print $6}')"
  [[ -n "${home_dir}" ]] || return 0
  ssh_dir="${home_dir}/.ssh"
  auth_path="${ssh_dir}/authorized_keys"

  install -d -m 0700 "${ssh_dir}" 2>/dev/null || return 0
  cp -f "${keys_file}" "${auth_path}" 2>/dev/null || return 0
  chmod 0600 "${auth_path}" 2>/dev/null || true
  if getent passwd "${login_user}" >/dev/null 2>&1; then
    local uid gid
    uid="$(getent passwd "${login_user}" | awk -F: '{print $3}')"
    gid="$(getent passwd "${login_user}" | awk -F: '{print $4}')"
    chown -R "${uid}:${gid}" "${ssh_dir}" 2>/dev/null || true
  fi
  echo "Kimodo SSH: installed authorized_keys for ${login_user} at ${auth_path}" >&2
}

# Collect public keys for the in-Pod sshd. The company gateway authenticates the
# website-registered key on the first hop, then re-authenticates against this
# process on port 22 as the notebook user (usually jovyan).
collect_ssh_authorized_keys() {
  local dest="$1"
  local -a sources=()
  local key_count=0
  local source_path=""
  local raw_key=""
  local normalized=""

  : >"${dest}"

  raw_key="${KIMODO_SSH_PUBLIC_KEY:-${SSH_PUBLIC_KEY:-${USER_SSH_PUBLIC_KEY:-}}}"
  if [[ -n "${raw_key}" ]]; then
    normalized="$(normalize_ssh_public_key_line "${raw_key}")"
    if [[ "${normalized}" == ssh-* || "${normalized}" == ecdsa-* ]]; then
      printf '%s\n' "${normalized}" >>"${dest}"
      sources+=("env:public-key")
    else
      echo "Kimodo SSH warning: public key env was set but did not look like an ssh public key." >&2
    fi
  fi

  for source_path in \
    ${KIMODO_SSH_AUTHORIZED_KEYS_FILE:-} \
    "/home/${NB_USER}/.ssh/authorized_keys" \
    /root/.ssh/authorized_keys \
    /etc/ssh/authorized_keys \
    /etc/ssh/authorized_keys/jovyan \
    /etc/ssh/authorized_keys/kimodo \
    /var/run/secrets/kimodo/authorized_keys \
    /var/run/secrets/kubernetes.io/ssh/authorized_keys
  do
    [[ -n "${source_path}" && -f "${source_path}" && -s "${source_path}" ]] || continue
    # Normalize each non-comment line in case the platform wrote a folded key.
    while IFS= read -r line || [[ -n "${line}" ]]; do
      [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
      normalized="$(normalize_ssh_public_key_line "${line}")"
      [[ -n "${normalized}" ]] || continue
      printf '%s\n' "${normalized}" >>"${dest}"
    done <"${source_path}"
    sources+=("file:${source_path}")
  done

  if [[ ! -s "${dest}" ]]; then
    return 1
  fi

  local cleaned
  cleaned="$(
    awk 'NF && $1 !~ /^#/ && !seen[$0]++ { print }' "${dest}"
  )"
  if [[ -z "${cleaned}" ]]; then
    : >"${dest}"
    return 1
  fi
  printf '%s\n' "${cleaned}" >"${dest}"
  chmod 0600 "${dest}"

  key_count="$(awk 'NF { n++ } END { print n+0 }' "${dest}")"
  if [[ "${key_count}" -le 0 ]]; then
    return 1
  fi

  echo "Kimodo SSH authorized_keys: ${key_count} key(s) from ${sources[*]}" >&2
  return 0
}

start_sshd() {
  [[ "${KIMODO_SSH_ENABLED:-1}" == "1" ]] || return 0
  command -v sshd >/dev/null 2>&1 || {
    echo "Kimodo SSH server not started: sshd is not installed." >&2
    return 0
  }
  [[ "$(id -u)" == 0 ]] || {
    echo "Kimodo SSH server not started: the container is not running as root." >&2
    return 0
  }

  ensure_notebook_user

  # Notebook platforms may make the image root filesystem read-only. Generate
  # per-Pod host keys under /tmp and never let an optional SSH failure terminate
  # the actual container workload.
  if ! (
    set -e
    runtime_dir="${KIMODO_SSH_RUNTIME_DIR:-/tmp/kimodo-sshd}"
    install -d -m 0700 "${runtime_dir}"
    if ! install -d -m 0755 /run/sshd 2>/dev/null; then
      install -d -m 0755 "${runtime_dir}/run"
      ln -sfn "${runtime_dir}/run" /run/sshd 2>/dev/null || true
    fi
    rm -f "${runtime_dir}"/ssh_host_* "${runtime_dir}/sshd.pid" "${runtime_dir}/authorized_keys"
    ssh-keygen -q -t ed25519 -N '' -f "${runtime_dir}/ssh_host_ed25519_key"
    ssh-keygen -q -t rsa -b 3072 -N '' -f "${runtime_dir}/ssh_host_rsa_key"

    sshd_args=(
      -E "${runtime_dir}/sshd.log"
      -h "${runtime_dir}/ssh_host_ed25519_key"
      -h "${runtime_dir}/ssh_host_rsa_key"
      -o "PidFile=${runtime_dir}/sshd.pid"
      -o UsePAM=no
      -o PrintMotd=no
      -o X11Forwarding=no
      -o PasswordAuthentication=no
      -o KbdInteractiveAuthentication=no
      -o PubkeyAuthentication=yes
      -o "PermitRootLogin=prohibit-password"
      -o StrictModes=no
    )

    if collect_ssh_authorized_keys "${runtime_dir}/authorized_keys"; then
      # OpenSSH rejects AuthorizedKeysFile under world-writable sticky dirs such
      # as /tmp for non-root users ("RSA key is not allowed"). Keep host keys in
      # /tmp, but publish authorized_keys to a safe path and user homes.
      auth_keys_file=""
      if cp -f "${runtime_dir}/authorized_keys" /etc/ssh/kimodo_authorized_keys 2>/dev/null; then
        chmod 0644 /etc/ssh/kimodo_authorized_keys 2>/dev/null || true
        auth_keys_file="/etc/ssh/kimodo_authorized_keys"
      fi
      install_authorized_keys_for_user root "${runtime_dir}/authorized_keys" || true
      install_authorized_keys_for_user "${NB_USER}" "${runtime_dir}/authorized_keys" || true
      if [[ -n "${auth_keys_file}" ]]; then
        sshd_args+=(
          -o "AuthorizedKeysFile=${auth_keys_file} .ssh/authorized_keys"
        )
      else
        # Fall back to the normal per-user ~/.ssh/authorized_keys locations.
        sshd_args+=(
          -o "AuthorizedKeysFile=.ssh/authorized_keys"
        )
      fi
      sshd_args+=(
        -o "PubkeyAcceptedAlgorithms=+ssh-rsa,rsa-sha2-256,rsa-sha2-512"
      )
    else
      echo "Kimodo SSH warning: no authorized_keys found." >&2
      echo "Kimodo SSH warning: gateway auth can succeed while Pod login fails with Permission denied (publickey)." >&2
      echo "Kimodo SSH warning: set KIMODO_SSH_PUBLIC_KEY or rely on platform-mounted /home/${NB_USER}/.ssh/authorized_keys." >&2
    fi

    /usr/sbin/sshd "${sshd_args[@]}"

    for _ in 1 2 3 4 5; do
      if ss -lnt 2>/dev/null | awk '{print $4}' | grep -Eq '(:|\.)22$'; then
        echo "Kimodo SSH server is listening on port 22 (notebook user: ${NB_USER})." >&2
        exit 0
      fi
      if command -v netstat >/dev/null 2>&1 && netstat -lnt 2>/dev/null | awk '{print $4}' | grep -Eq '(:|\.)22$'; then
        echo "Kimodo SSH server is listening on port 22 (notebook user: ${NB_USER})." >&2
        exit 0
      fi
      if [[ -f "${runtime_dir}/sshd.pid" ]] && kill -0 "$(cat "${runtime_dir}/sshd.pid")" 2>/dev/null; then
        echo "Kimodo SSH server started (pid $(cat "${runtime_dir}/sshd.pid"), notebook user: ${NB_USER})." >&2
        exit 0
      fi
      sleep 0.2
    done
    echo "Kimodo SSH server did not open port 22; see ${runtime_dir}/sshd.log" >&2
    if [[ -f "${runtime_dir}/sshd.log" ]]; then
      tail -n 40 "${runtime_dir}/sshd.log" >&2 || true
    fi
    exit 1
  ); then
    echo "Kimodo SSH server could not start; continuing without SSH." >&2
  fi
  return 0
}

# Kubernetes commonly sets runAsUser. A non-root process cannot and need not
# edit /etc/passwd; preserve the scheduler identity and execute directly.
if [[ "$(id -u)" != 0 ]]; then
  echo "Kimodo entrypoint: running as uid=$(id -u); SSH server requires root." >&2
  exec "$@"
fi

start_sshd

if [[ -z "${HOST_UID}" || -z "${HOST_GID}" ]]; then
  if [[ -d /workspace ]]; then
    HOST_UID="$(stat -c %u /workspace)"
    HOST_GID="$(stat -c %g /workspace)"
  else
    HOST_UID="${HOST_UID:-1000}"
    HOST_GID="${HOST_GID:-1000}"
  fi
fi

# On read-only root filesystems groupadd/useradd can fail; keep the workload
# alive and fall back to executing as the existing root identity when needed.
if ! getent group "${HOST_GID}" >/dev/null 2>&1; then
  groupadd -g "${HOST_GID}" "${HOST_USER}" 2>/dev/null || true
fi

if ! getent passwd "${HOST_UID}" >/dev/null 2>&1; then
  useradd -m -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/bash "${HOST_USER}" 2>/dev/null || true
fi

if command -v gosu >/dev/null 2>&1 && getent passwd "${HOST_UID}" >/dev/null 2>&1; then
  exec gosu "${HOST_UID}:${HOST_GID}" "$@"
fi

echo "Kimodo entrypoint: gosu/user mapping unavailable; continuing as root." >&2
exec "$@"
