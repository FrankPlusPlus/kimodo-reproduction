#!/usr/bin/env bash
set -euo pipefail

HOST_UID="${HOST_UID:-}"
HOST_GID="${HOST_GID:-}"
HOST_USER="${HOST_USER:-user}"

start_sshd() {
  [[ "${KIMODO_SSH_ENABLED:-1}" == "1" ]] || return 0
  command -v sshd >/dev/null 2>&1 || return 0
  [[ "$(id -u)" == 0 ]] || {
    echo "Kimodo SSH server not started: the container is not running as root." >&2
    return 0
  }

  # Notebook platforms may make the image root filesystem read-only. Generate
  # per-Pod host keys under /tmp and never let an optional SSH failure terminate
  # the actual container workload.
  if ! (
    set -e
    runtime_dir="${KIMODO_SSH_RUNTIME_DIR:-/tmp/kimodo-sshd}"
    install -d -m 0700 "${runtime_dir}"
    install -d -m 0755 /run/sshd
    rm -f "${runtime_dir}"/ssh_host_* "${runtime_dir}/sshd.pid"
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
    )
    if [[ -n "${KIMODO_SSH_PUBLIC_KEY:-}" ]]; then
      printf '%s\n' "${KIMODO_SSH_PUBLIC_KEY}" > "${runtime_dir}/authorized_keys"
      chmod 0600 "${runtime_dir}/authorized_keys"
      # The key lives below a world-writable sticky /tmp parent when the image
      # root is read-only. The private runtime directory itself is root-owned
      # mode 0700, so explicitly allow this reviewed absolute key path.
      sshd_args+=(
        -o "AuthorizedKeysFile=${runtime_dir}/authorized_keys"
        -o StrictModes=no
      )
    fi

    /usr/sbin/sshd "${sshd_args[@]}"
  ); then
    echo "Kimodo SSH server could not start; continuing without SSH." >&2
  fi
  return 0
}

# Kubernetes commonly sets runAsUser. A non-root process cannot and need not
# edit /etc/passwd; preserve the scheduler identity and execute directly.
if [[ "$(id -u)" != 0 ]]; then
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

if ! getent group "${HOST_GID}" >/dev/null 2>&1; then
  groupadd -g "${HOST_GID}" "${HOST_USER}"
fi

if ! getent passwd "${HOST_UID}" >/dev/null 2>&1; then
  useradd -m -u "${HOST_UID}" -g "${HOST_GID}" -s /bin/bash "${HOST_USER}"
fi

exec gosu "${HOST_UID}:${HOST_GID}" "$@"
