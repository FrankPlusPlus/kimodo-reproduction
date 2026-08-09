#!/usr/bin/env bash
# Source optional secret/config env files without overriding variables that are
# already set in the process environment (platform injection still wins).
#
# Default search order:
#   1) $KIMODO_ENV_FILE (explicit)
#   2) $KIMODO_CODE_ROOT/.env
#   3) $KIMODO_STORAGE_ROOT/secrets/kimodo.env
#
# Keep real keys out of Git. Prefer chmod 600 on the PVC file. Shared PVC is
# still readable by anyone with access to /home/share — treat it as shared.

kimodo_env_file_load() {
  local path="$1"
  local line key value loaded=0

  [[ -f "${path}" && -r "${path}" ]] || return 0

  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    if [[ "${line}" == export[[:space:]]* ]]; then
      line="${line#export}"
      line="${line#"${line%%[![:space:]]*}"}"
    fi
    [[ "${line}" == *=* ]] || continue
    key="${line%%=*}"
    value="${line#*=}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    # Do not clobber platform-injected or already-exported values.
    if [[ -n "${!key+x}" ]]; then
      continue
    fi
    if [[ "${value}" =~ ^\".*\"$ || "${value}" =~ ^\'.*\'$ ]]; then
      value="${value:1:${#value}-2}"
    fi
    export "${key}=${value}"
    loaded=$((loaded + 1))
  done <"${path}"

  if (( loaded > 0 )); then
    echo "Kimodo env: loaded ${loaded} unset variable(s) from ${path}" >&2
  fi
}

kimodo_load_env_files() {
  local code_root="${KIMODO_CODE_ROOT:-}"
  local storage_root="${KIMODO_STORAGE_ROOT:-}"
  local -a candidates=()

  if [[ -n "${KIMODO_ENV_FILE:-}" ]]; then
    candidates+=("${KIMODO_ENV_FILE}")
  fi
  if [[ -n "${code_root}" ]]; then
    candidates+=("${code_root}/.env")
  fi
  if [[ -n "${storage_root}" ]]; then
    candidates+=("${storage_root}/secrets/kimodo.env")
  fi

  local path
  for path in "${candidates[@]}"; do
    kimodo_env_file_load "${path}"
  done
}
