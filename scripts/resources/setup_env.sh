#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
bootstrap_python="${KIMODO_BOOTSTRAP_PYTHON:-python3}"
venv_dir="${KIMODO_VENV:-${project_root}/.venv}"
system_site_packages=false
flowmatching_repo=""
skip_flowmatching=false
with_motion_correction=false
dependency_lock="${project_root}/resources/dependencies.lock.yaml"
constraints_file="${KIMODO_PIP_CONSTRAINTS:-${project_root}/requirements-training-server.txt}"

usage() {
  echo "Usage: $0 [--system-site-packages] [--with-motion-correction]"
  echo "          [--flowmatching-repo /path/to/kimodo-flowmatching] [--skip-flowmatching]"
  echo "Environment overrides: KIMODO_BOOTSTRAP_PYTHON, KIMODO_VENV"
  echo "Dependency constraints: KIMODO_PIP_CONSTRAINTS (default: requirements-training-server.txt)"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system-site-packages)
      system_site_packages=true
      shift
      ;;
    --flowmatching-repo)
      if [[ $# -lt 2 ]]; then
        echo "--flowmatching-repo requires a path" >&2
        exit 2
      fi
      flowmatching_repo="$2"
      shift 2
      ;;
    --skip-flowmatching)
      skip_flowmatching=true
      shift
      ;;
    --with-motion-correction)
      with_motion_correction=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  if [[ "${system_site_packages}" == true ]]; then
    "${bootstrap_python}" -m venv --system-site-packages "${venv_dir}"
  else
    "${bootstrap_python}" -m venv "${venv_dir}"
  fi
else
  actual_system_site="$(${venv_dir}/bin/python - <<'PY'
import sys
print("true" if any("site-packages" in path and path.startswith(sys.base_prefix) for path in sys.path) and sys.prefix != sys.base_prefix else "false")
PY
)"
  if [[ "${actual_system_site}" != "${system_site_packages}" ]]; then
    echo "Existing venv isolation mode does not match this invocation: ${venv_dir}" >&2
    echo "Existing system-site-packages=${actual_system_site}, requested=${system_site_packages}." >&2
    echo "Use the matching flag or choose a new KIMODO_VENV; the script will not mutate an existing venv." >&2
    exit 2
  fi
fi

if [[ ! -f "${constraints_file}" ]]; then
  echo "Training constraints file is missing: ${constraints_file}" >&2
  exit 2
fi
if [[ "${with_motion_correction}" == true ]]; then
  command -v cmake >/dev/null || { echo "--with-motion-correction requires cmake" >&2; exit 2; }
  command -v c++ >/dev/null || { echo "--with-motion-correction requires a C++ compiler" >&2; exit 2; }
  "${venv_dir}/bin/python" -m pip install -c "${constraints_file}" -e "${project_root}[train]"
else
  SKIP_MOTION_CORRECTION_IN_SETUP=1 \
    "${venv_dir}/bin/python" -m pip install -c "${constraints_file}" -e "${project_root}[train]"
fi
if [[ "${skip_flowmatching}" == true && -n "${flowmatching_repo}" ]]; then
  echo "--skip-flowmatching and --flowmatching-repo are mutually exclusive" >&2
  exit 2
fi
if [[ "${skip_flowmatching}" != true && -z "${flowmatching_repo}" ]]; then
  flowmatching_repo="${project_root}/.deps/kimodo-flowmatching"
fi
expected_flow_revision="$(awk '/^[[:space:]]+revision:/ {print $2; exit}' "${dependency_lock}")"
flowmatching_remote="$(awk '/^[[:space:]]+remote:/ {print $2; exit}' "${dependency_lock}")"
if [[ "${skip_flowmatching}" != true && ! -d "${flowmatching_repo}/.git" ]]; then
  if [[ -e "${flowmatching_repo}" ]]; then
    echo "Default Flow Matching path exists but is not a Git checkout: ${flowmatching_repo}" >&2
    exit 2
  fi
  mkdir -p "$(dirname -- "${flowmatching_repo}")"
  git clone "${flowmatching_remote}" "${flowmatching_repo}"
  git -C "${flowmatching_repo}" checkout --detach "${expected_flow_revision}"
fi
if [[ "${skip_flowmatching}" != true ]]; then
  if [[ ! -f "${flowmatching_repo}/pyproject.toml" ]]; then
    echo "Flow Matching checkout is invalid: ${flowmatching_repo}" >&2
    exit 2
  fi
  actual_flow_revision="$(git -C "${flowmatching_repo}" rev-parse HEAD)"
  if [[ ! "${expected_flow_revision}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Invalid flowmatching revision in ${dependency_lock}" >&2
    exit 2
  fi
  if [[ "${actual_flow_revision}" != "${expected_flow_revision}" ]]; then
    echo "Flow Matching checkout revision mismatch." >&2
    echo "Expected: ${expected_flow_revision}" >&2
    echo "Actual:   ${actual_flow_revision}" >&2
    echo "Run: git -C ${flowmatching_repo} checkout ${expected_flow_revision}" >&2
    exit 2
  fi
  if [[ -n "$(git -C "${flowmatching_repo}" status --porcelain --untracked-files=all)" ]] \
      && [[ "${KIMODO_ALLOW_DIRTY_FLOWMATCHING:-0}" != 1 ]]; then
    echo "Flow Matching checkout is dirty; refuse an untracked converter implementation." >&2
    echo "Commit/stash changes, or explicitly set KIMODO_ALLOW_DIRTY_FLOWMATCHING=1 for development." >&2
    exit 2
  fi
  "${venv_dir}/bin/python" -m pip install -c "${constraints_file}" -e "${flowmatching_repo}"
fi
"${venv_dir}/bin/python" -m pip check
KIMODO_SETUP_PROJECT_ROOT="${project_root}" KIMODO_SETUP_VENV="${venv_dir}" \
  "${venv_dir}/bin/python" - <<'PY'
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import torch

project_root = Path(os.environ["KIMODO_SETUP_PROJECT_ROOT"]).resolve()
venv_dir = Path(os.environ["KIMODO_SETUP_VENV"]).resolve()
try:
    repo_commit = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
except (OSError, subprocess.CalledProcessError):
    repo_commit = None

packages = {}
for name in ("torch", "numpy", "transformers", "peft", "omegaconf", "huggingface-hub"):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass

receipt = {
    "schema_version": 1,
    "event": "kimodo_environment_ready",
    "python": platform.python_version(),
    "platform": platform.platform(),
    "repo_commit": repo_commit,
    "packages": packages,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "include_system_site_packages": any(
        "site-packages" in value and value.startswith(sys.base_prefix)
        for value in sys.path
    ) and sys.prefix != sys.base_prefix,
}
receipt_path = venv_dir / "kimodo-environment.json"
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(receipt, sort_keys=True))
PY
