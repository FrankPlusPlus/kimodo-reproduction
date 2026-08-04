#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
bootstrap_python="${KIMODO_BOOTSTRAP_PYTHON:-python3}"
venv_dir="${KIMODO_VENV:-${project_root}/.venv}"
system_site_packages=false
flowmatching_repo=""
with_motion_correction=false
dependency_lock="${project_root}/resources/dependencies.lock.yaml"
constraints_file="${KIMODO_PIP_CONSTRAINTS:-${project_root}/requirements-training-server.txt}"

usage() {
  echo "Usage: $0 [--system-site-packages] [--with-motion-correction] [--flowmatching-repo /path/to/kimodo-flowmatching]"
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
fi

"${venv_dir}/bin/python" -m pip install --upgrade pip
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
if [[ -n "${flowmatching_repo}" ]]; then
  if [[ ! -f "${flowmatching_repo}/pyproject.toml" ]]; then
    echo "Flow Matching checkout is invalid: ${flowmatching_repo}" >&2
    exit 2
  fi
  expected_flow_revision="$(awk '/^[[:space:]]+revision:/ {print $2; exit}' "${dependency_lock}")"
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
}
receipt_path = venv_dir / "kimodo-environment.json"
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(receipt, sort_keys=True))
PY
