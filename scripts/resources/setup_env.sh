#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
bootstrap_python="${KIMODO_BOOTSTRAP_PYTHON:-python3}"
venv_dir="${KIMODO_VENV:-${project_root}/.venv}"
system_site_packages=false
with_motion_correction=false
constraints_file="${KIMODO_PIP_CONSTRAINTS:-${project_root}/requirements-training-server.txt}"

usage() {
  echo "Usage: $0 [--system-site-packages] [--with-motion-correction]"
  echo "Environment overrides: KIMODO_BOOTSTRAP_PYTHON, KIMODO_VENV"
  echo "Dependency constraints: KIMODO_PIP_CONSTRAINTS (default: requirements-training-server.txt)"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system-site-packages)
      system_site_packages=true
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
