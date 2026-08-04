#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
bootstrap_python="${KIMODO_BOOTSTRAP_PYTHON:-python3}"
venv_dir="${KIMODO_VENV:-${project_root}/.venv}"
system_site_packages=false
flowmatching_repo=""
with_motion_correction=false

usage() {
  echo "Usage: $0 [--system-site-packages] [--with-motion-correction] [--flowmatching-repo /path/to/kimodo-flowmatching]"
  echo "Environment overrides: KIMODO_BOOTSTRAP_PYTHON, KIMODO_VENV"
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
if [[ "${with_motion_correction}" == true ]]; then
  command -v cmake >/dev/null || { echo "--with-motion-correction requires cmake" >&2; exit 2; }
  command -v c++ >/dev/null || { echo "--with-motion-correction requires a C++ compiler" >&2; exit 2; }
  "${venv_dir}/bin/python" -m pip install -e "${project_root}[train]"
else
  SKIP_MOTION_CORRECTION_IN_SETUP=1 \
    "${venv_dir}/bin/python" -m pip install -e "${project_root}[train]"
fi
if [[ -n "${flowmatching_repo}" ]]; then
  if [[ ! -f "${flowmatching_repo}/pyproject.toml" ]]; then
    echo "Flow Matching checkout is invalid: ${flowmatching_repo}" >&2
    exit 2
  fi
  "${venv_dir}/bin/python" -m pip install -e "${flowmatching_repo}"
fi
"${venv_dir}/bin/python" - <<'PY'
import json
import platform

import torch

print(
    json.dumps(
        {
            "event": "kimodo_environment_ready",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        },
        sort_keys=True,
    )
)
PY
