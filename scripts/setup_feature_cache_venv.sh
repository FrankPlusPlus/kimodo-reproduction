#!/usr/bin/env bash
# Install a CPU-only venv on the code PVC for feature-cache builds.
# Safe on GPU-less transfer/dev pods. Persists across pod restarts.
set -euo pipefail

CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
VENV="${KIMODO_FEATURE_CACHE_VENV:-${CODE_ROOT}/.venv-feature-cache}"
LOG_DIR="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}/feature-cache"
LOG="${LOG_DIR}/venv-setup.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "[setup] $(date -Is) code=${CODE_ROOT} venv=${VENV}"
python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -U pip
# Explicit CPU wheel index — do NOT pull multi-GB CUDA torch.
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir "numpy<2" scipy einops omegaconf pyyaml tqdm
python - <<'PY'
import torch
import numpy
import einops
import scipy

print(
    "deps_ok",
    torch.__version__,
    "cuda",
    torch.cuda.is_available(),
    "numpy",
    numpy.__version__,
)
PY
export PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${CODE_ROOT}"
python -c "from kimodo.training.feature_cache_cli import main; print('cli_ok')"
echo "[setup] READY $(date -Is)"
