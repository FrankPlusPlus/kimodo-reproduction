#!/usr/bin/env bash
# Interactive Kimodo demo on the 1xH200 kimodo-dev box.
# One Version dropdown: official SEED models plus auto-discovered training
# checkpoints (eval-exports and runs/*/checkpoints). Pick one, then Load model.
#
# Uses /home/jovyan/.venv-kimodo-demo if present. Do not Restart the 16-GPU
# training job to get demo packages.
#
# On the laptop, in a separate terminal:
#   ssh -L 7860:127.0.0.1:7860 kimodo-dev
# then open http://127.0.0.1:7860
set -euo pipefail

export KIMODO_CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
export KIMODO_STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"
export PYTHONPATH="${KIMODO_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export KIMODO_MODEL_ROOT="${KIMODO_MODEL_ROOT:-${KIMODO_STORAGE_ROOT}/models}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${KIMODO_MODEL_ROOT}/checkpoints}"
export HF_HOME="${HF_HOME:-${KIMODO_STORAGE_ROOT}/hf-cache}"
export TEXT_ENCODER_MODE="${TEXT_ENCODER_MODE:-local}"
export TEXT_ENCODER_DEVICE="${TEXT_ENCODER_DEVICE:-cuda:0}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TRANSFORMERS_ATTN_IMPLEMENTATION="${TRANSFORMERS_ATTN_IMPLEMENTATION:-eager}"
export TRITON_INTERPRET="${TRITON_INTERPRET:-1}"
export KIMODO_DEMO_SKIP_PREWARM="${KIMODO_DEMO_SKIP_PREWARM:-1}"
export KIMODO_LLM2VEC_FOUNDATION="${KIMODO_LLM2VEC_FOUNDATION:-${KIMODO_MODEL_ROOT}/llm2vec/foundation}"
export KIMODO_LLM2VEC_MNTP="${KIMODO_LLM2VEC_MNTP:-${KIMODO_MODEL_ROOT}/llm2vec/mntp-adapter}"
export KIMODO_LLM2VEC_SUPERVISED="${KIMODO_LLM2VEC_SUPERVISED:-${KIMODO_MODEL_ROOT}/llm2vec/supervised-adapter}"
export LOCAL_CACHE="${LOCAL_CACHE:-True}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export SERVER_NAME="${SERVER_NAME:-0.0.0.0}"
export SERVER_PORT="${SERVER_PORT:-7860}"
export KIMODO_DEMO_EXPORT_ROOTS="${KIMODO_DEMO_EXPORT_ROOTS:-${KIMODO_STORAGE_ROOT}/eval-exports}"
export KIMODO_DEMO_RUN_ROOTS="${KIMODO_DEMO_RUN_ROOTS:-${KIMODO_STORAGE_ROOT}/runs}"
export KIMODO_DEMO_EXPORT_CACHE="${KIMODO_DEMO_EXPORT_CACHE:-${KIMODO_STORAGE_ROOT}/eval-exports}"
export KIMODO_DEMO_AUTO_DISCOVER="${KIMODO_DEMO_AUTO_DISCOVER:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-${NO_PROXY}}"
if [[ -z "${http_proxy:-}${HTTP_PROXY:-}" ]] && ss -lnt 2>/dev/null | grep -q ':7993'; then
  export http_proxy="${KIMODO_DEMO_HTTP_PROXY:-http://127.0.0.1:7993}"
  export https_proxy="${https_proxy:-${http_proxy}}"
  export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
  export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"
fi

DEMO_PYTHON="${KIMODO_DEMO_PYTHON:-/home/jovyan/.venv-kimodo-demo/bin/python}"
if [[ ! -x "${DEMO_PYTHON}" ]]; then
  DEMO_PYTHON="$(command -v python3 || command -v python)"
fi

official_dir="${CHECKPOINT_DIR}/Kimodo-SOMA-SEED-v1.1"
default_model="${KIMODO_DEMO_MODEL:-kimodo-soma-seed}"

if ! "${DEMO_PYTHON}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  echo "CUDA torch unavailable on this pod (${DEMO_PYTHON})." >&2
  echo "Install into ${KIMODO_DEMO_PYTHON:-/home/jovyan/.venv-kimodo-demo}." >&2
  echo "Do not Restart the 16-GPU training job." >&2
  exit 2
fi
if ! "${DEMO_PYTHON}" -c "import viser, kimodo.demo"; then
  echo "viser/kimodo demo imports failed with ${DEMO_PYTHON}." >&2
  exit 2
fi
if [[ ! -f "${official_dir}/config.yaml" ]]; then
  echo "missing official SEED-v1.1 bundle: ${official_dir}" >&2
  exit 2
fi

echo "demo: python=${DEMO_PYTHON}"
echo "demo: official=${official_dir}"
echo "demo: default_model=${default_model} port=${SERVER_NAME}:${SERVER_PORT}"
echo "demo: export_roots=${KIMODO_DEMO_EXPORT_ROOTS}"
echo "demo: run_roots=${KIMODO_DEMO_RUN_ROOTS}"
echo "On laptop: ssh -L ${SERVER_PORT}:127.0.0.1:${SERVER_PORT} kimodo-dev"
echo "Then open http://127.0.0.1:${SERVER_PORT}"

cd "${KIMODO_CODE_ROOT}"
exec "${DEMO_PYTHON}" -m kimodo.demo --model "${default_model}"
