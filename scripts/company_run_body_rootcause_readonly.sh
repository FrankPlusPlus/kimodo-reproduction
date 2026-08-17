#!/usr/bin/env bash
# Run read-only root-cause probes R → S → U sequentially on 1 GPU.
#   exec bash /home/share/yzt/kimodo-reproduction/scripts/company_run_body_rootcause_readonly.sh
set -euo pipefail

export KIMODO_CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
export KIMODO_STORAGE_ROOT="${KIMODO_STORAGE_ROOT:-/home/share/yezitao-kimodo-reproduction}"

R_OUT="${KIMODO_STORAGE_ROOT}/diagnostics/body-rootcause-R-fullstack"
S_OUT="${KIMODO_STORAGE_ROOT}/diagnostics/body-rootcause-S-term-slopes"
U_OUT="${KIMODO_STORAGE_ROOT}/diagnostics/body-rootcause-U-gnorm-flip"

echo "=== Experiment R: full-stack attn/ffn ==="
KIMODO_PROBE_OUT_DIR="${R_OUT}" exec bash "${KIMODO_CODE_ROOT}/scripts/company_run_body_rootcause_R_fullstack.sh"

echo "=== Experiment S: per-term loss slopes ==="
KIMODO_PROBE_OUT_DIR="${S_OUT}" exec bash "${KIMODO_CODE_ROOT}/scripts/company_run_body_rootcause_S_term_slopes.sh"

echo "=== Experiment U: gnorm vs flip timeline ==="
python_bin="${KIMODO_PYTHON:-python3}"
full_stack_json="${R_OUT}/verdict.json"
if [[ ! -f "${full_stack_json}" ]]; then
  full_stack_json="${R_OUT}/rank-00.json"
fi
exec "${python_bin}" "${KIMODO_CODE_ROOT}/scripts/analyze_wd03_gnorm_flip_timeline.py" \
  --train-jsonl "${KIMODO_STORAGE_ROOT}/runs/v2-1m-hostnet-wd03-from650k/train.jsonl" \
  --full-stack-json "${full_stack_json}" \
  --output-dir "${U_OUT}"
