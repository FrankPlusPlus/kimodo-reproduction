#!/usr/bin/env bash
set -euo pipefail

# Canonical public entry point for V2 data work.  The specialized Python CLIs
# remain internal, testable building blocks because their provenance contracts
# are part of the published bundle; users should normally start here.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
stage="${1:-plan}"
shift || true

case "${stage}" in
  plan)
    cat <<'EOF'
prepare         deterministic V1 train rows -> timeline plan + LLM requests
pilot           small API generation pilot
audit-pilot     deterministic pilot quality report
generate        resumable full LLM generation
audit           deterministic full quality report + 1,200-row review sample
REVIEW-GATE     independent semantic review/remediation and immutable selection
bundle          selected responses -> cache + manifest + stats + inventory + preflight
package         verified train-ready bundle -> portable tar.zst + SHA-256
verify          verify every resource-state output in an existing final bundle
EOF
    ;;
  prepare|pilot|audit-pilot|generate|audit|manifest)
    exec "${script_dir}/internal/build_v2_llm.sh" "${stage}" "$@"
    ;;
  bundle)
    exec "${script_dir}/internal/build_v2_bundle.sh" all "$@"
    ;;
  package)
    exec "${script_dir}/internal/package_v2_bundle.sh" "$@"
    ;;
  status)
    exec "${script_dir}/internal/build_v2_bundle.sh" status "$@"
    ;;
  verify)
    final_root="${KIMODO_V2_FINAL_ROOT:-}"
    if [[ -z "${final_root}" ]]; then
      echo "KIMODO_V2_FINAL_ROOT is required for verify" >&2
      exit 2
    fi
    python_bin="${KIMODO_PYTHON:-${repo_root}/.venv/bin/python}"
    exec "${python_bin}" -m kimodo.data_pipeline.v2.v2_resource_state_cli \
      verify --root "${final_root}" "$@"
    ;;
  review-gate|REVIEW-GATE)
    echo "REVIEW-GATE is intentionally not automatic: audit the 1,200-row sample," >&2
    echo "resolve any major findings, finalize immutable responses, then create" >&2
    echo "KIMODO_LLM_RESPONSE_SELECTION before running the bundle stage." >&2
    exit 3
    ;;
  *)
    echo "usage: $0 {plan|prepare|pilot|audit-pilot|generate|audit|manifest|bundle|package|status|verify}" >&2
    exit 2
    ;;
esac
