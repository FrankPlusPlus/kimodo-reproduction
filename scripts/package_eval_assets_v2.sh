#!/usr/bin/env bash
# Package stratified 10% benchmark assets for company-server transfer (yezitao-kimodo-eval-v2).
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
python_bin="${KIMODO_PYTHON:-${project_root}/.venv/bin/python}"

yzt_root="${KIMODO_YZT_ROOT:-$(cd -- "${project_root}/.." && pwd)}"
package_name="${KIMODO_EVAL_PACKAGE_NAME:-yezitao-kimodo-eval-v2}"
benchmark_source="${KIMODO_BENCHMARK_STRATIFIED_ROOT:-${yzt_root}/kimodo-benchmark-stratified-10pct}"
official_summary="${KIMODO_OFFICIAL_STRATIFIED_SUMMARY:-${yzt_root}/kimodo-benchmark-results/core10-loss-domain-stratified-10pct/official-seed-v1.1/summary_rows.json}"
official_vs_nvidia="${KIMODO_OFFICIAL_VS_NVIDIA_JSON:-${yzt_root}/kimodo-benchmark-results/core10-loss-domain-stratified-10pct/official_vs_nvidia_full.json}"
v1_asset_root="${KIMODO_EVAL_V1_ROOT:-${yzt_root}/kimodo-portable-runtime/prepared/yezitao-kimodo-eval-v1}"
prepared_root="${KIMODO_PREPARED_ROOT:-${yzt_root}/kimodo-portable-runtime/prepared}"
package_root="${prepared_root}/${package_name}"
delivery_root="${prepared_root}/${package_name}.delivery"
compression_level="${KIMODO_ZSTD_LEVEL:-6}"
compression_threads="${KIMODO_ZSTD_THREADS:-8}"

command -v rsync >/dev/null || { echo "rsync is required" >&2; exit 2; }
command -v zstd >/dev/null || { echo "zstd is required" >&2; exit 2; }
[[ -x "${python_bin}" ]] || { echo "python not found: ${python_bin}" >&2; exit 2; }

for path in "${benchmark_source}" "${official_summary}" "${v1_asset_root}/model-sources.json" "${v1_asset_root}/scripts/download_eval_models.sh"; do
  [[ -e "${path}" ]] || { echo "missing required input: ${path}" >&2; exit 2; }
done

if [[ -d "${package_root}" ]]; then
  echo "refusing to overwrite existing package directory: ${package_root}" >&2
  exit 2
fi
if [[ -e "${delivery_root}" ]]; then
  echo "refusing to overwrite existing delivery directory: ${delivery_root}" >&2
  exit 2
fi

mkdir -p \
  "${package_root}/benchmark/stratified-10pct" \
  "${package_root}/baselines/official-seed-v1.1" \
  "${package_root}/references" \
  "${package_root}/config" \
  "${package_root}/scripts"

echo "[package-v2] copying benchmark tree (~2.6 GB) ..."
rsync -a --info=stats2 \
  "${benchmark_source}/content" \
  "${benchmark_source}/repetition" \
  "${benchmark_source}/proxy_manifest.json" \
  "${package_root}/benchmark/stratified-10pct/"

cp -- "${official_summary}" "${package_root}/baselines/official-seed-v1.1/summary_rows.json"
if [[ -f "${official_vs_nvidia}" ]]; then
  cp -- "${official_vs_nvidia}" "${package_root}/references/official_vs_nvidia_full.json"
fi
cp -- "${v1_asset_root}/model-sources.json" "${package_root}/model-sources.json"
cp -- "${v1_asset_root}/scripts/download_eval_models.sh" "${package_root}/scripts/download_eval_models.sh"
chmod +x "${package_root}/scripts/download_eval_models.sh"

cat >"${package_root}/config/eval.env.example" <<'EOF'
# Set this to the directory where the PVC is mounted inside the container.
export KIMODO_STORAGE_ROOT=/home/share/yezitao-kimodo-reproduction

# Archive extraction creates this fixed asset directory under KIMODO_STORAGE_ROOT.
export KIMODO_EVAL_ASSET_ROOT="${KIMODO_STORAGE_ROOT}/yezitao-kimodo-eval-v2"
export KIMODO_MODEL_ROOT="${KIMODO_STORAGE_ROOT}/models"
export HF_HOME="${KIMODO_STORAGE_ROOT}/hf-cache"

# Training run to monitor and a separate output directory for evaluation results.
export KIMODO_RUN_DIR="${KIMODO_STORAGE_ROOT}/runs/v2-1m-production"
export KIMODO_EVAL_ROOT="${KIMODO_STORAGE_ROOT}/eval-results/v2-1m-production"
export KIMODO_BENCHMARK_ROOT="${KIMODO_EVAL_ASSET_ROOT}/benchmark/stratified-10pct"
export KIMODO_OFFICIAL_BASELINE_SUMMARY="${KIMODO_EVAL_ASSET_ROOT}/baselines/official-seed-v1.1/summary_rows.json"

# Force the exact local LLM2Vec identity used by V2 instead of an implicit online fallback.
export TEXT_ENCODER_MODE=local
export KIMODO_LLM2VEC_FOUNDATION="${KIMODO_MODEL_ROOT}/llm2vec/foundation"
export KIMODO_LLM2VEC_MNTP="${KIMODO_MODEL_ROOT}/llm2vec/mntp-adapter"
export KIMODO_LLM2VEC_SUPERVISED="${KIMODO_MODEL_ROOT}/llm2vec/supervised-adapter"
export CHECKPOINT_DIR="${KIMODO_MODEL_ROOT}/checkpoints"
export LOCAL_CACHE=True

# Must match the packaged official baseline protocol for direct comparison.
export KIMODO_EVAL_DIFFUSION_STEPS=100
export KIMODO_EVAL_BATCH_SIZE=1
export KIMODO_EVAL_WORKERS=4
export KIMODO_EVAL_PAPER_PROTOCOL=1
export KIMODO_EVAL_TEXT_ENCODER_FP32=1
EOF

cat >"${package_root}/README.zh-CN.md" <<'EOF'
# yezitao-kimodo-eval-v2

可迁移的 **stratified 10% 官方 benchmark 子集**（2,269 cases）及同协议 Official SEED-v1.1 基线。
用于单卡 eval Pod 监控训练 checkpoint，或本地/公司侧消融复现。

## 包内内容

- `benchmark/stratified-10pct/`：分层 10% 子集，含完整 `meta.json`、约束文件、`gt_motion.npz` 与 `proxy_manifest.json`。
- `baselines/official-seed-v1.1/summary_rows.json`：同一子集、paper protocol、fp32 text encoder 下的 Official 基线。
- `references/official_vs_nvidia_full.json`：Official 子集结果 vs NVIDIA 全量表对照（可选参考）。
- `model-sources.json` / `scripts/download_eval_models.sh`：评测所需 LLM2Vec + TMR 模型下载脚本。
- `config/eval.env.example`：公司 PVC 路径与 watcher 环境变量。
- `payload.sha256`：解包后校验所有有效载荷。

相对 v1（128-case proxy），本子集约束 cm 已与 NVIDIA 全量表对齐（约 ±15%），更适合 A1/A2 消融结论。

## 1. 解包到 PVC

```bash
export KIMODO_STORAGE_ROOT=/home/share/yezitao-kimodo-reproduction
mkdir -p "$KIMODO_STORAGE_ROOT"
tar --use-compress-program=unzstd -xf yezitao-kimodo-eval-v2.tar.zst -C "$KIMODO_STORAGE_ROOT"
cd "$KIMODO_STORAGE_ROOT/yezitao-kimodo-eval-v2"
sha256sum -c payload.sha256
```

## 2. 下载模型到 PVC

与 v1 相同，约 16.5GB + HF cache；建议预留 35GB。

```bash
export KIMODO_STORAGE_ROOT=/home/share/yezitao-kimodo-reproduction
bash "$KIMODO_STORAGE_ROOT/yezitao-kimodo-eval-v2/scripts/download_eval_models.sh"
```

## 3. 启动 watcher 或单次 eval

```bash
cd /workspace/kimodo-reproduction
source /home/share/yezitao-kimodo-reproduction/yezitao-kimodo-eval-v2/config/eval.env.example
bash scripts/eval_company_watcher.sh
```

单次 eval 示例：

```bash
python -m kimodo.evaluation.eval_monitor_cli \
  --run-dir "$KIMODO_RUN_DIR" \
  --benchmark "$KIMODO_BENCHMARK_ROOT" \
  --output-root "$KIMODO_EVAL_ROOT" \
  --minimum-step 40000 --once --paper-protocol --text-encoder-fp32
```

## 协议固定项

- benchmark：stratified 10%（2,269 cases）
- diffusion steps：100
- eval batch size：1
- paper protocol：开启
- text encoder：**fp32**（与包内 Official 基线一致）
- `benchmark_inventory_sha256`：见 `package-metadata.json`

更改这些项后，结果不能再与包内 Official summary 直接作严格对比。
EOF

echo "[package-v2] writing package-metadata.json ..."
"${python_bin}" - "${package_root}" "${benchmark_source}" <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

package_root = Path(__import__("sys").argv[1])
benchmark_source = Path(__import__("sys").argv[2])
manifest = json.loads((benchmark_source / "proxy_manifest.json").read_text(encoding="utf-8"))
baseline = package_root / "baselines/official-seed-v1.1/summary_rows.json"
baseline_bytes = baseline.read_bytes()
case_counts = {}
for group in manifest.get("groups", []):
    parts = group["group"].split("/", 1)
    bucket = parts[0] + "/" + (parts[1].split("/", 1)[0] if len(parts) > 1 else parts[0])
    if "text2motion" in group["group"]:
        bucket = f"{parts[0]}/text2motion"
    elif "constraints" in group["group"]:
        bucket = f"{parts[0]}/" + ("constraints_withtext" if "constraints_withtext" in group["group"] else "constraints_notext")
    case_counts[bucket] = case_counts.get(bucket, 0) + group["selected_count"]

benchmark_bytes = 0
file_count = 0
gt_count = 0
for path in (package_root / "benchmark/stratified-10pct").rglob("*"):
    if path.is_file():
        file_count += 1
        benchmark_bytes += path.stat().st_size
        if path.name == "gt_motion.npz":
            gt_count += 1

metadata = {
    "benchmark": {
        "bytes": benchmark_bytes,
        "case_count": manifest.get("selected_case_count", gt_count),
        "case_counts": case_counts,
        "file_count": file_count,
        "inventory_sha256": "bd7db29d0d388f3428541a2ddd8c60179907dc4e541625429196b5bf4e11552d",
        "kind": "stratified-10pct",
        "manifest": "benchmark/stratified-10pct/proxy_manifest.json",
        "source_full_suite_cases": 22474,
    },
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "official_baseline": {
        "bytes": len(baseline_bytes),
        "diffusion_steps": 100,
        "file": "baselines/official-seed-v1.1/summary_rows.json",
        "model": "nvidia/Kimodo-SOMA-SEED-v1.1",
        "paper_protocol": True,
        "sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "text_encoder_precision": "fp32",
    },
    "package": "yezitao-kimodo-eval-v2",
    "predecessor": "yezitao-kimodo-eval-v1",
    "purpose": "Stratified 10% official benchmark subset and matching official SEED-v1.1 baseline.",
    "schema_version": 2,
    "scope": {
        "contains_eval_code": False,
        "contains_model_weights": False,
        "eval_code_location": "project Docker image",
        "full_official_benchmark": False,
        "model_download_script": "scripts/download_eval_models.sh",
    },
    "v2_text_encoder_revisions": {
        "foundation": "53346005fb0ef11d3b6a83b12c895cca40156b6c",
        "mntp": "31474e395ada192e8ed1586db6be79fb3b70c9c0",
        "supervised": "baa8ebf04a1c2500e61288e7dad65e8ae42601a7",
    },
}
(package_root / "package-metadata.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({"case_count": metadata["benchmark"]["case_count"], "bytes": metadata["benchmark"]["bytes"]}, indent=2))
PY

echo "[package-v2] writing payload.sha256 ..."
(
  cd -- "${package_root}"
  find . -type f ! -name payload.sha256 | LC_ALL=C sort | while read -r relpath; do
    relpath="${relpath#./}"
    sha256sum -- "${relpath}"
  done
) >"${package_root}/payload.sha256"

staging="$(mktemp -d --tmpdir="${prepared_root}" ".${package_name}.delivery.building.XXXXXX")"
archive_name="${package_name}.tar.zst"
archive_path="${staging}/${archive_name}"
cleanup() { [[ -n "${staging:-}" ]] && rm -rf -- "${staging}"; }
trap cleanup EXIT

echo "[package-v2] creating ${archive_name} (zstd -${compression_level}, ${compression_threads} threads) ..."
tar --format=posix --hard-dereference --sort=name \
  --pax-option=delete=atime,delete=ctime \
  --owner=0 --group=0 --numeric-owner --mode='u+rwX,go+rX,go-w' \
  -cf - -C "${prepared_root}" -- "${package_name}" \
  | zstd -q -T"${compression_threads}" "-${compression_level}" --check -o "${archive_path}"

zstd -q -t -- "${archive_path}"
(
  cd -- "${staging}"
  sha256sum -- "${archive_name}" >"${archive_name}.sha256"
)

archive_sha256="$(awk '{print $1}' "${staging}/${archive_name}.sha256")"
archive_bytes="$(stat -c '%s' "${archive_path}")"
payload_files="$(grep -c '  ' "${package_root}/payload.sha256" || true)"
extracted_bytes="$(du -sb "${package_root}" | awk '{print $1}')"

"${python_bin}" - "${staging}/${archive_name}.metadata.json" "${package_name}" "${archive_name}" "${archive_sha256}" "${archive_bytes}" "${payload_files}" "${extracted_bytes}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
record = {
    "archive": sys.argv[3],
    "benchmark_cases": 2269,
    "bytes": int(sys.argv[5]),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "extracted_bytes": int(sys.argv[7]),
    "extracted_top_level": sys.argv[2],
    "payload_files": int(sys.argv[6]),
    "schema_version": 2,
    "sha256": sys.argv[4],
}
out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

mkdir -p -- "${delivery_root}"
mv -- "${staging}/${archive_name}" "${staging}/${archive_name}.sha256" "${staging}/${archive_name}.metadata.json" "${delivery_root}/"
staging=""
trap - EXIT

echo
echo "[package-v2] ready:"
echo "  directory: ${package_root}"
echo "  delivery:  ${delivery_root}/${archive_name}"
echo "  sha256:    ${archive_sha256}"
echo "  size:      $(numfmt --to=iec-i --suffix=B "${archive_bytes}" 2>/dev/null || echo "${archive_bytes} bytes")"
