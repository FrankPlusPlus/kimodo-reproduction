# Benchmark-oriented V2 数据配方

V2 的目标是提升公开 Kimodo Motion Generation Benchmark 所覆盖的文本组合与约束能力，同时保持训练源严格来自官方 BONES-SEED train whitelist。它是可审计的工程配方，不冒充 NVIDIA 未公开的完整训练数据配方。

## 文本数据变化

- 保留 V1 的 898,205 条 full-motion 描述和 318,647 条 single-event 描述。
- 删除 190,332 条机械拼接的 `A Then, B` combined rows。
- 从 V1 train event annotations 重建连续 2–5 event spans：整数 30fps 帧边界、最长 300 帧、相邻未标注 gap 不超过 45 帧。
- 按公开 benchmark 的 multi-event 结构确定性选择 223,889 个 semantic spans：
  - 2 events：175,999
  - 3 events：40,681
  - 4 events：6,246
  - 5 events：963
- 先覆盖不同 ordered source-text tuple，再 round-robin 填充重复 tuple；最终只需 70,169 个 LLM 请求。
- 当前生产方案使用 `mimo-v2.5-pro`，一次 API 调用批量处理 16 个逻辑样本；同一 Pro 模型再做一次语义 self-judge，逐项检查动作顺序、方向、身体部位、物体、交互和次数。self-judge 不是独立模型评审。被拒绝或多次 JSON/API 失败的极少数请求使用完整保留源事件文本的确定性 fallback，并在逐行 provenance 和汇总计数中显式标记；最终仍需人工分层抽检。

预期 V2 raw manifest 为 1,440,741 rows。文本阶段没有生成新 motion，也没有实现论文所述但未公开配方的 cross-motion diffusion transitions。因此 `paper_parity_gate.eligible` 必须保持 `false`。

## Phase-2 约束变化

V2 保留论文公开的原五类 constraint curriculum，同时让 constrained samples 中 25% 走公开 benchmark 的 13-leaf coverage lane：

- strict endpoint inbetweening：只约束 `[0, T-1]`；
- full-sequence root paths：约束 `0..T-1`；
- 固定双脚、双手、双手双脚 EE sets；
- 四种公开 mixture，包括 `RightHand + LeftFoot` 的特殊组合；
- benchmark sparse counts 最大 9，经验幂次 0.45；原 paper lane 仍保持 1→20 curriculum、arbitrary EE 和 foot contacts。

覆盖概率和 sparse power 是工程假设，位于 `configs/overlays/benchmark_v2_constraints.yaml`。当前版本对齐 constraint shape，但尚未实现 3–10 秒均匀 duration-aware raw crop；该项需要单独 ablation，不能在 normalized constraint tensor 上事后切片。

## 30k 训练验证配方

V2 的 30k 端到端消融使用 `configs/training/kimodo_soma_seed_v2_30k.yaml`，与公司 1M production
profile 独立。该配置保持此前验证的
`20k Phase 1 + 10k Phase 2`、Adam-atan2、`lr=1e-5`、论文七项权重、Smooth-L1 `beta=1`、
target-root FK 和 root-to-body detach，只做两项与 V2 目标直接相关的显式工程选择：

- 六项 representation direct loss 在 normalized feature domain 计算；
- Phase 2 启用公开 benchmark 13-leaf coverage lane。

FK 不随 direct domain 改变：rotation 和 target 会在进入骨架前反归一化，FK joint error 始终在物理米制
空间计算。canonical `kimodo_soma_seed_public.yaml` 继续保留 physical direct-loss baseline，避免把本次
30k 工程选择倒推为 NVIDIA 未公开的训练事实。

本地两张 H200、global batch 512 的启动选择为：

```bash
KIMODO_TWO_GPU_CONFIG=configs/training/kimodo_soma_seed_v2_30k.yaml \
KIMODO_TRAINING_OVERLAY=configs/overlays/two_h200_gb512.yaml \
KIMODO_PATHS_CONFIG=/path/to/local-v2.paths.yaml \
scripts/train_two_gpu_seed.sh
```

V2 manifest、inventory、stats 由 schema-v1 paths 文件提供。公司镜像使用独立的
`kimodo_soma_seed_v2_1m_16h200.yaml`，不会覆盖这个本地短步配方。

## 构建顺序（MiMo 2.5 Pro）

仓库统一入口是 `scripts/v2_pipeline.sh`。它收敛了用户需要记忆的命令面，但不会删除或绕过内部的
manifest、cache、lineage、quality、inventory 和 publish CLI；这些工具的输出哈希属于成品审计链。
先执行 `scripts/v2_pipeline.sh plan` 可以看到完整阶段。`REVIEW-GATE` 是有意保留的人工门禁，不能把
LLM 生成成功误当成语义质量合格。

密钥只能通过环境变量或 Kubernetes Secret 注入，不能写进仓库、镜像、命令行参数或 bundle。下面用
交互式隐藏输入；实际 CI/Kubernetes 应改为 Secret：

```bash
read -rsp 'MiMo API key: ' PRODUCT_GRAPH_LLM_API_KEY && echo
export PRODUCT_GRAPH_LLM_API_KEY
export PRODUCT_GRAPH_LLM_BASE_URL=https://api.xiaomimimo.com/v1
export PRODUCT_GRAPH_LLM_MODEL=mimo-v2.5-pro
```

先生成 train-only timeline plan。历史 staging 中 `qwen.requests` 只是旧文件名，内容本身是
provider-neutral schema；新目录建议命名为 `llm.requests`：

```bash
kimodo_prepare_timeline_v2 \
  --source-manifest /mnt/kimodo/data/adopted-legacy-soma30-v1/train.raw.jsonl \
  --train-split artifacts/benchmark-metadata/splits/train_split_paths.txt \
  --output-plan /mnt/kimodo/data/benchmark-v2-soma30-v2.2.building/provenance/timeline.selected.v2.2.jsonl \
  --output-requests /mnt/kimodo/data/benchmark-v2-soma30-v2.2.building/provenance/llm.requests.v2.2.jsonl
```

等价的统一入口为：

```bash
KIMODO_STORAGE_ROOT=/mnt/kimodo \
KIMODO_LLM_REQUESTS=/mnt/kimodo/data/benchmark-v2-soma30-v2.2.building/provenance/llm.requests.v2.2.jsonl \
scripts/v2_pipeline.sh prepare
```

先做 64 条分层 pilot，审阅输出与账单后再启动全量。pilot 和全量使用不同输出，不能把 pilot 文件直接
当成全量 shard：

```bash
kimodo_generate_llm_v2 \
  --requests /mnt/kimodo/data/benchmark-v2-soma30-v2.2.building/provenance/llm.requests.v2.2.jsonl \
  --model mimo-v2.5-pro --judge-model mimo-v2.5-pro \
  --batch-size 16 --concurrency 8 --requests-per-minute 90 \
  --max-requests 64 \
  --output /mnt/kimodo/data/benchmark-v2-soma30-v2.2.building/provenance/mimo.responses.pilot.jsonl

kimodo_audit_llm_v2 \
  --requests /mnt/kimodo/data/benchmark-v2-soma30-v2.2.building/provenance/llm.requests.v2.2.jsonl \
  --responses /mnt/kimodo/data/benchmark-v2-soma30-v2.2.building/provenance/mimo.responses.pilot.jsonl \
  --allow-partial --report-only \
  --report /mnt/kimodo/data/benchmark-v2-soma30-v2.2.building/provenance/mimo.quality.pilot.json \
  --review-sample /mnt/kimodo/data/benchmark-v2-soma30-v2.2.building/provenance/mimo.review.pilot.jsonl
```

pilot 通过人工查看后启动全量。脚本默认关闭 MiMo thinking、要求 JSON object、限速、指数退避，输出
`.partial` 可用原命令断点续跑；每次成功 API 调用另有不含密钥的 receipts ledger：

```bash
kimodo_generate_llm_v2 \
  --requests /mnt/kimodo/data/benchmark-v2-soma30-v2.2.building/provenance/llm.requests.v2.2.jsonl \
  --model mimo-v2.5-pro --judge-model mimo-v2.5-pro \
  --batch-size 16 --concurrency 8 --requests-per-minute 90 \
  --output /mnt/kimodo/data/benchmark-v2-soma30-v2.2.building/provenance/mimo.responses.v2.2.jsonl

kimodo_audit_llm_v2 \
  --requests /mnt/kimodo/data/benchmark-v2-soma30-v2.2.building/provenance/llm.requests.v2.2.jsonl \
  --responses /mnt/kimodo/data/benchmark-v2-soma30-v2.2.building/provenance/mimo.responses.v2.2.jsonl \
  --report /mnt/kimodo/data/benchmark-v2-soma30-v2.2.building/provenance/mimo.quality.v2.2.json \
  --review-sample /mnt/kimodo/data/benchmark-v2-soma30-v2.2.building/provenance/mimo.review.v2.2.jsonl

kimodo_build_manifest_v2 \
  --source-manifest /mnt/kimodo/data/adopted-legacy-soma30-v1/train.raw.jsonl \
  --plan /mnt/kimodo/data/benchmark-v2-soma30-v2.2.building/provenance/timeline.selected.v2.2.jsonl \
  --responses /mnt/kimodo/data/benchmark-v2-soma30-v2.2.building/provenance/mimo.responses.v2.2.jsonl \
  --train-split artifacts/benchmark-metadata/splits/train_split_paths.txt \
  --expected-model mimo-v2.5-pro --expected-revision provider-managed \
  --output /mnt/kimodo/data/benchmark-v2-soma30-v2.2.building/train.raw.jsonl
```

本地 Qwen3-32B 的 `kimodo_generate_qwen_v2` 仍作为离线 fallback 保留，但 MiMo 产物会诚实标记为
`timeline_multi_llm`，不会冒充 Qwen 数据。全量质量 gate 通过后，再依次运行
`kimodo_cache_manifest_v2 extract`、`kimodo_cache_text`、
`kimodo_cache_manifest_v2 compose --llm-cached-manifest ...`、V2 stats 重算、
`kimodo_reference_inventory` 全内容验证和真实 batch preflight。只有这些门禁全部通过后，staging 目录
才能原子改名为 train-ready bundle。

## PVC 与权限

V2 staging 可用 hardlink 复用同一文件系统上的 immutable V1 motion/text cache，但迁移归档必须解引用 hardlink。最终目录和子目录需要 group traverse，文件需要 group read；Kubernetes Pod 应配置与 PVC 一致的 `runAsUser`/`fsGroup`。训练启动前必须由真实容器 UID 完成 manifest、随机 motion、embedding 和 inventory 的读取预检。
