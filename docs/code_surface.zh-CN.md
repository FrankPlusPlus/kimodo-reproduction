# Kimodo 代码面与清理边界

本文区分“稳定公共入口”“核心实现”和“审计/历史工具”。目标是减少使用者需要记忆的入口，而不是为了
文件数量好看而删除可复现证据。

## 稳定公共入口

| 任务 | 入口 |
|---|---|
| 公司镜像 tag / digest / 用法说明 | `docs/docker_images.zh-CN.md` |
| 容器模式选择 | `scripts/container_start.sh` |
| V2 数据构建 | `scripts/v2_pipeline.sh` |
| 本地环境和资源初始化 | `scripts/bootstrap_training.sh` |
| 通用单机/多机训练 | `scripts/train_distributed.sh` |
| 公司训练（环境变量控制拓扑，默认 16-rank） | `scripts/train_company.sh` |
| 数据准备或绑定 | `scripts/prepare_container.sh` |
| 合成数据两步 smoke | `scripts/smoke_train.sh` |
| 训练导出评测 | `scripts/eval_company_watcher.sh` |
| 官方 baseline 评测 | `scripts/eval_official_baseline.sh` |
| 分层 10% benchmark 构建 | `scripts/build_benchmark_stratified_proxy.sh` |
| A1 stratified 重跑 | `scripts/core10_stratified_benchmark_pipeline.sh` |
| Official 子集 vs NVIDIA 全量表 | `scripts/compare_official_subset_to_nvidia_full.py` |

Docker/Kubernetes 用户通常只需要以上入口。Python CLI 仍可用于测试、故障恢复和精确重放某一阶段，
但不再要求用户手工记住整条调用链。

`scripts/internal/` 保存 V2 的 LLM、bundle、package 和 watcher 分阶段实现；它们由
`scripts/v2_pipeline.sh` 调度，不属于普通用户入口。

旧的 `prepare_and_train_two_gpu_seed.sh` compatibility wrapper 已删除：它没有仓库内调用者，且把资源准备
和训练重新耦合在一次进程中。对应能力分别由 `bootstrap_training.sh` / `prepare_container.sh` 与
`train_two_gpu_seed.sh` 提供。

## 必须保留的核心

### 训练运行时

- `cli.py`、`config.py`、`data.py`；
- `modeling.py`、`constraints.py`、`losses.py`、`engine.py`；
- `checkpoint.py`、`ema.py`、`optim.py`、`run_lock.py`。

这些模块同时消费 V1/V2 manifest。数据版本差异由 paths、manifest、stats 和训练 YAML 表达，不应为每个
bundle 复制一套 trainer。

### 通用离线数据管线：`kimodo/data_pipeline/`

- `manifest_cli.py`、`text_cache_cli.py`、`stats_cli.py`；
- `reference_inventory.py` / `reference_inventory_cli.py`；
- 原子发布的公共实现位于 `kimodo/common/file_permissions.py`；训练 provenance 仍位于
  `kimodo/training/provenance.py`。

### V2 核心链：`kimodo/data_pipeline/v2/`

- `timeline_multi_cli.py`；
- `llm_api_augmentation_cli.py`、`llm_quality_cli.py`；
- `response_selection_cli.py`；
- `v2_manifest_cli.py`、`v2_cached_manifest_cli.py`、`v2_lineage_cli.py`；
- `v2_bundle_publish_cli.py`、`v2_resource_state_cli.py`。

### 评测链：`kimodo/evaluation/`

- `validation_cli.py`、`eval_monitor_cli.py`；
- `benchmark/` 下生成、embedding、指标和汇总脚本。

## 不应直接删除、但可降级为内部工具

- `qwen_augmentation_cli.py`：MiMo 不可用时的离线 fallback；
- `semantic_count_repair_cli.py`、`semantic_response_finalize_cli.py`；
- `independent_review_remediation_cli.py`、`duplicate_response_repair_cli.py`；
- `kimodo/devtools/` 下的 `core_subset_cli.py`、`smoke_fixture_cli.py`；
- `kimodo/evaluation/benchmark_cli.py`；
- `manifest_overlay_cli.py`。

前四类记录了已经发生过的语义修复和成品 provenance。实现现已从训练热路径迁入
`kimodo/data_pipeline/v2/`，但成品中已经发布的 generator identity 和 schema 字段保持不变，因此现有
V2 bundle 仍可逐哈希验证。后三类分别服务快速验证、性能基准和显式数据消融，也不属于正式训练热路径。

## 当前仍然有意留在链外的部分

1. LLM pilot 的人工阅读、1,200 条独立语义复核和 major adjudication；这是 fail-closed 质量门禁。
2. API key、LLM2Vec/Qwen 权重和公司 PVC；它们必须由 Secret 或挂载提供，不能进入仓库或镜像。
3. 官方未公开的 cross-motion diffusion transition 配方；V2 不伪造该步骤。

因此，“统一入口”不等于“取消门禁”。`scripts/v2_pipeline.sh plan` 中的 `REVIEW-GATE` 必须由明确的审阅
结果和 immutable response selection 才能越过。

## 已完成的安全迁移

1. `kimodo/training/` 只保留 trainer、Dataset、模型/loss/optimizer、checkpoint/EMA 和运行锁；
2. 离线数据、V2、评测和开发工具分别迁入独立包；
3. console-script 名称保持不变，仅更新其 Python module target；
4. 既有 provenance identity 保持不变，新实现路径由代码清单和测试单独约束。

不要以“没有被生产训练 import”为唯一删除依据：离线构建、断点恢复、成品审计和历史重放同样属于本项目
的可交付能力。
