# Kimodo 训练复现：历史工程验收报告

> 本文是早期、较窄的“工程链路可运行”验收快照，下面的测试数不能作为当前测试总数或完整论文复现结论。它能支持的最强结论是：相关实现合约和合成闭环在当时的测试中通过；真实 BONES-SEED 训练、论文规模训练、私有 Rigplay 数据和论文指标均未由这份报告验证。严格边界见 `paper_training_parity_audit.md`。

## 1. 结论与边界

当时的历史结论是：**工程实现测试通过**，论文严格复现为 **CONDITIONAL / BLOCKED / NOT-TESTED**。不对 P0/P1/P2 数量作零问题声明：论文和公开仓库未披露的选择仍然是有效复现嫌疑，不能因为单元测试通过而排除。

本报告中的 `PASS` 只表示所列工程合约在记录的测试环境中通过。基于 Kimodo 论文、公开 denoiser/推理代码、官方配置和发布 checkpoint，项目形成了可训练、可恢复、可导出、可接公开 benchmark 的工程实现。这不表示获得 NVIDIA 官方 trainer，不表示未披露选择已对齐，也不表示已复现私有 Rigplay 数据上的论文数值。

原始 clone 保持不变：`/Users/frank/Documents/kimodo`，基线 commit 为 `1aece8c124d73d255ceff5086d983b844c9f4e94`。所有实现位于本 reproduction 目录。

## 2. 已实现清单

1. 训练数据：BONES-SEED CSV/parquet、官方 split、full/event/combined-event manifest、30 Hz 转换、时间裁剪、随机 heading、变长 padding、float32 text cache。
2. 表示与模型：SOMA30 369D、root/body 两阶段 denoiser、constraint overwrite + binary mask concat、官方 checkpoint strict load。
3. diffusion/loss：cosine DDPM、1000 steps、`x0` prediction、六项 representation SmoothL1 + FK，总权重 `[10,2,10,3,10,4,5]`。
4. 两阶段课程：500k text-only + 500k constraint；五类约束；10/25/65 no/two/one categorical；稀疏关键帧 1→20；text dropout 与四类 CFG 分支日志。
5. 优化：Everett Adam-atan2 参考公式 `4/π·λ·atan2(m,λ√v)`，工程默认 λ=8；AMP、clip、有效帧加权 accumulation/DDP。
6. 可靠性：EMA 0.995/10、原子 checkpoint、schema 3、每-rank RNG、真实 micro-index、epoch/batch 精确恢复、受保护 milestone。
7. provenance：manifest 及引用 motion/embedding、来源 metadata/split hash、stats、骨架资产、官方 bundle、整个 `kimodo/**/*.py`、训练 YAML、依赖清单与 benchmark 入口。
8. 导出/评测：自包含 EMA inference bundle；`--checkpoint-bundle` 接入官方生成脚本；合成 generate→evaluate→parse 闭环。
9. 工具：train、manifest、text cache、stats、smoke fixture 五条训练相关 CLI；生产和 tiny 两套结构化配置。
10. 回归：官方 checkpoint、两阶段 tiny、epoch 边界、配置/数据变更拒绝、2-rank DDP exact resume、不等长梯度、约束 mask、EMA 和 bundle reload。

## 3. 最终 gate 证据

| Gate | 历史测试结果 | 验收证据与边界 |
|---|---|---|
| G1 官方权重兼容 | CONTRACT TEST PASS | 408 tensors strict load；283,281,777 参数；证明发布架构/checkpoint 兼容，不证明训练 recipe 相同 |
| G2 数据与表示 | FIXTURE TEST PASS | full/event/combined fixture 读取；369D；padding；stats metadata；未验证官方 stats 或完整真实数据分布 |
| G3 训练数学 | UNIT TEST PASS | DDPM `x0`、七项 loss、Adam-atan2 数值 oracle；reduction/loss domain/优化器其余参数未公开 |
| G4/G5 两阶段 | TINY TEST PASS | tiny 日志完成 Phase 1 step 1 与 Phase 2 step 2；不是 500k+500k 结果 |
| G6 conditioning | CONTRACT TEST PASS | text drop、五类 pattern、四类 CFG branch 跨 rank 日志；family 内部分布仍是 reconstruction |
| G7 恢复 | ENGINEERING TEST PASS | 单进程、accumulation epoch-boundary、2-rank resume；只验证恢复机制 |
| G8 CLI/config | ENGINEERING TEST PASS | production dry-run、非法配置/变更硬失败、tiny 从空目录可运行 |
| G9 benchmark | SYNTHETIC INTERFACE PASS | 2 motions、2 cases、12 summary rows；评测前后模型 hash 不变；不是公开完整 benchmark 或论文私有指标 |
| G10 provenance/docs | SNAPSHOT PASS | schema 3；当时的训练输入/代码 hash 和假设记录；不代表后续修改自动受此快照覆盖 |

当时的历史测试快照（不是当前测试总数）：

```text
pytest -q
15 passed, 1 skipped

KIMODO_OFFICIAL_BUNDLE=... pytest -q tests/training/test_official_checkpoint.py
2 passed
```

其中常规套件的一个 skip 是必须显式提供 1.1 GB 官方资产的 gate。2-rank Gloo 测试需要允许本机回环通信；受限沙箱中的 `uv_bind EPERM` 属于环境权限，不是断言失败。

独立 benchmark 闭环：

```text
Generated 2 motions
Evaluated 2 motion folders
Rows: 12, testcases: 2
model sha256 before/after:
6ecca82e87d810ccf57e01eff96a6fc086867348f0956c7a1ff34fadb5ac25f2
```

## 4. 冻结的官方资产

| 资产 | SHA-256 |
|---|---|
| `Kimodo-SOMA-SEED-v1.1/model.safetensors` | `ba8145cd5c8a3340b236fb2dce030ce5b151d1ef2c8491cfd2dc4d5f7c42b177` |
| `Kimodo-SOMA-SEED-v1.1/config.yaml` | `905664ad05779b0e28c391b85dc81c9de166418bd5f471ef605f75ab746ce391` |
| `train_split_paths.txt`（128,351 行） | `9fd3d85c1be6c44234f17b86840c71a6cbf36aa5d2294547fb5b89d7a624c5b8` |

## 5. 仍属 UNKNOWN 的论文细节

- direct loss 的 physical/normalized 域、SmoothL1 β、精确 reduction 与 FK 坐标/root 约定；
- Kimodo 实际 Adam-atan2 λ/betas/weight decay/warmup/scheduler；
- 训练精度、gradient clipping、seed；
- dropout 覆盖范围、Phase 边界 optimizer/EMA 行为；root→body 生产 profile 按公开 denoiser 的 training-mode branch 设为 detach，论文 “end-to-end” 不足以确定私有 trainer 是否允许跨 bridge 梯度；
- 官方 stats、clip/crop、caption/timeline、原文/paraphrase/stitch 的混合分布；
- Qwen3 paraphrase 与 transition stitching recipe；
- 五类 constraint 的 family/关键帧/dense span/heading/冲突精确分布；
- checkpoint 选择、early stopping 与私有测试协议。

对应实现均有显式配置或文档标记，不得把当前工程默认倒推成官方设置。

## 6. 外部阻塞，不属于代码缺陷

1. 在这份历史快照形成时，BONES-SEED 下载因 gated license 返回 401，未执行真数据训练；
   后续 2026-08-04 已在获授权的本地数据上完成 3-step 工程短训，见 `h200_training_benchmark.md`。
2. 完整 1M steps、global batch 2048、16×A100-80GB 尚未实际运行。
3. 论文 RP 约 700h 数据及 Sec. 6 私有测试集不公开，无法严格数值复现。
4. 合成 benchmark 使用 stub 文本/TMR embedding，只证明接口闭环；真实指标仍需正式 LLM2Vec、TMR 和 BONES-SEED。

后续短训只解除“真实 loader/反向/optimizer 能否走通”的工程疑问；完整训练、公开 benchmark 和论文
私有实验仍需按运行手册执行，并保留所有 provenance 输出。
