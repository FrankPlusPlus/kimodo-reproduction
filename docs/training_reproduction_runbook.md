# Kimodo 训练复现运行手册

本目录实现的是基于论文、发布配置、checkpoint 和公开推理代码重建的训练系统，不是 NVIDIA 官方训练源码。证据与默认假设逐项见 `training_reproduction_spec.md`；公开代码接口约束见 `code_training_contract.md`。

## 1. 当前复现边界

| 层级 | 当前状态 | 含义 |
|---|---|---|
| 发布架构与权重兼容 | 已验证 | 官方 `Kimodo-SOMA-SEED-v1.1` 的 408 个权重张量严格加载；模型为 283,281,777 参数；CPU 前向/反向均为有限值 |
| 训练机制复现 | 已实现并烟测 | DDPM `x0` 预测、七项 loss、两阶段 curriculum、五类约束、text dropout、Adam-atan2、EMA、DDP、断点恢复、导出 |
| BONES-SEED 数据闭环 | 代码已实现；真实数据受门禁阻塞 | manifest、时间子片段、官方 split、stats、文本缓存均有入口；本环境未获 gated 数据访问权 |
| 论文训练数值复现 | 不能宣称 | 原论文 RP 数据为专有；SEED 的精确采样混合、LLM paraphrase/stitch 数据、loss 域和完整 optimizer 超参未公开 |
| 官方 benchmark 闭环 | 接口已接通；真实运行受数据阻塞 | 导出 bundle 可直接传给 `benchmark/generate_eval.py --checkpoint-bundle`；构建 GT 仍需 BONES-SEED |

## 2. 环境

```bash
python3.12 -m venv .venv
SKIP_MOTION_CORRECTION_IN_SETUP=1 .venv/bin/pip install -e '.[train]'
```

若只训练预缓存的文本 embedding，不需要在训练进程加载 PEFT/Transformers。生成缓存时才需要安装项目的完整文本编码依赖，并准备约 8B LLM2Vec 模型。

先用可提交配置跑一次从零开始的 CPU 两阶段闭环：

```bash
.venv/bin/kimodo_create_smoke_fixture
.venv/bin/kimodo_train --config configs/training/kimodo_tiny_smoke.yaml
```

第一条命令确定性生成 `tests/fixtures/training` 下的 motion、16D text embedding、manifest 和 stats，并拒绝覆盖已有目录；第二条执行 1 step Phase 1 + 1 step Phase 2、checkpoint/resume 基础路径及 EMA bundle 导出。该 fixture 只验证工程路径，不代表真实训练数据。

## 3. 准备 BONES-SEED

先在 Hugging Face 接受 BONES-SEED 的 gated license，再下载并解压 `soma_uniform.tar.gz`。下载官方 benchmark 仓库中的 `splits/train_split_paths.txt`，并准备：

- `metadata/seed_metadata_v004.csv` 或 parquet；
- `metadata/seed_metadata_v002_temporal_labels.jsonl`；
- `soma_uniform/bvh/...`；
- benchmark 的官方 train split。

构建 full clip、单 event 与相邻双 event manifest：

```bash
.venv/bin/kimodo_build_manifest \
  --metadata /data/bones-seed/metadata/seed_metadata_v004.csv \
  --temporal-labels /data/bones-seed/metadata/seed_metadata_v002_temporal_labels.jsonl \
  --split-file /data/kimodo-benchmark/splits/train_split_paths.txt \
  --dataset-root /data/bones-seed \
  --skeleton soma_uniform \
  --source-fps 120 \
  --output data/bones_seed/train.raw.jsonl
```

`--full-repeats`、`--event-repeats`、`--combined-event-repeats` 是显式的工程采样权重。论文只说按预设分布混合，却没有公布概率；默认均为 1，不能称为官方比例。构建器同时写入 `train.raw.jsonl.metadata.json`，冻结 metadata、timeline、split 的绝对路径、大小和 SHA-256。跨 motion 的 stitched clips 和 Qwen3-32B paraphrases 也未发布，本实现不会伪造它们。

生产配置启用 `paper_method_strict=true` 和 `data.require_paper_data_parity=true`。因此，上述普通 manifest 会被有意拒绝；只有同时包含 Qwen3-32B paraphrase 行、由非增强 diffusion checkpoint 生成的跨 motion transition 行，以及完整 hash/provenance 的 manifest 才能进入严格 paper profile。若只想运行公开数据工程基线，必须显式设置 `paper_method_strict=false` 和 `data.require_paper_data_parity=false`，并把结果标为 `engineering reconstruction`，不能标为论文方法完整复现。

## 4. 缓存冻结的 LLM2Vec 条件

训练只读取 float32 `.npy` embedding。revision 不仅写入 metadata，也实际传入 tokenizer、base model 和 PEFT adapter 的 `from_pretrained`：

```bash
.venv/bin/kimodo_cache_text \
  --manifest data/bones_seed/train.raw.jsonl \
  --output-manifest data/bones_seed/train.cached.jsonl \
  --cache-dir data/bones_seed/text_cache \
  --provider local \
  --base-revision <commit-sha> \
  --peft-revision <commit-sha>
```

如复用官方 text-encoder 服务，可改为 `--provider api --api-url ...`。API 服务自身的模型 revision 必须另行固定并记录。

## 5. 计算 normalization stats

```bash
.venv/bin/kimodo_compute_stats \
  --manifest data/bones_seed/train.cached.jsonl \
  --output data/bones_seed/stats/soma30-30fps \
  --split train \
  --skeleton-joints 30 \
  --fps 30
```

输出必须为：

```text
stats/
  global_root/{mean,std}.npy   # 5
  local_root/{mean,std}.npy    # 4
  body/{mean,std}.npy          # 364
  stats.metadata.json          # manifest hash、唯一 clip/帧数、heading 假设
```

论文未披露 stats 拟合方法。当前工程默认对每个唯一 motion/time-span 枚举全部不重叠的最长 10 秒窗口，每窗独立做首帧 root 归零和确定性均匀 heading；不足两帧的尾窗会与前窗重新平衡。该策略覆盖全部训练帧且可重复，但必须标成 `[DEFAULT]`。

## 6. 从头训练、论文 Phase 2 恢复与实验性后训练

从头训练：

```bash
torchrun --standalone --nproc_per_node=16 -m kimodo.training.cli \
  --config configs/training/kimodo_soma_seed_reproduction.yaml \
  --set data.manifest=data/bones_seed/train.cached.jsonl \
  --set model.stats_path=data/bones_seed/stats/soma30-30fps
```

上述命令在缺少论文增强资产时会 fail closed。公开 BONES-SEED 工程基线可附加 `--set paper_method_strict=false --set data.require_paper_data_parity=false`；这些 override 会主动降低复现等级。

配置默认复现论文公开数值：16 层/每 stage、8 heads、latent 1024、FFN 2048、post-norm、1000 diffusion steps、500k+500k、global batch 2048（16×local 128）、lr `2e-5`、EMA 0.995/每 10 step。Adam-atan2 使用 Kimodo 所引用论文的参考公式 `4/π·λ·atan2(m, λ√v)` 和实验值 `λ=8`；Kimodo 自身没有披露 λ、betas、weight decay 或 schedule，因此仍属于 `[DEFAULT]`。实际显存不足时减小 local batch 并增加 `runtime.gradient_accumulation_steps`，确保 `world_size × batch_size × accumulation = 2048`。

论文 Phase 2 只能从本训练器在 500k 保存的 full-state checkpoint 恢复，保证 model、optimizer、EMA、RNG 与数据位置连续：

```bash
.venv/bin/kimodo_train \
  --config outputs/kimodo-soma-seed-reproduction/config.resolved.yaml \
  --set runtime.resume=outputs/kimodo-soma-seed-reproduction/checkpoints/step-000500000.pt
```

官方发布的 `Kimodo-SOMA-SEED-v1.1` 是最终 EMA 推理权重，不是 500k Phase 1 full-state checkpoint；它没有 optimizer/scaler/RNG。因此只能作为实验性后训练初始化，不能称为复现论文 Phase 2。示例：

```bash
.venv/bin/kimodo_train \
  --config configs/training/kimodo_soma_seed_reproduction.yaml \
  --set model.checkpoint_dir=/checkpoints/Kimodo-SOMA-SEED-v1.1 \
  --set data.manifest=data/bones_seed/train.cached.jsonl \
  --set paper_method_strict=false \
  --set data.require_paper_data_parity=false \
  --set curriculum.phase1_steps=0 \
  --set curriculum.phase2_steps=10000 \
  --set runtime.max_steps_override=10000 \
  --set runtime.output_dir=outputs/experimental-seed-v1.1-posttrain
```

这会新建 optimizer 和 EMA（EMA 初值来自加载的最终模型），适合自定义约束/域适配实验，但结果属于新的后训练模型。`checkpoint_dir` 中应有官方 `config.yaml`、`model.safetensors` 和 `stats/motion/...`；权重加载严格检查全部参数，不使用 `strict=False`。

一般训练恢复：

```bash
.venv/bin/kimodo_train \
  --config outputs/kimodo-soma-seed-reproduction/config.resolved.yaml \
  --set runtime.resume=outputs/kimodo-soma-seed-reproduction/checkpoints/step-000500000.pt
```

实际 latest 路径记录在 `checkpoints/latest.txt`。checkpoint 包含 online/EMA model、optimizer、scaler、epoch/batch、每个 DDP rank 的 Python/NumPy/Torch RNG、resolved config，以及 manifest 引用的数据/文本 embedding/来源 metadata、stats、骨架资产、官方 bundle 和关键代码的 SHA-256。任一训练关键输入变化都会拒绝恢复。为保证 `set_epoch` 的确定性，当前实现明确禁用 `persistent_workers=true`。

## 7. 两个必须保留的消融开关

- `loss.direct_feature_domain=physical|normalized`：论文未说明直接六项 loss 的计算域。工程默认 `physical`；FK 始终在物理域。
- `model.detach_root_for_body=false|true`：论文明确称 interleaved two-stage denoiser 端到端训练，因此 paper profile 默认 `false`；`true` 只保留为公开代码训练分支兼容消融。

在作者给出信息前，两项都必须写入实验记录，不得倒推成“官方设置”。

## 8. 导出与官方 benchmark 闭环

训练结束会生成：

```text
outputs/.../exports/step-XXXXXXXXX/
  config.yaml
  model.pt              # EMA raw denoiser state
  stats/
  TRAINING_PROVENANCE.txt
```

该 bundle 已做严格重新实例化测试。接入公开 benchmark：

```bash
python benchmark/create_benchmark.py /data/kimodo-testsuite \
  --dataset /data/bones-seed/soma_uniform

python benchmark/generate_eval.py \
  --benchmark /data/kimodo-testsuite \
  --output outputs/eval-generated \
  --checkpoint-bundle outputs/.../exports/step-001000000 \
  --diffusion_steps 100

python benchmark/embed_folder.py outputs/eval-generated
python benchmark/evaluate_folder.py outputs/eval-generated --paper-protocol
```

`--paper-protocol` 会另写完整集合 retrieval/FID、EE rotation geodesic error、generated smooth-root mean error 和 pelvis-to-smooth-root pointwise p95；不会覆盖或混入公开 benchmark 的旧列。公开 benchmark 与论文 Sec. 6 的私有测试集仍不是同一套，因此只能验证指标实现和公开闭环，不能据此声称复现论文表格。

## 9. 验收命令

```bash
# 合成数据：表示、五类约束、loss、optimizer、梯度策略、导出、两阶段训练、精确恢复
.venv/bin/python -m pytest -q tests/training

# 1.1 GB 官方 checkpoint 的严格 load/forward/backward gate（单独运行，
# 避免和会启动本地 Gloo 进程的 DDP 测试争夺资源）
KIMODO_OFFICIAL_BUNDLE=/checkpoints/Kimodo-SOMA-SEED-v1.1 \
  .venv/bin/python -m pytest -q tests/training/test_official_checkpoint.py

# 配置静态检查
.venv/bin/kimodo_train \
  --config configs/training/kimodo_soma_seed_reproduction.yaml --dry-run
```

本轮最终集成自检结果：常规套件 `42 passed, 1 skipped`（skip 是需显式资产路径的官方 gate）；官方 checkpoint 文件使用真实资产路径单独运行 `2 passed`。仓库套件内包含 Figure 9 tensor/梯度合同、paper-data fail-closed gate、重采样与 stats 全覆盖、论文评测数值 oracle、两阶段 tiny smoke、2-rank CPU DDP 连续训练/断点恢复，以及不等长序列全局有效帧梯度等价测试。独立 verifier 已复跑并批准论文明确训练方法的严格代码门禁；BONES-SEED 真数据和论文私有测试集仍未执行，最终状态以 `paper_training_parity_audit.md` 为准。

## 10. 完成清单

- [x] 论文/代码/官方 config 三方对齐的架构与表示合同
- [x] DDPM 训练、`x0` 预测、七项 smooth-L1/FK loss
- [x] Phase 1/2、五类 constraint pattern、text dropout、dropout 切换
- [x] Adam-atan2、EMA、DDP、AMP、gradient clipping
- [x] manifest、官方 split、timeline subclip、文本缓存、全覆盖 stats、严格 paper-data provenance gate
- [x] full-state checkpoint、provenance hash、精确 mid-epoch resume
- [x] EMA 推理 bundle 与公开 benchmark loader
- [x] 官方 v1.1 strict load + forward + backward
- [ ] Qwen3-32B paraphrase 与 diffusion-transition stitched clips：缺少官方 prompt/revision、transition checkpoint 和生成协议；严格 profile 会阻断
- [ ] BONES-SEED 真数据 DataLoader/一步训练：需要用户接受 license 并提供授权数据
- [ ] 16×A100、global batch 2048、完整 1M steps：需要约 16×A100-80GB 级资源
- [ ] 论文 RP 数值表：专有 700h 数据及若干 recipe 细节未公开，客观不可严格复现
