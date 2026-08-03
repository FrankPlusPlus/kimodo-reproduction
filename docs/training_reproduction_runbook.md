# Kimodo 训练复现运行手册

本目录实现的是基于论文、发布配置、checkpoint 和公开推理代码重建的训练系统，不是 NVIDIA 官方训练源码。证据与默认假设逐项见 `training_reproduction_spec.md`；公开代码接口约束见 `code_training_contract.md`。

## 1. 当前复现边界

| 层级 | 当前状态 | 含义 |
|---|---|---|
| 发布架构与权重兼容 | 已验证 | 官方 `Kimodo-SOMA-SEED-v1.1` 的 408 个权重张量严格加载；模型为 283,281,777 参数；CPU 前向/反向均为有限值 |
| 训练机制复现 | 已实现并烟测 | DDPM `x0` 预测、七项 loss、两阶段 curriculum、五类约束、text dropout、Adam-atan2、EMA、DDP、断点恢复、导出 |
| BONES-SEED 数据闭环 | 数据已下载，生产缓存生成中 | 官方 archive 已核验并解压 142,220 个 BVH；官方 split 128,351 条中 metadata 可解析 128,315 条，36 条缺失会写入 sidecar |
| 论文训练数值复现 | 不能宣称 | 原论文 RP 数据为专有；SEED 的精确采样混合、LLM paraphrase/stitch 数据、loss 域和完整 optimizer 超参未公开 |
| 官方 benchmark 闭环 | 接口已接通；真实运行受数据阻塞 | 导出 bundle 可直接传给 `benchmark/generate_eval.py --checkpoint-bundle`；构建 GT 仍需 BONES-SEED |

## 2. 环境

```bash
# 本服务器复用系统 CUDA 13 / PyTorch 2.13，环境已建在：
source .venv/bin/activate
python -m pip check
```

从空目录重建同类共享环境时（不要覆盖正在使用的 `.venv`）：

```bash
python3.14 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-training-server.txt
.venv/bin/python -m pip install -e . -e ../kimodo-flowmatching
.venv/bin/python -m pip check
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

`--system-site-packages` 会复用管理员维护的 CUDA/PyTorch，避免每个租户复制数 GB wheel；
代价是系统环境升级可能造成漂移。因此 `requirements-training-server.txt` 固定本次通过
测试的版本，正式训练前必须重新执行 `pip check`、完整测试和 CUDA 版本打印。

若只训练预缓存的文本 embedding，不需要在训练进程加载 PEFT/Transformers。生成缓存时才需要安装项目的完整文本编码依赖，并准备约 8B LLM2Vec 模型。

先用可提交配置跑一次从零开始的 CPU 两阶段闭环：

```bash
.venv/bin/kimodo_create_smoke_fixture
.venv/bin/kimodo_train --config configs/training/kimodo_tiny_smoke.yaml
```

第一条命令确定性生成 `tests/fixtures/training` 下的 motion、16D text embedding、manifest 和 stats，并拒绝覆盖已有目录；第二条执行 1 step Phase 1 + 1 step Phase 2、checkpoint/resume 基础路径及 EMA bundle 导出。该 fixture 只验证工程路径，不代表真实训练数据。

## 3. 准备 BONES-SEED

新机器需先接受 BONES-SEED license，再下载并解压。当前服务器已经完成该步骤：
`/home/yezitao/data/yzt/seed` 中有 archive、metadata 和 142,220 个解压 BVH；官方
benchmark split 位于本仓库 `artifacts/benchmark-metadata/splits/`。复建时需要：

- `metadata/seed_metadata_v004.csv` 或 parquet；
- `metadata/seed_metadata_v002_temporal_labels.jsonl`；
- `soma_uniform/bvh/...`；
- benchmark 的官方 train split。

构建 full clip、单 event 与相邻双 event manifest：

```bash
.venv/bin/kimodo_build_manifest \
  --metadata /home/yezitao/data/yzt/seed/metadata/seed_metadata_v004.csv \
  --temporal-labels /home/yezitao/data/yzt/seed/metadata/seed_metadata_v002_temporal_labels.jsonl \
  --split-file /home/yezitao/PublicWorkspace/yzt/kimodo-reproduction/artifacts/benchmark-metadata/splits/train_split_paths.txt \
  --dataset-root /home/yezitao/data/yzt/seed \
  --motion-cache-root /home/yezitao/PublicWorkspace/yzt/kimodo-training-data/motions/soma30-30fps \
  --motion-cache-fps 30 \
  --skeleton soma_uniform --source-fps 120 \
  --output /home/yezitao/PublicWorkspace/yzt/kimodo-training-data/train.raw.jsonl
```

当前服务器已经有且只应保留一份 `kimodo-fm-prepare-bones` 转换进程；它完成前不要再
启动第二份转换。完成后的唯一顺序是：构建 `train.raw.jsonl` → LLM2Vec 文本缓存生成
`train.cached.jsonl` → repro stats/inventory → FM bridge/inventory/stats → 两卡 dry-run/短训。
Qwen3-32B 下载用于研究未公开的数据增强边界，不阻塞公开 BONES-SEED baseline。
LLM2Vec foundation 使用 NousResearch 对论文 Meta-Llama foundation 的公开逐分片等价重发布；
精确来源、revision 和上游等价哈希记录在 `configs/models.server.lock.json`。

`--full-repeats`、`--event-repeats`、`--combined-event-repeats` 是显式的工程采样权重。论文只说按预设分布混合，却没有公布概率；默认均为 1，不能称为官方比例。构建器同时写入 `train.raw.jsonl.metadata.json`，冻结 metadata、timeline、split 的绝对路径、大小和 SHA-256。跨 motion 的 stitched clips 和 Qwen3-32B paraphrases 也未发布，本实现不会伪造它们。

生产配置启用 `paper_method_strict=true` 和 `data.require_paper_data_parity=true`。因此，上述普通 manifest 会被有意拒绝；只有同时包含 Qwen3-32B paraphrase 行、由非增强 diffusion checkpoint 生成的跨 motion transition 行，以及完整 hash/provenance 的 manifest 才能进入严格 paper profile。若只想运行公开数据工程基线，必须显式设置 `paper_method_strict=false` 和 `data.require_paper_data_parity=false`，并把结果标为 `engineering reconstruction`，不能标为论文方法完整复现。

## 4. 缓存冻结的 LLM2Vec 条件

训练只读取 float32 `.npy` embedding。revision 不仅写入 metadata，也实际传入 tokenizer、base model 和 PEFT adapter 的 `from_pretrained`：

```bash
.venv/bin/kimodo_cache_text \
  --manifest ../kimodo-training-data/train.raw.jsonl \
  --output-manifest ../kimodo-training-data/train.cached.jsonl \
  --cache-dir ../kimodo-training-data/text-cache \
  --provider local \
  --foundation-model /home/yezitao/data/yzt/kimodo-repro/models/llm2vec/foundation \
  --foundation-revision 53346005fb0ef11d3b6a83b12c895cca40156b6c \
  --mntp-model /home/yezitao/data/yzt/kimodo-repro/models/llm2vec/mntp-adapter \
  --mntp-revision 31474e395ada192e8ed1586db6be79fb3b70c9c0 \
  --supervised-model /home/yezitao/data/yzt/kimodo-repro/models/llm2vec/supervised-adapter \
  --supervised-revision baa8ebf04a1c2500e61288e7dad65e8ae42601a7
```

Meta 原仓库为 gated 且当前账号未获批准。服务器改用
`NousResearch/Meta-Llama-3-8B-Instruct@53346005...`：四个 BF16 权重分片、
`config.json` 和 `tokenizer.json` 均与论文所用
`meta-llama/Meta-Llama-3-8B-Instruct@8afb486c...` 一致。它不是独立训练的新模型，
仍须遵守 Llama 3 license。两个公开 adapter 保持原 revision；不要改用 GGUF、GPTQ、
AWQ 或二次微调权重。

如复用官方 text-encoder 服务，可改为 `--provider api --api-url ...`。API 服务自身的模型 revision 必须另行固定并记录。

## 5. 计算 normalization stats

```bash
.venv/bin/kimodo_compute_stats \
  --manifest /home/yezitao/PublicWorkspace/yzt/kimodo-training-data/train.cached.jsonl \
  --output /home/yezitao/PublicWorkspace/yzt/kimodo-training-data/stats/repro-soma30-30fps \
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

正式训练前需一次性建立并完整校验 manifest 引用清单。`build` 会读取并 SHA-256
所有唯一 motion、embedding 和来源 sidecar；训练启动只校验 manifest、inventory 与
metadata 的聚合摘要，不会再次扫描数百 GB：

```bash
.venv/bin/kimodo_reference_inventory build \
  --manifest /home/yezitao/PublicWorkspace/yzt/kimodo-training-data/train.cached.jsonl \
  --output /home/yezitao/PublicWorkspace/yzt/kimodo-training-data/train.cached.references.jsonl

# 数据迁移、恢复长期实验或怀疑资产被改动时，独立执行全量复验：
.venv/bin/kimodo_reference_inventory verify \
  --manifest /home/yezitao/PublicWorkspace/yzt/kimodo-training-data/train.cached.jsonl \
  --inventory /home/yezitao/PublicWorkspace/yzt/kimodo-training-data/train.cached.references.jsonl
```

生产配置使用 `data.reference_verification=inventory` 并强制 inventory 和
`.metadata.json` sidecar 同时存在。小型测试仍可使用 `full`，但大数据正式训练不得让
trainer 每次启动重新逐文件哈希。

## 6. 从头训练、论文 Phase 2 恢复与实验性后训练

从头训练：

```bash
torchrun --standalone --nproc_per_node=16 -m kimodo.training.cli \
  --config configs/training/kimodo_soma_seed_reproduction.yaml \
  --set data.manifest=/home/yezitao/PublicWorkspace/yzt/kimodo-training-data/train.cached.jsonl \
  --set data.reference_inventory=/home/yezitao/PublicWorkspace/yzt/kimodo-training-data/train.cached.references.jsonl \
  --set model.stats_path=/home/yezitao/PublicWorkspace/yzt/kimodo-training-data/stats/repro-soma30-30fps
```

上述命令在缺少论文增强资产时会 fail closed。公开 BONES-SEED 工程基线可附加 `--set paper_method_strict=false --set data.require_paper_data_parity=false`；这些 override 会主动降低复现等级。

两卡服务器使用独立配置和安全 launcher。launcher 不选择物理 GPU；必须由调度器或
租户显式设置可见设备，并且 PyTorch 最终只能看到两张卡：

```bash
export CUDA_VISIBLE_DEVICES=<allocated-device-a>,<allocated-device-b>
export KIMODO_TRAINING_DATA=/home/yezitao/PublicWorkspace/yzt/kimodo-training-data
export KIMODO_LOCAL_ROOT=/home/yezitao/PublicWorkspace/yzt/kimodo-reproduction
scripts/train_two_gpu_seed.sh
```

launcher 默认使用可训练的公开数据 engineering profile：它显式设置
`paper_method_strict=false`、`data.require_paper_data_parity=false`，但模型、loss、optimizer、
两阶段 curriculum 和 EMA 数值均与 strict two-GPU profile 相同。它不能标成完整论文数据
recipe 复现。获得带完整 provenance 的 Qwen/transition 增强 manifest 后，可改用严格 profile：

```bash
export KIMODO_SHARED_ROOT=/home/yezitao/data/yzt/kimodo-repro
export KIMODO_TWO_GPU_CONFIG="$PWD/configs/training/kimodo_soma_seed_two_gpu.yaml"
scripts/train_two_gpu_seed.sh
```

严格 profile 保持 `paper_method_strict=true` 和数据增强门禁，只设置
`runtime.enforce_paper_scale=false`。两个 profile 都采用保守起点
`32 × 2 ranks × 4 accumulation = global batch 256`。GPU 型号、可见设备、world size、
local batch、accumulation 和 effective global batch 会写入 provenance。正式长跑前应在
真实 300 帧 batch 上完成显存/吞吐 profiling；如果修改 batch/accumulation，必须创建
新的正式 run，不得用训练关键配置不同的 checkpoint 强行恢复。
这里的增强门禁只能证明 manifest 自身的 schema、hash 和 provenance 一致，不能证明
未公开的 Qwen prompt、采样 mixture 或 transition 协议与 NVIDIA 私有 recipe 相同；
即使门禁通过，也仍不能宣称完整论文数据分布复现。
新训练会拒绝写入非空 `runtime.output_dir`；断点续训必须显式设置
`runtime.resume`。滚动保留最近三个普通 checkpoint，同时永久保护 500k 阶段边界、
每 100k milestone 和最终 checkpoint。

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
  --set model.checkpoint_dir=/home/yezitao/data/yzt/kimodo-repro/models/Kimodo-SOMA-SEED-v1.1 \
  --set data.manifest=/home/yezitao/PublicWorkspace/yzt/kimodo-training-data/train.cached.jsonl \
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

实际 latest 路径记录在 `checkpoints/latest.txt`。checkpoint 包含 online/EMA model、optimizer、scaler、epoch/batch、每个 DDP rank 的 Python/NumPy/Torch RNG、resolved config，以及 manifest/inventory 聚合摘要、stats、骨架资产、官方 bundle、关键代码和硬件 scale 记录。任一训练关键输入变化都会拒绝恢复。逐个 motion/embedding 的 SHA-256 位于预生成 inventory；需要内容级复验时运行上面的独立 `verify`，而不是在 trainer 启动时扫描。为保证 `set_epoch` 的确定性，当前实现明确禁用 `persistent_workers=true`。

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
KIMODO_OFFICIAL_BUNDLE=/home/yezitao/data/yzt/kimodo-repro/models/Kimodo-SOMA-SEED-v1.1 \
  .venv/bin/python -m pytest -q tests/training/test_official_checkpoint.py

# 配置静态检查
.venv/bin/kimodo_train \
  --config configs/training/kimodo_soma_seed_reproduction.yaml --dry-run
```

本轮最终集成自检结果：常规套件 `50 passed, 1 skipped`（skip 是需显式资产路径的官方 gate）；官方 checkpoint 文件使用真实资产路径单独运行 `2 passed`。仓库套件内包含 Figure 9 tensor/梯度合同、paper-data fail-closed gate、重采样与 stats 全覆盖、论文评测数值 oracle、两阶段 tiny smoke、2-rank CPU DDP 连续训练/断点恢复，以及不等长序列全局有效帧梯度等价测试。真实 BONES-SEED 已下载，30fps 生产缓存正在生成；完整 DataLoader/一步训练仍需缓存、文本 embedding、manifest 与 stats 全部闭环。论文私有测试集仍不可用，最终状态以 `paper_training_parity_audit.md` 为准。

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
- [ ] BONES-SEED 真数据 DataLoader/一步训练：数据已授权下载；等待 30fps cache、LLM2Vec cache、manifest、inventory 与 stats 完成
- [ ] 16×A100、global batch 2048、完整 1M steps：需要约 16×A100-80GB 级资源
- [ ] 论文 RP 数值表：专有 700h 数据及若干 recipe 细节未公开，客观不可严格复现
