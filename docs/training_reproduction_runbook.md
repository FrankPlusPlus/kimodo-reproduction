# Kimodo repro：clone 后资源准备与训练

本仓库是基于论文、公开配置和推理代码重建的训练系统，不是 NVIDIA 原始训练源码。
公开 BONES-SEED 可形成可训练的工程基线；论文未公开的 Qwen prompt/mixture、跨动作
transition 数据和若干 optimizer/loss 细节不能被脚本补齐。

## 配置分层

工程明确分为三类 YAML，其中前两类共同构成资源配置层：

| 文件 | 职责 | 是否包含机器路径 |
|---|---|---:|
| `resources/catalog.public.yaml` | 远端 repo、完整 revision、文件大小/SHA-256、资源分组 | 否 |
| `<storage-root>/config/resources.paths.yaml` | 下载目标、已有资源、预处理输出位置 | 是；仓库外 |
| `<storage-root>/config/repro.paths.yaml` | 训练实际读取的 manifest/inventory/stats 和 run 目录 | 是；脚本生成 |
| `configs/training/*.yaml` | 方法、模型、loss、课程与 optimizer | 否 |
| `configs/overlays/*.yaml` | 卡数、local batch、gradient accumulation、workers | 否 |

合并优先级固定为：base training YAML → paths YAML → hardware overlay → CLI `--set`。
paths YAML 采用白名单，放入 batch size、学习率等字段会直接报错。

## 新服务器最短流程

先在 Hugging Face 接受 `bones-studio/seed` 的 license，并用正常的 HF credential store
或 `HF_TOKEN` 登录。token 不写入任何 YAML、receipt 或日志。BONES converter 属于本仓库，
流程不 clone、安装或导入 Flow Matching 等第二个项目：

```bash
git clone https://github.com/FrankPlusPlus/kimodo-reproduction.git /work/repro
cd /work/repro
proxy_on  # 仅服务器需要代理时
scripts/bootstrap_training.sh --storage-root /shared/kimodo --hf-login
```

脚本把机器配置写入 `/shared/kimodo/config/`，数据和 run 均放在指定 storage root：

- raw archives 和固定模型可以放共享/NFS；
- `pipeline.prepared_root` 应放训练节点低延迟存储；
- `pipeline.run_root` 放 checkpoint/run；重要 milestone 再归档共享盘；
- 建议为最小训练流水线预留 230–260 GB；Qwen 另需约 65.5 GB。

需要逐阶段控制时，仍可先用 `resources ... init` 生成 YAML，再运行：

```bash
scripts/resources/resources.sh --paths /shared/kimodo/config/resources.paths.yaml plan
scripts/resources/resources.sh --paths /shared/kimodo/config/resources.paths.yaml fetch
scripts/resources/resources.sh --paths /shared/kimodo/config/resources.paths.yaml prepare
```

也可以执行一条命令：

```bash
scripts/resources/resources.sh --paths /shared/kimodo/config/resources.paths.yaml all
```

默认 `train-minimal` 只包含：

- `bones-studio/seed@2f59b207...` 的 SOMA Uniform archive/metadata/timeline；
- `nvidia/Kimodo-Motion-Gen-Benchmark@2727f526...` 的官方 train split；
- `NousResearch/Meta-Llama-3-8B-Instruct@53346005...`；
- McGill LLM2Vec MNTP 与 supervised adapters。

它不会下载 Qwen 或官方 Kimodo inference checkpoint。可选资源必须显式指定
`paper-exploration` 或 `official-oracle` 分组。

## 已有资源直接复用

完整的旧 cache 可用 `bootstrap_training.sh --legacy-root ...` 做一次 verified adoption；已经
train-ready 的 portable bundle 搬到新服务器后可用 `--prepared-root ...` 完整校验并绑定，均无需
重新运行 8B encoder。可直接复制的三种部署流程见
[`portable_training_setup.md`](portable_training_setup.md)。单个 pinned source 也可继续用下面的
`existing_path` 方式复用。

在 `resources/paths.local.yaml` 中将对应项设置为：

```yaml
resources:
  bones_seed:
    destination: null
    existing_path: /shared/datasets/bones-seed
  llm2vec_foundation:
    destination: null
    existing_path: /shared/models/llm2vec/foundation
```

`existing_path` 会逐文件核对 size 和 SHA-256，通过后零复制复用；失败时不会修改共享
目录或退化为重新下载。其他资源同理。`destination` 模式支持 Hugging Face 分片断点续传，
并为同一资源加锁，避免两个进程同时写。

## prepare 生成什么

流水线按顺序完成：

1. 安全解压 SOMA Uniform archive，拒绝路径穿越和链接逃逸；
2. BVH SOMA77/120 Hz → SOMA30/30 Hz canonical NPZ；
3. 构建 portable `train.raw.jsonl`；
4. 离线运行 Llama-3 8B + MNTP + supervised adapter，生成 FP32 `[1,4096]` cache；
5. 构建 portable `train.cached.jsonl`；
6. 拟合 repro normalization stats；
7. 构建并完整验证 reference inventory；
8. 写 `resource-state.json` 和 `pipeline.repro_paths_yaml`。

manifest、embedding 与 inventory 引用均相对 prepared bundle；把整个 prepared root 移到
另一挂载点后，只需调整 paths YAML。cache key 绑定模型 repo/revision、内容 hash、实际编码
实现和数值依赖，不再绑定服务器绝对路径、无关文档 commit 或 model-lock 的本机路径。

重复运行会验证并复用完整阶段。manifest/sidecar、inventory/metadata 或 stats 出现孤立/不完整
状态时会 fail closed，不会静默覆盖。LLM2Vec 单文件缓存为原子写，中断后可继续复用已完成项。

## 两张 H200 训练

资源完成后，把生成的 paths YAML 传给 launcher：

```bash
export CUDA_VISIBLE_DEVICES=0,2
export KIMODO_PATHS_CONFIG=/path/from/pipeline/repro.paths.yaml

# 只解析配置分层；完整资源内容门禁由前面的 prepare/verify 执行
scripts/train_two_gpu_seed.sh --dry-run

# 十步真实 smoke
scripts/train_two_gpu_seed.sh --set runtime.max_steps_override=10

# 正式 500k + 500k
scripts/train_two_gpu_seed.sh
```

launcher 默认接受名称中包含 `H200` 的两张卡（SXM/NVL 均可）。非 H200 测试可显式设置
`KIMODO_EXPECTED_GPU_PATTERN`；需要审计精确型号时设置 `KIMODO_EXPECTED_GPU_NAME`。

默认 overlay 为 `configs/overlays/two_h200_gb2048.yaml`：两 ranks × local batch 128 ×
accumulation 8 = effective global batch 2048。硬件不同可复制 overlay 修改 batch/workers，
不应复制整份方法 YAML。

训练只读取已缓存的 `[1,4096]` 句向量，不会加载 Llama、MNTP、supervised adapter 或 Qwen。

训练进程会在 output dir 写 `.kimodo-active-run.lock`。同一主机上 PID 已退出、PID 被复用或主机
已重启时，下一次启动会保守地自动回收；其他主机或无法解析的 owner 不会被猜测为死亡。可先检查：

```bash
python -m kimodo.training.run_lock inspect /path/to/run
python -m kimodo.training.run_lock clear-stale /path/to/run
# 对 remote/unknown owner，先在调度器/对应节点确认任务已终止，再逐字复制 inspect 输出的 token：
python -m kimodo.training.run_lock clear /path/to/run --expected-token <token>
```

`clear-stale` 只删除本机能够证明已失效的锁；显式 `clear` 必须给出当前随机 token，避免误删
已经被另一个训练进程替换的锁，但外部 liveness 判断仍由操作者负责。

## 与其他训练项目的边界

本流程到 `repro.paths.yaml` 和 Kimodo 训练产物为止，不负责生成其他项目的 manifest、stats 或路径文件。
若另一个项目需要相同原始数据，应按它自己的数据契约独立准备；不要把本仓库安装进对方环境，也不要把
对方仓库作为本项目的 converter 依赖。

## 复现边界

- pinned 官方 split 有 128,351 个 key；当前 pinned `seed_metadata_v004.csv` 可解析其中
  128,315 个，固定缺 36 个。流水线锁定三项计数和缺失集合 SHA-256，任何上游漂移都会
  fail closed；这不是静默 `allow-missing`；
- public profile 是 BONES-SEED engineering reconstruction；
- `configs/training/kimodo_soma_seed_reproduction.yaml` 是 paper-data strict gate，缺少未公开
  增强资产时应当失败；
- Qwen 权重只是可选研究工具，单独下载它不能恢复论文数据；
- 官方 `Kimodo-SOMA-SEED-v1.1` 是 EMA inference bundle，不是可恢复 optimizer/RNG 的
  Phase-1 full-state checkpoint。

论文逐项对齐与未知项见 `training_reproduction_spec.md` 和
`paper_training_parity_audit.md`；H200 测速见 `h200_training_benchmark.md`。
