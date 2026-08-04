# 从 Git clone 到可训练：可搬迁部署手册

本文只解决一个工程目标：在新的 Linux GPU 服务器上 clone 仓库后，以脚本准备或复用全部训练资源，
最后得到一个可直接传给 trainer 的路径 YAML。模型方法参数仍在版本化训练 YAML 中；服务器路径、令牌、
数据和 checkpoint 不进入 Git。

## 1. 一条命令负责什么

入口是 `scripts/bootstrap_training.sh`。它依次完成：

1. 创建隔离 `.venv`，按 `requirements-training-server.txt` 安装并执行 `pip check`；
2. 从 `resources/dependencies.lock.yaml` 读取 FM 仓库和精确 commit，自动 clone 到
   `.deps/kimodo-flowmatching`，以 detached HEAD 校验后安装；
3. 生成 `<storage-root>/config/resources.paths.yaml`；
4. 下载并逐文件校验固定 revision 的 BONES-SEED 和 LLM2Vec 资源，或校验/绑定已有资源；
5. 构建 SOMA30/30fps motion、相对路径 manifest、离线 `[1,4096]` 句向量 cache、stats 和
   reference inventory；
6. 只有完整 hash、schema、stats 和真实 CPU batch preflight 全部通过后，才写
   `resource-state.json: repro_train_ready`；
7. 输出 `<storage-root>/config/repro.paths.yaml`，trainer 只需读取该文件。

训练时不会加载 Llama/Qwen。LLM2Vec foundation、MNTP adapter 和 supervised adapter 只在首次离线生成
句向量时使用；已有已验证 cache 可直接迁移，不会冒充新 encoder 重新盖章。

## 2. 前置条件

- Linux、Git、可用的 Python（推荐站点支持的 3.11/3.12；本机也验收了 Python 3.14.6）；
- 可写的共享存储；全量首次构建保守预留 230–260GB，run 目录还需为每个 full-state checkpoint
  预留约 4.6GB；
- 已在 BONES-SEED 页面接受 gated license；bootstrap 可在创建 venv 后执行 `hf auth login`；
- 若服务器访问外网需要代理，先在当前 shell 执行 `proxy_on`。脚本不会把代理或 token 写入 YAML、
  receipt 或 Git。

## 3. 场景 A：全新服务器，从官方资源构建

```bash
git clone https://github.com/FrankPlusPlus/kimodo-reproduction.git
cd kimodo-reproduction

proxy_on  # 仅在该服务器需要代理时执行
scripts/bootstrap_training.sh --storage-root /shared/kimodo --hf-login
```

默认只准备资源，不会意外启动 1M-step 作业。完成后先检查真实 batch，再训练：

```bash
.venv/bin/python -m kimodo.training.cli \
  --config configs/training/kimodo_soma_seed_public.yaml \
  --paths /shared/kimodo/config/repro.paths.yaml \
  --overlay configs/overlays/two_h200_gb2048.yaml \
  --preflight

CUDA_VISIBLE_DEVICES=0,1 \
KIMODO_PATHS_CONFIG=/shared/kimodo/config/repro.paths.yaml \
scripts/train_two_gpu_seed.sh
```

也可以在准备成功后显式要求立即训练：

```bash
scripts/bootstrap_training.sh \
  --storage-root /shared/kimodo \
  --gpus 0,1 \
  --train
```

## 4. 场景 B：当前服务器已有旧 cache，不重跑 LLM2Vec

旧 bundle 必须含 `train.raw.jsonl`、`train.cached.jsonl`、`motions/`、`text-cache/` 和完整 stats。
同一文件系统推荐 `hardlink`，几乎不增加数据块占用；跨文件系统用 `copy`。

```bash
scripts/bootstrap_training.sh \
  --storage-root /shared/kimodo \
  --legacy-root /shared/kimodo-training-data \
  --conversion-inventory /shared/conversion/soma30-30fps.inventory.jsonl \
  --asset-mode hardlink
```

adoption 会验证旧 sidecar 中的清洗文本 SHA、cache key、provider identity、`float32 [1,4096]`、文件
SHA 和 motion frame count，然后生成相对路径 schema-5 bundle。它保留旧 provider provenance，绝不加载
8B encoder。首次遍历 140 万行和数十万个 inode 需要数分钟，这是一次性迁移成本。

## 5. 场景 C：把 train-ready bundle 搬到另一台服务器

prepared bundle 内所有训练引用都是相对路径，可整体复制或挂载：

```bash
rsync -a --info=progress2 \
  /shared/kimodo/prepared/adopted-legacy-soma30-v1/ \
  new-server:/shared/kimodo-prepared/
```

在新服务器 clone 后直接绑定；不会下载 BONES-SEED 或运行 LLM2Vec：

```bash
scripts/bootstrap_training.sh \
  --storage-root /shared/kimodo-runtime \
  --prepared-root /shared/kimodo-prepared
```

`bind-prepared` 会重新 hash manifest、inventory、全部引用和六个 stats 文件，发现漏拷或损坏就失败。
校验成功后生成 `/shared/kimodo-runtime/config/repro.paths.yaml`。若环境已经另外装好，也可只调用：

```bash
.venv/bin/python -m kimodo.resources.cli \
  --catalog resources/catalog.public.yaml \
  bind-prepared \
  --prepared-root /shared/kimodo-prepared \
  --run-root /shared/kimodo-runtime/runs \
  --output /shared/kimodo-runtime/config/repro.paths.yaml
```

## 6. 两层 YAML 和目录职责

| 内容 | 位置 | 是否进 Git |
|---|---|---|
| 固定 repo/revision/文件 hash | `resources/catalog.public.yaml` | 是 |
| FM 精确依赖 commit | `resources/dependencies.lock.yaml` | 是 |
| 下载、已有资源和 prepare 输出位置 | `<storage-root>/config/resources.paths.yaml` | 否 |
| trainer 的 manifest/stats/output/resume 路径 | `<storage-root>/config/repro.paths.yaml` | 否 |
| 论文方法参数 | `configs/training/*.yaml` | 是 |
| 卡数、local batch、梯度累计、workers | `configs/overlays/*.yaml` | 是 |

这样做有两个用途：有现成资源时只替换 paths YAML；换硬件时只替换 overlay。数据位置不会污染论文方法
配置，硬件 batch 也不会改写资源 receipt。

## 7. 两卡 H200 与断点续训

默认 overlay 是 `2 ranks × local batch 128 × accumulation 8 = global batch 2048`。Linux 上显式使用
`fork` 启动 CPU-only DataLoader workers，避免 Python 3.14 的 `forkserver` 把 1.4M 行 dataset 为每个
worker 序列化一遍。manifest 首次扫描仍是固定启动成本，不应计入稳态单 step 时间。

短训：

```bash
CUDA_VISIBLE_DEVICES=0,2 \
KIMODO_PATHS_CONFIG=/shared/kimodo/config/repro.paths.yaml \
scripts/train_two_gpu_seed.sh \
  --set runtime.output_dir=/shared/kimodo/runs/smoke-001 \
  --set runtime.max_steps_override=2 \
  --set runtime.checkpoint_every=1
```

原地 resume：

```bash
CUDA_VISIBLE_DEVICES=0,2 \
KIMODO_PATHS_CONFIG=/shared/kimodo/config/repro.paths.yaml \
scripts/train_two_gpu_seed.sh \
  --set runtime.output_dir=/shared/kimodo/runs/smoke-001 \
  --set runtime.resume=/shared/kimodo/runs/smoke-001/checkpoints/step-000000002.pt \
  --set runtime.max_steps_override=10
```

fresh run 要求空 output；in-place resume 要求 checkpoint 属于该 run，并校验训练关键配置、代码、数据、
stats、world size 和每-rank RNG。并发写同一 run 会被独占锁拒绝，已有同 step checkpoint 不会被覆盖。

## 8. 当前真实验收与论文边界

本机的 portable adoption 已完整验证 `1,407,184` 行、`128,315` 个唯一 motion、`132,972` 个唯一
embedding；真实 preflight 组装了 `[128,300,369]` motion 和 `[128,1,4096]` text batch；两张 H200
完成 step 1 并从 4.53GB full-state checkpoint 原地恢复到 step 2。

这证明公开 BONES-SEED 工程已经 train-ready，不证明 1M steps 的最终收敛或论文私有 Sec. 6 数值。
论文未公开/不可得的 Qwen paraphrase 配方、跨动作 transition 混合、私有 Rigplay/native-27 与评测协议
仍是复现边界；public profile 明确是工程复刻，strict profile 会在缺这些资产时 fail closed。
