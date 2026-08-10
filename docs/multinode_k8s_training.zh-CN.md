# 两机 16×H200 的容器化与多机 DDP 部署

本文对应本仓库当前实现，物理资源是两台 8×H200，训练契约是 16 个 DDP rank。Kubernetes Pod 的
切分由公司平台决定：可以是 `2 Pods × 8 GPU`、`4 Pods × 4 GPU` 或 `16 Pods × 1 GPU`。一个 Pod
不能跨物理节点，因此在每台只有 8 卡时，`1 Pod × 16 GPU` 不可调度。

## 1. 谁负责什么

```text
提交端/跳板机
  └─ 创建数据 Job 和训练 Job，本身不参加计算

共享持久化存储（PVC 挂到容器 /home/share）
  ├─ yzt/kimodo-reproduction/：常驻代码（KIMODO_CODE_ROOT，训练从此处启动）
  └─ yezitao-kimodo-reproduction/
      ├─ benchmark-v2-soma30-v2.2/：manifest、motion、text cache、stats、repro.paths.yaml
      ├─ feature-cache/v1/：离线 motion 特征（可选，见 docs/motion_feature_cache.zh-CN.md）
      ├─ yezitao-kimodo-eval-v1/：固定 benchmark proxy 与官方基线
      └─ runs/：日志、checkpoint、export（rank 0 可写）

Pod/launcher group 0..N-1
  └─ 每 Pod 启动 M 个 local ranks；N × M 必须等于 16
             └──── NCCL：机内 NVLink/NVSwitch，机间 IB/RoCE ────┘
```

Kubernetes 负责 Pod、GPU、网络、DNS、存储和失败重启；`torchrun` 负责在每个 Pod 生成相应进程并
给出 `RANK/WORLD_SIZE/LOCAL_RANK`；训练代码使用这些变量初始化 NCCL/DDP。应用代码不实现 TCP、
RDMA、ring all-reduce 或梯度发送。

## 2. 当前代码为什么已经能跨机

`kimodo/training/engine.py` 已经完成：

- 从环境读取 `WORLD_SIZE/RANK/LOCAL_RANK`，GPU 训练选择 NCCL；
- 一个进程绑定一个本机 GPU，并用 `DistributedDataParallel` 包装模型；
- `DistributedSampler` 按 16 个 rank 切分样本；
- 梯度累计期间使用 DDP 同步，指标和有效帧数用 collective 汇总；
- 只有全局 rank 0 写 config、日志、checkpoint 和推理 export；
- checkpoint 前收集每个 rank 的 RNG，支持同一 world size 的精确恢复。

原来的 `scripts/train_two_gpu_seed.sh` 不能跨机只是因为它硬编码了 `--standalone`、2 个本地进程和
“恰好两张可见 GPU”的检查。多机入口改用 `scripts/train_distributed.sh`。

## 3. 镜像中放什么

历史 tag、digest、推荐用法见 [`docs/docker_images.zh-CN.md`](./docker_images.zh-CN.md)
（PVC：`/home/share/yzt/kimodo-reproduction/docs/docker_images.zh-CN.md`）。当前训练推荐
`hub.inner.ai.kingsoft.com/hh-678395/kimodo-reproduction:v8`（`:pvc-train` 同内容）。

镜像提供 Python/CUDA/NCCL 运行时、依赖、RDMA 工具，以及 `/workspace/scripts/container_start.sh`
启动器。公司训练默认从共享盘代码目录启动，而不是改镜像内 `/workspace`：

```text
KIMODO_CODE_ROOT=/home/share/yzt/kimodo-reproduction
KIMODO_STORAGE_ROOT=/home/share/yezitao-kimodo-reproduction
```

构建时仍会把仓库复制进 `/workspace` 以便安装依赖和 smoke test；运行时若 PVC 代码存在，
launcher 会设置 `PYTHONPATH` 并 exec `$KIMODO_CODE_ROOT/scripts/...`。日常改代码只更新 PVC；
只有依赖/系统环境变化才重打镜像。不要把数百 GB 数据、token、checkpoint 打进镜像。
镜像默认 `idle`，创建 Pod 不必立刻具备 16 张 H200 或 RDMA。
正式作业设置受支持的 `KIMODO_CONTAINER_MODE`，或覆盖容器命令：

```text
idle           默认保活，不训练
train-company  公司训练；拓扑由环境变量控制（默认 2x8 / 16 rank）
train-local    实验室两卡训练
prepare        PVC 数据准备或绑定
preflight      只读取并验证一个真实 batch
eval-watch     监控训练导出并运行 benchmark
eval-official  生成固定官方 baseline
```

显式 Pod `command` 的优先级最高；例如直接调用 `scripts/train_company.sh` 时无需设置 mode。
拓扑与短跑用环境变量控制，例如 2x3 冒烟：`KIMODO_NNODES=2` `KIMODO_NPROC_PER_NODE=3`
`KIMODO_EXPECTED_WORLD_SIZE=6` `KIMODO_BATCH_SIZE=8` `KIMODO_MAX_STEPS=5`。

```bash
docker build -t REGISTRY/yezitao-kimodo-train:GIT_SHA .
docker push REGISTRY/yezitao-kimodo-train:GIT_SHA
```

镜像 tag 最好使用 commit SHA 或不可变 digest。基础镜像中的 CUDA、PyTorch 和 NCCL 必须与 H200
节点驱动、容器运行时及集群 RDMA 软件栈一起做验收；不能只凭“容器里能看到 GPU”判断 IB 已生效。

## 4. 数据处理不要跟每个训练 Pod 绑在一起

推荐将数据准备做成独立 Job：

```bash
KIMODO_PYTHON=python \
KIMODO_STORAGE_ROOT=/mnt/kimodo \
/workspace/scripts/prepare_container.sh
```

从原始资源首次执行时，会初始化资源路径、下载并校验固定 revision 的资源、转换 motion、生成离线文本
embedding、manifest、stats、inventory 和 `/mnt/kimodo/config/repro.paths.yaml`。接入其他 train-ready
prepared bundle 时，设置 `KIMODO_PREPARED_ROOT=/mnt/source/prepared-bundle`，脚本只验证并绑定，不重跑
LLM2Vec。

本项目交付的 `benchmark-v2-soma30-v2.2.tar.zst` 已经是 train-ready portable bundle，并自带使用
`KIMODO_DATA_ROOT`/`KIMODO_RUN_ROOT` 的 `repro.paths.yaml`。把它直接解压到
`${KIMODO_STORAGE_ROOT}/benchmark-v2-soma30-v2.2` 后，公司训练入口会直接使用包内 paths 文件；不需要再跑
prepare Job。prepare/bind 流程只用于从原始资源重建数据或接入其他 prepared bundle。

实践中的作业顺序应是：

1. 数据准备 Job（一次，必要时用 1 张 GPU 生成文本 cache）；
2. 数据 preflight Job（不启动 16 卡，只验证一个真实 CPU batch）；
3. 两节点训练 Job；
4. 可选验证/导出 Job。

不要在各训练 Pod 的 initContainer 中各跑一次完整 prepare。那会竞争共享目录、重复占 GPU，并让每次
训练重启都重新检查大数据。initContainer 只适合做“路径存在、容量足够、DNS 可解析”之类的轻检查。

## 5. 训练 Pod 必须获得的契约

所有 Pod 使用同一个不可变镜像和相同命令：

```bash
/workspace/scripts/train_company.sh
```

如果公司平台不允许覆盖容器命令、只允许注入环境变量，则设置：

```text
KIMODO_CONTAINER_MODE=train-company
```

默认 `2×8` 切分的公共环境：

```text
KIMODO_NNODES=2
KIMODO_NPROC_PER_NODE=8
KIMODO_CODE_ROOT=/home/share/yzt/kimodo-reproduction
KIMODO_STORAGE_ROOT=/home/share/yezitao-kimodo-reproduction
KIMODO_NODE_RANK=0|1
MASTER_ADDR=<node-rank-0 的稳定 Pod DNS 或 IP>
MASTER_PORT=29500
```

分布式三件套含义：

- `KIMODO_NODE_RANK`：当前 Pod 是第几台训练机（0 或 1）。两机各不相同。
- `MASTER_ADDR`：rank0 那台机的地址（Pod DNS/IP），所有 Pod 填同一个，用来会合。
- `MASTER_PORT`：会合端口，默认 `29500`，两机相同且网络要通。

所有训练 Pod 把同一块共享 PVC 挂到 `/home/share`。代码在 `yzt/kimodo-reproduction`，
数据在 `yezitao-kimodo-reproduction`。

标准布局下，company launcher 会从 `KIMODO_STORAGE_ROOT` 自动推导 data、paths、run、
benchmark proxy 和 eval 输出目录；只有实际目录布局不同时才覆盖对应变量。W&B project、group、
train/benchmark run ID 同样有稳定默认值，通常不需要填写。

密钥（`WANDB_API_KEY`、洗数据用的 `PRODUCT_GRAPH_LLM_API_KEY`）优先由平台 Secret 注入。
若平台不便注入，可把 gitignored 的 `.env` 放在 PVC 代码根（见仓库 `.env.example`）：

```bash
cp .env.example /home/share/yzt/kimodo-reproduction/.env
chmod 600 /home/share/yzt/kimodo-reproduction/.env
# 编辑填入 key；勿提交 Git
```

`container_start.sh` / `train_company.sh` 会自动加载该文件中**尚未设置**的变量。
共享盘上的明文 key 对有 share 权限的人可见，能走平台 Secret 仍更安全。训练 Pod 与
eval-watch Pod 可共用同一 `.env`；只有 global rank 0 / eval Pod 会上报 W&B。

若平台采用每物理节点两个 4-GPU Pod，则设置 `KIMODO_NNODES=4`、
`KIMODO_NPROC_PER_NODE=4` 和四个唯一 node rank。这里的 `KIMODO_NNODES` 是 launcher/Pod 数，
不是物理机器数。production YAML 还会在 trainer 内再次强制检查 `world_size=16` 和
`effective_global_batch=2048`，因此绕过 launcher 也不会静默跑成错误规模。所有 Pod 把同一个共享卷
挂到所有 Pod 中相同的 `/mnt/kimodo`。如果公司的实际挂载点不同，只设置
`KIMODO_STORAGE_ROOT` 和 paths/run 变量，不要把个人目录写入镜像。如果平台的 PyTorch
runtime 已经替你执行 `torchrun` 并直接注入每个训练进程的 `RANK/WORLD_SIZE/LOCAL_RANK`，不要再次
套 `train_distributed.sh`；此时容器命令直接使用：

```bash
python -m kimodo.training.cli \
  --config /workspace/configs/training/kimodo_soma_seed_v2_1m_16h200.yaml \
  --paths /home/share/yezitao-kimodo-reproduction/benchmark-v2-soma30-v2.2/repro.paths.yaml \
  --set runtime.output_dir=/home/share/yezitao-kimodo-reproduction/runs/v2-1m-production
```

这是上线前必须向平台管理员确认的第一件事：平台是“一 Pod 一节点，由镜像内部 torchrun”，还是“平台
已经 torchrun 到每个进程”。两层 torchrun 会启动错误数量的进程。

## 6. batch 怎么从两卡换算到 16 卡

本工程的有效全局 batch 为：

```text
world_size × per-rank batch_size × gradient_accumulation_steps
```

本地两卡 overlay 是 `2 × 128 × 8 = 2048`；公司完整 production YAML 是
`16 × 128 × 1 = 2048`，无需额外硬件 overlay。若真实序列分布让单卡 batch 128 OOM，可以保持
全局 2048 改成 `16 × 64 × 2` 或
`16 × 32 × 4`。这里的 `batch_size` 是每个进程/每张卡，不是每节点也不是全局值。

DDP 会在每张卡保留完整模型、optimizer 和 EMA。16 卡提升数据并行吞吐，并不会把模型显存自动分摊到
16 张卡；只有模型单卡放不下时才需要 FSDP/DeepSpeed ZeRO，而当前 283M 模型不因“多机”本身需要它们。

## 7. IB/RoCE 通信由谁配置

训练代码只选择 NCCL backend。真正让 NCCL 使用高速网络，需要平台层同时满足：

- 两节点的 GPU/NIC 拓扑、驱动、OFED/Inbox verbs 和 NCCL 版本兼容；
- Pod 能看到 RDMA 设备（通常由 NVIDIA Network Operator、SR-IOV 或 RDMA device plugin 提供）；
- CNI/NetworkPolicy/防火墙允许节点间 rendezvous 和 NCCL 数据连接；
- RoCE 集群还要由网络管理员配置 PFC/ECN、GID/traffic class 等；这些不能从训练代码猜；
- 如使用 GPUDirect RDMA，IOMMU、DMA-BUF/peer memory、容器权限和拓扑也要满足平台要求。

公司启动器通过 `scripts/nccl_rdma_env.sh` 配置 NCCL/RDMA。默认 `KIMODO_NCCL_ENV_MODE=respect`：
平台已注入的 `NCCL_IB_*` 不覆盖，只在未设置时补 `NCCL_IB_HCA=mlx5`。

若日志出现 `ibv_modify_qp ... errno 19`（平台常注入 `NCCL_IB_GID_INDEX=3` 且为
`::ffff:` IPv4-mapped GID），在启动脚本里改用强制模式（**无需重打镜像**）：

```bash
export KIMODO_NCCL_ENV_MODE=force-auto   # 用 sysfs 猜 RoCEv2 GID + rail HCA
# 或扫 index：export KIMODO_NCCL_ENV_MODE=force-gid=0   # 再试 1..7
unset NCCL_IB_DISABLE
export KIMODO_NCCL_DEBUG=1
```

完整粘贴模板见 `scripts/company_start_ib_trials.sh`。Pod 内也可先跑
`bash scripts/probe_rdma_gids.sh` 看 GID 表。同事 VeRL 0.8 镜像栈（CUDA13 +
`libibverbs*`/`rdma-core`）与当前 `:v8` 对齐；差异通常在平台注入的 GID/HCA，不在缺包。

首个 smoke 可临时打开：

```text
KIMODO_NCCL_DEBUG=1
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET
```

日志应明确出现 IB/verbs；如果只出现 Socket，机间通信大概率退化到了 TCP。`NCCL_IB_DISABLE=1`
仅作分界，不要当正式长训方案。

## 8. 推荐上线顺序

1. 在一个 H200 Pod 中执行 `nvidia-smi`、导入 torch、检查申请的 GPU 数全部可见；
2. 跨两台物理节点运行 NCCL tests（至少 all-reduce），分别验证 TCP 基线和 IB/RoCE 带宽；
3. 用 trainer `--preflight` 验证共享数据；
4. 两机 16 卡先跑 200 optimizer steps；
5. 检查日志中的 `world_size=16`、`effective_global_batch=2048`、每 Pod local rank 数符合资源申请；
6. kill 一个训练 Pod，验证控制器的 gang restart 行为和从共享 checkpoint 恢复；
7. 再提交正式新 output directory 的长训。

示例 smoke 参数：

```bash
KIMODO_RUN_DIR=/mnt/kimodo/runs/ddp16-smoke-001 \
KIMODO_AUTO_RESUME=0 \
/workspace/scripts/train_company.sh \
  --set runtime.max_steps_override=200 \
  --set runtime.checkpoint_every=100 \
  --set runtime.log_every=1
```

本 trainer 的精确 resume 要求训练关键配置、代码/data/stats fingerprint、world size 和 per-rank RNG 数量
一致。因此两卡 checkpoint 不能直接作为 16 卡的“精确续训”checkpoint；16 卡作业应 fresh start，或者
另行实现只加载模型权重、不恢复 optimizer/RNG 的 warm-start 语义。16 卡之间原地 resume 则保持
`world_size=16`。公司 launcher 默认从同一 `KIMODO_RUN_DIR/checkpoints/latest.txt` 自动恢复；最终
checkpoint 和完整 EMA bundle 都已存在时才直接成功退出，否则会恢复并补齐导出。控制器必须 gang
restart 全部 Pod，不能只留下部分旧 rank。

## 9. 常见卡住位置

| 现象 | 最先检查 |
|---|---|
| rank 数不足 | 所有 Pod 是否启动；`MASTER_ADDR/PORT` 是否一致；torchrun node rank 是否唯一 |
| `Address already in use` | rank 0 DNS 指错、同节点重复启动 launcher、多个作业共用了 hostNetwork 端口 |
| NCCL init hang | Pod 间 DNS/端口/NetworkPolicy、错误 NIC、RDMA device 未挂载、两节点 NCCL/driver 不一致 |
| 能跑但很慢 | NCCL 日志是否回落 Socket；数据盘 IOPS；16 rank 同时扫描 manifest；DataLoader worker 总数 |
| checkpoint 找不到 | 所有 Pod 是否挂载同一 PVC 和相同绝对路径；rank 0 是否有写权限 |
| 一启动主机内存暴涨 | 每 rank 都会解析大 manifest；先减少 `num_workers` 只影响 worker，再评估 manifest 索引格式 |
| OOM | `batch_size` 是每卡；改为 64/2 或 32/4，保持全局 batch 2048 |

特别注意当前约 140 万行 manifest：16 个 rank 会各自建立 dataset 索引并在启动时产生共享存储元数据压力。
这不影响 DDP 正确性，但可能成为 16 卡扩展后的主要启动/内存瓶颈。正式长训前应记录每 Pod RSS、manifest
加载时长、数据等待占比和存储 IOPS；若明显超标，再把 manifest 改成可 mmap 的二进制索引或做节点级
缓存，而不是先调 NCCL 参数。
