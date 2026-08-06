# 两机 16×H200 的容器化与多机 DDP 部署

本文对应本仓库当前实现，目标拓扑是两个 Kubernetes 训练 Pod、每个 Pod 看到 8 张 H200，合计
16 个训练进程。它不假定集群使用哪一种训练控制器；Kubeflow Trainer、旧 PyTorchJob、Volcano、
Kueue/JobSet 或公司自研平台都可以，只要满足本文的启动与网络契约。

## 1. 谁负责什么

```text
提交端/跳板机
  └─ 创建数据 Job 和训练 Job，本身不参加计算

共享持久化存储（PVC/NFS/Lustre/并行文件系统）
  ├─ prepared/：manifest、motion、text cache、stats（训练时只读）
  ├─ config/repro.paths.yaml（训练时只读）
  └─ runs/：日志、checkpoint、export（rank 0 可写）

训练 Pod 0（node rank 0）              训练 Pod 1（node rank 1）
  ├─ local rank 0 → GPU 0               ├─ local rank 0 → GPU 0
  ├─ ...                                ├─ ...
  └─ local rank 7 → GPU 7               └─ local rank 7 → GPU 7
             └──── NCCL：机内 NVLink/NVSwitch，机间 IB/RoCE ────┘
```

Kubernetes 负责 Pod、GPU、网络、DNS、存储和失败重启；`torchrun` 负责在每个节点生成 8 个进程并
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

镜像应包含代码、Python/CUDA/NCCL 运行时、依赖、训练 YAML 和启动脚本；不要包含数百 GB 数据、
Hugging Face token、运行日志或 checkpoint。当前 Dockerfile 已复制 `configs/`、`resources/`、
`scripts/`，默认命令为多机启动器。

```bash
docker build -t REGISTRY/kimodo-train:GIT_SHA .
docker push REGISTRY/kimodo-train:GIT_SHA
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

首次执行会初始化资源路径、下载并校验固定 revision 的资源、转换 motion、生成离线文本 embedding、
manifest、stats、inventory 和 `/mnt/kimodo/config/repro.paths.yaml`。如果已经把 train-ready prepared
bundle 搬到共享盘，设置 `KIMODO_PREPARED_ROOT=/mnt/source/prepared-bundle`，脚本只验证并绑定，不重跑
LLM2Vec。

实践中的作业顺序应是：

1. 数据准备 Job（一次，必要时用 1 张 GPU 生成文本 cache）；
2. 数据 preflight Job（不启动 16 卡，只验证一个真实 CPU batch）；
3. 两节点训练 Job；
4. 可选验证/导出 Job。

不要在两个训练 Pod 的 initContainer 中各跑一次完整 prepare。那会竞争共享目录、重复占 GPU，并让每次
训练重启都重新检查大数据。initContainer 只适合做“路径存在、容量足够、DNS 可解析”之类的轻检查。

## 5. 两个训练 Pod 必须获得的契约

两个 Pod 使用同一个镜像和相同命令：

```bash
/workspace/scripts/train_distributed.sh \
  --set runtime.output_dir=/mnt/kimodo/runs/experiment-001
```

公共环境：

```text
KIMODO_NNODES=2
KIMODO_NPROC_PER_NODE=8
KIMODO_PATHS_CONFIG=/mnt/kimodo/config/repro.paths.yaml
KIMODO_TRAINING_OVERLAY=/workspace/configs/overlays/two_node_16_h200_gb2048.yaml
MASTER_ADDR=<node-rank-0 的稳定 Pod DNS 或 IP>
MASTER_PORT=29500
```

每 Pod 唯一环境：

```text
Pod 0: KIMODO_NODE_RANK=0
Pod 1: KIMODO_NODE_RANK=1
```

每个 Pod 请求 `nvidia.com/gpu: 8`，并把同一个共享卷挂到相同的 `/mnt/kimodo`。如果平台的 PyTorch
runtime 已经替你执行 `torchrun` 并直接注入每个训练进程的 `RANK/WORLD_SIZE/LOCAL_RANK`，不要再次
套 `train_distributed.sh`；此时容器命令直接使用：

```bash
python -m kimodo.training.cli \
  --config /workspace/configs/training/kimodo_soma_seed_public.yaml \
  --paths /mnt/kimodo/config/repro.paths.yaml \
  --overlay /workspace/configs/overlays/two_node_16_h200_gb2048.yaml \
  --set runtime.output_dir=/mnt/kimodo/runs/experiment-001
```

这是上线前必须向平台管理员确认的第一件事：平台是“一 Pod 一节点，由镜像内部 torchrun”，还是“平台
已经 torchrun 到每个进程”。两层 torchrun 会启动错误数量的进程。

## 6. batch 怎么从两卡换算到 16 卡

本工程的有效全局 batch 为：

```text
world_size × per-rank batch_size × gradient_accumulation_steps
```

原两卡 overlay 是 `2 × 128 × 8 = 2048`；新 overlay 是 `16 × 128 × 1 = 2048`，所以无需再累计
8 次。若真实序列分布让单卡 batch 128 OOM，可以保持全局 2048 改成 `16 × 64 × 2` 或
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

不要在仓库中盲目硬编码 `NCCL_SOCKET_IFNAME=eth0`、`NCCL_IB_HCA=mlx5_0` 或
`NCCL_IB_GID_INDEX`；不同集群名称和 RoCE 配置不同。先让 NCCL 自动探测，首个 smoke job 临时设置：

```text
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET
```

日志应明确出现 IB/verbs 或平台的 NCCL network plugin；如果只出现 Socket，代码仍能跑，但机间通信
大概率退化到了 TCP。确认后将调试日志关闭，永久调优参数交给集群级 NCCL 配置或平台模板。

## 8. 推荐上线顺序

1. 在一个 H200 Pod 中执行 `nvidia-smi`、导入 torch、检查 8 卡可见；
2. 两 Pod 运行 NCCL tests（至少 all-reduce），分别验证 TCP 基线和 IB/RoCE 带宽；
3. 用 trainer `--preflight` 验证共享数据；
4. 两机 16 卡只跑 2–10 optimizer steps，`checkpoint_every=1`；
5. 检查日志中的 `world_size=16`、`effective_global_batch=2048`、每个节点 8 个 local rank；
6. kill 一个训练 Pod，验证控制器的 gang restart 行为和从共享 checkpoint 恢复；
7. 再提交正式新 output directory 的长训。

示例 smoke 参数：

```bash
/workspace/scripts/train_distributed.sh \
  --set runtime.output_dir=/mnt/kimodo/runs/ddp16-smoke-001 \
  --set runtime.max_steps_override=2 \
  --set runtime.checkpoint_every=1 \
  --set runtime.log_every=1
```

本 trainer 的精确 resume 要求训练关键配置、代码/data/stats fingerprint、world size 和 per-rank RNG 数量
一致。因此两卡 checkpoint 不能直接作为 16 卡的“精确续训”checkpoint；16 卡作业应 fresh start，或者
另行实现只加载模型权重、不恢复 optimizer/RNG 的 warm-start 语义。16 卡之间原地 resume 则保持
`world_size=16` 并显式指定同一 run 下的 checkpoint。

## 9. 常见卡住位置

| 现象 | 最先检查 |
|---|---|
| 只有 8 个 rank | 第二个 Pod 是否启动；两个 Pod 的 `MASTER_ADDR/PORT` 是否一致；node rank 是否唯一 |
| `Address already in use` | rank 0 DNS 指错、同节点重复启动 launcher、多个作业共用了 hostNetwork 端口 |
| NCCL init hang | Pod 间 DNS/端口/NetworkPolicy、错误 NIC、RDMA device 未挂载、两节点 NCCL/driver 不一致 |
| 能跑但很慢 | NCCL 日志是否回落 Socket；数据盘 IOPS；16 rank 同时扫描 manifest；DataLoader worker 总数 |
| checkpoint 找不到 | 两 Pod 是否挂载同一 PVC 和相同绝对路径；rank 0 是否有写权限 |
| 一启动主机内存暴涨 | 每 rank 都会解析大 manifest；先减少 `num_workers` 只影响 worker，再评估 manifest 索引格式 |
| OOM | `batch_size` 是每卡；改为 64/2 或 32/4，保持全局 batch 2048 |

特别注意当前约 140 万行 manifest：16 个 rank 会各自建立 dataset 索引并在启动时产生共享存储元数据压力。
这不影响 DDP 正确性，但可能成为 16 卡扩展后的主要启动/内存瓶颈。正式长训前应记录每 Pod RSS、manifest
加载时长、数据等待占比和存储 IOPS；若明显超标，再把 manifest 改成可 mmap 的二进制索引或做节点级
缓存，而不是先调 NCCL 参数。
