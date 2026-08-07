# 训练期间的公开 Benchmark 监控

训练进程只负责每 10 step 写健康指标、每 10k step 保存 trainer checkpoint、每
100k step 原子发布 EMA inference bundle。真实生成评测由独立的单 GPU eval Pod
监听共享 PV 完成，不导入活跃 trainer，也不参与 DDP collective，因此不会改变
loss、梯度、随机数或训练吞吐。

## 前置资产

准备一个固定、分层、不可变的公开 benchmark proxy。它应同时覆盖 content 与
repetition，以及 overview、timeline single、timeline multi、constraints with text、
constraints without text；约束样本至少覆盖 root、end-effector、full-body 和 mixed。
每个样本目录必须已有 `meta.json`、`gt_motion.npz`，有约束时还需
`constraints.json`。完整公开套件适合 500k/1M 节点，不适合每 100k 高频运行。

## 先生成官方基线

在独立单 GPU Pod 中，对完全相同的 proxy 只运行一次 released
`Kimodo-SOMA-SEED-v1.1`：

```bash
KIMODO_BENCHMARK_ROOT=/mnt/kimodo/eval/benchmark/proxy \
KIMODO_OFFICIAL_EVAL_ROOT=/mnt/kimodo/eval/official-seed-v1.1 \
/workspace/scripts/eval_official_baseline.sh
```

严格可比模式保持 `batch_size=1`、DDIM 100、同一 text encoder/TMR 精度且不启用
postprocess。官方模型只校准评测资产和目标线，不会进入或污染训练。

## 启动旁路监控

使用与训练相同的镜像，但覆盖默认命令，在一个独立单 GPU eval Pod 中运行：

```bash
KIMODO_RUN_DIR=/mnt/kimodo/runs/v2-1m-production \
KIMODO_BENCHMARK_ROOT=/mnt/kimodo/eval/benchmark/proxy \
KIMODO_EVAL_ROOT=/mnt/kimodo/eval/v2-1m \
KIMODO_OFFICIAL_BASELINE_SUMMARY=/mnt/kimodo/eval/official-seed-v1.1/summary_rows.json \
/workspace/scripts/eval_company_watcher.sh
```

`KIMODO_EVAL_ROOT` 应位于独立 eval PVC 或节点本地 NVMe；benchmark proxy 也应在 eval
侧只读挂载。监控器只从训练 PV 顺序读取一次约 1 GB 的 EMA bundle，避免生成结果和
TMR 中间文件与 checkpoint 写入争抢同一个 PV。proxy 的完整内容 hash 只在首次运行
计算，之后用文件清单/大小/mtime 快速校验是否发生变化。

监控器只消费完整的 `exports/step-NNNNNNNNN/`，在隔离 `.building` 目录中依次执行
生成、TMR embedding、官方 metrics 和汇总；全部成功后才原子发布
`step-NNNNNNNNN/complete.json`。失败记录在 `failed-step-*`，不会停止训练。
便于 dashboard 或人工轮询的扁平指标写入 `history.jsonl`，最近一次状态原子更新到
`latest.json`。

## 观察与告警

重点分别观察两个 split 的 R@3、FID、foot skate、contact consistency，以及
full-body、end-effector、root mean/p95。`complete.json` 同时记录模型权重 hash、
proxy 资产 hash、生成协议和相对官方基线的逐指标差值。

默认趋势告警要求同一指标连续两个 100k 区间显著恶化，避免单次生成方差触发误停：
R@3 每次下降超过 2 个百分点、contact 每次下降超过 0.005，位置/滑步每次恶化超过
`max(10%, 0.5 cm 或 cm/s)`。告警只建议人工检查，不自动杀死训练。训练 loss 继续下降
而 content/repetition benchmark 同时连续变差，是目标偏移；content 下降而 repetition
稳定或上升，是过拟合信号；Phase 2 约束改善但文本或运动质量恶化，是多目标失衡。

`--paper-protocol` 的完整集合 retrieval/FID 与旧版 group 内聚合是两套口径，结果会
分别记录，不能合并成同一条曲线。V1/V2 的训练 manifest 不同，但导出的 SOMA bundle
接口相同，因此可共用同一个 proxy 和监控器。

## W&B 统一监控

W&B 在没有 Key 时默认关闭；注入 `WANDB_API_KEY` 后自动启用。只有训练 global rank 0 建立 `train` run，独立 eval Pod 建立
`benchmark` run。两个 run 使用同一 project/group，并以训练 `global_step` 对齐曲线。
原有 `train.jsonl`、`history.jsonl`、Pod 日志和 checkpoint 不依赖 W&B，网络故障默认只会
停用远端上报，不会停止训练。若要求 W&B 不可用时必须终止任务，可额外设置
`KIMODO_WANDB_REQUIRED=1`。

在线模式下，训练和 eval Pod 最少只需注入同一个 Secret：

```text
WANDB_API_KEY=<由平台 Secret 注入，不写入镜像或启动脚本>
```

默认 project 是 `kimodo-reproduction`；group 根据 `KIMODO_RUN_DIR` 自动生成，训练和
benchmark 的 run ID/name 也会分别稳定生成为 `...-train` 和 `...-benchmark`，Pod 重启后自动
续接。因此下面这些都只是可选覆盖项：

```text
WANDB_PROJECT=kimodo-production
WANDB_ENTITY=<团队或组织名>
KIMODO_WANDB_GROUP=v2-1m-20260807
KIMODO_WANDB_TRAIN_RUN_ID=v2-1m-20260807-train
KIMODO_WANDB_TRAIN_RUN_NAME=v2-1m-train
KIMODO_WANDB_BENCHMARK_RUN_ID=v2-1m-20260807-benchmark
KIMODO_WANDB_BENCHMARK_RUN_NAME=v2-1m-benchmark
```

训练 run 上报 loss、加权 loss、学习率、梯度、吞吐、显存、conditioning/data 分布、
checkpoint 与 EMA export 事件；benchmark run 上报扁平化的公开指标、告警数、评测协议和
成功/失败状态。默认不上传约 1 GB 的模型/checkpoint artifact，避免阻塞训练和占用 W&B 存储。
无外网环境只设置 `WANDB_MODE=offline` 也会启用本地记录，之后从持久卷中的 `.wandb` 目录同步。
