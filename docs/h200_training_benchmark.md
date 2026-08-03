# Kimodo H200 training benchmark

测量日期：2026-08-03。结论先行：论文最佳模型的 batch 是 2048，不是
4096；4096 是 LLM2Vec embedding 的宽度。两张 H200 上推荐
`local batch=128, accumulation=8`。正式 Phase 1 测量约 3.93 秒/optimizer-step；
它只比显存激进的 `256/4` 慢约 5%，但每卡峰值 reserved memory 从约 105 GiB
降至 56 GiB。把独立测得的两个 phase 各按 500k step 计入后，1M step 的理想线性
估算约 41.8 天。

## 论文口径与计步

论文第 4.3 节明确写明最佳模型使用 16 张 A100 80GB、global batch 2048；第 6.3
节的 scaling 设置为 4/8/16 GPUs 对应 512/1024/2048。训练配置里的
`runtime.batch_size` 是每 rank 的 micro-batch：

```text
effective_global_batch = world_size * local_micro_batch * accumulation
```

一个 optimizer-step 包含 `accumulation` 个 forward/backward micro-step。论文没有
讨论 gradient accumulation；本复现将论文的 500k+500k step 对齐为 optimizer
update，而不是 micro-step。

## 稳态结果

以下均为完整 282M/283M 两阶段 denoiser、300 帧、BF16、Adam-atan2、全部七项
loss、gradient clip、DDP 和 EMA 配置。每 rank 使用 16 个 DataLoader workers、
pin memory 和 prefetch 2。计时窗不含 checkpoint/export。正式结果会统一测量 10 个
optimizer-step，因此每个窗口都恰好包含一次 EMA 更新。fixture 足够大，计时窗不跨
epoch。下表是 Phase 1（dropout 0.1）速率。

| H200 | local B | accum | global B | 秒/optimizer-step | samples/s | 每卡峰值 reserved | 1M steps 理想线性估算 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 32 | 32 | 2048 | 4.612 | 444.0 | 20.5 GiB | 53.4 天 |
| 2 | 64 | 16 | 2048 | 4.153 | 493.2 | 31.6 GiB | 48.1 天 |
| 2 | 128 | 8 | 2048 | 3.928 | 521.4 | 55.8 GiB | 45.5 天 |
| 2 | 256 | 4 | 2048 | 3.740 | 547.7 | 105.0 GiB | 43.3 天 |
| 1 | 256 | 8 | 2048 | 7.545 | 271.4 | 104.2 GiB | 87.3 天 |
| 2 | 256 | 8 | 4096 | 7.551 | 542.5 | 105.0 GiB | 87.4 天 |

两卡相对单卡、同 global batch 2048 的实测加速约 1.99 倍。global batch 从 2048
加到 4096 时，每个 optimizer-step 的样本数翻倍，时间也近似翻倍，吞吐保持不变。

Phase 2 的 constraint curriculum 与 dropout=0 也做了相同口径的独立稳态测量：

| H200 | local B | accum | global B | 秒/optimizer-step | samples/s | 每卡峰值 reserved |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 128 | 8 | 2048 | 3.288 | 622.8 | 48.0 GiB |
| 2 | 256 | 4 | 2048 | 3.374 | 607.0 | 88.7 GiB |

按 500k Phase 1 + 500k Phase 2 线性相加，B=128/A=8 约 41.8 天，B=256/A=4
约 41.2 天；后者只节省约 1.4% 总时间，却显著压缩共享节点上的显存余量。

## 显存边界

单卡、300 帧、A=1 的独立进程扫描中，local B=320 可以运行但 peak reserved
约 127.4/139.8 GiB，超过 85% production-safe 线；B=384 明确 OOM。B=256 的
峰值约 103--105 GiB，因此是容量意义上的 safe 上限，不是默认推荐值。

LLM2Vec 与 Qwen3-32B 均为离线预处理工具，不进入训练进程。训练只读取缓存的
`[1,4096]` float32 embedding，B=256 时约 4 MiB/卡，已包含在上述峰值内。

## 已修复的数据瓶颈

通用 NPZ loader 原先会在随机 crop 前对整条 canonical motion 运行一次
`complete_motion_dict`，包括 FK、速度/接触和 SciPy multigrid ADMM root smoother；
dataset 随后只保留 local rotations/root positions，并在 crop 后把整套计算再做一次。
`num_workers=0` 时这个重复路径曾把 E=256 拖到约 13.3 秒/step。

训练 loader 现在只对已经验证的同 FPS Kimodo NPZ 使用 raw fast path，直接读取
`local_rot_mats` 和 `root_positions`，crop 后仍按原顺序执行唯一一次 representation
构造、origin translation、随机 heading 和 normalization。这个优化不改变训练张量
或随机数顺序。扩大 fixture 并做等量 micro-step 稳态复测后，正式 Phase 1 的
两卡 B=128/A=8 为 3.928 秒/optimizer-step、521.4 samples/s；旧的
`0.524 秒/step` 是短 fixture、不同累计窗口的早期诊断数，不能与 optimizer-step
口径比较，已从结论中撤除。

本表的 motion 数值为固定形状合成数据，文件已在 page cache；它准确覆盖模型、
CPU representation 和 DDP 路径，但真实多文件 manifest 的冷 I/O、变长 padding、
checkpoint 和共享服务器干扰仍需在 text cache/stats 完成后再做短实训复核。物理 0
号 H200 测量时另有其他用户约 4.5 GiB 常驻进程，因此默认选择 56 GiB 的 B=128
方案，保留更充足的租户安全余量。
