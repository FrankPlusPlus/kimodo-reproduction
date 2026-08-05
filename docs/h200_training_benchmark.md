# Kimodo H200 training benchmark

## 2026-08-04 修复后短测

在第一轮 provenance/cache/run-lock 修复后的工作树上，用物理 GPU `0,2` 再次执行
`local micro-batch=256`、`gradient accumulation=4`、两 ranks，因此 effective global batch
为 2048。BF16、完整两阶段 283M denoiser、七项 loss、Adam-atan2 和 DDP 训练数学均保持不变。

- 真实 adopted BONES-SEED manifest（1,407,184 rows）完成 3 个 Phase-1 optimizer steps；loss
  全部有限。第 1 步含 worker 启动与首次 I/O，为 22.387 s；第 2、3 步增量分别为 3.702 s、
  3.768 s，即约 553.2、543.5 samples/s。这里只能证明短训可运行并给出现场速度，不能推断收敛。
- 固定 300-frame benchmark 用 1 warmup + 2 measured steps 测得 3.712 s/optimizer-step、
  551.7 samples/s；每卡 peak allocated 104.4 GiB、peak reserved 104.9 GiB。
- 物理 0 号卡当时另有其他用户约 4.5 GiB 常驻、采样时 SM 利用率为 0；所以这是共享卡现场短测，
  不是独占节点正式 benchmark。原始机器可读结果为
  `outputs/benchmarks/results/audit-20260804-phase1-b256-a4-3step.json`，真实训练输出为
  `outputs/runs/audit-20260804-b256-a4-real-3step/`。

短窗口没有覆盖 EMA 的第 10-step 更新，也刻意不把 checkpoint/export I/O 算入 benchmark
计时；真实 3-step run 在结束后写出了 full-state 格式 checkpoint 和 inference export，
但本次没有从该 checkpoint 续训，且 `ema.num_updates=0`，因此 export 只证明结构可加载，
不能代表已吸收这 3 步更新的 EMA 模型质量。

短训结束后又修正了 stats producer 的传递依赖闭包和 run-lock 清理竞态；这些修改不改变
模型、loss、optimizer 或训练 engine，且已通过完整回归，但没有用最终工作树重新跑这次
真实 3-step。因此该结果是训练数学链路证据，不是最终工作树逐文件 provenance 全等证明。

## 2026-08-03 较早的正式测量

下节测量日期为 2026-08-03。结论先行：论文最佳模型的 batch 是 2048，不是
4096；4096 是 LLM2Vec embedding 的宽度。两张 H200 上推荐
`local batch=128, accumulation=8`。最终 detached-bridge 配置的正式 Phase 1 测量约
3.92 秒/optimizer-step；
它只比显存激进的 `256/4` 慢约 5%，但每卡峰值 reserved memory 从约 105 GiB
降至 56 GiB。把独立测得的两个 phase 各按 500k step 计入后，1M step 的理想线性
估算约 41.4 天。

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
epoch。最终表来自 clean commit `4089fb46393ec7c9206c3f111f899f4f9fba44c3`，原始
JSON 为忽略提交的 `outputs/benchmarks/results/formal-4089fb4-*.json`。下表是
Phase 1（dropout 0.1）速率。

| H200 | local B | accum | global B | 秒/optimizer-step | samples/s | 每卡峰值 reserved | 1M steps 理想线性估算 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 32 | 32 | 2048 | 4.559 | 449.3 | 19.7 GiB | 52.8 天 |
| 2 | 64 | 16 | 2048 | 4.117 | 497.5 | 31.6 GiB | 47.7 天 |
| 2 | 128 | 8 | 2048 | 3.917 | 522.9 | 56.0 GiB | 45.3 天 |
| 2 | 256 | 4 | 2048 | 3.729 | 549.2 | 105.0 GiB | 43.2 天 |
| 1 | 256 | 8 | 2048 | 7.545 | 271.4 | 104.2 GiB | 87.3 天 |
| 2 | 256 | 8 | 4096 | 7.551 | 542.5 | 105.0 GiB | 87.4 天 |

两卡相对单卡、同 global batch 2048 的实测加速约 1.99 倍。global batch 从 2048
加到 4096 时，每个 optimizer-step 的样本数翻倍，时间也近似翻倍，吞吐保持不变。

Phase 2 的 constraint curriculum 与 dropout=0 也做了相同口径的独立稳态测量：

| H200 | local B | accum | global B | 秒/optimizer-step | samples/s | 每卡峰值 reserved |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 128 | 8 | 2048 | 3.234 | 633.3 | 48.0 GiB |
| 2 | 256 | 4 | 2048 | 3.316 | 617.7 | 89.1 GiB |

按 500k Phase 1 + 500k Phase 2 线性相加，B=128/A=8 约 41.4 天，B=256/A=4
约 40.8 天；后者只节省约 1.5% 总时间，却显著压缩共享节点上的显存余量。

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
两卡 B=128/A=8 为 3.917 秒/optimizer-step、522.9 samples/s；旧的
`0.524 秒/step` 是短 fixture、不同累计窗口的早期诊断数，不能与 optimizer-step
口径比较，已从结论中撤除。

本表的 motion 数值为固定形状合成数据，文件已在 page cache；它准确覆盖模型、
CPU representation 和 DDP 路径，但真实多文件 manifest 的冷 I/O、变长 padding、
checkpoint 和共享服务器干扰仍需在 text cache/stats 完成后再做短实训复核。物理 0
号 H200 测量时另有其他用户约 4.5 GiB 常驻进程，因此默认选择 56 GiB 的 B=128
方案，保留更充足的租户安全余量。
