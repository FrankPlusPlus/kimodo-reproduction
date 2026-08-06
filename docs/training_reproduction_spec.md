# Kimodo 训练复现规格

> 版本：Mission v1，2026-08-02  
> 目标：把论文与公开推理代码中可验证的信息整理成可直接编码的训练合同；凡论文/代码未披露的值，均明确标为建议默认或未知。  
> 论文来源：`/Users/frank/Downloads/kimodo.pdf`（下文页码为 PDF 印刷页码）。  
> 代码来源：本仓库当前工作副本。公开仓库没有训练入口、训练 Dataset、loss、optimizer、EMA 或 checkpoint-save 实现；本规格因此不是“官方训练代码转录”。

## 0. 结论与复现边界

### 0.1 可以复现什么

1. **架构兼容复现**：可以。公开代码完整给出了 motion representation、constraint imputation、root/body denoiser、cosine diffusion、DDIM 和 separated CFG；公开 checkpoint 的 `config.yaml` 给出了发布模型的结构参数。
2. **工程训练复现**：可以。在 BONES-SEED 与官方公开 benchmark split 上，可按本文建议默认实现一个可训练、可恢复、可评测的 Kimodo-compatible 系统。
3. **论文 RP 模型数值复现**：目前不可以保证。Bones Rigplay 全量训练数据是专有数据；训练 sampler、loss reduction/计算域、Adam-atan2 完整超参、精度、seed、scheduler、checkpoint 选择规则和增强混合比例均未公开。

### 0.2 “后训练”的准确界定

Kimodo 主 denoiser 明确采用连续的两阶段课程：前 500k step 是 text-to-motion pre-training，后 500k step 在同一模型上继续做 constraint curriculum training（论文 Sec. 4.3 p.11、Sec. 6.2 p.14）。广义上 Phase 2 是后训练；它不是 RLHF/DPO、ControlNet、LoRA 或 reward-model 后训练。Phase 2 沿用同一 `x0` 重建目标，没有论文证据表明加入新 reward 或独立 constraint loss。

### 0.3 Provenance 标签

| 标签 | 含义 | 实现约束 |
|---|---|---|
| **[PAPER]** | 论文明确陈述、公式、表或图 | 可称“官方论文值” |
| **[CODE]** | 当前公开代码可直接证明 | 可称“公开实现行为”；不自动等同训练时行为 |
| **[ARTIFACT]** | NVIDIA 发布 checkpoint/config/model card 或公开 benchmark 文档 | 可称“发布模型配置/公开评测约定” |
| **[INFERRED]** | 由多个事实推断，但未被作者明确承诺 | 必须可配置，不得硬编码成官方值 |
| **[DEFAULT]** | 本规格为补齐缺口提出的领域默认 | 必须在配置和日志里显示 `reproduction_default` |
| **[UNKNOWN]** | 当前材料无法确定 | 需要实验、作者确认或新材料 |

公开 artifact 锚点：发布模型结构参数以 NVIDIA 的
[`Kimodo-SOMA-RP-v1.1/config.yaml`](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1/blob/6c9233af1180b8151e3c4703477104af5dce9dd5/config.yaml)
固定 revision 为准；模型族和数据说明以
[`Kimodo-SOMA-RP-v1.1` model card](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1)
为准。训练实现开始前应把这些小型元数据下载到 experiment manifest 并记录 SHA-256，避免 `main` 漂移。

## 1. 一页训练合同

以下是实现者应先落地的最小主路径。所有带 `DEFAULT` 的项都必须可被配置覆盖。

```yaml
target: open_seed_engineering_reproduction
data:
  dataset: BONES-SEED                    # DEFAULT；RP 需要合法数据授权
  split_source: kimodo_benchmark_splits # ARTIFACT
  fps: 30                               # PAPER / released-model ARTIFACT
  max_seconds: 10                       # PAPER
  min_seconds: 1                        # DEFAULT
  motion_source_repeats:                # IMPLEMENTED BASELINE；不是官方比例
    full_clip: 1
    atomic_single: 1
    atomic_combined: 1
    stitched: 0                         # BLOCKED：缺 transition model/asset
  text_sources:
    original: enabled
    qwen_paraphrase: blocked             # BLOCKED：缺官方 prompt/revision/cache
  normalize_from_train_split_only: true # DEFAULT + leakage prevention
model:
  skeleton: SOMA30
  motion_dim: 369
  diffusion_steps: 1000
  prediction: x0
  beta_schedule: cosine_0.008_max_0.999
  denoiser:
    stages: [global_root, local_root_conditioned_body]
    latent_dim: 1024
    ff_size: 2048
    layers_per_stage: 16
    heads: 8
    activation: gelu
    norm_first: false
    text_slots: 50
    text_dim: 4096
    input_first_heading: true
    constraint_mode: concat_mask
text:
  encoder: McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp
  peft: McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised
  frozen: true
  cache_dtype: float32
loss:
  smooth_l1_beta: 1.0                   # DEFAULT；论文只说 smooth L1
  component_weights:
    root_position: 10.0
    root_heading: 2.0
    joint_position: 10.0
    joint_velocity: 3.0
    joint_rotation_6d: 10.0
    foot_contact: 4.0
    fk_joint_position: 5.0
curriculum:
  phase1_steps: 500000
  phase2_steps: 500000
  text_drop_probability: 0.10
  phase1_transformer_dropout: 0.10
  phase1_pe_dropout: 0.10              # DEFAULT：论文未区分两种 dropout
  phase2_transformer_dropout: 0.00
  phase2_pe_dropout: 0.00
  phase2_no_constraint_probability: 0.10
  phase2_two_pattern_probability: 0.25
  phase2_sparse_keyframes_max: [1, 20]
optimizer:
  name: AdamAtan2
  learning_rate: 2.0e-5
  betas: [0.9, 0.999]                   # DEFAULT
  weight_decay: 0.0                    # DEFAULT
  schedule: constant                   # DEFAULT；论文未报 scheduler
  grad_clip_norm: 1.0                  # DEFAULT；稳定性保护
distributed:
  global_batch_size: 2048
  world_size: 16
  local_batch_size: 128                 # INFERRED：若无 grad accumulation
  precision: bf16_mixed                # DEFAULT
ema:
  update_every_steps: 10
  decay: 0.995
checkpoint:
  save_every_steps: 10000               # DEFAULT
  save_phase_boundary: true
  inference_weights: ema
eval:
  ddim_steps: 100
  cfg_text: 2.0
  cfg_constraint: 2.0
  postprocess: false
```

## 2. 关键参数 provenance 总表

| 训练关键项 | 值/行为 | Provenance | 证据与实现说明 |
|---|---:|---|---|
| RP 数据规模 | 700 h、170 人 | PAPER | Sec. 3 p.6 |
| 论文 train/test split | 按 unique behavior 留出 10%；约 5k 测试 motion | PAPER | Sec. 6.1 p.13 |
| 发布模型帧率 | 30 fps | PAPER + ARTIFACT | Sec. 4.3 p.10-11；公开 `config.yaml` |
| 论文消融帧率 | 20 fps | PAPER | Sec. 6.1 p.13、Table 1 p.14；不可与发布模型混用 |
| 最大训练长度 | 10 s | PAPER | Sec. 2.1 p.5、Sec. 4.3 p.10 |
| 最小长度/长度分布 | 未披露 | UNKNOWN | 建议 1-10 s，按可用 clip 随机 crop |
| motion rep | `[r_p,r_a,j_p,j_v,j_a,f]` 语义 | PAPER | Sec. 4.1 p.8 |
| 公开代码实际 feature 顺序 | root pos、heading、joint pos、**rotation 6D、velocity**、contact | CODE | `kimodo/motion_rep/reps/kimodo_motionrep.py:34-41`；顺序与论文列举不同，checkpoint 兼容实现必须跟代码 |
| root smoothing | XZ 平滑、Y 保留 | PAPER + CODE | Sec. 4.1 p.8；`smooth_root.py:201-234` |
| smoothing margin | 0.06 m | CODE | `smooth_root.py:215-221`；论文未给数值 |
| foot contact 标签 | 4 个，速度 <0.15 m/s 且高度 <0.10 m | CODE | `kimodo_motionrep.py:90-92`、`feet.py:11-60` |
| normalize | `(x-mean)/sqrt(std^2+1e-5)`，global/local/body split stats | CODE | `stats.py:15-77`、`reps/base.py:72-83` |
| heading augmentation | frame-0 heading 随机 | PAPER + CODE | Sec. 4.2 p.10、Sec. 4.3 p.10-11；`reps/base.py:193-204` |
| text encoder | LLM2Vec，4096D | PAPER + CODE | Sec. 4.2 p.9；`load_model.py:22-33` |
| text encoder frozen | `eval` + `requires_grad=False` + `no_grad` | CODE | `llm2vec_wrapper.py:49-52,65-82` |
| 发布模型 text slots | 1 embedding + 49 zero tokens = 50 slots | PAPER + ARTIFACT + CODE | Sec. 4.2 p.9；公开 config `num_text_tokens_override: 50`；`backbone.py:110-114,159-170` |
| 训练 text embedding 精度 | float32 | ARTIFACT | `docs/source/benchmark/results.md:7` 明确 v1.1 Kimodo/TMR 训练使用 float32 embedding |
| diffusion target | clean motion `x0` | PAPER | Sec. 4 Background p.7、Sec. 4.3 Eq. (1) p.10 |
| base diffusion steps | 1000 | PAPER + ARTIFACT | Sec. 4.3 p.10；发布 config |
| beta schedule | cosine，offset .008，max beta .999 | CODE | `model/diffusion.py:12-26`；论文只给 DDPM/T=1000 |
| constraint conditioning | overwrite noisy features，再 concat binary mask | PAPER + CODE | Sec. 4.2 p.9、Fig. 9；`twostage_denoiser.py:98-105` |
| root/body architecture | 两个 transformer；root global，body 接 local-root | PAPER + CODE | Sec. 4.2 p.9-10；`twostage_denoiser.py:36-61,107-152` |
| 每 stage 层/头/宽度 | 16 / 8 / 1024；总 282M | PAPER | Sec. 4.2 p.10 |
| FFN/activation/norm | 2048 / GELU / post-norm | ARTIFACT + CODE | 发布 config；`backbone.py:121-134` |
| root-to-body 梯度 | 生产 profile detach；不 detach 仅作消融 | CODE + PAPER-AMBIGUOUS | 公开 denoiser 的 training-mode branch 对 conversion 使用 `no_grad`/`detach`；Sec. 5 p.12 的 “trains end-to-end” 未说明 body loss 是否必须跨 bridge 反传，也不能证明私有 trainer 的 autograd 行为 |
| loss 权重 | `[10,2,10,3,10,4,5]` | PAPER | Eq. (1) 及其后文字，p.10 |
| loss reduction/物理或 normalized 域 | 未披露 | UNKNOWN | 必须做配置和消融，见 6.3 |
| optimizer | Adam-atan2，lr `2e-5` | PAPER | Sec. 4.3 p.10 |
| betas/weight decay/scheduler/warmup | 未披露 | UNKNOWN | 本规格建议 `.9/.999`、0、constant、无 warmup |
| 总训练 | 2×500k = 1M step | PAPER | Sec. 4.3 p.11 |
| Phase 1 | text-only、无 constraints | PAPER | Sec. 4.3 p.11 |
| Phase 2 | text + sampled kinematic constraints | PAPER | Sec. 4.3 p.11 |
| 两 pattern 混合 | 25% | PAPER | Sec. 4.3 p.11 |
| 无 constraint | Phase 2 中 10% | PAPER | Sec. 4.3 p.11 |
| sparse keyframe curriculum | 最大 keyframe 数 1→20 线性增长，并偏向少 keyframe | PAPER | Sec. 4.3 p.11；具体分布 UNKNOWN |
| model dropout | Phase 1 0.1；Phase 2 0 | PAPER + UNKNOWN | Sec. 4.3 p.11；代码把 transformer/PE dropout 分成两个参数，论文未说明 0.1 是否同时作用二者；默认同时设置 |
| text dropout | 两阶段均 10% | PAPER | Sec. 4.3 p.11 |
| EMA | 每 10 step，decay .995，推理用 EMA | PAPER | Sec. 4.3 p.11 |
| global batch | 2048 / 16×A100 80GB | PAPER | Sec. 4.3 p.10-11 |
| precision/accumulation/clip | 未披露 | UNKNOWN | 建议 bf16、按需 accumulation、norm 1.0 |
| inference | DDIM 100；separated CFG 2/2 | PAPER + CODE | Sec. 4.4 p.11；`cfg.py:94-129`、`kimodo_model.py:607-633` |
| 论文公平评测后处理 | 关闭 | PAPER | Sec. 4.4 p.11 |
| paper/public constraint metric 一致性 | 不完全一致 | CODE + ARTIFACT | 当前 `ContraintFollow` 只算 EE position，未算 rotation；root2d 直接比较 posed pelvis 与 smooth-root target，见 `metrics/constraints.py:58-78`；公开结果 EE rotation 为 `-` |
| checkpoint 选择/保存频率 | 未披露 | UNKNOWN | 建议最后一个 EMA 为主结果，并保留固定间隔快照 |

## 3. 数据与预处理

### 3.1 数据版本和 split

**官方论文事实**：Bones Rigplay 含 700 小时 optical mocap、170 名参与者；每个 clip 有 overview description，并切成带 fine-grained description 的 atomic sub-clips（Sec. 3 p.6）。论文量化使用 native 27-joint skeleton；SOMA、Unitree G1、SMPL-X 是将数据 retarget 后分别训练的变体（Sec. 3 p.6）。

**公开工程事实**：公开文档说明每种 skeleton 单独训练；SOMA 内部训练/预测使用统一比例的 `somaskel30`，对外转为 77 joints（`docs/source/key_concepts/skeleton.md:3-20`）。公开 benchmark 给出 BONES-SEED 的 `train_split_paths.txt`、content test、repetition test（`docs/source/benchmark/introduction.md:13-22`）。

**实现合同**：

- `target=paper_rp`：必须提供合法 Rigplay 数据及对应 split manifest；没有时立即失败，不得默默换成 SEED。
- `target=open_seed_engineering_reproduction`：使用公开 benchmark split；绝不以 test clips 计算 normalization、训练 TMR、做 early stopping 或生成增强。
- 每次运行记录 dataset 许可证、文件清单 SHA-256、split 文件 SHA-256、retarget 版本和 skeleton rest-pose hash。
- 为每个 skeleton 训练独立 checkpoint；不得假定 SOMA→G1/SMPL-X warm-start，论文与代码均未公开此做法。

### 3.2 原始 motion 到统一 skeleton

输入最少包含：local joint rotations `[T,J,3,3]`、root translation `[T,3]`、fps、文本和 clip/event 边界。预处理顺序：

1. 按目标 skeleton 的 rest orientation 转换 local rotations；用 FK 得到 global rotations/positions。
2. 重采样到 30 fps；旋转用 quaternion slerp，translation 用线性/三次插值。**[DEFAULT]** 论文未披露重采样滤波器。
3. 将 frame 0 的 smoothed root XZ 平移到 `(0,0)`；Y 保留绝对高度。论文 Sec. 4.3 p.10 说 root position 置于首帧原点，公开约束文档明确 XZ 规范（`docs/source/user_guide/constraints.md:21-36`）。
4. 不把首帧 heading 固定为零；训练时额外采样目标 heading `U[0,2π)` 并整体旋转。代码实现见 `reps/base.py:193-204`。
5. 生成 Sec. 4.1/4.2 所需 feature，最后用训练 split stats normalize。

### 3.3 Motion representation 的精确实现

公开代码定义以下顺序（`kimodo_motionrep.py:34-48`）：

1. `smooth_root_pos`: 3，全球坐标；XZ 经 smoothing，Y 保留。
2. `global_root_heading`: 2，`[cos ψ, sin ψ]`。
3. `local_joints_positions`: `3J`；XZ 相对 smoothed root，Y 为全局高度。代码通过 `hips_offset` 构造（`kimodo_motionrep.py:76-89`）。
4. `global_rot_data`: `6J`，global joint rotation 的 continuous 6D。
5. `velocities`: `3J`，global joint position 的每秒速度；末帧复制前一速度（`feature_utils.py:38-72`）。
6. `foot_contacts`: 4，左 heel/toe、右 heel/toe。

总维度 `D = 3+2+3J+6J+3J+4 = 12J+9`：SOMA30 `D=369`，G1-34 `D=417`，SMPLX-22 `D=273`。global root 为前 5 维；body 为 `D-5`。local root 为 `[heading angular velocity(1), planar velocity(2), absolute Y(1)]`，共 4 维（`reps/base.py:113-157`）。

平滑器公开代码使用 ADMM/multigrid，XZ margin 固定 `0.06 m`，Y 不变（`smooth_root.py:142-234`）。精确复现 checkpoint-compatible feature 时，应直接复用该实现；若批量离线预处理，则必须做数值回归测试，容差 `1e-5`。

Foot contact 标签使用速度与高度联合阈值：`speed < 0.15 m/s && y < 0.10 m`（`kimodo_motionrep.py:90-92`、`feet.py:38-59`）。

### 3.4 Normalization

公开代码要求 `stats/motion/{global_root,local_root,body}/{mean,std}.npy`，并将 global-root/body 拼成模型输入 stats（`reps/base.py:19-33,72-83`）。公式为：

`x_norm = (x - mean) / sqrt(std^2 + 1e-5)`（`stats.py:15-20,65-77`）。

**[DEFAULT]**：stats 仅从 train split 的所有有效帧估计，使用 float64 accumulator，落盘 float32；contact 也按同式归一化。保存 frame count、clip count、每块 mean/std checksum。local-root stats 必须由同一批 global-root features 经公开 `global_root_to_local_root` 产生，不能从别的 corpus 借用。

### 3.5 文本和 motion augmentation

**[PAPER] Sec. 3 p.7**：

- Qwen3-32B 把描述 paraphrase 为统一的 `A [subject]...` 结构并生成不同细节层级。
- 随机拼接成对 motion clips；使用一个在 non-augmented dataset 上训练的 diffusion model 生成短 transition。
- 最终训练随机混合 full clips、single/combined atomic sub-clips、stitched clips、original descriptions 和 LLM paraphrases。

**[UNKNOWN]**：两轴混合比例、atomic 合并长度、transition 长度、Qwen prompt/temperature/seed、preliminary transition model 是否初始化最终模型、增强数据去重规则均未披露。

**当前已实现的工程 baseline**：

- manifest 构建器对 full、single atomic、相邻 combined atomic 使用显式 repeat count；当前默认都是 `1`，不是官方混合比例。
- combined atomic 当前只组合两个相邻且来自同一 motion 的 events；不是此前设计草案中的 2-4 个随机连续 events。
- 当前没有生成 stitched transition，也没有生成 Qwen3-32B paraphrase。普通 baseline 因而只有原始文本；严格 paper-data profile 会 fail closed。
- 实验必须标为 `engineering reconstruction / no_stitched_augmentation / no_qwen_paraphrase`，不可称论文数据方法完整复现。

**尚未落地的候选方案**：full/single/combined/stitched=`0.30/0.30/0.20/0.20`、original/paraphrase=`0.50/0.50`、combined 2-4 events、transition 5-15 frames。这些只是待消融的设计候选，不是当前配置，不是代码默认，也不是论文官方值。未来若实现，所有 LLM paraphrase 必须离线生成、版本化并经过人工/规则抽检；不得在训练 loop 内请求 LLM。

### 3.6 Variable-length batch

论文只说每 batch 含 variable-length sequences 且 loss 相应 mask（Sec. 4.3 p.10）。

**[DEFAULT]**：先选 motion source，再从合法 event/clip 中 crop 1-10 s；保留原事件边界，按 batch 内最长序列右 padding。`x_pad_mask=True` 表示有效帧，公开 backbone 最终取反传给 PyTorch padding mask（`backbone.py:136-150,213-229`）。所有 loss 的 numerator 与 denominator 都排除 padding；velocity/FK 也不得跨 padding 边界。

## 4. 文本 embedding

### 4.1 Encoder 身份与冻结

论文使用 LLM2Vec 4096D（Sec. 4.2 p.9）。公开 loader 指定：

- base: `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`
- PEFT: `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised`
- output dim 4096

证据：`kimodo/model/load_model.py:22-33`。公开 wrapper 将 encoder 设为 eval、冻结全部参数，并在 `no_grad` 下编码（`llm2vec_wrapper.py:49-52,65-82`）。因此训练实现必须把 text encoder 当作固定特征提取器，不把它加入 optimizer。

### 4.2 单向量 + 49 extra tokens

公开 wrapper mean-pools 为每条文本一个向量并返回 `[B,1,4096]`（`llm2vec_wrapper.py:84-94`）。发布 config 的 `num_text_tokens_override=50` 使 backbone padding 到 50 slots；论文 Sec. 4.2 p.9 对应一个 `ctext` 加 `P=49` 个 all-zero extra tokens。`use_text_mask=false` 会让全部 50 slots 可参与 attention（`backbone.py:159-186`）。zero token 先经过带 bias 的 `nn.Linear(4096,1024)`，所以进入 transformer 后不是恒零 latent；这与论文“输入 extra tokens 为零”不矛盾。

### 4.3 精度与缓存

公开 benchmark 文档说明 Kimodo-SOMA-v1.1 和 TMR 训练使用 float32 embeddings（`docs/source/benchmark/results.md:7`）。

**实现合同**：离线缓存 float32 embedding；缓存 key 包含原始 UTF-8 文本、sanitize 版本、base/PEFT revision、pooling config。LLM2Vec wrapper 内部 batch size 固定 1 以确保 repeatability（`llm2vec_wrapper.py:71-80`）。训练读取后可在投影前保持 FP32；AMP 仅作用于 denoiser。

Text dropout 不是随机删 token：**[DEFAULT，匹配公开 CFG 空文本语义]** 把 embedding 清零并把 text pad mask 全设 False。公开推理对空字符串也是该行为（`kimodo_model.py:591-599`）。

## 5. Diffusion 与条件输入

### 5.1 Forward process

论文按 DDPM 从 uniform timestep `t∈{1,...,T}` 采样噪声，denoiser 预测 clean motion `x0`，`T=1000`（Sec. 4 Background p.7、Sec. 4.3 p.10）。公开实现使用 cosine alpha-bar：

`alpha_bar(t)=cos(((t+0.008)/1.008)*π/2)^2`，`beta<=0.999`（`diffusion.py:12-26`）。

训练实现直接复用 `Diffusion.q_sample`（`diffusion.py:96-110`），在 normalized `x0` 上加标准高斯噪声。timestep tensor 使用公开实现的 0-based `[0,999]`，论文的 1-based 只是记号。

### 5.2 Constraint imputation

由 GT `x0` 构造 target feature `x_tgt` 与同形 binary mask `m`。每个 diffusion step 输入：

`x_imputed = m*x_tgt + (1-m)*x_t`

`x_in = concat(x_imputed, m)`

论文见 Sec. 4.2 p.9、Fig. 9；代码见 `twostage_denoiser.py:98-105`。`x_tgt` 必须用与 `x0` 相同的 normalization。mask 不 normalize。无 constraints 时 target/mask 均为零。

全身 position constraint 同时必须约束 smoothed root XZ，否则局部 joint position 无参照；公开代码会主动报错（`kimodo_motionrep.py:284-302`）。position/rotation/root mask 的确切 feature 写入逻辑应直接复用 `kimodo_motionrep.py:222-306`。

## 6. Denoiser 与训练目标

### 6.1 Root/body 两 stage

发布架构每 stage 是 PyTorch `TransformerEncoder`：16 layers、8 heads、latent 1024；论文报告两 stage 总计 282M learnable params（Sec. 4.2 p.10）。发布 config 另给 FFN 2048、GELU、`norm_first=false`、heading input、50 text slots。公开构造见 `backbone.py:60-134`。

Prefix token 顺序按代码是 `[50 text slots, timestep, first-heading]`，再接 pose tokens（`backbone.py:168-220`）。所有 token 加 sinusoidal PE（`backbone.py:235-280`）。

SOMA30 + concat mask 的 shape 合同：

- Stage 1 input `2D=738`，output global root `5`。
- global root prediction 转为 normalized local root `4`。
- Stage 2 input `local root 4 + noisy/imputed body 364 + original full mask 369 = 737`，output body `364`。
- 最终 output concat 为 `369`。

对应构造：`twostage_denoiser.py:30-61,107-152`。

### 6.2 公开实现与论文的梯度差异

论文 Sec. 5 p.12 说 interleaved two-stage denoiser “trains end-to-end”，但没有明确规定 body loss 必须穿过 global→local bridge。官方公开的 denoiser 实现在 training mode 对该 conversion 使用 `no_grad` 并 `detach`；公开仓库没有发布完整 trainer，因此这只能证明已发布 denoiser branch 的行为。两个 stage 仍可在同一 forward、总 loss 和 optimizer step 中联合训练。`detach_root_for_body=true` 时 body loss 不经 local-root condition 反传到 root stage；`false` 时使用梯度耦合 bridge。

**实现决定 [CODE + PAPER-AMBIGUOUS]**：生产 profile 默认 `detach_root_for_body=true`，匹配已发布 denoiser 的 training-mode branch。这与论文所述的两 stage 联合训练不冲突，但论文未公开 bridge autograd 细节，所以不将私有 trainer 标为已对齐。`false` 保留为梯度耦合消融；建议小规模 A/B，但不得把任一私有 trainer 的 autograd 行为宣称为论文明确事实。

### 6.3 Loss

论文 Eq. (1), p.10：

`L = 10 L_root_pos + 2 L_root_heading + 10 L_joint_pos + 3 L_joint_vel + 10 L_joint_rot + 4 L_contact + 5 L_FK_pos`

每项使用 smooth L1；FK 从预测 joint rotations 计算 joint positions。变量长度用 mask。论文没有给 SmoothL1 `beta`、每项 reduction、是否在 normalized/physical 域计算、6D rotation 是否先投影、FK 项如何从 global rotations 转 local rotations。

**[DEFAULT] 可编码定义**：

1. diffusion 输入/输出始终 normalized。
2. direct six component loss 默认在 **normalized feature domain** 计算，每项先对有效 `[B,T,component_dims]` 做 mean，再乘论文权重；`SmoothL1(beta=1.0)`。FK 仍在反归一化后的物理空间计算。
3. rotation direct loss 对代码存储的 global 6D 表示计算。
4. FK loss：6D→rotation matrix→按 skeleton 转 local rotations→FK；与 GT global joint positions 比较，单位 m，对有效 batch/time/joint/xyz 做 mean。
5. contact target 为 0/1；使用同一 SmoothL1，不额外 BCE。
6. 记录七个未加权 loss、七个加权 loss 和 total；任何 NaN 立即保存 crash checkpoint。

**已完成的二选一消融**：`loss_domain={physical,normalized_direct_physical_fk}`。项目选择后者作为 V1/V2 新训练的统一工程默认；physical 仅保留作旧 30K 历史对照。该结论不得倒推成 NVIDIA 未公开的官方事实。

## 7. Phase 1 / Phase 2 curriculum

### 7.1 Phase 1：0-499,999

- text-to-motion only，无 kinematic constraints。**[PAPER]**
- model dropout `0.1`。**[PAPER]** 公开 backbone 有 transformer 与 positional-encoding 两个 dropout；论文未区分，默认两者都设 0.1 **[DEFAULT]**。
- text input 以 `0.10` 概率 drop，训练 conditional/unconditional 分支。**[PAPER]**
- 所有 7 项 `x0` reconstruction loss 仍计算。**[INFERRED]** Sec. 4.3 未给 phase-specific loss，最小假设是同一目标。

### 7.2 Phase 2：500,000-999,999

- 从 Phase 1 的同一 model/optimizer/EMA/global_step 继续；不重置 optimizer。**[INFERRED]** “trained in two phases” 暗示连续训练，但论文未明确 optimizer state；因此必须在配置中记录。
- dropout 改为 `0.0`。**[PAPER]** 发布 checkpoint config 的 transformer/PE dropout 均为 0，和最终阶段一致。
- 10% 样本无 constraints。**[PAPER]**
- 25% 样本混合两种 constraint patterns。**[PAPER]**
- sparse constraint 最大 keyframe 数随 Phase 2 progress 从 1 线性增至 20，并偏向更少 keyframe。**[PAPER]**
- text dropout 仍为 10%。**[PAPER]**

若 no-constraint 与 text-drop 独立采样，期望 batch 组成是 joint 81%、constraint-only 9%、text-only 9%、unconditional 1%。这是 **[INFERRED]**，但正好支持 separated CFG 的三个必要分支；默认采用并在日志统计实际比例。

### 7.3 Constraint pattern sampler

论文列出的 pattern（Sec. 4.3 p.11）：

1. sparse full-body joint positions；
2. sparse hands/feet 的随机子集 positions/rotations；
3. sparse 2D root position/heading waypoints；
4. dense 2D root position/heading paths；
5. sparse foot-contact configuration。

**[UNKNOWN]**：五类基础概率、dense path 长度、是否同时约束 heading、手脚子集概率、position/rotation 拆分概率、foot-contact keyframe 规则、两 pattern 是否允许同类、冲突覆盖顺序。

**[DEFAULT] sampler 算法**：

```text
sample categorical mode: no constraints 10%, two patterns 25%, one pattern 65%
if constrained:
    n_patterns = 2 for the two-pattern mode, otherwise 1
    sample n_patterns without replacement, equal weights over 5 families
    kmax = 1 + floor(19 * phase2_progress)
    for sparse family: sample K from P(K=k) proportional to 1/k, k=1..kmax
    sample unique valid frames, then sort
    build observed features/mask from the same GT x0
```

Family-specific defaults：

- full-body：K 个 frame，写入所有 joint positions，同时写 smoothed root XZ、root Y、heading。
- end-effector：从 `{L/R hand,L/R foot}` 非空子集均匀采样；同时写 position+global rotation。
- sparse root：K 个 frame，position 总是写，heading 以 `0.5` 概率写。
- dense root：随机连续区间，长度在有效 clip 的 20%-80% 均匀采样；逐帧写 root position，heading 以 `0.5` 概率写。
- foot contact：K 个 frame；仅写 contact feature mask。公开 inference constraint class 没有 foot-contact authoring API，因此训练实现需新增内部 sampler，不修改本公开 inference schema。

从同一 GT 生成的重叠 constraint 值理论上一致；mask 用 OR。若浮点变换后不一致，按 full-body > end-effector > root > contact 的固定优先级，并计数报警。这个优先级是 **[DEFAULT]**。

## 8. Optimizer、分布式和 EMA

### 8.1 Adam-atan2

**[PAPER]** Adam-atan2，learning rate `2e-5`（Sec. 4.3 p.10）。公开依赖没有提供 optimizer 实现，训练项目必须显式 vendor/锁定一个实现。

**[DEFAULT]**：采用 Kimodo 引用的 Everett et al. Adam-atan2 stretched 定义 `4/π·λ·atan2(m, λ√v)` 及其论文实验值 `λ=8`，标准 moment betas `(0.9,0.999)`、无 weight decay、constant LR、无 warmup。配置必须保存 `atan2_lambda`，不可只写 `AdamAtan2` 名称；Kimodo 本身未披露 λ。对 embedding/normalization 参数不做优化；所有 denoiser parameter 在同一 parameter group。

梯度 global-norm clip `1.0` 是稳定性默认，不是论文值。建议后续诊断记录每 step 的 pre/post clip norm，并在 99.9 percentile 长期低于阈值时做关闭 clip 的 ablation；当前 trainer 尚未记录这两个诊断量。

### 8.2 Batch 与精度

最佳模型论文配置：global batch 2048，16×A100 SXM4 80GB，30 fps（Sec. 4.3 p.10-11）。若无 accumulation，则 local batch=128/GPU，这是 **[INFERRED]**。

**[DEFAULT]** DDP + bf16 autocast + FP32 optimizer state；按显存选择 microbatch，并 accumulation 到精确 global batch 2048。梯度 accumulation 期间 loss 必须按 global valid-element denominator 等价归一，不能简单平均不同长度 microbatch。建议后续记录 effective frames/step，因为相同 motion batch 可能有不同 frame 数；当前日志尚未包含该字段。

### 8.3 EMA

每 10 optimizer steps 更新一次，`ema = 0.995*ema + 0.005*model`；两阶段全程连续，测试用 EMA（Sec. 4.3 p.11）。**[DEFAULT]** EMA 第一次更新前从 online model 深拷贝；仅跟踪 trainable floating parameters/buffers，不跟踪冻结 LLM2Vec。Phase 切换不重置 EMA。

## 9. Checkpoint 与恢复

论文仅说明推理使用 EMA，没有保存/选择规则。公开 loader 支持 `.safetensors` 或普通 PyTorch state dict，并能去除 `denoiser.backbone.` prefix（`model/loading.py:44-68`、`twostage_denoiser.py:66-71`）。发布模型目录期望 `config.yaml`、`model.safetensors`、split stats（`load_model.py:177-206`）。

**[DEFAULT] checkpoint 内容**：

- online denoiser state；EMA denoiser state；
- optimizer state、AMP/scaler（若有）、global optimizer step、microstep；
- phase 和 phase-local progress；
- Python/NumPy/PyTorch CPU/CUDA RNG；sampler RNG 与 dataloader epoch；
- data/split/text-cache/stats/skeleton hash；完整 resolved config；
- 当前 dropout/PE dropout、Adam-atan2 变体参数；
- 代码 revision/dirty-state 与数据/实现 provenance。当前 trainer 没有验证循环，因此不保存“最近验证指标”。

当前实现保存：每 `checkpoint_every` step、Phase 1 边界和最终 step 的 full-state checkpoint；检测到 non-finite 时另存 diagnostic（不是 exact-resume 点）。保留最近若干普通 checkpoint，并保留 phase boundary/final milestone。当前没有 best-proxy/验证最优跟踪，也不承诺一般异常退出都能保存。最终另导出 EMA-only `model.pt`、推理 `config.yaml` 和 stats；训练恢复必须使用 full-state checkpoint。

**选择规则 [DEFAULT]**：论文数值复现主结果使用 **1M-step final EMA**，避免把未公开 early-stopping 伪装成官方规则；验证最优仅作诊断/次要报告。

## 10. Evaluation 与验收

### 10.1 论文协议

论文 Sec. 6.1 p.13：按 novel behaviors 划分 10% test，约 5k motions；三组测试为 overview text、fine-grained text、text+constraint suite。文本指标 R@3 和 FID 使用在完整 Rigplay（包括 train/test split）训练的 TMR。constraint 指标包含 full-body position、end-effector position/rotation、2D root 与 pelvis-to-smoothed-root p95；另报 foot skate/contact。

论文消融默认 20 fps、8 GPU/batch 1024；最佳演示模型 30 fps、16 GPU。Table 1/2 数字不能直接作为 30 fps 发布 checkpoint 的逐项验收阈值。

### 10.2 公开 benchmark 协议

公开 suite 有 22,474 cases，分 content/repetition，含 text-only、constraint-only、text+constraint（`docs/source/benchmark/introduction.md:24-102`）。生成脚本默认 DDIM 100、batch 32，但公开结果要求 batch=1 以逐 case seed 精确复现（`benchmark/generate_eval.py:41-63,327-349`；`docs/source/benchmark/results.md:3-12`）。

公开 pipeline 使用 TMR 预训练模型嵌入（`benchmark/embed_folder.py:4-8,58-76`），并计算 TMR、foot、contact 与 constraint metrics（`benchmark/evaluate_folder.py:21-40,293-305`）。

**协议差异警告**：公开 benchmark 文档已说明它不是论文 Sec. 6.1 的 exact test suite（`docs/source/benchmark/introduction.md:7-11`）。旧 `ContraintFollow` 仍保留公开 benchmark 的历史口径；`benchmark/evaluate_folder.py --paper-protocol` 另行计算 generated smooth-root 对输入 smooth-root 的平均误差、EE global-rotation geodesic error、pelvis-to-smooth-root 的 pointwise p95，以及完整 prompt set 的 retrieval/FID，并使用独立 `paper_*` 字段。两套列不得混用；缺少论文私有测试集时仍不能声称复现 Table 1/2 数值。

### 10.3 训练中验证计划 [DEFAULT]

- 每 10k step：固定 1k-case proxy；DDIM 50 用于趋势，另抽 100 cases 用 DDIM 100。
- 每 50k step 和 Phase 边界：完整可承受 validation；DDIM 100，`w_text=w_constraint=2`，不做 postprocess。
- 统一固定 seeds；同时保存 online 与 EMA 结果，但主报告 EMA。
- Phase 1 关注 text R@3/FID、foot skate/contact；Phase 2 增加所有 constraint errors。
- 训练结束跑完整公开 benchmark，batch=1；分别报告 float32 text embedding 的主结果和 bf16 compatibility 结果。

### 10.4 最低工程验收

1. **Feature round-trip**：local rotation/root → feature → inverse，root/joint rotation/position 误差在数值容差内；SOMA30 输出维度 369。
2. **Diffusion identity tests**：`q_sample(t=0)`、schedule monotonic、DDIM shape/dtype/device；固定 seed 输出 deterministic。
3. **Constraint imputation tests**：每种 mask 只覆盖预期 feature；full-body position 自动含 root reference；无 constraint 与零 mask 等价。
4. **Gradient tests**：生产 profile 默认 `detach=true`，body loss 不得经 local-root bridge 更新 root stage，而 root loss仍须更新 root stage；`detach=false` 梯度耦合消融中 body loss必须能穿过 bridge。
5. **CFG dropout coverage**：Phase 2 长期统计接近 joint 81%、constraint-only 9%、text-only 9%、unconditional 1%。
6. **Resume equivalence**：同 seed 连续 100 step 与 50+resume+50 的权重/optimizer/EMA 在容差内一致。
7. **Overfit smoke**：32 clips 可把 reconstruction/FK loss 显著压低，并在约束 frame 接近 GT。

## 11. 论文消融锚点与预期方向

Table 1 p.14 的 20 fps、8 GPU ablation 给出 curriculum 的方向性证据：Full Model 相比 No Train Curriculum，text metrics 接近，但 constraint errors 明显更低：full-body `2.67 vs 5.80 cm`、end-effector pos `3.09 vs 6.59 cm`、2D root `2.90 vs 5.71 cm`、pelvis p95 `9.7 vs 15.5 cm`。因此 Phase 2 sampler/无约束比例/渐进 keyframe 是一级验收对象；若工程实现 text 指标正常但 constraint error 不下降，优先检查 sampler、mask normalization 和 Phase 2 dropout，而不是先加新 loss。

Table 2 p.15 说明 data/model/batch 扩展均有收益；full release 是 282M、batch 2048。小资源复现只能称 scale-down proxy，不应承诺论文数值。

## 12. 明确非训练项

- DDIM、separated CFG 是推理；公开公式实现见 `cfg.py:94-129`。
- multi-prompt 是分段生成、overlap constraints 和 blending，不是 multi-prompt 训练；论文 Sec. 4.4 p.11，公开实现 `kimodo_model.py:123-378`。
- foot locking、IK 和短 motion optimization 是 output post-processing；论文 Sec. 4.4 p.11 明确 Sec. 6 实验不使用。
- G1 demonstrations 之后训练的 ProtoMotions physics policy 是下游任务，不是 Kimodo 后训练（Fig. 6/Sec. 2.4 p.6）。
- Qwen3-32B 是离线 paraphrase 工具；没有证据表明 Kimodo 微调 Qwen。
- TMR 在 Rigplay 上单独训练，服务评测；不是生成模型的一部分。
- 论文明确 Kimodo 不依赖额外 ControlNet fine-tuning、test-time guidance/optimization 或 RL（Sec. 5 p.12）。

## 13. 仍需回答的阻塞问题（最多 3 个）

1. **目标数据与 split**：是否有 Bones Rigplay 的合法访问权，以及论文使用的 exact train/test manifest、retargeted assets 和 metadata？没有则只能验收 BONES-SEED 工程复现，不能验收 RP 数值复现。
2. **核心训练 recipe**：作者能否提供 loss reduction/计算域、Adam-atan2 完整变体与 betas/weight-decay/schedule、训练精度、gradient clipping、seed、Phase 边界 optimizer/EMA 行为？这些参数足以造成不可忽略的训练差异。
3. **增强与 constraint sampler**：作者能否提供 Qwen paraphrase prompt/缓存、stitched transition 数据/预训练 transition model，以及 motion/text mixture 与五类 constraint pattern 的完整概率分布？没有这些材料，数据分布和 Phase 2 控制分布都只能近似。

## 14. 实现顺序

1. 锁定数据目标、split、skeleton 和 licenses；产出 manifest/hash。
2. 复用公开 skeleton/FK/motion-rep，完成离线 feature + stats + round-trip tests。
3. 冻结 LLM2Vec，生成 FP32 text cache。
4. 复用公开 Diffusion/TwostageDenoiser；实现论文 Eq. (1) loss 与 provenance-aware config。
5. 先跑 32-clip overfit，再跑 10k-step Phase-1 smoke。
6. 实现 constraint sampler 与 Phase 2，验证 CFG 四类样本比例。
7. 加入 DDP、Adam-atan2、EMA、full-state checkpoint 和 resume-equivalence test。
8. 运行 50k proxy 消融：loss domain、root detach、constraint sparse distribution。
9. 固化选择后启动 1M-step 训练；不再根据 test split 调参。
10. 以 final EMA、DDIM100、CFG2/2、无 postprocess 跑完整评测并报告与官方边界。
