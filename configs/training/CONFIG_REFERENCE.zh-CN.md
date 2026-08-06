# Kimodo 训练配置逐字段说明

本文解释同目录下三份训练方法配置：

- `kimodo_soma_seed_public.yaml`：公开 BONES-SEED 上可以实际训练的工程重建配置；
- `kimodo_soma_seed_reproduction.yaml`：论文数据严格门禁配置；公开数据缺少未发布增强资产，因此默认会失败；
- `kimodo_tiny_smoke.yaml`：只用于验证代码链路的微型配置，不能用于有效训练。

本文主要逐字段解释 `kimodo_soma_seed_public.yaml`。严格配置除了
`paper_method_strict=true`、`data.require_paper_data_parity=true` 以及说明性注释外，当前数值与 public
配置相同。tiny-smoke 则故意缩小模型、数据和训练步数。

## 1. 先理解：训练最终不是只读取这一份 YAML

训练配置按以下顺序合并，越靠右优先级越高：

```text
TrainingConfig 代码默认值
  → --config 方法 YAML
  → --paths 机器路径 YAML
  → --overlay 硬件/实验 overlay（可重复）
  → --set key=value 命令行覆盖（可重复）
```

例如两张 H200 的启动命令实际是：

```bash
python -m kimodo.training.cli \
  --config configs/training/kimodo_soma_seed_public.yaml \
  --paths /storage/kimodo/config/repro.paths.yaml \
  --overlay configs/overlays/two_h200_gb2048.yaml
```

四层的职责不同：

| 层 | 回答的问题 | 典型字段 |
|---|---|---|
| 方法 YAML | 模型和训练算法是什么 | 层数、loss 权重、学习率、课程阶段 |
| paths YAML | 数据和输出在这台机器的哪里 | manifest、stats、checkpoint、output_dir |
| hardware overlay | 这组硬件怎样承载方法 | 每卡 batch、累积步数、workers、精度 |
| `--set` | 这一次临时实验改什么 | 最大步数、日志频率等 |

`--paths` 采用白名单，只能提供路径字段，不能修改学习率、模型宽度或 loss。普通 `--overlay` 可以覆盖训练
字段，所以正式运行应保留 resolved config 和 provenance 进行审计。

训练启动后，合并完成的实际配置会写入：

```text
<runtime.output_dir>/config.resolved.yaml
```

因此排查一次运行时，应看 `config.resolved.yaml`，不能只看 base YAML。

## 2. 阅读本文需要的基本名词

### 2.1 frame、clip、sample、batch、step

- **frame（帧）**：某一时刻的完整人体姿态。30 FPS 表示一秒 30 帧。
- **clip（动作片段）**：连续多帧组成的动作，例如 5 秒行走是约 150 帧。
- **manifest row / sample（样本）**：manifest 的一行。它引用某个动作文件、一个时间区间和一段文字。
  同一个动作文件可以因为不同文字或不同时间区间形成多个训练样本。
- **micro-batch**：一次 forward/backward 输入一个 rank 的样本集合。
- **optimizer step**：真正更新一次模型参数。使用梯度累积时，要处理多个 micro-batch 才产生一个
  optimizer step。
- **rank**：一个分布式训练进程，通常一张 GPU 对应一个 rank。

有效 global batch 的计算是：

```text
world_size × 每 rank batch_size × gradient_accumulation_steps
```

例如两张卡、每卡 128、累积 8 次：`2 × 128 × 8 = 2048`。

### 2.2 skeleton、root、joint、local/global rotation

- **skeleton（骨架）**：关节及父子连接关系。SOMA30 表示这里训练使用 30 个关节。
- **joint（关节）**：骨架节点，例如 pelvis、膝、踝、手腕。
- **root（根关节）**：整棵骨架的根，通常是 pelvis/骨盆；root 的空间移动带动整个人体移动。
- **local rotation（局部旋转）**：关节相对父关节的旋转。
- **global rotation（全局旋转）**：把祖先关节旋转逐级合成后，该关节在世界坐标系中的朝向。
- **forward kinematics，FK（正向运动学）**：已知骨架长度、root 位置和各关节局部旋转，沿父子链计算
  每个关节的全局位置和全局旋转。

### 2.3 diffusion、denoiser、conditioning

- **diffusion（扩散模型）**：训练时给干净动作逐步加噪，模型学习从带噪动作恢复干净动作。
- **timestep**：扩散噪声等级。这里从 `0..999` 随机采样。
- **denoiser（去噪网络）**：输入带噪动作、文字、timestep 和可选动作约束，输出干净动作预测。
- **conditioning（条件）**：指导生成的外部信息，例如文字、root 路径、关键姿势、手脚位置。
- **mask-imputation**：用布尔 mask 标记哪些动作特征是用户已知约束，并把这些已知值覆盖进带噪输入。

## 3. 顶层字段

### `schema_version: 1`

配置文件格式版本，不是模型版本。代码只接受 schema 1。它用于防止未来字段语义改变后，旧配置被静默
解释成新格式。

### `paper_method_strict: false`

控制“论文明确披露的字段是否允许被覆盖”。

- `false`：公开工程配置；允许硬件测试、短跑等显式调整。
- `true`：锁定代码已经编码的论文明确值，例如 30 FPS、10 秒、两阶段各 500k steps、loss 权重、
  Adam-atan2、EMA 等。

严格模式不等于“已经复现论文”。它还要求 `data.require_paper_data_parity=true`，而公开 BONES-SEED
manifest 缺少论文未发布的 Qwen paraphrase 和跨动作 transition 数据，因此会 fail closed。

严格模式还可以检查论文训练规模：当 `runtime.enforce_paper_scale=true` 时，要求 16 ranks 且有效 global
batch 为 2048。两卡工程重建通过 hardware overlay 只关闭这个规模门禁，不改变模型数学。

## 4. `data`：训练样本怎样读取

```yaml
data:
  manifest: ""
  split: train
  fps: 30
  max_seconds: 10.0
  min_frames: 2
  num_workers: 8
  multiprocessing_context: fork
  pin_memory: true
  prefetch_factor: 2
  persistent_workers: false
  require_cached_text: true
  reference_verification: inventory
  reference_inventory: null
  require_paper_data_parity: false
```

### `manifest`

训练样本清单 JSONL 的路径。JSONL 是“一行一个 JSON 对象”的文本格式。一行通常引用：

- 样本 ID；
- canonical motion NPZ；
- 文字描述；
- split；
- 可选的起止时间；
- 已缓存的 LLM2Vec embedding。

base YAML 故意留空，因为它是机器路径，不属于训练方法。资源 pipeline 生成的 `repro.paths.yaml` 会通过
`--paths` 填入真实值。正式训练读取的是 `train.cached.jsonl`，不是 `train.raw.jsonl`。

### `split: train`

只读取 manifest 中 `split == "train"` 的行。split 是数据集划分，目的是让训练集和评测集不共享动作，
避免数据泄漏。

### `fps: 30`

训练数据时间采样率：每秒 30 帧。它必须与 `model.fps` 相同。若原始动作是 120 FPS，重采样必须在离线
预处理完成，训练器不会临时改变 FPS。

FPS 会影响：

- 秒和帧之间的换算；
- 速度特征的数值；
- 10 秒最多对应多少帧；
- temporal event 的 `start_time/end_time` 裁剪位置。

### `max_seconds: 10.0`

单个训练样本最长 10 秒，即 30 FPS 下最多 300 帧。更长的动作在每个 epoch 中按稳定随机种子选择一个
连续窗口；不是永远只取开头，也不是在离线 cache 中提前切死。

### `min_frames: 2`

最短接受 2 帧。速度至少需要相邻两帧才能定义，所以 2 是表示层技术下限。论文只披露最长 10 秒，
没有披露最短动作长度；因此该值是 reconstruction 选择，不是论文事实。

已知 `frame_count` 的过短行会在 Dataset 初始化时确定性排除，而不是让 DataLoader worker 运行到一半
才报错。

### `num_workers: 8`

每个训练 rank 的 DataLoader CPU worker 数。两卡且设置 16 时，通常会产生约 32 个 worker。worker 负责：

- 读取 NPZ；
- temporal crop；
- FK 和 369D 特征构建；
- root/heading 增强；
- 读取文字 `.npy`。

它不是动作离线转换的 `pipeline.motion_workers`，也不是 stats 的 `pipeline.stats_workers`。

### `multiprocessing_context: fork`

DataLoader worker 的启动方式：

- `fork`：Linux 下复制进程地址空间，未修改内存通过 copy-on-write 共享；适合已经解析出的 140 万行
  manifest，避免每个 worker 重新序列化数 GB Python 对象。
- `spawn`：全新 Python 进程，跨平台但要 pickle Dataset。
- `forkserver`：由专用 server fork；Python 3.14 在部分 POSIX 环境默认使用它。
- `auto`：让 PyTorch/Python 选择。

这里 worker 只做 CPU 数据处理，不访问 CUDA，所以生产 Linux 配置选择 `fork`。

### `pin_memory: true`

让 DataLoader 把 batch 放到 page-locked CPU 内存，GPU 可更高效地异步拷贝。它会增加锁页内存占用；
纯 CPU 训练通常设为 false。

### `prefetch_factor: 2`

每个 worker 预取的 batch 数。总预取规模大致是 `num_workers × prefetch_factor` 个 batch。增大可能掩盖
磁盘延迟，也可能显著增加 CPU 内存和文件 I/O 压力。`num_workers=0` 时该值不会传给 DataLoader。

### `persistent_workers: false`

是否在一个 epoch 结束后保留 worker。当前实现强制为 false，因为 Dataset 的 `set_epoch()` 会改变随机裁剪
种子；持久 worker 持有的 Dataset 副本不会自动收到主进程的新 epoch，从而破坏预期增强和精确 resume。

### `require_cached_text: true`

要求每个 manifest row 都有离线文字 embedding。生产训练不会加载 8B LLM2Vec，而是直接读取 float32
`[1,4096]` `.npy`。设为 true 可以在启动时提前发现漏缓存行。

### `reference_verification: inventory`

启动时如何校验 manifest 引用：

- `full`：现场重新 hash 每个 motion 和 embedding；适合小测试数据，真实数据会很慢。
- `inventory`：信任 prepare 阶段已经完整构建并验证的 content-addressed inventory；启动时只核对 manifest、
  inventory 和 inventory metadata 的身份。

`inventory` 不是“不校验”，而是把昂贵的几十 GB 全量校验前移到 prepare 阶段。

### `reference_inventory: null`

引用清单的路径。base YAML 留空，由 `--paths` 生成层填入
`train.cached.references.jsonl`。当 verification 为 `inventory` 时不能为空。

### `require_paper_data_parity: false`

是否要求 manifest 声明并证明至少包含：

- Qwen3-32B paraphrase 行及 prompt/model provenance；
- 两个不同 source motion 拼接并由 transition model 衔接的行及 provenance。

public 配置为 false，诚实表示它使用公开工程数据。strict 配置为 true，因此普通 BONES-SEED manifest 会
被拒绝。这个门禁只能检查 schema 和自洽性，无法证明未公开 prompt、mixture 与 NVIDIA 私有 recipe 一致。

## 5. `model`：去噪模型和动作表示

```yaml
model:
  checkpoint_dir: null
  checkpoint_weights: null
  skeleton_joints: 30
  stats_path: ""
  fps: 30
  num_diffusion_steps: 1000
  motion_mask_mode: concat
  llm_dim: 4096
  llm_tokens: 1
  num_text_tokens_override: 50
  latent_dim: 1024
  ff_size: 2048
  num_layers: 16
  num_heads: 8
  activation: gelu
  norm_first: false
  input_first_heading_angle: true
  use_text_mask: false
  detach_root_for_body: true
```

### `checkpoint_dir`

可选的官方 inference bundle 目录。该目录应包含官方 `config.yaml`、唯一一份权重文件及它引用的 stats。
设置后，模型对象图和结构字段从官方 bundle 构建，用于兼容性初始化。

它不是 trainer resume。官方 inference bundle 没有 optimizer、global step、RNG 和完整 EMA 训练状态。

### `checkpoint_weights`

可选的“只加载模型权重”路径。当前 YAML 的模型结构仍由下面字段创建，然后对 state dict 做 strict load。
适合从一个兼容 denoiser 初始化新实验，也不是精确 resume。

三种概念不要混淆：

| 字段 | 加载什么 | 能否精确续训 |
|---|---|---:|
| `model.checkpoint_dir` | 官方推理 bundle 的结构和权重 | 否 |
| `model.checkpoint_weights` | 兼容 denoiser state dict | 否 |
| `runtime.resume` | trainer full-state checkpoint | 是 |

### `skeleton_joints: 30`

使用 SOMA30 骨架。每帧动作表示宽度因此是 369：root 5 维加 body 364 维。这里只填数量并不意味着任意
30 关节骨架都兼容；真实语义还依赖注册表中的关节顺序、父子关系、rest offsets 和坐标约定。

### `stats_path`

normalization stats 目录。应包含：

```text
global_root/{mean,std}.npy  # 每个 [5]
local_root/{mean,std}.npy   # 每个 [4]
body/{mean,std}.npy         # 每个 [364]
stats.metadata.json
```

base YAML 留空，由路径层填入。它必须与当前 manifest、FPS、骨架和预处理策略匹配。

### `fps: 30`

模型动作表示的 FPS。必须等于 `data.fps`。速度、foot contact 检测和时间裁剪都依赖它。

### `num_diffusion_steps: 1000`

前向扩散链的离散噪声等级数量。训练时每个样本均匀抽一个 `t∈[0,999]`，从干净动作 `x0` 生成带噪
动作 `xt`，网络直接预测 `x0`。这里使用公开实现的 cosine schedule。

这不是推理一定要跑 1000 次；推理采样器可以选择子序列减少 denoising evaluations。

### `motion_mask_mode: concat`

约束条件的注入方式。对被约束的 `(frame, feature)`：

1. 用 clean observed value 覆盖 noisy motion 对应位置；
2. 把同形状的 0/1 mask 拼到模型输入特征末尾。

模型因此既看到约束值，也知道哪些值是用户指定的。代码只接受 `concat` 作为论文对齐模式。

### `llm_dim: 4096`

每个 LLM2Vec 文字向量的宽度。cached manifest 中的 embedding 最后一维必须严格等于 4096。

### `llm_tokens: 1`

真实文字 encoder 输出一个 sentence token，形状为 `[1,4096]`。这描述实际有内容的 token 数。

### `num_text_tokens_override: 50`

进入 transformer 前，把文字 token 序列固定到 50 个位置。当前输入是 1 个语义向量，剩余 49 个位置补零。
这保持公开模型/论文的 prefix 长度契约。它不是把一句话重新切成 50 个词，也不是再次调用 tokenizer。

### `latent_dim: 1024`

transformer 内部每个 token 的隐藏宽度 `d_model`。动作输入、文字向量、timestep 和 heading 都先投影到
1024 维，再进入 self-attention。

### `ff_size: 2048`

每层 transformer 中前馈网络（FFN）的中间宽度。典型层结构可粗略理解为：

```text
self-attention → 1024→2048→1024 的逐 token MLP
```

它不是动作特征宽度，也不是 batch size。

### `num_layers: 16`

每个 stage 的 transformer encoder 层数。Kimodo 有 root stage 和 body stage，各 16 层，不是两者合计
16 层。

### `num_heads: 8`

多头注意力头数。1024 维隐藏空间分成 8 个注意力子空间，每头宽度 128。头数必须整除 latent dim。

### `activation: gelu`

FFN 使用 GELU 非线性。它比硬截断的 ReLU 更平滑，是 transformer 常见选择。

### `norm_first: false`

PyTorch `TransformerEncoderLayer` 的归一化顺序：

- `false`：post-norm，子层残差相加后做 LayerNorm；
- `true`：pre-norm，进入子层前做 LayerNorm。

该字段会改变网络函数和 checkpoint 兼容性，不能随意切换。

### `input_first_heading_angle: true`

把样本第一帧的朝向角编码为 `[cos θ, sin θ]`，投影成一个额外 prefix token。这样模型明确知道动作坐标系
相对世界的初始朝向。

### `use_text_mask: false`

是否让 transformer padding mask 屏蔽补齐的文字 token。公开实现为 false，因此固定的 50 个文字位置都
参与 attention；补零位置经过线性层和 transformer 后仍能作为兼容的 prefix slots。改成 true 会改变官方
checkpoint 的行为。

### `detach_root_for_body: true`

Kimodo 是两阶段去噪器：

```text
带噪 369D + mask
  → root transformer 预测 global root 5D
  → global root 转成 local root 4D
  → body transformer 预测 body 364D
```

`true` 表示训练时切断 root→local-root→body 这条桥上的梯度：body loss 不会通过该桥反向更新 root
transformer；root transformer 仍通过自己的 root/loss 项训练。两个 stage 仍在同一次 forward 和同一个
optimizer step 中联合更新。

`false` 是允许 body loss 穿过 bridge 的消融实验，不是当前公开代码默认值。

## 6. `curriculum`：从文本生成过渡到约束生成

curriculum 是训练课程：训练早期先学基本动作分布，后期再加入复杂控制条件。

```yaml
curriculum:
  phase1_steps: 500000
  phase2_steps: 500000
  phase1_dropout: 0.1
  phase2_dropout: 0.0
  text_dropout_probability: 0.1
  no_constraint_probability: 0.1
  mix_two_probability: 0.25
  sparse_keyframes_min: 1
  sparse_keyframes_max: 20
  sparse_count_power: 1.0
  dense_path_min_fraction: 0.2
  dense_path_max_fraction: 0.8
  root_heading_probability: 0.5
```

### `phase1_steps: 500000`

前 500k optimizer steps。constraint sampler 返回空 mask，模型主要学习文字条件下恢复动作。这里的 step
是参数更新次数，不是 DataLoader batch 次数；梯度累积会让一个 step 包含多个 micro-batch。

### `phase2_steps: 500000`

后 500k optimizer steps。启用五类在线动作约束采样。默认总训练步数是两阶段之和 1,000,000。

### `phase1_dropout: 0.1` / `phase2_dropout: 0.0`

动态设置两个 transformer 中普通 dropout 和 positional-encoding dropout。Phase 1 为 10%，Phase 2
关闭模型 dropout。这与下面的 text dropout 是两件事；Phase 2 的文字条件仍可能被丢弃。

### `text_dropout_probability: 0.1`

每个样本以 10% 概率把文字 embedding 置零、文字 mask 置空。模型因而同时学习有文字和无文字分支，
推理时可做 classifier-free guidance（CFG）。

### `no_constraint_probability: 0.1`

Phase 2 中每个样本有 10% 概率不提供动作约束。

### `mix_two_probability: 0.25`

Phase 2 中每个样本有 25% 概率同时抽两个不同约束 pattern。结合上一字段，当前分布为：

```text
10% 无动作约束
25% 两种动作约束
65% 一种动作约束
```

五个 pattern family 当前均匀选择；论文没有披露 family 概率，这是 reconstruction 选择。

### `sparse_keyframes_min: 1` / `sparse_keyframes_max: 20`

稀疏约束的关键帧数范围。Phase 2 刚开始时允许的最大关键帧数约为 1，随训练进度线性增长，到 Phase 2
末尾增至 20。这让模型先学少量控制点，再学更密集控制。

### `sparse_count_power: 1.0`

在 `1..当前最大值` 中抽关键帧数量时，数量 `k` 的权重与 `k^-power` 成正比。值为 1 时，小数量比大数量
更常见。该精确分布未由论文披露。

### `dense_path_min_fraction: 0.2` / `dense_path_max_fraction: 0.8`

`root_dense` pattern 选择一个连续时间区间，长度为有效动作的 20% 到 80%，在该区间逐帧约束 root 的
地面平面轨迹 XZ。

### `root_heading_probability: 0.5`

生成 root sparse/dense 位置约束时，有 50% 概率同时约束 root heading。否则只约束地面路径，不强制人物
面朝哪个方向。

### 五类约束 pattern 是什么

| pattern | 含义 |
|---|---|
| `full_body_sparse` | 少数帧给定 root、朝向和全身关节位置，相当于姿势关键帧 |
| `end_effector_sparse` | 少数帧约束手或脚的位置/旋转；end effector 是运动链末端 |
| `root_sparse` | 少数帧约束骨盆在地面 XZ 的 waypoint，可选朝向 |
| `root_dense` | 连续一段时间逐帧约束骨盆路径，可选朝向 |
| `foot_contact_sparse` | 少数帧指定脚跟/脚尖是否接触地面 |

## 7. `loss`：预测错在哪里、各错误有多重要

网络预测干净的 369D 动作。loss 包含六个直接特征项和一个 FK 几何一致性项。

```yaml
loss:
  direct_feature_domain: normalized
  smooth_l1_beta: 1.0
  root_position: 10.0
  root_heading: 2.0
  joint_position: 10.0
  joint_velocity: 3.0
  joint_rotation: 10.0
  foot_contact: 4.0
  forward_kinematics: 5.0
```

总 loss 概念上是：

```text
10 Lroot_pos + 2 Lheading + 10 Ljoint_pos + 3 Lvelocity
+ 10 Lrotation + 4 Lcontact + 5 LFK
```

所有项只在 `valid_frames=true` 的帧上计算，padding 不参与；先按每帧特征分量平均，再按全局有效帧数
归一化，使变长 batch、梯度累积和 DDP 的数学权重一致。

### `direct_feature_domain: normalized`

- `physical`：先反归一化，再在米、速度、6D rotation、contact 等原始数值域计算六个直接 loss。
- `normalized`：直接在 z-score 后的模型空间计算六项。

FK loss 无论该字段取什么都在反归一化后的物理空间计算。论文没有披露六项 direct loss 的确切域；
本项目消融后统一选择 normalized 作为 V1/V2 新训练默认，physical 只用于显式的历史对照。

### `smooth_l1_beta: 1.0`

Smooth L1/Huber loss 的转折阈值。小误差区域近似二次函数，提供平滑梯度；大误差区域近似 L1，降低离群值
影响。这里的 beta 作用于所选 domain 的特征数值。

### `root_position: 10.0`

平滑 root XYZ 轨迹的权重。root 轨迹决定整个人在世界中的移动。

### `root_heading: 2.0`

root 朝向 `[cos θ, sin θ]` 的权重。用二维单位向量避免角度在 `-π/π` 处不连续。

### `joint_position: 10.0`

30 个关节位置特征的权重。XZ 相对平滑 root，Y 保留全局高度；它直接约束身体形状和肢体位置。

### `joint_velocity: 3.0`

关节全局速度的权重。速度项帮助动作连续、减少抖动，但权重低于位置和旋转。

### `joint_rotation: 10.0`

30 个关节全局 6D rotation 表示的权重。6D rotation 会再转换成合法旋转矩阵。

### `foot_contact: 4.0`

左/右 heel、toe 四个接触通道的权重。接触信息帮助模型区分脚落地和腾空，减少脚滑。

### `forward_kinematics: 5.0`

把预测的 global 6D rotation 转成 rotation matrix，再转 local rotations，经骨架 FK 得到关节位置，与目标
关节位置比较。这一项把“旋转看似接近”和“旋转真正产生正确身体几何”联系起来。

## 8. `optimizer`：怎样更新参数

```yaml
optimizer:
  name: adam_atan2
  learning_rate: 2.0e-5
  betas: [0.9, 0.999]
  weight_decay: 0.0
  atan2_lambda: 8.0
  gradient_clip_norm: 1.0
```

### `name: adam_atan2`

使用 Adam-atan2。它保留 Adam 的一阶/二阶动量，但用有界的 `atan2` 形式代替传统 Adam 的
`m/(sqrt(v)+eps)` 更新。也可显式选择 `adamw` 做消融。

论文明确了 Adam-atan2，但没有公开 lambda、betas、weight decay 等全部细节。

### `learning_rate: 2.0e-5`

每次 optimizer step 的基本更新尺度。当前没有 scheduler 或 warmup，学习率保持常数。学习率 2e-5 是
论文明确值；“无 scheduler/warmup”是重建选择。

### `betas: [0.9, 0.999]`

一阶动量和二阶平方动量的指数衰减系数：

- beta1 越大，梯度方向平均越平滑；
- beta2 越大，梯度平方尺度估计变化越慢。

这两个具体值未由 Kimodo 论文披露。

### `weight_decay: 0.0`

解耦权重衰减。0 表示关闭。非零时参数会在梯度更新之外按比例收缩。

### `atan2_lambda: 8.0`

Adam-atan2 公式中控制 atan2 饱和尺度的超参数。8 来自参考实现的实验值，不是 Kimodo 论文公开值。

### `gradient_clip_norm: 1.0`

optimizer step 前把全模型梯度总范数裁到最多 1，降低偶发大梯度造成数值爆炸的风险。设为 null 可关闭。

## 9. `ema`：用于推理的平滑模型

```yaml
ema:
  enabled: true
  decay: 0.995
  update_every: 10
```

EMA 是 Exponential Moving Average（指数移动平均）。它维护一份不参与反向传播的 shadow 权重：

```text
shadow = 0.995 × shadow + 0.005 × online_model
```

### `enabled: true`

开启 shadow model。trainer checkpoint 同时保存 online 和 EMA；inference export 使用 EMA。

### `decay: 0.995`

越接近 1，平均窗口越长、权重变化越平滑。EMA 初始值是 online model 的完整拷贝。

### `update_every: 10`

每 10 个成功的 optimizer steps 更新一次 EMA。AMP 因 overflow 跳过的 step 不计为成功更新。

## 10. `runtime`：运行、分布式、日志与恢复

```yaml
runtime:
  output_dir: ""
  seed: 1234
  device: auto
  precision: bf16
  batch_size: 128
  gradient_accumulation_steps: 1
  log_every: 10
  checkpoint_every: 10000
  milestone_every: 100000
  keep_last_checkpoints: 3
  resume: null
  initial_global_step: 0
  distributed: auto
  enforce_paper_scale: true
  dry_run: false
  max_steps_override: null
```

### `output_dir`

本次 run 的目录，base YAML 留空，由 paths YAML 填入。新训练要求目录为空；同一路径已有不相关内容会
拒绝启动。典型内容：

```text
config.resolved.yaml
provenance.json
train.jsonl
checkpoints/
exports/
```

### `seed: 1234`

控制 Python、NumPy、Torch、CUDA、DistributedSampler、crop、heading、constraint、diffusion noise 等
随机过程。每个 rank 会派生不同但可复现的随机流。精确 resume 还会保存每个 rank 的 RNG state。

### `device: auto`

设备选择：优先 CUDA，其次 MPS，否则 CPU。在 torchrun 下，每个 rank 使用对应 `LOCAL_RANK` 的 CUDA
设备。也可显式写 `cpu`、`cuda`、`cuda:1` 等，但分布式通常保留 auto。

### `precision: bf16`

autocast 精度：

- `fp32`：范围和精度都高，显存与计算开销最大；
- `bf16`：指数范围接近 FP32，尾数较短，现代训练 GPU 上通常稳定且高效；
- `fp16`：指数范围较小，需要 GradScaler，overflow 时可能跳过更新。

模型参数和部分关键计数不一定都永久存成该 dtype；这是 forward autocast 策略。

### `batch_size: 128`

每个 rank、每个 micro-step 的样本数，不是 global batch。变长动作会 pad 到该 micro-batch 最长序列。

### `gradient_accumulation_steps: 1`

累积多少个 micro-batch 后更新一次参数。两卡 overlay 改成 8，从而达到有效 global batch 2048。
DDP 在非边界 micro-step 使用 `no_sync()`，边界时按所有 rank 的全局有效帧数统一归一化梯度。

### `log_every: 10`

每多少个成功 optimizer steps 向 `train.jsonl` 写一次 loss、吞吐和课程统计。

### `checkpoint_every: 10000`

每 10k optimizer steps 保存一次 full-state trainer checkpoint。

### `milestone_every: 100000`

每 100k step 标记一个受保护里程碑 checkpoint。Phase 1 边界和最终 step 也受保护，不参与普通
keep-last 清理。设为 0 可关闭周期性 milestone 保护。

### `keep_last_checkpoints: 3`

对非受保护普通 checkpoint，只保留最近 3 个。milestone、phase boundary 和最终 checkpoint 不计入这
三个，因此总文件数可能明显大于 3。

### `resume: null`

full-state checkpoint 路径。恢复内容包括：

- online model；
- optimizer；
- EMA；
- AMP scaler；
- global step、epoch、batch/micro index；
- resolved config 和 provenance；
- 每个 rank 的 Python/NumPy/Torch/CUDA RNG。

它由 paths YAML 白名单允许，因此可以在部署配置中填写。

### `resume_mode`（代码默认 `in_place`，base YAML 未显式写出）

这是 `TrainingConfig` 的有效字段，即使当前 YAML 省略也会由结构化默认值补上：

- `in_place`：checkpoint 必须来自当前 `output_dir/checkpoints`，继续原 run；
- `fork`：允许从父 checkpoint 在一个全新空目录启动分支 run，并记录 parent hash lineage。

`fork` 必须和非空 `runtime.resume` 一起使用。

### `initial_global_step: 0`

不加载 checkpoint 时从哪个 curriculum step 开始。正常训练必须为 0；它主要用于受控 benchmark/诊断，
不是 resume 的替代品，因为它不恢复模型、optimizer 和 RNG。strict 模式拒绝非零值。

### `distributed: auto`

是否启用 DistributedDataParallel：

- `auto`：`WORLD_SIZE>1` 时开启；
- true 风格值：要求在 torchrun 环境中；
- false 风格值：关闭。

CUDA 使用 NCCL，CPU 使用 Gloo。

### `expected_world_size: null` / `expected_global_batch: null`

可选部署门禁，和 `paper_method_strict` 无关。公司 16-H200 profile 分别设为 16 和 2048；实际 torchrun
规模或 `world_size × per-rank batch × accumulation` 不匹配时，trainer 在分配模型前立即拒绝启动。
本地短训 profile 保持 null，不受该门禁影响。

### `enforce_paper_scale: true`

仅在 `paper_method_strict=true` 时生效：要求 16 ranks 和有效 global batch 2048。public 配置本身不会触发
严格规模检查；两卡 overlay 仍显式设为 false，以清楚记录这是硬件规模重建。

### `dry_run: false`

只解析、合并、验证并打印配置，不构建训练运行时。CLI 的 `--dry-run` 会等价地覆盖此字段为 true。
dry-run 不要求真实数据路径存在；它不是数据 preflight。

完整数据合同检查应使用 `--preflight`，它会解析 manifest/inventory 并在 CPU 上构造一个代表 batch，但
不会分配约 283M 参数的 denoiser。

### `max_steps_override: null`

临时把总训练终点改成指定 step，常用于 10-step smoke 或 benchmark：

```bash
--set runtime.max_steps_override=10
```

它表示“总终点是 10”，不是“额外再跑 10 步”。strict 模式要求为 null，以免短跑被误标为论文训练。

## 11. public、strict、tiny-smoke 到底怎么选

### 正常公开训练

```text
configs/training/kimodo_soma_seed_public.yaml
```

它保持当前已编码的论文模型/loss/curriculum 数值，同时明确：数据 mixture 和未公开 augmentation 不是
论文 exact recipe。

### 论文资产门禁审计

```text
configs/training/kimodo_soma_seed_reproduction.yaml
```

只有当外部 manifest 真正提供 Qwen paraphrase 和 stitched transition 行及 provenance 时才可能通过。
即使通过，也只证明 schema 自洽，不能证明 NVIDIA 未公开 recipe 的真实性。

### 安装和代码 smoke

```text
configs/training/kimodo_tiny_smoke.yaml
```

它使用 32 维、1 层的小模型、16 维文本 fixture、CPU、2 个总 steps。它回答“代码能否 forward、backward、
checkpoint”，不回答训练质量、论文贴合或大规模性能。

## 12. 常用命令和检查顺序

仅查看合并配置：

```bash
python -m kimodo.training.cli \
  --config configs/training/kimodo_soma_seed_public.yaml \
  --paths /storage/kimodo/config/repro.paths.yaml \
  --overlay configs/overlays/two_h200_gb2048.yaml \
  --dry-run
```

检查完整数据并构造 CPU batch：

```bash
python -m kimodo.training.cli \
  --config configs/training/kimodo_soma_seed_public.yaml \
  --paths /storage/kimodo/config/repro.paths.yaml \
  --preflight
```

两卡 10-step 真实 smoke：

```bash
KIMODO_PATHS_CONFIG=/storage/kimodo/config/repro.paths.yaml \
CUDA_VISIBLE_DEVICES=0,1 \
scripts/train_two_gpu_seed.sh \
  --set runtime.max_steps_override=10
```

正式训练前至少检查：

1. `resource-state.json` 的 status 是 `repro_train_ready`；
2. paths YAML 指向预期 manifest、inventory、stats 和全新的 output dir；
3. dry-run 的 effective batch 与预期一致；
4. 运行开始后归档 `config.resolved.yaml` 和 `provenance.json`；
5. 不把 public profile 的结果标注成私有 Rigplay 或严格论文数值复现。
