# Kimodo 完整训练管线复现汇报 PPT 骨架

## 1. 汇报定位

**标题：** Kimodo 完整训练管线复现：从模型原理、工程实现到效果验证

**核心叙事：**

> 为什么复现 → 原始 Kimodo 是什么 → 模型为什么这样设计 → 原理如何落到代码 → 数据和训练如何运行 → 结果是否可信 → 成果如何交付

本提纲不按汇报时长删减内容，也不预设必须控制在多少页。制作 PPT 时应保证下面的内容链条完整，再根据实际材料决定某个主题用一页还是多页展开。

**统一口径：**

- 本项目是基于公开论文、公开推理代码和公开配置完成的 clean-room 训练重建，不是 NVIDIA 原始训练源码。
- 始终区分三类信息：论文明确公开、公开模型代码能够确认、本仓库为补齐训练链路作出的工程选择。
- 结构复现、训练复现、效果复现是三件不同的事，分别给出结论和证据。
- 所有性能、成本和完成度结论必须由配置、日志、checkpoint、评测报告或演示支撑。
- 尚未完成或尚未验证的内容使用 `[待填]`、`[进行中]` 或 `[未验证]`，不提前写成结论。

## 2. 章节与评分项对应关系

| 评分项 | 权重 | PPT 章节 | 核心问题 |
|---|---:|---|---|
| 目标与业务价值 | 15 | 第一部分 | 为什么复现、复现什么、怎样算成功 |
| 核心成果完成度 | 30 | 第二至第五部分 | 模型、数据、代码和训练链路是否完整 |
| 效果与验证证据 | 25 | 第六部分 | 结果如何、结论是否有可追溯证据 |
| 方案与实现质量 | 20 | 第七部分 | 技术决策是否合理、实现是否可靠 |
| 协作与交付 | 10 | 第八部分 | 过程是否留痕、成果能否被他人接手 |

## 3. PPT 逐页骨架

### 封面：Kimodo 完整训练管线复现

- 副标题：从模型原理、工程实现到效果验证。
- 汇报人、团队、日期。
- 本次实际复现对象：Kimodo-SOMA-SEED，或最终采用的模型版本。
- 使用一张具有代表性的文本/约束控制动作生成图作为背景。

### 第 1 页：执行摘要——本项目完成了什么

**回答的问题：项目最终交付了什么？**

- 一句话说明 Kimodo：由文本和运动学约束共同控制的 3D 人体/机器人动作扩散模型。
- 复现链路：公开数据 → 训练 bundle → 模型训练 → checkpoint/EMA → benchmark → 推理演示。
- 当前状态：数据、模型、训练、评测、交付分别标记为已完成、进行中或未验证。
- 核心结果：`[训练数据规模]`、`[训练 step]`、`[最终 checkpoint]`、`[核心指标]`、`[资源消耗]`。
- 核心交付：代码、配置、数据合同、容器、启动脚本、日志、权重、评测报告和文档。
- 用一张端到端总览图说明整个项目，而不是直接进入局部技术细节。

---

## 第一部分：目标与业务价值

### 第 2 页：任务背景——Kimodo 要解决什么问题

- 动作生成任务的输入：自然语言、全身关键帧、末端位置/旋转、二维路径、二维路点等条件。
- 输出：时序连续、语义正确、动作自然并满足运动学约束的 3D 动作序列。
- 传统 text-to-motion 只能表达“做什么”，实际创作和机器人任务还需要精确控制“在哪里、何时、以什么姿态做”。
- Kimodo 的核心价值是用同一个生成模型统一文本控制与多种稀疏/稠密运动学控制。
- 下游价值：动画制作、交互式动作编辑、机器人动作生成、仿真和策略训练数据构建。

**图示：** 文本和多类约束 → Kimodo → 人体/机器人动作。

### 第 3 页：原始 Kimodo 与本次复现对象

- 介绍 Kimodo 全称和模型定位：kinematic motion diffusion。
- 官方模型覆盖 SOMA、Unitree G1、SMPL-X 等骨架以及 RP、BONES-SEED 等训练数据条件。
- 区分官方 RP 模型约 700 小时商业友好动捕数据与公开 BONES-SEED 约 288 小时数据。
- 明确本次选择的模型、骨架、数据集、代码基线和 checkpoint 基线。
- 说明为什么选择这一复现对象：公开可获得、能够形成闭环、可以与官方 checkpoint/benchmark 对比。
- 如果 V1 和 V2 数据配方都存在，说明最终生产训练使用哪个版本以及另一个版本的作用。

### 第 4 页：复现动机与项目价值

- 公开仓库主要提供推理、Demo 和 benchmark，没有发布 NVIDIA 原始训练源码。
- 仅能运行官方 checkpoint 不能回答模型如何训练、能否修改数据、能否扩展控制条件等问题。
- 本项目要获得的是自主的数据构建、训练、恢复、评测和模型迭代能力。
- 技术价值：验证论文机制和公开模型行为，补齐训练缺口。
- 工程价值：形成可迁移、可恢复、可审计的分布式训练系统。
- 后续价值：支持新的文本标注、约束分布、骨架或下游机器人任务。

### 第 5 页：复现范围、边界与成功标准

| 复现层级 | 目标 | 验收证据 |
|---|---|---|
| 原理复现 | 解释动作表示、扩散、条件注入、两级 Transformer 和损失 | 架构图、公式、shape ledger、代码映射 |
| 数据复现 | 从公开资产构建 train-ready bundle | manifest、数据统计、质量门禁、revision |
| 模型复现 | 训练态 forward、loss 和梯度路由与目标设计一致 | parity 测试、单 batch 验证、源码定位 |
| 训练复现 | Phase 1/2 可运行、续训、保存并导出 EMA | 日志、曲线、checkpoint、resume 记录 |
| 效果复现 | 当前模型相对基线改善，并与官方模型进行公平比较 | benchmark、对照实验、案例和失败分析 |
| 工程复现 | 新环境能够按文档重新部署和运行 | 容器、锁定配置、脚本、运行手册 |

- 明确不在本次范围内的内容，以及这些内容是否影响最终结论。
- 分别定义“链路跑通”“完整训练结束”“效果达到目标”的标准。
- 公开数据条件下的结果不能表述为对 700 小时 RP 模型的同条件复现。

---

## 第二部分：Kimodo 模型原理

### 第 6 页：问题形式化——模型的输入、输出和学习目标

- 输入动作序列记为 \(x_0\)，批量形状为 `[B,T,369]`。
- 文本条件提供动作语义；运动学约束提供指定帧、指定关节或 root 的目标值。
- 扩散训练从 \(x_0\) 构造带噪动作 \(x_t\)。
- denoiser 接收带噪动作、约束值、约束 mask、文本、扩散 timestep 和初始朝向。
- 模型直接预测完整 clean motion \(\hat{x}_0\)，而不是预测噪声 \(\epsilon\)。
- 训练目标是让 \(\hat{x}_0\) 同时满足动作重建、物理几何和控制条件。

### 第 7 页：模型总体结构

- 文本由 LLM2Vec 离线编码为条件 embedding。
- clean motion 经前向扩散得到 noisy motion。
- 在线约束采样器生成 observed motion 和 motion mask。
- 约束值覆盖到 noisy motion，并与 binary mask 一起进入 denoiser。
- Root Transformer 先预测全局 root trajectory。
- global root 转换为 local root 条件后，Body Transformer 预测身体动作。
- root 和 body 输出拼成完整的 369 维 \(x_0\) prediction。
- 七项损失监督模型，optimizer 和 EMA 完成参数更新与稳定权重积累。

**图示：** 用一张完整模型图串联 text、diffusion、constraint、Root、Body 和 loss。

### 第 8 页：骨架、坐标系与动作表示

- 介绍实际使用的 SOMA 骨架和当前训练投影到的 30 个关节。
- 说明 global coordinate、root-relative/local coordinate、heading 和关节旋转的关系。
- 解释首帧 root XZ 平移归零与随机 heading augmentation 的目的。
- 说明 FK 在原始旋转/root position、动作特征构建和物理空间 loss 中分别承担什么作用。
- 说明 6D rotation representation 相对欧拉角/四元数的作用。
- 必要时补充模型输入骨架与公开 SOMA77 资产之间的映射边界。

### 第 9 页：369 维动作特征——模型到底预测什么

| Feature block | 单帧维度 | 含义 |
|---|---:|---|
| smooth root position | 3 | 平滑 root 的三维全局位置 |
| global root heading | 2 | 朝向的 \((\cos\theta,\sin\theta)\) |
| local joint positions | 90 | 30 个关节的 root-relative 位置 |
| global 6D rotations | 180 | 30 个关节的全局旋转表示 |
| joint velocities | 90 | 30 个关节的全局速度 |
| foot contacts | 4 | 四通道足接触状态 |
| **合计** | **369** | global root 5 + body 364 |

- Root Transformer 输出全局 root `[B,T,5]`。
- Body Transformer 使用的 local root 为 `[角速度, XZ 平面速度, root 高度]`，共 4 维。
- 解释位置、旋转、速度和足接触同时建模的必要性。
- 展示一个动作样本由原始旋转/root position 转成 369 维特征的过程。

### 第 10 页：文本条件与 52 个 conditioning prefix

- 文本离线编码通常得到 `[B,1,4096]` 的 LLM2Vec sentence embedding。
- backbone 补充 49 个 zero extra slots，总共形成 50 个 text slots。
- 每个 stage 使用独立的 `Linear(4096→1024)` 将文本投影到 Transformer latent。
- diffusion timestep 经过 sinusoidal embedding 和 MLP，形成 1 个 time token。
- 初始 heading 的 cos/sin 经投影形成 1 个 heading token。
- prefix 顺序为 50 个 text slots、1 个 timestep token、1 个 heading token，再接 T 个 motion tokens。
- 解释 zero slots 经 bias、位置编码和 self-attention 后为什么能成为内部信息交换空间。
- Root 和 Body 使用同类条件，但参数并不共享。

### 第 11 页：前向扩散与 \(x_0\) 预测

- 从 clean motion \(x_0\) 均匀采样 diffusion timestep \(t\) 和 Gaussian noise \(\epsilon\)。
- 使用 cosine noise schedule 构造 \(x_t\)。
- 每个 batch 样本独立采样 timestep 和 noise。
- padding frame 不参与有效 loss 和有限差分，但 tensor 仍保持 batch 对齐。
- 模型输出是 clean motion \(\hat{x}_0\)，不是 noise prediction。
- 解释 \(x_0\) prediction 对动作约束覆盖和多项几何 loss 的便利性。
- 对比训练时的单步随机加噪与推理时的多步反向去噪。

### 第 12 页：运动学约束如何表示和注入

- 约束采样器输出两张同形张量：`observed_motion [B,T,369]` 和 `motion_mask [B,T,369]`。
- 使用下式把已知约束覆盖到 noisy motion：

  \[
  \tilde{x}_t=m\odot x_{target}+(1-m)\odot x_t
  \]

- 将 motion mask 沿 feature 维拼到模型输入，使模型知道哪些坐标是约束值。
- 约束 mask 不是 loss mask；模型仍预测所有有效帧和全部 369 维特征。
- 展示 text-only、full-body keyframe、end-effector、root path、root waypoint 等 pattern 如何映射到 mask。
- 区分稀疏约束、稠密约束、单 pattern 和双 pattern。
- 说明约束值的坐标系、时间范围和 feature slice 必须与模型表示一致。

### 第 13 页：Root Transformer——先预测全局运动轨迹

- 输入为 imputed noisy motion 369 维与完整 mask 369 维拼接，得到 `[B,T,738]`。
- 经 `Linear(738→1024)` 投影为 motion token。
- 与 52 个 prefix 拼接，形成 `[B,T+52,1024]`。
- Root Transformer：16 layers、8 heads、latent 1024、FFN 2048。
- 丢弃 prefix 对应输出，只保留 T 个 pose token。
- 经 `Linear(1024→5)` 输出 global root prediction。
- Root 虽只输出 5 维，但能够看到完整 noisy body 和完整约束 mask，因此可与身体和约束协调。

### 第 14 页：Global root 到 local root 的转换与梯度边界

- 将 normalized global root 反归一化，恢复三维位置和 heading。
- 使用长度感知的有限差分计算角速度和平面速度，不跨 padding 帧。
- 拼接 root 高度后得到 `[B,T,4]` local root，并按 local-root stats 重新归一化。
- 训练默认对 local root 执行 stop-gradient/detach。
- Body forward 数值上依赖 Root prediction，但 Body loss 默认不会经此桥更新 Root Transformer。
- 解释这一行为来自公开 denoiser 的 training-mode forward，公开报告未完整说明官方 trainer 的 autograd 细节。
- 说明如果关闭 detach，梯度路由将发生什么变化，并将其列为可验证的消融项。

### 第 15 页：Body Transformer——在根轨迹条件下生成全身动作

- 从 imputed noisy motion 去掉 5 维 global root，得到 364 维 body slice。
- predicted local root 4 维与 noisy/imputed body 364 维拼成 368 维 body base。
- 再拼接完整 369 维 mask，得到 `[B,T,737]`。
- 经 `Linear(737→1024)` 和 52 个 prefix 进入独立 Body Transformer。
- Body 输出 `[B,T,364]`，与 Root 的 5 维结果拼成 `[B,T,369]`。
- Body 不读取 Root hidden state，只读取 Root 最终预测转换得到的 local root。
- 解释这种两级结构如何解耦全局轨迹与局部姿态，同时保持二者协调。

### 第 16 页：七项训练损失与梯度路由

- 直接损失使用 valid-frame masked Smooth-L1。
- 七项损失及论文权重：
  - root position：10；
  - root heading：2；
  - joint position：10；
  - joint velocity：3；
  - joint rotation：10；
  - foot contact：4；
  - differentiable FK：5。
- root position 和 root heading 直接监督 Root Transformer。
- 其余 body direct losses 与 FK 监督 Body Transformer。
- FK loss 将预测旋转转换到物理关节空间，约束最终骨架几何。
- padding 由 valid frames 屏蔽，DDP 下按全局有效帧数归一化。
- 展示 detach 存在时和关闭时的完整梯度流向。

### 第 17 页：两阶段训练课程

| 项目 | Phase 1 | Phase 2 |
|---|---|---|
| step 范围 | 0～499,999 | 500,000～999,999 |
| 动作约束 | text-only，motion mask 为 0 | 90% constrained，10% no constraint |
| Transformer/attention/PE dropout | 0.1 | 0.0 |
| text conditioning dropout | 10% | 10% |
| 模型结构与 loss | 相同 | 相同 |

- Phase 1 先学习文本到自然动作的生成能力。
- Phase 2 在已有动作先验上学习运动学约束遵循。
- Phase 2 移除模型 dropout，避免直接覆盖到 noisy input 的约束被 dropout 干扰。
- Phase 2 在线采样论文约束大类和本项目 benchmark-oriented lane。
- 说明 phase 切换时 optimizer、EMA、global step 和 checkpoint 如何连续。

### 第 18 页：推理与反向去噪流程

- 从随机噪声动作序列开始，并准备文本、initial heading、约束值和 mask。
- 每个 diffusion step 重复执行约束覆盖和 Root/Body 两级 denoising。
- 解释模型如何在反向过程中持续保持已知约束。
- 说明 classifier-free guidance 的条件/无条件分支及 CFG scale 的影响。
- 输出 369 维动作反归一化后转换为骨架姿态和可视化/下游格式。
- 展示纯文本、关键帧、end-effector、root path 等推理输入的差异。
- 说明 diffusion steps、随机种子、CFG、动作长度对结果和成本的影响。

---

## 第三部分：从原理到代码实现

### 第 19 页：论文、公开代码与训练重建的关系

- 论文给出模型思想、动作表示语义、条件注入、两阶段课程和七项 loss。
- 公开模型代码给出 checkpoint-sensitive backbone、prefix、Root/Body forward、输入维度和 detach 行为。
- 本项目补齐 Dataset、constraint sampler、loss reduction、optimizer、DDP、checkpoint、EMA 和评测编排。
- 对每项实现标注信息来源，避免把工程补全误称为官方未公开 recipe。
- 展示一张“三层证据来源 → 当前实现”的映射图。

### 第 20 页：模型原理到源码的映射

| 原理/模块 | 主要实现 |
|---|---|
| 数据读取、裁剪与 collate | `kimodo/training/data.py` |
| 369 维动作表示 | `kimodo/motion_rep/reps/kimodo_motionrep.py` |
| normalization | `kimodo/motion_rep/stats.py` |
| Phase 2 约束采样 | `kimodo/training/constraints.py` |
| diffusion | `kimodo/model/diffusion.py` |
| prefix、padding mask 和 Transformer | `kimodo/model/backbone.py` |
| Root/Body 两级 forward 与 detach | `kimodo/model/twostage_denoiser.py` |
| 七项 loss | `kimodo/training/losses.py` |
| 训练循环 | `kimodo/training/engine.py` |
| optimizer、EMA、checkpoint | `kimodo/training/optim.py`、`ema.py`、`checkpoint.py` |

- 用关键 forward 伪代码串起真实调用顺序。
- 标出每个模块的输入/输出 shape。
- 标出对应的 parity test、单元测试或 smoke test。

### 第 21 页：一次训练 step 的真实执行路径

1. Dataset 根据 manifest row 加载 motion、text embedding 和 metadata。
2. Collate 完成 padding，生成 clean motion、valid frames 和 lengths。
3. 文本 conditioning dropout 与 Phase 2 constraint sampling。
4. 采样 timestep/noise，生成 noisy motion。
5. 约束值覆盖并拼接 mask。
6. Root forward → global-to-local → detach → Body forward。
7. 拼接 \(\hat{x}_0\)，计算七项 loss numerator 和 valid-frame denominator。
8. DDP reduction、backward、gradient clipping 和 finite check。
9. optimizer step、global step 更新、EMA、日志与 checkpoint。

**图示：** 使用代码调用链和 tensor shape 双层泳道图。

### 第 22 页：实现一致性与差异清单

| 项目 | 论文/官方 | 本次实现 | 一致性结论 | 影响 |
|---|---|---|---|---|
| motion representation | `[填入依据]` | 369 维 SOMA30 | `[待填]` | `[待填]` |
| prefix 和 backbone | `[填入依据]` | `[当前实现]` | `[待填]` | `[待填]` |
| Root/Body forward | `[填入依据]` | `[当前实现]` | `[待填]` | `[待填]` |
| loss 与权重 | `[填入依据]` | `[当前实现]` | `[待填]` | `[待填]` |
| 数据规模与构成 | `[官方条件]` | `[本次条件]` | 不同/部分一致 | `[待填]` |
| 训练超参数 | 部分未公开 | `[本次配置]` | 工程补全 | `[待填]` |
| 评测版本 | `[官方版本]` | `[当前版本]` | `[待填]` | `[待填]` |

- 明确哪些部分达到结构等价、哪些只能做合理重建、哪些仍缺少证据。

---

## 第四部分：训练数据管线

### 第 23 页：数据来源、组成与合规边界

- 列出实际使用的数据集、版本、许可证、下载来源和访问条件。
- 说明 BONES-SEED 的动作时长、样本数、文本覆盖率和骨架构成，以最终数据报告为准。
- 列出基础文本、事件级文本、timeline multi-prompt 和 LLM 增强文本的组成。
- 说明 train/validation/test 或 benchmark 的隔离方式，防止数据泄漏。
- 说明数据和模型资产不能直接提交到仓库的原因，以及 PVC/外部存储的管理方式。
- 明确训练结论只适用于实际使用的数据版本与过滤规则。

### 第 24 页：离线数据构建总流程

- 资源下载与完整性校验。
- 原始 motion 解析和统一 canonical 格式。
- 骨架转换、裁剪、异常样本过滤和动作质量检查。
- 文本清洗、事件/timeline 构造和 LLM 增强。
- LLM2Vec 离线编码与 cache revision 绑定。
- 训练统计量计算：global root、local root、body normalization stats。
- 生成 manifest JSONL 和 train-ready bundle。
- 执行质量门禁、发布 bundle，并记录 provenance。

**图示：** 原始数据 → canonical assets → feature/text cache → manifest/bundle。

### 第 25 页：Manifest 与 train-ready bundle 合同

- Manifest 是样本索引和协议，不是大 tensor 本身。
- 每行应关联 motion、text embedding、裁剪区间、样本 ID、mixture source 和 revision 信息。
- motion、text cache 和 normalization stats 通过显式路径和版本绑定。
- V1/V2 通过同一 manifest 合同进入训练，避免 trainer 绑定某一种数据构建方式。
- bundle 搬迁后能够重新校验和 rebind，不需要重做全部预处理。
- 展示一条真实 manifest row 和对应文件，但隐去敏感路径或凭证。

### 第 26 页：单个样本如何变成模型输入

- 读取 canonical local rotations 和 root positions。
- 映射到训练骨架并按 manifest 时间段裁剪。
- 必要时随机裁剪到最大 300 帧。
- FK 和 feature 构造得到 raw motion `[Ti,369]`。
- 首帧 root XZ 平移、随机 heading augmentation 和 normalization。
- 加载 cached text embedding `[1,4096]`。
- 返回 clean motion、text、length、initial heading、id 和数据来源等字段。
- 展示每一步前后的 shape 和数值语义。

### 第 27 页：Batch、变长序列与 DDP 数据语义

- Collate 按当前 batch 最大长度 padding，得到 `[B,T,369]`。
- `valid_frames [B,T]` 同时用于 attention padding、loss mask 和 lengths 计算。
- `lengths` 用于约束采样和 global-to-local finite difference。
- 文本 embedding padding 与 motion padding 分开处理。
- DistributedSampler 按 rank 分配 manifest rows。
- 变长动作下不能简单按 batch size 平均 loss，必须按全局有效帧数归一化。
- 说明 shuffle、seed、epoch 和 resume 后如何维持可复现的数据顺序。

### 第 28 页：Phase 2 在线约束采样与数据混合

- 展示论文约束大类、单/双 pattern 和无约束样本的顶层概率。
- 展示 sparse keyframe 数量、dense span、root heading 等采样规则。
- 展示 V2 benchmark-oriented lane 与 13 个 leaf 的作用。
- 说明 observed motion 和 mask 如何从 clean motion 在线构造。
- 在线约束不需要为每种模式提前保存一份 motion，从而避免 bundle 爆炸。
- 区分论文明确公开的概率、本仓库选择的概率以及用于验证的临时配置。
- 展示不同约束 pattern 的 mask 可视化。

### 第 29 页：数据质量门禁与数据统计

- 检查文件存在性、manifest 唯一性、shape、dtype、NaN/Inf 和长度范围。
- 检查骨架、关节顺序、FPS、坐标系和 rotation validity。
- 检查文本空值、重复、时间段覆盖和 cache revision。
- 检查训练/验证泄漏以及 mixture source 比例。
- 展示构建前后样本数、帧数、动作时长、文本类型和过滤原因。
- 展示失败样本的隔离、修复和重新发布流程。
- 数据统计必须对应最终用于训练的 bundle，而不是中间版本。

---

## 第五部分：训练系统与核心成果

### 第 30 页：训练配置与超参数

- 模型尺寸、层数、hidden size、heads 和参数量。
- 最大序列长度、每 rank batch、world size、gradient accumulation 和 global batch。
- BF16/FP16、gradient clipping、随机种子和 dropout。
- Adam-atan2、learning rate `2e-5`、betas、lambda、weight decay 和 scheduler 设置。
- Phase 1/2 steps、checkpoint 周期、EMA 周期与 decay。
- 文本 dropout、约束分布和 loss 权重。
- 列出官方公开值、本次配置值和选择依据，不只罗列最终数字。

### 第 31 页：单机、单节点 DDP 与多机训练架构

- 本地/单卡 smoke 用于验证数据、forward、backward 和 checkpoint。
- 两卡或单节点 DDP 用于验证梯度同步、loss 归一化和 resume。
- 生产训练拓扑：实际节点数、GPU 型号和数量、网络、共享 PVC 和容器。
- 每个 rank 的数据、日志和 checkpoint 责任。
- NCCL/RDMA/IB 由哪个基础设施层负责，训练代码负责什么。
- 容器中包含代码和依赖，数据、token 和大型模型资产从外部挂载。
- 展示 rank、sampler、all-reduce、checkpoint writer 和共享存储关系。

### 第 32 页：Optimizer、数值稳定性与 EMA

- 解释 Adam-atan2 与本次使用参数的来源和边界。
- BF16 训练不启用 GradScaler，FP16 配置才启用。
- 每个有效 optimizer step 前执行 global gradient norm 和 finite check。
- 非有限 loss/gradient 时保存诊断 checkpoint 并 fail fast。
- EMA 每 10 个成功 step 更新，decay 为 0.995，跨 Phase 1/2 不重置。
- 浮点 state 插值，非浮点 buffer 复制；EMA 覆盖完整 model state。
- 推理和 benchmark 使用 EMA state，说明与瞬时训练权重的差别。

### 第 33 页：Checkpoint、断点续训与实验状态恢复

- checkpoint 中保存 model、EMA、optimizer、global step、phase、随机状态和 provenance。
- 说明 latest/periodic/diagnostic/final checkpoint 的用途。
- resume 后验证 step、optimizer state、EMA、随机数和数据顺序是否连续。
- DDP 下只允许指定 rank 写主 checkpoint，避免并发覆盖。
- run lock 防止同一输出目录被多个训练任务误用。
- 展示中断前后 loss、step 和权重的一致性验证。
- 说明 checkpoint 容量、保留策略和归档方式。

### 第 34 页：日志、监控与训练期间评测

- 训练日志：total loss、七项 loss、learning rate、gradient norm、吞吐和 step time。
- 系统监控：GPU 显存/利用率、CPU、I/O、网络通信和错误日志。
- 数据监控：有效帧数、序列长度、mixture source 和约束 pattern 分布。
- 定期导出 EMA checkpoint 并由旁路 benchmark worker 评测。
- 评测结果与 checkpoint step、配置和代码 revision 绑定。
- 设定异常告警：loss 非有限、吞吐下降、checkpoint 失败、指标退化。
- 展示实际监控界面、曲线或日志证据。

### 第 35 页：端到端完成度与可运行证据

| 环节 | 状态 | 证据 | 未完成项/风险 |
|---|---|---|---|
| 数据下载与 bundle 构建 | `[待填]` | 资源记录、manifest、质量报告 | `[待填]` |
| 单卡 smoke | `[待填]` | 日志、loss、checkpoint | `[待填]` |
| DDP 与断点续训 | `[待填]` | 测试、step/状态一致性 | `[待填]` |
| Phase 1 完整训练 | `[待填]` | 曲线、EMA checkpoint | `[待填]` |
| Phase 2 完整训练 | `[待填]` | 曲线、EMA checkpoint | `[待填]` |
| 官方 benchmark | `[待填]` | 配置、结果 bundle | `[待填]` |
| 推理与可视化 | `[待填]` | 视频、motion NPZ、约束文件 | `[待填]` |

- 每项状态必须链接到可追溯证据。
- “代码已实现”“smoke 跑通”“完整训练完成”“效果达标”分别报告，不合并成一个完成度。

### 第 36 页：端到端演示链路

- 展示最终实验配置、代码 revision 和数据 manifest。
- 展示数据校验与一个 batch 的关键 shape。
- 展示训练日志、checkpoint 目录和 resume 记录。
- 加载最终 EMA checkpoint。
- 分别展示 text-only、full-body、end-effector、root path/waypoint 等生成结果。
- 展示同一 checkpoint 的自动 benchmark 结果。
- 说明演示输入、随机种子、CFG、diffusion steps 和输出文件，确保结果可重复。

---

## 第六部分：效果与验证证据

### 第 37 页：验证策略——如何证明实现正确

- 数据层：shape、统计量、时间裁剪、骨架转换和 cache revision 测试。
- 模型层：官方 checkpoint 加载、forward shape、prefix 顺序和 padding mask 测试。
- 论文核心 parity：369 维表示、约束覆盖、两级 forward、七项 loss 和 phase 行为。
- 训练层：单 batch 过拟合、smoke、梯度有限性、EMA 和 checkpoint 测试。
- 分布式层：单卡/多卡 loss 语义、global valid-frame normalization 和 DDP resume。
- 评测层：官方基线重跑、指标版本与历史修复对齐。
- 工程层：配置合同、入口脚本、容器启动和资源路径测试。

### 第 38 页：Benchmark、评估协议与比较对象

- 比较对象：随机初始化/未训练模型、Phase 1、Phase 2、官方 checkpoint 和必要的外部 baseline。
- 评测数据：Kimodo Motion Generation Benchmark 与固定内部验证集。
- 文本语义指标：TMR 等实际启用指标。
- 约束遵循指标：root、end-effector、关键帧、路径和旋转误差。
- 动作质量指标：foot skate、失败率、有效样本率和人工审阅。
- 固定随机种子、样本数、动作长度、diffusion steps、CFG 和评估代码版本。
- 明确训练集/验证集/benchmark 的隔离关系和公平比较条件。
- 指标必须标注越高越好或越低越好。

### 第 39 页：官方基线复验

- 使用官方 checkpoint 在当前代码和当前 benchmark 版本下重新生成结果。
- 将复验结果与官方报告结果对比，确认评测链没有系统性偏差。
- 记录模型版本、benchmark revision、推理参数和指标实现修复。
- 如果无法复现官方指标，先分析评测版本、模型版本和生成配置，不直接归因于训练模型。
- 官方基线复验结果作为后续训练 checkpoint 对比的统一参照。

### 第 40 页：核心定量结果

| 模型/阶段 | 文本语义指标 | 约束指标 | 动作质量指标 | 有效率 | 训练成本 |
|---|---:|---:|---:|---:|---:|
| 官方 checkpoint | Overview FID 0.119；R@3 见 Official v1 表 | EE ~4.7–4.9 cm；全身 ~4.0 cm | skate / contact 见 Official 表 | `[待填]` | 未公开/已知值 |
| 随机初始化或早期基线 | `[待填]` | `[待填]` | `[待填]` | `[待填]` | `[待填]` |
| Phase 1 checkpoint（500k） | content Overview FID 0.110 / R@3 96.7%（当时文本最好点） | 全身仍 ~66 cm（尚未学约束） | `[待填完整表]` | `[待填]` | 500k step |
| Phase 2 当前可报点（wd03 **750k**） | Overview FID **0.116** / R@3 **98.9%** | with-text 全身 4.83 cm，EE **9.58 cm** | skate 4.46 cm/s，contact 0.969 | `[待填]` | 16×H200 至 750k |
| Phase 2 800k（已坍塌，勿当成果） | Overview FID 0.137 / R@3 96.7% | 全身 8.57 cm，EE 12.49 cm | skate 5.37，contact 0.926 | — | — |
| 与官方差距 | Overview 文本已打平或略优 | **EE 仍约 2×** | skate 略差 | — | — |

数字来源与消融历程见 [`training_collapse_ablation.zh-CN.md`](training_collapse_ablation.zh-CN.md)。750k 是目前能对外报的 Phase 2 点，不是 1M 完成。

- 同时报样本数、均值、方差或置信区间，不只展示最好一次。
- 将总体指标拆到各约束类型，避免平均数掩盖失败类别。
- 对每个差距给出数据、训练、推理或评测层面的证据链。

### 第 41 页：训练曲线与收敛分析

- 展示 total loss 和七项子 loss 的 train/validation 曲线。
- 标记 warm start、resume、Phase 1/2 切换、异常中断和重要配置变化。
- 展示 learning rate、gradient norm、EMA 与非 EMA benchmark 的变化。
- **必须标出两次坍塌是同一条路：** 原线 690k 已抹干净、696.8k loss 锁死 ~10.4；kf-smooth 695k 刚翻脸；wd03 过了 696.6k 后约 800k 再炸（jsonl 806340）。wd=0.3 推迟发病，没有换病。
- 将 benchmark 随 checkpoint 的变化与训练 loss 放在一起：wd03 Overview 700k 0.119 → 750k 0.116 → 800k 0.137。质量在 gnorm 锁死前已经开始退。
- 检查不同 rank、不同 seed 或重复实验的稳定性；冻 K=7 换 seed 仍在 696.6k 起飞，说明不是单次坏 batch。
- 细节表和对照 run 目录见 [`training_collapse_ablation.zh-CN.md`](training_collapse_ablation.zh-CN.md) 文首汇报口径和第 15 节建议页 A/C。

### 第 42 页：消融与因果验证

- text-only Phase 1 与完整 Phase 2：验证约束课程的作用。
- 无 extra text slots 与 50 text slots：验证条件容量的作用。
- 关闭某项关键 loss：分析各 feature 和物理约束的贡献。
- EMA 与非 EMA 权重：验证生成稳定性差异。
- root-to-body detach 开/关：验证梯度边界选择。
- 官方约束 lanes 与 V2 benchmark-oriented lane：验证数据分布设计。
- 不同 CFG、diffusion steps、序列长度和数据规模的敏感性。
- **Phase 2 中后期坍塌消融（已做，汇报必须单列）：** 平滑课表、LR 1e-5、800 cap、冻 K=7、换 seed、wd=0.3、LR 3e-6。结论：延迟家族穷尽；近因是末层注意力残差对冲；发病时七项损失不发奖金；锁死后才离不开抹掉。对照表见 [`training_collapse_ablation.zh-CN.md`](training_collapse_ablation.zh-CN.md) 第 1 节总表和第 15 节。
- 对无法做完整规模的实验，标注为缩小规模趋势验证，不外推成最终性能结论。

### 第 43 页：定性结果与约束可视化

- 纯文本生成：语义一致性、动作自然性和多样性。
- 多段文本/timeline：不同时间区间的语义切换和动作过渡。
- full-body keyframes：指定帧姿态命中情况。
- end-effector：手脚位置/旋转遵循情况。
- root path/waypoints：轨迹跟随和朝向一致性。
- 组合约束：文本与多个运动学条件同时存在时的协调能力。
- 每类同时展示输入条件、官方结果、复现结果和定量误差。

### 第 44 页：失败案例与误差归因

- 文本语义偏差或复杂描述遗漏。
- 稀疏约束之间的不自然过渡。
- end-effector 漂移或旋转误差（750k with-text EE **9.58 cm**，官方约 4.7–4.9 cm，这是效果差距，不是坍塌）。
- root path 命中但身体动作不自然。
- foot sliding、穿地、自碰撞或异常速度。
- 长序列退化、边界帧不连续和多 prompt 切换失败。
- **训练动力学失败（与上面质量差距分开讲）：** 最后一层 `x` 与 `attn(x)` 对着干（750k 余弦翻到 0，救援 800k −0.82），post-norm LN 放大回传，Adam-atan2 不收步。不是抄关键帧，不是 FK/6D 奇点，不是 800k 的约束时钟。
- 将原因归入数据覆盖、约束分布、模型容量、loss、采样参数或评测实现；坍塌归入 **post-norm + 尺度不变优化器**，不要写成「K=8 容量不够」。
- 每类失败提出对应的验证实验和修复方向。坍塌：可报点是 750k。不要把拿掉 clip、pre-norm 或第八项损失写成论文复现。见消融文档第 13 节。

### 第 45 页：结果结论与可信度边界

- 已被数据支持的结论是什么。坍塌专项见 [`training_collapse_ablation.zh-CN.md`](training_collapse_ablation.zh-CN.md) 文首汇报口径：结构能训、696.6k 与 750k 后是同一条抹掉路、延迟家族穷尽、750k 是质量点、近因是残差对冲、大梯度是 1/σ、发病时损失不发奖金、发病是慢漂。不要把拿掉 clip、最小散度或加第八项损失写成根因。
- 仅在当前骨架、数据、训练 step 和推理配置下成立的结论是什么。
- 哪些指标已接近官方模型，哪些仍存在显著差距：Overview 文本已打平；**EE 仍约 2×**；1M / K=20 **未跑通**。
- 哪些差距可以由数据规模或公开信息缺失解释，哪些仍需实验确认。官方同结构到过 K=20，未公开优化器/正则，不能写成「16 层必然炸」。
- 当前结果是否足以证明训练链路正确，是否足以证明效果复现，分别回答。链路（数据→训练→评测）已跑通到 750k；效果复现未完成（约束差、课程未到 20、800k 再炸）。
- 主动列出负面结果和不确定性，不只展示最佳案例。不要把救援 795k 或父 run 800k 当成果图。

---

## 第七部分：方案与实现质量

### 第 46 页：关键技术决策与依据

| 决策 | 本项目方案 | 依据与收益 | 代价/边界 |
|---|---|---|---|
| 训练代码来源 | clean-room reconstruction | 公开仓库缺少原始 trainer | 不等同官方 recipe |
| 数据接口 | manifest + 可迁移 bundle | 统一 V1/V2、支持迁移和审计 | 需严格校验 revision |
| 文本处理 | 离线缓存 embedding | 避免训练时重复运行文本编码器 | cache 必须绑定 encoder 版本 |
| 训练精度 | BF16 | H200 上节省显存并保持稳定 | 依赖硬件支持 |
| 分布式 loss | 全局有效帧归一化 | 变长动作跨 rank 语义一致 | 实现复杂度增加 |
| 权重选择 | EMA checkpoint | 提高扩散生成稳定性 | 增加状态和存储开销 |
| 错误策略 | finite check + fail fast | 防止无效训练长期运行 | 需要完善恢复流程 |

- 每个关键选择说明候选方案、采用依据、验证方法和适用边界。
- 标明选择来自论文、公开代码还是本项目工程判断。

### 第 47 页：实现质量与测试体系

- 配置与代码分离，训练行为由显式 YAML 和 overlay 控制。
- 资源 catalog 与机器路径分离，凭证不写入配置、脚本或镜像。
- 模块边界覆盖数据、模型、训练、评测和资源管理。
- 单元测试、合同测试、paper parity、DDP resume、benchmark parity 和工程回归测试组成验证矩阵。
- 对 checkpoint-sensitive 结构建立加载测试，防止无意修改模型拓扑。
- 对数据 revision 和文本 cache 建立一致性检查。
- 对公共入口和内部审计工具划定边界，避免交付面不断膨胀。

### 第 48 页：性能、成本与可扩展性

- 报告数据预处理耗时、训练吞吐、GPU 利用率、峰值显存和 checkpoint 存储。
- 报告单 step、单 epoch/固定 step 区间和完整训练的实际资源消耗。
- 分析 text cache、I/O、变长 padding、通信和 checkpoint 对吞吐的影响。
- 说明从单卡到多机的 batch 换算和学习语义是否保持一致。
- 说明增加数据、序列长度、骨架或约束类型时的扩展点和成本。
- 成本只报告实际测量值；预测值单独标注假设和计算方法。

### 第 49 页：风险、合规与适用边界

- 技术风险：官方 recipe 未完全公开、训练规模大、数据和评测版本可能变化。
- 数据风险：数据质量、文本噪声、分布偏差、训练/评测泄漏。
- 工程风险：多机通信、共享存储、checkpoint 容量和长任务恢复。
- 模型风险：动作不物理、自碰撞、约束冲突和长序列不稳定。
- 合规风险：BONES-SEED gated license、官方模型许可、第三方文本模型许可和凭证管理。
- 当前结论只覆盖实际复现的数据、骨架、代码 revision、硬件和配置。
- 不把 SOMA/BONES-SEED 结果直接外推到 RP 数据、G1、SMPL-X 或真实机器人控制。

---

## 第八部分：协作、留痕与交付

### 第 50 页：项目过程与关键决策留痕

- 展示项目阶段：论文/代码理解、数据构建、训练实现、验证、生产训练和评测。
- 说明数据准备、模型实现、基础设施、训练运行和评测验收的责任边界。
- Git commit/tag、实验 ID、配置 hash、manifest revision、镜像和 checkpoint 一一对应。
- 关键技术决策记录包含背景、候选方案、选择依据、风险和验证结果。
- 失败实验保留配置、日志、结论和后续处理，避免重复踩坑。
- 重要修复说明影响范围，并重新运行对应验证。

### 第 51 页：交付资产清单

- 环境：`Dockerfile`、锁定依赖、镜像版本和启动说明。
- 资源：catalog、paths 示例、下载/校验脚本和许可证说明。
- 数据：V1/V2 bundle、manifest、normalization stats、质量报告和 provenance。
- 模型：训练代码、配置、checkpoint、EMA 导出和模型加载说明。
- 训练：单卡、多卡、多机脚本，resume、监控和故障处理手册。
- 评测：官方基线、benchmark 配置、指标报告、生成结果和审阅记录。
- 文档：数据配方、完整张量流、多机部署、benchmark 监控和代码边界。
- 说明哪些资产进 Git，哪些放共享存储，哪些只能通过受控地址获取。

### 第 52 页：他人如何从零接手并复跑

1. 获取代码并确认 commit/tag。
2. 准备容器或锁定环境。
3. 配置机器本地资源路径和访问权限。
4. 下载或绑定已有数据/模型资源。
5. 校验 train-ready bundle 与 manifest。
6. 运行单卡 smoke 和测试矩阵。
7. 启动单节点/多机训练或从 checkpoint 恢复。
8. 导出 EMA checkpoint。
9. 运行固定 benchmark 和推理样例。
10. 根据实验 ID 找到配置、日志、权重和结果报告。

- 给出实际公共命令入口和文档链接。
- 说明常见失败点、排查顺序和需要联系的负责人。

---

## 第九部分：总结

### 第 53 页：对照成功标准给出最终结论

| 目标 | 结论 | 关键证据 | 尚存问题 |
|---|---|---|---|
| 模型原理可解释 | `[完成/部分完成]` | 架构、shape、loss、梯度与源码映射 | `[待填]` |
| 数据链路可复现 | `[完成/部分完成]` | bundle、manifest、统计与质量报告 | `[待填]` |
| 模型实现正确 | `[完成/部分完成]` | parity、加载、forward 与训练测试 | `[待填]` |
| 训练链路可运行 | `[完成/部分完成]` | 日志、曲线、resume、checkpoint | `[待填]` |
| 效果得到验证 | `[完成/部分完成]` | benchmark、对照、消融和案例 | `[待填]` |
| 成果可以交付 | `[完成/部分完成]` | 代码、配置、资产、文档和复跑记录 | `[待填]` |

- 最终结论必须分别回答“链路是否完整”“效果是否复现”“是否可交接”。
- 总结最重要的技术贡献、工程贡献和验证结论。
- 对未完成项给出明确后续动作和验收标准。

**结束句：**

> 本项目交付的不只是一个可推理的权重，而是一套从公开数据、模型机制到分布式训练、效果验证均可追溯和再次执行的 Kimodo 训练系统；同时对官方未公开细节和本项目工程补全保持清晰边界。

## 4. 后备明细材料

以下内容可作为详细页或答疑材料，与正文中的结论保持一致：

1. Phase 2 forward 的完整 shape ledger。
2. 369 维 feature index 和骨架关节映射。
3. 七项 loss 的公式、权重、reduction 和梯度路由。
4. Phase 2 约束 pattern、概率与 mask 示例。
5. 训练超参数、global batch 换算和硬件拓扑。
6. 论文、公开代码和本项目工程选择的逐项差异表。
7. 数据集统计、许可证、过滤规则和质量门禁明细。
8. DDP resume、finite check、checkpoint 与 EMA 验证记录。
9. Benchmark 指标定义、版本、生成参数和评估命令。
10. 全部定量结果、分组指标和统计显著性。
11. 更多成功/失败生成视频与对应约束文件。
12. 代码目录、公共命令、测试矩阵和故障排查手册。

## 5. PPT 制作规则

- 每页标题尽量直接表达结论，例如“Root/Body 两级建模把全局轨迹与局部动作解耦”，避免只写“模型介绍”。
- 一个页面只承载一个逻辑问题；内容较多时继续拆页，不为控制页数把多个问题挤在一起。
- 模型原理页统一使用颜色：root、body、text、constraint、noise 各一种颜色，并在所有图中保持一致。
- 原理图要同时给出数据语义和关键 tensor shape，不能只有模块名称。
- 代码页只展示能说明核心机制的片段，同时标出文件、类/函数和输入输出。
- 结果页标明数据版本、checkpoint step、随机种子、推理参数、指标版本和指标方向。
- 完成度只使用可核验状态：“代码存在”不等于“训练完成”，“生成过样例”不等于“效果复现”。
- 失败实验、负面案例和官方未公开边界主动展示，保证结论可信。
- 动态结果同时准备关键帧、输入约束和数值指标，避免视频与证据脱节。
