# 消融实验任务本（core10 统一数据）

目标：在同一份 core10 数据、同一评测协议下，回答四个问题——(1) direct loss 的计算域怎么选；
(2) Flow Matching 相对 DDPM 以及 FM 内部数学选择的收益；(3) V2 自然多事件文本相对 V1 机械拼接
的回报；(4) Phase 2 benchmark 13-leaf 约束 lane 的工程假设是否成立。每个实验一次只动一个变量，
全部结论以公开 benchmark proxy 指标为准，不以训练 loss 为准。

---

## 0. 共同设定（所有实验共享，先固定再开跑）

### 数据（唯一来源，不得中途更换）

| 项 | 值 |
|---|---|
| 训练集 | `kimodo-core10-v1/core10.cached.jsonl`（sha `561698ab…43ad`） |
| 规模 | 10.0h · 4,751 motions · 52,592 rows（train 46,796 / validation 5,796，take 分组零泄漏） |
| 统计量 | DDPM 侧用 `kimodo-core10-v1/stats/train-soma30-30fps`；FM 侧必须用自家 stats v5 重拟合 |
| FM 侧准入 | 需先完成 core10 → FM manifest v3 单向转换（秒→帧、逐行 motion SHA、NPZ 注入 `fps`+`semantic_contract_json`、按 FM 规则重算 text-cache key），再过 `prepare_resources.sh` 全量校验 |

### 训练预算与硬件

- DDPM（kimodo-reproduction）：默认 30k steps = 20k Phase 1 + 10k Phase 2，
  `configs/training/kimodo_soma_seed_v2_30k.yaml` + `configs/overlays/two_h200_gb512.yaml`，
  Adam-atan2、lr 1e-5、论文七项权重、Smooth-L1 beta=1、target-root FK、root→body detach。
  **例外：实验一（loss 域）定论 run 用 40k = 30k Phase 1 + 10k Phase 2**（见实验一），
  Phase 2 步数各实验保持一致。
- FM（kimodo-flowmatching）：`configs/training/soma30_base.yaml` 系，改 `phase1_steps=20000`、
  `phase2_steps=10000` 对齐预算；等 global batch（512）通过 batch/accumulation 配平。
- 两侧 backbone 容量对齐记录在案（hidden/layers/heads/text_dim），不对齐处显式写进结论的限制条款。

### 评测协议（与 core10-loss-domain-128 现有结果保持同口径）

- 128-case 官方 testsuite proxy，覆盖 content/repetition × overview/timeline_single/timeline_multi/
  constraints_withtext/constraints_notext。
- `paper_protocol`、generation batch=1、fp32 text encoder、无 postprocess。
- DDPM 生成 DDIM 100；FM 生成按实验二的 NFE 网格，主表取 NFE=100 对齐项。
- 参照线：released `Kimodo-SOMA-SEED-v1.1` 在同一 proxy 上的一次性结果（已存在于
  `kimodo-benchmark-results/core10-loss-domain-128/official-seed-v1.1`）。
- 主指标：TMR `t2m_R/R03`、`t2m_sim`、`FID/gen_gt`（content 与 repetition 分列）；
  运动质量 foot skate 系列 + `foot_contact_consistency`；约束 `constraint_*` mean/p95；
  FM 额外记录 NFE 与单样本延迟。

### 变更控制与记账

- 每个 arm 至少 seed=1234；结论关键对比补 seed=2025 做第二次（预算允许时）。
- 每个 run 在 `kimodo-benchmark-results/<experiment>/registry.jsonl` 登记一行：
  run id、config sha、data sha、stats sha、seed、EMA bundle sha、结果目录。
- 判读纪律：单指标单 run 差异小于官方基线 run-to-run 方差的，不做结论；
  content 与 repetition 方向不一致时如实分开表述。

---

## 实验一：Direct loss 计算域消融（40k 步定论）

**问题**：六项 representation direct loss 在 normalized feature domain 还是 physical domain 计算？
（FK loss 恒在物理米制域，不参与此消融。）

**预算**：本实验定论 run 使用 **40k steps = 30k Phase 1 + 10k Phase 2**。Phase 2 保持 10k 与其他
实验完全同口径，只延长 Phase 1——direct-loss 域的差异主要在文本驱动训练期显现，30k 结果可能
尚未拉开或尚未收敛稳定。

**现状**：`kimodo-benchmark-results/core10-loss-domain-128/` 已有三个 30k arm + 官方基线：
`old-physical-direct-30k`、`new-normalized-direct-30k`、`raw-normalized-direct-30k`，
以及 `comparison.json` / `*.tables.md`。30k 结果保留作趋势参考，**最终结论以 40k run 为准**。

**任务**

- [ ] 先汇总现有 30k comparison.json 成单页趋势表：normalized vs physical 在 content/repetition
  各类别的 R@3、FID、foot skate、constraint 误差差值，标注相对官方基线的差距，确认哪两个 arm
  值得进 40k 定论。
- [ ] 两个定论 arm（physical vs normalized）各跑 40k（seed=1234），结果目录
  `core10-loss-domain-128/<arm>-40k/`；方向与 30k 一致且差异超方差，即可定论。
- [ ] 若 40k 两 arm 差异仍在方差内，补 seed=2025 各一次再判；仍无差异则按「维持 physical」收尾。
- [ ] 把最终选择写回 `configs/training/` 注释与 `benchmark_v2_data_recipe.zh-CN.md`，
  说明 canonical `kimodo_soma_seed_public.yaml` 保留 physical baseline 的原因。

**判据**：normalized 域若在多数类别 R@3/FID 占优且运动质量不回退，则后续配方沿用 normalized；
差异在方差内则维持 physical（与论文表述更接近，减少一条工程假设）。

---

## 实验二：Flow Matching 数学方法消融（本任务本核心）

### 2a. 生成范式主对比：DDPM (x0 + DDIM) vs Rectified Flow (velocity + ODE)

**Arms**（同 core10、同预算、同评测）：

| Arm | 仓库 | 关键设定 |
|---|---|---|
| A1 DDPM-x0 | kimodo-reproduction | 实验一选出的 direct-loss 域，DDIM 100 |
| A2 FM-rectified | kimodo-flowmatching | `path=linear_noise_to_data`，Heun 50（NFE 100） |

**公平性规则**（写死，不许赛后调整）：
- 主表固定 NFE=100 对齐（DDIM 100 步 vs Heun 50 步）。
- 附表做 NFE 网格 {8, 16, 32, 100} 与固定延迟两条曲线；Heun n 步计 2n 次场评估，不与 n 步 DDIM 混称。
- 文本编码器、proxy、TMR 权重、精度完全一致。

**任务**

- [ ] 完成 FM 数据准入（见 §0），先跑 README 验收门 2–3：真实数据 decode/encode 检查 + 小规模
  overfit 可视化，确认没有表示层 bug 再进 30k。
- [ ] A1、A2 各跑 30k，产出 EMA bundle，接入同一 proxy 评测。
- [ ] 产出 NFE-质量曲线与延迟-质量曲线（FM 侧 Euler/Heun 都记）。

**判据**：FM 在低 NFE（8–32）区间若以明显优势追平或超过 DDIM 同预算质量，即为 FM 的核心卖点成立；
NFE=100 处两者应大体相当，若 FM 明显落后则先查训练配平而不是下结论。

### 2b. FM 内部数学选择

| 变量 | Arms | 备注 |
|---|---|---|
| 概率路径/耦合 | linear rectified（默认） vs OT-CFM minibatch 耦合 | OT-CFM 是 roadmap 明确的待消融项，需先实现（Sinkhorn 或 exact 小 batch OT） |
| 时间采样 | Uniform(0,1)（默认） vs logit-normal(0,1) | 实现成本低，社区反复报告有效，值得一测 |
| 求解器 | Euler vs Heun，NFE 匹配 | 推理期变量，同一 checkpoint 上扫，不需要重训 |

**任务**

- [ ] 实现 OT-CFM 耦合（独立 config 开关，默认关闭，带单元测试进 FM 仓库测试套件）。
- [ ] 实现 logit-normal 时间采样开关。
- [ ] 在 A2 基础上各改一个变量重训 30k：A3 = OT-CFM，A4 = logit-normal。
- [ ] 求解器对比在 A2–A4 的 checkpoint 上直接扫 NFE 网格，不重训。

**判据**：以 NFE=16 与 NFE=100 两个工作点的 R@3/FID 决定默认配方；OT-CFM 若无收益如实记录
（10h 小数据上 OT 耦合可能不显著，这本身是有价值的负结果）。

---

## 实验三：文本数据配方消融——V1 机械拼接 vs V2 自然多事件（推荐，研究价值最高的数据侧问题）

**为什么值得做**：V2 花了真金白银（LLM 生成 + 双重评审）把 19 万条 `A Then, B` 机械拼接换成
自然多事件描述，但「这笔投入在 benchmark 上换回多少 timeline_multi 提升」还没有受控证据。
公开 benchmark 的 `timeline_multi` 恰好就是 LLM 自然拼接风格，这是能直接对外讲的可比结果；
V2 bundle 里的 `stats-v1-ablation` 目录说明这条消融本来就在计划内。

**Arms**（同一批 4,751 个 motion，只换多事件文本行）：

| Arm | 多事件文本 | 构建方式 |
|---|---|---|
| B1 core10-v1 | `combined_events` 机械 `Then,`（7,292 行） | 现成 |
| B2 core10-v2 | V2 `timeline_multi_llm` 自然改写 | 需从 V2 bundle 按 core10 的 motion 集合切一个变体 |

**任务**

- [ ] 写 core10-v2 切分脚本：复用 core10-v1 的 motion 选择（保证两 arm motion 完全一致），
  full/event 行原样保留，把 combined 行替换为该 motion 集合上的 V2 timeline 行；
  行数与时长差异写进 receipt（V2 span 覆盖可能不完全等于 combined 覆盖，必须如实记录）。
- [ ] 用实验一选出的 loss 域配置各跑 30k（B1 可直接复用实验一对应 arm 的结果）。
- [ ] 评测重点看 `timeline_multi`（content 与 repetition），同时确认 overview/timeline_single 不回退。

**判据**：B2 在 timeline_multi 的 R@3 / t2m_sim 显著提升且其他类别持平，即证明 V2 数据配方
在下游成立，可写进 V2 交付的价值论证；若无差异，1M 生产训练的数据决策需要重新审视。

---

## 实验四：Phase 2 约束 curriculum 消融——paper lane vs benchmark 13-leaf lane（推荐）

**为什么值得做**：`configs/overlays/benchmark_v2_constraints.yaml` 里 25% 概率走 13-leaf coverage
lane、sparse 最大 9、幂次 0.45，这些都是文档明确标注的**工程假设而非论文事实**
（`benchmark_v2_data_recipe.zh-CN.md`、`training_phase2_dataflow.zh-CN.md` 都写了「不能表述成
benchmark train distribution parity」）。1M 生产训练已经带着这组假设在跑，需要受控证据。

**Arms**（Phase 1 完全相同，只在 Phase 2 分叉）：

| Arm | Phase 2 约束采样 |
|---|---|
| C1 paper lane | 纯论文公开五类 curriculum（1→20 sparse、arbitrary EE、foot contacts） |
| C2 benchmark lane | C1 + 25% 13-leaf coverage lane（现行 overlay） |
| C3（可选） | coverage 概率 50%，检验剂量效应 |

**任务**

- [ ] 从同一个 Phase 1 checkpoint 分叉三个 Phase 2（各 10k steps），排除 Phase 1 方差。
- [ ] 评测重点：`constraints_withtext` / `constraints_notext` 全部约束类别的 mean/p95 误差、
  `constraint_root2d_acc`，以及 text-following 三类是否回退。
- [ ] 若 C2 相对 C1 无收益，评估把 overlay 从 1M 生产配方移除的成本。

**判据**：C2 在 mixture / 13-leaf 相关约束类别误差显著下降且文本指标持平 → 保留 overlay 并在文档
中把假设升级为「有 core10 消融证据的选择」；否则回退 paper lane，减少与论文的偏离面。

---

## 执行顺序与依赖

```text
E1 定论（physical / normalized 各 1 个 40k run）
   └─→ 确定 direct-loss 域 ──┐
FM 数据准入 + 验收门 2–3      ├─→ E2a（A1 用 E1 winner 的 30k 口径新训或复用，A2 新训）
                              │      └─→ E2b（A3、A4 重训；solver 扫描免训）
core10-v2 切分脚本            ├─→ E3（B1 复用，B2 新训）
                              └─→ E4（共享 Phase 1，三个 Phase 2 分叉）
```

新训练量合计：E1 为 2 个 40k run（≈2.7 个 30k 当量，必要时再加 seed），其余约 6.5 个 30k run
（A2、A3、A4、B2、C1/C2/C3 的 Phase 2 ×3 按 1.5 个 run 计），两张 H200 上按现有 30k 吞吐排期。
评测统一走 proxy-128 流水线，结果目录按
`kimodo-benchmark-results/core10-<experiment>-128/<arm>/` 组织，与 loss-domain 现有布局一致。

## 备选池（本轮不做，记录理由）

- **x0 vs velocity 参数化（DDPM 内部）**：与 E2a 部分重叠，等 FM 主对比出结果再决定是否值得拆分。
- **两阶段 root/body vs 单阶段**：改动模型面过大，30k 预算下难以公平配平，留给专门的架构消融。
- **文本编码器替换**：牵动全部 text cache 与 TMR 口径，成本高且不影响当前两个核心问题。
- **时长桶重平衡（3–10s duration-aware crop）**：文档已标注需要单独 ablation，依赖 raw crop
  实现，排在 E4 之后。
