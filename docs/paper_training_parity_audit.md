# Kimodo 论文训练方法逐项审计

## 1. 验收口径与当前结论

本审计只把论文明确写出的计算、阶段和数值称为 `PAPER`。公开 checkpoint/config/推理代码提供但论文没有写出的事实称为 `ARTIFACT`。为了闭合训练而选择的值称为 `ASSUMED`；缺少必要资产或协议时称为 `BLOCKED`。

当前结论分成两个不能混淆的层级：**论文明确训练方法的代码与门禁为 PASS**；**完整论文实验为 BLOCKED / NOT-TESTED**。核心模型与训练数学已达到“论文明确项逐条对齐并有契约测试”，但论文明确使用的 Qwen3-32B paraphrase 和 diffusion-transition cross-motion stitching 尚无可生成资产，因此完整 paper-data profile 会主动拒绝启动。该阻断不能用普通 BONES-SEED manifest 静默绕过。

这不是 NVIDIA 官方训练源码，也不能证明私有 trainer 的未披露细节。

## 2. 本轮发现并修复的问题

1. **两阶段梯度语义**：论文 Sec. 5 p.12 说 interleaved two-stage denoiser `trains end-to-end`，但官方公开训练代码明确在 root→body conversion 使用 `no_grad + detach`。生产 profile 默认 `detach_root_for_body=true`，把 end-to-end 解释为两个 stage 在同一 forward/loss/update 中联合训练；`false` 是梯度耦合消融。
2. **两阶段计算缺少直接证据**：新增 Figure 9 tensor-level test，逐项检查约束覆盖、mask 拼接、完整输入进入 root stage、global-root 转 local-root、imputed noisy body 进入 body stage、最终 root/body 拼接及端到端梯度。
3. **数据增强可被静默遗漏**：生产配置新增 `require_paper_data_parity=true` 并接入 trainer。缺少 Qwen paraphrase、stitched transition 或其 provenance 时 fail closed。
4. **manifest provenance 太弱**：普通构建器逐行记录原始文本/时间标签来源，sidecar 明确列出缺失增强；严格 gate 校验 manifest SHA-256、Qwen3-32B 归属、prompt hash、两条不同 source motion、source time ranges、transition checkpoint hash 和 transition frame range。
5. **重采样时间轴不一致**：整数倍率分支曾忽略目标帧数，非整数分支曾用 `linspace` 把末帧强行对齐而改变播放速度。现统一使用 half-open 物理时间网格；root 线性插值、rotation quaternion SLERP。
6. **stats 覆盖不足**：曾只从每个长 clip 取一个随机 10 秒窗口。现按唯一 motion/time-span 枚举覆盖全部有效帧的不重叠窗口，每窗独立 root-origin 和稳定随机 heading，并把该策略标为论文未披露默认。
7. **multi-prompt 默认 API 错误**：修复 `num_samples=None` 导致 `range(None)`、标量帧数按 batch 而不是 prompt 段数展开的问题，并增加输入校验。
8. **论文评测口径缺失**：新增隔离的 `--paper-protocol`，计算 EE SO(3) 旋转误差、generated smooth-root mean、pelvis pointwise p95，以及完整 prompt set 的 retrieval/FID；旧公开 benchmark 指标保持不变。

## 3. 论文明确训练条款逐项映射

状态含义：`VERIFIED` 表示有当前代码、直接测试并已通过独立复验；`ASSUMED` 表示论文未披露；`BLOCKED` 表示缺少不可伪造的资产或协议；`NOT-TESTED` 表示实现或配置存在，但尚未完成论文规模实跑。

| ID | 论文条款 | 当前实现证据 | 状态 |
|---|---|---|---|
| DATA-01 | Rigplay 约 700h、170 subjects | 本工程真实目标是 BONES-SEED/SOMA30；两者在文档和配置中分开 | `BLOCKED`：私有 RP 不可用 |
| DATA-02 | full motion、single/combined atomic sub-clips | `manifest_cli.py` 生成 full、single event、相邻 combined event | `VERIFIED`；组合长度/比例 `ASSUMED` |
| DATA-03 | Qwen3-32B 按统一 `A [subject]...` 结构、多细节层级改写 | 严格 schema/gate 已有；无官方 prompt/revision 与生成资产 | `BLOCKED` |
| DATA-04 | 随机跨 motion 拼接，用 non-augmented diffusion model 生成短 transition | 严格 schema/gate 已有；无 transition checkpoint、长度和生成协议 | `BLOCKED` |
| DATA-05 | 按预设分布混合原始/增强 motion 与文本 | repeat 权重和实际来源可追踪 | 官方比例 `ASSUMED` |
| PRE-01 | motion 最长裁到 10 秒 | `MotionManifestDataset` 随机裁剪到 `fps*10` | `VERIFIED` |
| PRE-02 | 第一帧 smoothed-root 位于 XZ 原点 | `translate_2d_to_zero` | `VERIFIED` |
| PRE-03 | 第一帧 heading 随机化 | uniform target heading 后 `rotate_to` | `VERIFIED`；分布形状 `ASSUMED` |
| REP-01 | `[r_p,r_a,j_p,j_v,j_a,f]` 语义、smoothed root、global 6D rotations、contact | 公开 motion-representation 实现与 checkpoint shape | `VERIFIED/ARTIFACT` |
| TXT-01 | LLM2Vec 4096D | text cache 与 model contract 强制宽度 | `VERIFIED`；具体 revision 必须外部固定 |
| TXT-02 | 1 个文本 embedding + 49 个零 token | backbone 固定到 50 text slots | `VERIFIED` |
| STAGE-01 | `x_tilde=m*x_tgt+(1-m)*x_t`，再 concat mask | `TwostageDenoiser.forward` | `VERIFIED` tensor oracle |
| STAGE-02 | root stage 看完整 noisy/imputed motion，输出 global root | root model input/output contract | `VERIFIED` tensor oracle |
| STAGE-03 | global root prediction 用有限差分转 local `[heading vel, planar vel, Y]` | `global_root_to_local_root` | `VERIFIED/ARTIFACT`；差分边界未披露 |
| STAGE-04 | predicted local root 与 `x_in` body features进入 body stage | body input contract | `VERIFIED` tensor oracle |
| STAGE-05 | 最终输出 concat predicted global root 与 predicted body | output contract | `VERIFIED` |
| STAGE-06 | 两个 transformer，各 16 layers、8 heads、latent 1024 | production config + official strict-load shape | `VERIFIED`；FFN/GELU/post-norm 为 `ARTIFACT` |
| STAGE-07 | interleaved two-stage joint training | production profile `detach_root_for_body=true` | `VERIFIED-CODE`；`false` 梯度耦合路径另有测试 |
| DIFF-01 | DDPM；每步均匀采样 `t` 和 Gaussian noise；预测 clean `x0` | engine + `Diffusion.q_sample` | `VERIFIED` formula oracle |
| DIFF-02 | `T=1000` | config validation 固定 1000 | `VERIFIED` |
| LOSS-01 | 六项 representation Smooth-L1 + FK Smooth-L1 | `KimodoLoss` | `VERIFIED` |
| LOSS-02 | 权重 `[10,2,10,3,10,4,5]` | production config + loss-weight test | `VERIFIED` |
| LOSS-03 | variable-length batch 的 loss 必须 mask | valid-frame numerator/global denominator | `VERIFIED`，含不等长 accumulation 等价测试 |
| CUR-01 | Phase 1 500k text-only，无 constraint | global-step curriculum | `VERIFIED` |
| CUR-02 | Phase 2 500k，text + sampled constraints | global-step curriculum | `VERIFIED` |
| CUR-03 | 五类 pattern：full-body sparse、hands/feet sparse、root sparse、root dense、contact sparse | `ConstraintCurriculumSampler` | `VERIFIED-SEMANTIC`；family 内精确分布 `ASSUMED` |
| CUR-04 | Phase 2 中 10% 无 constraint、25% 两种 pattern | categorical branch | `VERIFIED` 统计测试 |
| CUR-05 | sparse 最大 keyframe 数从 1 线性增到 20，偏向较少 | linear cap + `1/k` sampler | 线性/偏少 `VERIFIED`；`1/k` 为 `ASSUMED` |
| REG-01 | Phase 1 dropout 0.1；Phase 2 去掉 dropout | 动态更新 transformer/attention/PE dropout | `VERIFIED`；覆盖 PE 为 `ASSUMED` |
| REG-02 | 两阶段 text input 均 10% dropout | engine 独立采样并记录 CFG branch | `VERIFIED` |
| OPT-01 | Adam-atan2，lr `2e-5` | vendored optimizer + numeric oracle | `VERIFIED`；lambda/betas/WD/schedule `ASSUMED` |
| EMA-01 | decay 0.995，每 10 optimizer step 更新，推理用 EMA | trainer + export bundle | `VERIFIED` |
| SCALE-01 | global batch 2048，16xA100-80GB，30fps | production config 为 local 128，16 rank 时 2048 | `NOT-TESTED`：未做 16-GPU/1M-step 运行 |

## 4. 论文未披露但训练闭环必须选择的实现

下列项目不得写成官方值；它们必须保留配置、写入 resolved config/checkpoint provenance，并在有真数据后做消融：

- direct loss 使用 physical 还是 normalized domain；
- Smooth-L1 beta、分量 reduction、FK root convention；
- cosine beta schedule 的 offset/max-beta；
- Adam-atan2 lambda/betas/weight decay、warmup/scheduler；
- BF16、gradient clipping、seed、microbatch/accumulation；
- Phase 边界是否连续保留 optimizer/EMA；
- 五类 constraint family 概率、hand/foot 子集规则、position/rotation 组合概率；
- sparse count 的 `1/k`、dense path 20%-80%、heading 0.5；
- normalization stats 的窗口、旋转采样与去重策略；
- 数据 source/text mixture、Qwen prompt/revision/temperature/seed、transition 长度和选择规则。

当前默认采用可复现、可覆盖和 fail-closed 的工程实现；“业内合理”不等于“已验证最优”。真正选择默认值需要至少运行预先定义的 proxy ablation，并且不能在论文私有 test set 上调参。

## 5. 严格 paper profile 与工程 baseline

`configs/training/kimodo_soma_seed_reproduction.yaml` 是严格方法 profile：

- `paper_method_strict=true`，覆盖论文明确数值和语义的 override 会被配置校验拒绝；
- `model.detach_root_for_body=true`；
- `data.require_paper_data_parity=true`；
- 缺少 paper augmentations 或 provenance 时拒绝训练。
- 运行时必须是 16 ranks，且 `world_size * local_batch * accumulation = 2048`；strict profile 禁止用 `max_steps_override` 缩短论文 1M-step 训练。

若使用普通 BONES-SEED 数据调试训练代码，必须显式覆盖：

```bash
--set paper_method_strict=false --set data.require_paper_data_parity=false
```

此时输出只能标为 `engineering reconstruction / public-data baseline`。

## 6. 当前验收证据与不能通过的 gate

已经有单元/集成证据覆盖：Figure 9 两阶段 tensor 数据流和梯度、DDPM forward 公式、Eq. (1) loss/权重、五类约束、phase/mix 频率、重采样、stats 全覆盖、严格 manifest gate、tiny 两阶段训练、单机与 2-rank resume、官方 checkpoint strict load。

最终集成自检为 `50 passed, 1 skipped`，官方 1.1 GB checkpoint gate 单独为 `2 passed`。独立 verifier 已在最终文件状态上复跑严格配置门禁和完整测试套件，并将“论文明确训练方法的严格代码门禁与文档披露”审批为 `PASS`；论文完整数据、算力与私有评测实验仍为 `BLOCKED/NOT-TESTED`。严格 manifest gate 只校验自声明 schema、hash 与 provenance，不能证明未公开 prompt、mixture 或 transition 协议与官方私有 recipe 相同。

以下仍不能标为 `verified`：

- 真 BONES-SEED DataLoader/一步训练；
- Qwen3-32B paraphrase 生成质量；
- non-augmented transition model 的训练和 stitched asset 生成；
- 16xA100-80GB、global batch 2048、完整 1M steps；
- 私有 Rigplay 训练和论文约 5k test suite；
- paper evaluator 已通过合成数值 oracle，但未在论文私有约 5k test suite 上运行，因此论文表格数值仍不可验证。

因此最终合格措辞是：**核心训练方法按论文明确条款完成实现与合成验证；完整数据方法和论文数值复现仍受外部资产/算力阻断。**
