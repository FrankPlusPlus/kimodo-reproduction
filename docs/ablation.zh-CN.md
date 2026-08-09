# Kimodo 消融实验记录

本文记录 Core10 工程验证线上的消融实验设计、结果与解读。

**主评测资产（2026-08-09 起）：** 分层 10% 官方 testsuite 子集
`kimodo-benchmark-stratified-10pct`（2,269 cases，paper protocol，100 diffusion steps，
generation batch size 1，text encoder fp32）。

**历史快速 proxy（128 cases）：** 仍保留于 `kimodo-benchmark-proxy-128`，仅用于训练期
~5 min 趋势监控；**不再作为消融结论依据**。

---

## Benchmark 资产：分层 10% 官方子集（主评测）

### 为何替换 128 proxy

128-case proxy 的问题（见下文「历史 128 proxy」）：

- Text2motion 每组仅 4 motions → R@3 顶格、FID 与全量差 10× 以上
- 约束类仅 0.78%/组 → Official 绝对 cm 与 NVIDIA 全量表偏差 10–60%
- 组内 case 选取脚本未入库 → 不可完全复现

新子集目标：**在可接受的 eval 成本下，尽量逼近官方 22,474 全量 aggregate 指标**，同时保持
**同协议下的消融可比性**。

### 采样策略（stratified proportional，可复现）

| 参数 | 值 |
|---|---|
| 母集 | `kimodo-benchmark-metadata/testsuite`（22,474 cases） |
| 采样率 | **10% / 叶子组**（`rate=0.10`） |
| 组内下限 | constraint 组 `min=10`；text2motion 组 `min=40`（10% 已更高，实际由 rate 决定） |
| 随机性 | **确定性**：`sha256(seed, leaf_group, case_id)` 排序取 top-k |
| Seed | `20260809` |
| 输出规模 | **2,269 cases**（≈10.1% of full） |
| 叶子组 | 58 组全覆盖，每组约 10%（constraint 26/256，overview 92/917 等） |

与 128 对比：

| | 128 proxy | stratified 10% |
|---|---:|---:|
| Cases | 128 | 2,269 |
| Text overview (content) | 4 | 92 |
| Constraint 组 (content/path_2dpos) | 2 | 26 |
| 单次 eval（单卡估时） | ~5 min | ~90 min |
| 对标 NVIDIA 全量 | 不可靠 | **设计上逼近**（需 Official 跑分验证） |

### 构建流程

```bash
export KIMODO_BENCHMARK_METADATA=/path/to/kimodo-benchmark-metadata/testsuite
export KIMODO_SEED_DATASET=/storage/data/metaiot_data/yzt/seed/soma_uniform
export KIMODO_BENCHMARK_PROXY128=/path/to/kimodo-benchmark-proxy-128  # 复用已有 gt

bash scripts/build_benchmark_stratified_proxy.sh
```

步骤：

1. **`kimodo/devtools/benchmark_subset_cli.py`**：按上表策略生成 case 列表，拷贝
   `meta.json` / `seed_*.json`；从 128 proxy **预拷贝已有 `gt_motion.npz`**（128 例免重建）
2. **`benchmark/create_benchmark.py`**：对其余 ~2,141 例从 BONES-SEED 生成 `gt_motion.npz`
   与 `constraints.json`（8 workers，约 30–60 min）
3. 写入 **`proxy_manifest.json`**（含每组 full/selected case IDs、seed、rate、content hash）

输出目录：`kimodo-benchmark-stratified-10pct/`

构建完成后 inventory hash（2026-08-09）：

`bd7db29d0d388f3428541a2ddd8c60179907dc4e541625429196b5bf4e11552d`

### 完整性校验

- Manifest：`proxy_manifest.json`（构建 receipt）
- Eval 指纹：`benchmark_inventory_sha256`（`meta.json` + `constraints.json` + `gt_motion.npz`）
- 构建完成后写入 eval 输出根目录的 `benchmark_inventory.json`

### 验证：Official vs NVIDIA 全量表

构建并完成 Official 基线后：

```bash
python scripts/compare_official_subset_to_nvidia_full.py \
  kimodo-benchmark-results/core10-loss-domain-stratified-10pct/official-seed-v1.1/summary_rows.json \
  --output kimodo-benchmark-results/core10-loss-domain-stratified-10pct/official_vs_nvidia_full.json
```

**验收标准（工程约定）：**

| 指标族 | 相对 NVIDIA docs 全量表 |
|---|---|
| Constraint FB/EE/Root cm | 多数 bucket **±15% 以内** |
| Foot skate / contact | **±10% 以内** |
| R@3 / FID | 趋势一致；FID 绝对值应明显优于 128 proxy |

**Official 基线实测（`official_vs_nvidia_full.json`，2026-08-09）：**

| Bucket | Root2D Δ | EE Δ | FB Δ | 结论 |
|---|---:|---:|---:|---|
| content / constraints_withtext | −7% | +3% | +3% | ✅ 通过 |
| content / constraints_notext | −2% | −7% | +3% | ✅ 通过 |
| content / text overview contact | −0.8% | — | — | ✅ 通过 |
| content / text overview skate | −7% | — | — | ✅ 通过 |

约束 cm 已与 NVIDIA 全量表对齐（~4.6 cm vs ~5.0 cm），显著优于 128 proxy 的系统性偏差。

**2026-08-09 实测（`official_vs_nvidia_full.json`）：**

| 指标 | stratified 10% | NVIDIA 全量 | Δ |
|---|---:|---:|---|
| Root2D cm（content/with text） | 4.6 | 5.0 | **−7%** |
| EE cm | 3.9 | 3.8 | +3% |
| FB cm | 3.5 | 3.4 | +3% |
| Contact | 0.977 | 0.977 | ≈0 |
| Skate cm/s | 3.8 | 4.1 | −7% |
| R@3 % | 98.9 | 81.1 | +22%（子集 N 更小，R@3 方差更大） |
| FID | 0.125 | 0.035 | +257%（仍远优于 128 proxy 的 inflated FID） |

约束类 **通过 ±15% 验收**；子集可作为 Official 对照与消融主评测。

### 消融重跑

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/core10_stratified_benchmark_pipeline.sh
```

输出：

- `kimodo-benchmark-results/core10-loss-domain-stratified-10pct/official-seed-v1.1/`
- `.../old-physical-direct-40k/`
- `.../new-normalized-direct-40k/`
- `comparison-40k-stratified.json`

---

## Benchmark 资产：128-case proxy（历史 / 快速监控）

### 定位：engineering proxy，不是官方全量表

| 层级 | 路径 / 来源 | 规模 | 本仓库用途 |
|---|---|---:|---|
| NVIDIA 公开 metadata | `kimodo-benchmark-metadata/testsuite/` | 22,474 testcase（仅 meta/seed，无 `gt_motion`） | 采样母集 |
| 官方全量 testsuite | HuggingFace `nvidia/Kimodo-Motion-Gen-Benchmark` + BONES-SEED + `create_benchmark.py` | 22,474 cases，~26GB `gt_motion` | 论文级最终复测目标；**本仓库尚未完整构建** |
| **128-case proxy（历史）** | `kimodo-benchmark-proxy-128/` | 128 cases | **训练期快速监控 only** |
| **10% stratified（当前主评测）** | `kimodo-benchmark-stratified-10pct/` | 2,269 cases | **A1/A2 消融与 Official 对照** |

设计目标（见 `docs/training_benchmark_monitor.zh-CN.md`）：在训练/消融迭代中提供**固定、
分层覆盖、不可变**的公开 benchmark 子集，单次单卡 eval ~5 min，而不是每轮跑 15–20 GPU·h
的全量 suite。

### 采样策略：按官方 hierarchy 的分层定额（stratified quota）

128 例 **不是** 对 22,474 的简单随机 0.57% 抽样，而是 **保留官方目录结构，在每个
最细叶子组内取固定配额**：

| 维度 | 全量官方 testsuite | 128 proxy 配额 |
|---|---:|---|
| Split | `content` + `repetition` | 各 64 cases |
| 顶层 category | `text2motion`、`constraints_withtext`、`constraints_notext` | 全覆盖 |
| 约束叶子 subtype | 13 组（`end-effectors/*`×3、`fullbody/*`×2、`root/*`×4、`mixture/*`×4） | **每组 2 cases** × 2 split × 2 文本模式 = **104 cases** |
| Text2motion 子类 | `overview` / `timeline_single` / `timeline_multi` | **每子类 4 cases** × 2 split = **24 cases** |
| **合计** | 22,474 | **128**（= 104 + 24） |

与全量的覆盖比例（实测，proxy ID 均为 metadata 母集子集）：

| 叶子组示例 | proxy / 全量 |
|---|---:|
| 每个 constraint 叶子组（如 `content/constraints_notext/root/path_2dpos`） | 2 / 256（0.78%） |
| `content/text2motion/overview` | 4 / 917 |
| `repetition/text2motion/overview` | 4 / 2380 |

**分层逻辑（为何这样设计）：**

1. **结构完整性优先**：NVIDIA 官方 benchmark 按 split → 任务类型 → 约束族 → 稀疏模式
   分层组织（见 `kimodo/docs/source/benchmark/introduction.md`）。proxy 保证 **58 个叶子组
   每组至少 1–2 个样本**，避免消融只测到某一类约束。
2. **约束 vs 文本配额不同**：约束类 metric 是 per-motion 平均，2 例/组已能反映趋势；
   text2motion 的 R@3/FID 在 paper protocol 下需要 **整组 motion 池** 才能算 retrieval/FID，
   因此每组保留 4 例（全量 overview 类 testcase 通常也聚合为 4 motions/testcase）。
3. **固定 ID 列表、非每次重采样**：选定 case ID 后写入资产包，后续所有 run 复用同一
   128 例；eval monitor 用 content hash 检测 proxy 是否被篡改。

**当前缺口（128）：** 构建脚本曾引用 `prepare_public_benchmark_proxy.sh` 但未入库；
组内 ID 以资产包 manifest 为准。

**状态：** 已被 stratified 10% 子集取代用于消融结论；128 仍可用于 watcher 高频监控。

### 构建做法（资产包流程）

1. **母集**：NVIDIA 公开 benchmark metadata（`Kimodo-Motion-Gen-Benchmark`），与官方 eval
   pipeline 相同的 split/category 层级。
2. **GT motion**：对每个选中 testcase，通过 BONES-SEED + `benchmark/create_benchmark.py`
   生成 `gt_motion.npz`（proxy 内已包含，无需 eval 时再跑 create）。
3. **子集拷贝**：按上节配额从完整 testsuite **拷贝** `meta.json`、`constraints.json`（若有）、
   `gt_motion.npz` 及 seed 侧车文件，保持路径与官方一致。
4. **打包**：`yezitao-kimodo-eval-v1`（`package-metadata.json`，`created_utc: 2026-08-07`）
   含 `benchmark/proxy-128/` + 同 proxy 上跑出的 Official SEED-v1.1 `summary_rows.json`。
5. **本地路径**：解压/同步后为 `kimodo-benchmark-proxy-128/`；消融 pipeline 与
   `eval_official_baseline.sh` / `eval_monitor_cli` 均指向该目录。

每个 case 目录至少包含：`meta.json`、`gt_motion.npz`；约束类另有 `constraints.json`。

### 完整性校验（immutable asset）

Eval monitor 对 proxy 做指纹（`kimodo/evaluation/eval_monitor_cli.py`）：

- 哈希输入：所有 `meta.json`、`constraints.json`、`gt_motion.npz` 的路径 + 内容 SHA256
- 本地固定值：`benchmark_inventory_sha256 = a7b67fa0fef10d71b8dfbd3952d08655a66f234161b26f4d29bb56296c073851`
- 首次 eval 写入 `benchmark_inventory.json`；之后仅校验 stat signature，proxy 变更则拒绝继续

### 科学性评估：何时可信、何时不可信

**适合（本 proxy 的设计目标）：**

| 用途 | 理由 |
|---|---|
| **同 proxy 内消融排序**（physical vs normalized vs lane0.25） | 固定 128 例 + 同协议 + 同 Official 对照 |
| **约束类趋势**（root2d、FB/EE cm、foot skate/contact） | 58 组分层覆盖；per-motion 指标对小 N 相对稳 |
| **训练期高频监控** | 成本 ~5 min vs 全量 ~15–20 GPU·h |

**不适合 / 需降级表述：**

| 用途 | 理由 |
|---|---|
| **对标 NVIDIA 文档全量表** | 128 绝对值与 22,474 不等价；Official 在 proxy 上 ~5 cm，全量表 ~3.2–3.4 cm |
| **R@3 / FID 绝对值** | text 类每组仅 4 motions，R@3 易顶到 100%，FID 方差极大（Official overview FID：全量 0.035 vs proxy 0.489） |
| **宣称「达到官方 benchmark 水平」** | proxy 是 engineering trend panel，不是 statistical estimator of full-suite mean |
| **未覆盖的稀有 subtype** | 每组仅 2 例，无法代表组内 256 例的尾部难度分布 |

**结论：** 128 采样策略对 **消融相对比较** 是合理且可辩护的工程选择（分层定额 + 固定资产）；
对 **绝对 benchmark 排名** 不科学，必须全量复测。

### Official SEED-v1.1：128 proxy vs 官方全量（参考）

本地 **没有** Official 在 22,474 全量上的实测；下表为 **proxy 重跑** vs **NVIDIA docs 全量表**
（同 metric 定义，不同样本集）：

| 指标族 | 128 proxy vs 全量 | 典型偏差 |
|---|---|---|
| Foot skate / contact | 较接近 | Overview skate ±1–16% |
| Constraint 位置误差（cm） | 同量级，content 上常偏高 | FB +10–60%；EE 常 ±10% |
| R@3 / FID | **不可比** | R@3 顶格；FID 可差一个数量级 |

因此：`comparison-40k.json` 里的 Official 列是 **「同 proxy、同协议」参考线**，不是
NVIDIA 公开发布的全量 benchmark 分数。

### 评测协议（消融统一）

| 项目 | A1/A2 消融 run | 资产包内 Official 基线 |
|---|---|---|
| Pipeline | `generate_eval` → `embed_folder` → `evaluate_folder --paper-protocol` → `parse_folder` | 同左 |
| Diffusion steps | 100 | 100 |
| Batch size | 1 | 1 |
| Text encoder | **fp32**（`--text_encoder_fp32`） | **bf16**（`eval_official_baseline.sh` 默认） |
| Postprocess | 关 | 关 |

消融与资产包 Official 基线存在 **fp32 vs bf16** 微小协议差；相对比较仍以各 arm 同 fp32 为准。
重跑 Official 时可设 `KIMODO_EVAL_TEXT_ENCODER_FP32=1` 完全对齐。

### Proxy case manifest（58 组 × 选定 ID）

完整 128 ID 按叶子组列出（与 `kimodo-benchmark-proxy-128/` 一致）：

| 叶子组 | Case IDs |
|---|---|
| `content/constraints_notext/end-effectors/feet_posrot` | 0022, 0227 |
| `content/constraints_notext/end-effectors/hands_feet_posrot` | 0120, 0157 |
| `content/constraints_notext/end-effectors/hands_posrot` | 0034, 0129 |
| `content/constraints_notext/fullbody/inbetweening` | 0074, 0091 |
| `content/constraints_notext/fullbody/random` | 0039, 0239 |
| `content/constraints_notext/mixture/root_ee_hands_feet_posrot_fullbody` | 0013, 0224 |
| `content/constraints_notext/mixture/root_ee_hands_posrot` | 0038, 0054 |
| `content/constraints_notext/mixture/root_ee_hands_posrot_fullbody` | 0212, 0229 |
| `content/constraints_notext/mixture/root_path_fullbody` | 0141, 0248 |
| `content/constraints_notext/root/path_2dpos` | 0100, 0245 |
| `content/constraints_notext/root/path_2dposrot` | 0065, 0075 |
| `content/constraints_notext/root/waypoint_2dpos` | 0056, 0115 |
| `content/constraints_notext/root/waypoint_2dposrot` | 0045, 0156 |
| `content/constraints_withtext/end-effectors/feet_posrot` | 0101, 0241 |
| `content/constraints_withtext/end-effectors/hands_feet_posrot` | 0003, 0132 |
| `content/constraints_withtext/end-effectors/hands_posrot` | 0157, 0177 |
| `content/constraints_withtext/fullbody/inbetweening` | 0222, 0248 |
| `content/constraints_withtext/fullbody/random` | 0225, 0226 |
| `content/constraints_withtext/mixture/root_ee_hands_feet_posrot_fullbody` | 0015, 0075 |
| `content/constraints_withtext/mixture/root_ee_hands_posrot` | 0082, 0218 |
| `content/constraints_withtext/mixture/root_ee_hands_posrot_fullbody` | 0027, 0108 |
| `content/constraints_withtext/mixture/root_path_fullbody` | 0101, 0111 |
| `content/constraints_withtext/root/path_2dpos` | 0231, 0238 |
| `content/constraints_withtext/root/path_2dposrot` | 0082, 0191 |
| `content/constraints_withtext/root/waypoint_2dpos` | 0176, 0242 |
| `content/constraints_withtext/root/waypoint_2dposrot` | 0000, 0214 |
| `content/text2motion/overview` | 0001, 0084, 0258, 0343 |
| `content/text2motion/timeline_multi` | 0218, 0362, 0541, 0649 |
| `content/text2motion/timeline_single` | 0077, 0110, 0117, 0197 |
| `repetition/constraints_notext/end-effectors/feet_posrot` | 0169, 0230 |
| `repetition/constraints_notext/end-effectors/hands_feet_posrot` | 0083, 0221 |
| `repetition/constraints_notext/end-effectors/hands_posrot` | 0060, 0065 |
| `repetition/constraints_notext/fullbody/inbetweening` | 0134, 0230 |
| `repetition/constraints_notext/fullbody/random` | 0030, 0117 |
| `repetition/constraints_notext/mixture/root_ee_hands_feet_posrot_fullbody` | 0102, 0143 |
| `repetition/constraints_notext/mixture/root_ee_hands_posrot` | 0109, 0171 |
| `repetition/constraints_notext/mixture/root_ee_hands_posrot_fullbody` | 0072, 0099 |
| `repetition/constraints_notext/mixture/root_path_fullbody` | 0111, 0229 |
| `repetition/constraints_notext/root/path_2dpos` | 0118, 0230 |
| `repetition/constraints_notext/root/path_2dposrot` | 0145, 0231 |
| `repetition/constraints_notext/root/waypoint_2dpos` | 0139, 0198 |
| `repetition/constraints_notext/root/waypoint_2dposrot` | 0126, 0185 |
| `repetition/constraints_withtext/end-effectors/feet_posrot` | 0170, 0196 |
| `repetition/constraints_withtext/end-effectors/hands_feet_posrot` | 0166, 0250 |
| `repetition/constraints_withtext/end-effectors/hands_posrot` | 0116, 0194 |
| `repetition/constraints_withtext/fullbody/inbetweening` | 0097, 0252 |
| `repetition/constraints_withtext/fullbody/random` | 0100, 0223 |
| `repetition/constraints_withtext/mixture/root_ee_hands_feet_posrot_fullbody` | 0190, 0235 |
| `repetition/constraints_withtext/mixture/root_ee_hands_posrot` | 0089, 0146 |
| `repetition/constraints_withtext/mixture/root_ee_hands_posrot_fullbody` | 0173, 0211 |
| `repetition/constraints_withtext/mixture/root_path_fullbody` | 0051, 0225 |
| `repetition/constraints_withtext/root/path_2dpos` | 0075, 0241 |
| `repetition/constraints_withtext/root/path_2dposrot` | 0066, 0216 |
| `repetition/constraints_withtext/root/waypoint_2dpos` | 0104, 0110 |
| `repetition/constraints_withtext/root/waypoint_2dposrot` | 0019, 0255 |
| `repetition/text2motion/overview` | 0309, 0552, 0858, 1335 |
| `repetition/text2motion/timeline_multi` | 0028, 0190, 0755, 1172 |
| `repetition/text2motion/timeline_single` | 0169, 0760, 1900, 2311 |

---

## 实验索引

| ID | 消融轴 | 状态 | 结果文件 |
|---|---|---|---|
| A1 | **Loss domain**：`physical` vs `normalized` direct feature loss | ✅ 40k stratified 10%（2026-08-09） | `comparison-40k-stratified.json` |
| A2 | **Benchmark constraint lane**：`benchmark_coverage_probability` 0 vs 0.25 | ✅ 40k（128 proxy；stratified 待重评） | `comparison-benchmark-lane-40k.json` |
| A3 | Phase 2 长度 / detach / EMA 等 | ⏳ 未做 | 见 `ppt_outline.zh-CN.md` 第 42 页 |

---

## A1：Loss domain 消融（physical vs normalized）

> **评测资产：** §1.3 为 **stratified 10% 主结果**（2026-08-09，`comparison-40k-stratified.json`）。
> §1.3-H 及 §1.4–1.5 保留 **128 proxy 历史** 供对照；§1.6–1.7 结论以 stratified 为准。

### 1.1 在消融什么

配置项：`loss.direct_feature_domain`，取值为 `physical` 或 `normalized`。

| 模式 | 六项 direct loss 计算域 | FK loss |
|---|---|---|
| **physical** | 反归一化到物理量（米、弧度等）后算 Smooth-L1 | **不变**：始终在反归一化后的物理关节空间 |
| **normalized** | 在归一化 feature tensor 上算 Smooth-L1 | **不变**：同上 |

六项 direct loss 对应：root position、root heading、joint position、joint rotation、joint velocity、foot contact。

**重要边界：**

- FK **不参与**本消融对比——两种模式下 FK 数值完全相同（见 `kimodo/training/losses.py` 与
  `tests/training/test_contracts.py::test_diffusion_loss_and_adam_atan2_contract`）。
- 训练日志里的 `loss/total` **不可跨 arm 比较**（physical ≈ 0.10 vs normalized ≈ 0.51 是量纲差异，不是优劣）。
- 官方默认 recipe（V2 / reproduction profile）使用 **normalized** direct loss + **physical** FK；
  physical direct 是 `kimodo_soma_seed_public.yaml` 保留的 baseline 对照。

### 1.2 实验设定

| 项目 | 值 |
|---|---|
| 数据 | Core10 manifest（`kimodo-core10-v1/core10.cached.jsonl`） |
| 基础 config | `configs/training/kimodo_soma_seed_public.yaml` |
| Overlay | `configs/overlays/two_h200_gb512.yaml` + `configs/experiments/validation_core10_from_scratch.yaml` |
| 步数 schedule | Phase 1 = 20k，Phase 2 = 10k（checkpoint 内锁定）；通过 `runtime.max_steps_override=40000` 将 Phase 2 延长到等效 **20k+20k = 40k** |
| 硬件 | 2× H200，`CUDA_VISIBLE_DEVICES=0,2`，effective global batch = 512 |
| Benchmark lane | **`benchmark_coverage_probability: 0.0`**（两 arm 相同，未启用 V2 13-leaf pattern） |
| 对照 baseline | Official SEED v1.1（`Kimodo-SOMA-SEED-v1.1`，非 Core10 训练） |

**Run 目录：**

| Arm | 训练目录 | 40k benchmark 输出 |
|---|---|---|
| physical | `kimodo-validation-runs/core10-from-scratch-20k-10k` | `.../core10-loss-domain-stratified-10pct/old-physical-direct-40k/` |
| normalized | `kimodo-validation-runs/core10-normalized-20k-10k` | `.../core10-loss-domain-stratified-10pct/new-normalized-direct-40k/` |

**汇总 JSON：**

- 30k（128 历史）：`kimodo-benchmark-results/core10-loss-domain-128/comparison.json`
- 40k stratified：`kimodo-benchmark-results/core10-loss-domain-stratified-10pct/comparison-40k-stratified.json`
- 40k（128 历史）：`kimodo-benchmark-results/core10-loss-domain-128/comparison-40k.json`

### 1.3 主结果表（stratified 10%，content 桶，40k）

#### Text-following（content/overview）

| 指标 | Official | Physical 40k | Normalized 40k | 更优 |
|---|---:|---:|---:|---|
| R@3 ↑ | 98.9 | **77.2** | 68.5 | physical |
| FID (gen vs gt) ↓ | 0.125 | **0.360** | 0.373 | physical |
| Foot contact ↑ | 0.977 | 0.790 | **0.819** | normalized |
| Foot skate (cm/s) ↓ | 3.8 | 17.8 | **13.7** | normalized |
| t2m_sim ↑ | 0.924 | 0.780 | **0.781** | ≈平 |

#### Constraints（content/constraints，内部 metric）

| 指标 | Official | Physical 40k | Normalized 40k | 更优 |
|---|---:|---:|---:|---|
| root2d_acc ↑ | — | 0.407 | **0.428** | normalized |
| root2d_err ↓ | — | **0.338** | 0.386 | physical |
| end_effector err ↓ | — | **0.405** | 0.443 | physical |
| fullbody keyframe err ↓ | — | **0.376** | 0.378 | ≈平 |

#### Constraints（paper table，cm，with text）

| 指标 | Official | Physical 40k | Normalized 40k | 更优 |
|---|---:|---:|---:|---|
| 2D Root Pos ↓ | **4.6** | **27.7** | 30.7 | physical |
| End-Effector Pos ↓ | **3.9** | **40.4** | 45.5 | physical |
| Full-Body Pos ↓ | **3.5** | 38.8 | **38.0** | normalized |

> Stratified 上 Official 约束 cm 已与 NVIDIA 全量表对齐（~4.6 cm）；Core10 两 arm 仍约 **6–8×** 差距。

### 1.3-H 主结果表（128 proxy 历史，content 桶，40k）

#### Text-following（content/overview）

| 指标 | Official | Physical 40k | Normalized 40k | 更优 |
|---|---:|---:|---:|---|
| R@3 ↑ | 100 | 100 | 100 | 平 |
| FID (gen vs gt) ↓ | 0.489 | 1.157 | **1.061** | normalized |
| Foot contact ↑ | 0.973 | 0.765 | **0.787** | normalized |
| Foot skate ratio ↓ | 0.060 | 0.380 | **0.319** | normalized |
| t2m_sim ↑ | 0.886 | 0.746 | **0.780** | normalized |

#### Constraints（content/constraints，内部 metric）

| 指标 | Official | Physical 40k | Normalized 40k | 更优 |
|---|---:|---:|---:|---|
| root2d_acc ↑ | **0.918** | 0.394 | 0.376 | physical |
| root2d_err ↓ | **0.051** | 0.344 | 0.357 | physical |
| end_effector err ↓ | **0.036** | 0.430 | 0.550 | physical |
| fullbody keyframe err ↓ | **0.048** | **0.467** | 0.478 | physical |

#### Constraints（paper table，cm，with text，row 0）

| 指标 | Official | Physical 40k | Normalized 40k | 更优 |
|---|---:|---:|---:|---|
| Full-Body Pos ↓ | **5.4** | 44.4 | **37.4** | normalized |
| End-Effector Pos ↓ | **3.8** | **39.8** | 51.9 | physical |
| 2D Root Pos ↓ | **5.5** | **38.5** | 41.8 | physical |

### 1.4 30k → 40k 变化（同一 arm 内）

延长 Phase 2（+10k step）对 **约束类** 改善最大；text 类相对稳定。

**Physical arm：**

| 指标 | 30k | 40k | 变化 |
|---|---:|---:|---|
| root2d_err | 0.593 | 0.344 | **−42%** |
| fullbody keyframe err | 1.138 | 0.467 | **−59%** |
| FB pos (cm) | 103.2 | 44.4 | **−57%** |
| FID (overview) | 1.169 | 1.157 | ≈ 平 |

**Normalized arm：**

| 指标 | 30k | 40k | 变化 |
|---|---:|---:|---|
| root2d_err | 0.616 | 0.357 | **−42%** |
| fullbody keyframe err | 1.076 | 0.478 | **−56%** |
| FB pos (cm) | 73.6 | 37.4 | **−49%** |
| FID (overview) | 1.388 | 1.061 | **−24%** |

### 1.5 全维度 head-to-head（40k，12 个 benchmark bucket 内所有标量 metric）

| 统计 | Physical 赢 | Normalized 赢 | 平 |
|---|---:|---:|---:|
| 单项 metric 胜负 | 39 | **86** | 215 |

按大类相对 Official 的贴近度（越高越好）：

| 大类 | Physical | Normalized |
|---|---:|---:|
| Text / content | 0.804 | **0.805** |
| Text / repetition | **0.816** | 0.815 |
| Constraints（全体） | 0.621 | **0.635** |

> 大量「平」来自 R@3 均为 100 等离散指标；Normalized 在 FID、foot 等连续指标上赢面更大。

### 1.6 效果解读

#### （1）两 arm 都远未接近 Official

在 stratified 10% 上，Official 约束 cm 已验证接近 NVIDIA 全量（~4.6 cm）。Core10 physical/normalized
仍约 **28–31 cm vs 4.6 cm（6–8×）**，`root2d_acc` 约 0.41 vs Official ~0.92 量级。不能外推为
「换 loss domain 即可复现官方约束水平」。

#### （2）Loss domain 差异 **存在但次要**（stratified 10%）

在相同数据、相同步数、相同 `benchmark_coverage_probability=0` 的前提下：

- **Physical** 在 **R@3、FID（content overview）** 上略优（77.2 vs 68.5；0.360 vs 0.373）——
  与 128 proxy 上 R@3=100 饱和不同，stratified 子集更能分辨 text 检索差异；
- **Normalized** 在 **foot contact、skate** 上 consistently 略好；
- **Physical** 在 **root / EE 约束误差**（internal metric 与 paper-table cm）上 consistently 略好；
- **Full-body cm** 互有胜负（content with text：normalized 38.0 vs physical 38.8）。

#### （3）30k 与 40k 的交叉现象

- **30k 时** normalized 的 `root2d_acc` 略高于 physical（0.354 vs 0.309）；
- **40k 时** physical 在 root/EE 上反超。

说明延长 Phase 2 后，physical 在 **空间约束梯度** 上的收益更快显现；normalized 则在 **整体 motion 质量**
（FID、foot）上保持优势。

#### （4）FK 不在本消融范围内

用户若看到 training loss 中 `loss/forward_kinematics` 项，两 arm 行为一致。讨论「FK 是否也 normalized」
属于 **新的 loss 设计实验**，不是本 A1 已覆盖的内容；官方 recipe 刻意保持 FK 在物理空间。

### 1.7 结论：哪个更优？

**没有单一「全面更好」的答案，取决于优化目标：**

| 若优先考虑… | 推荐 arm | 理由 |
|---|---|---|
| **R@3 / FID（content text）** | **Physical** | stratified：77.2 vs 68.5 R@3；0.360 vs 0.373 FID |
| **Foot contact / skate** | **Normalized** | contact 0.819 vs 0.790；skate 13.7 vs 17.8 cm/s |
| **Root path / EE 约束贴合** | **Physical** | root2d_err、EE cm、internal EE err 均优于 normalized |
| **Full-body 位置误差（paper table cm）** | **≈平 / Normalized 略优** | FB 38.0 vs 38.8（content with text） |
| **PPT 默认推荐表述** | **Normalized（官方默认）** | 与官方 recipe 一致；physical 在 R@3/约束子项上可作补充 |

**一句话结论：**

> 在 Core10 40k、stratified 10% 主评测下，**两 arm 差距比 128 proxy 更小、更可信**：
> physical 略优 R@3/FID 与 root/EE 约束；normalized 略优 foot 指标并与官方默认一致。
> 两者相对 Official 的 constraint gap 仍很大，**loss domain 不是主因**——更可能是 Phase 2 太短、
> 未启用 benchmark constraint lane、训练数据规模等（见 A2 计划）。

### 1.8 可信度边界

- 本实验是 **Core10 子集 + 40k step** 的趋势验证，**不能**外推为 full-data 1M 的最终性能排序。
- 两 arm 均未启用 `benchmark_v2_constraints.yaml`；constraint bucket 评测形态与训练分布不一致。
- Official SEED 对照来自 **不同训练 run**，不是 Core10 上的 checkpoint 对比。
- 128-case proxy 的采样策略、与全量表的偏差：见 **「历史 128 proxy」**；主评测见 **「分层 10%」**。

---

## A2：Benchmark constraint lane 消融（lane=0 vs 0.25）

> **评测资产：** 当前数字来自 **128 proxy**（`comparison-benchmark-lane-40k.json`，2026-08-09）。
> 主评测应以 stratified 10% 重评为准；128 上 R@3 饱和，约束趋势可参考但绝对值不可对标全量。

### 2.1 在消融什么

在 **同一 Core10 数据、同一 normalized direct loss、同一 40k 步数** 下，只改变 Phase 2 约束采样：

| Arm | `benchmark_coverage_probability` | 含义 |
|---|---:|---|
| lane=0（A1 normalized） | 0.0 | 仅论文五类 paper-single / paper-two |
| lane=0.25（A2） | 0.25 | 有约束样本中 25% 走公开 benchmark 13-leaf 形状（约占 Phase 2 全部样本 22.5%） |

**不改变：** 训练数据、loss domain、总约束比例（仍约 90%）、text dropout。

### 2.2 实验设定

| 项目 | 值 |
|---|---|
| Loss domain | **normalized**（与 A1 推荐一致） |
| 新增 overlay | `configs/overlays/benchmark_v2_constraints.yaml` |
| 步数 | 40k（20k P1 + 20k P2，`max_steps_override=40000`） |
| 训练 | **from scratch**（不可从 A1 checkpoint resume） |
| 对照 | A1 `normalized@40k lane=0` |

**Run 目录：** `kimodo-validation-runs/core10-normalized-benchmark-lane-40k`  
**结果：** `kimodo-benchmark-results/core10-loss-domain-128/comparison-benchmark-lane-40k.json`

### 2.3 主结果表（128 proxy，content 桶，40k）

#### Text-following（content/overview）

| 指标 | Official | lane=0 | lane=0.25 | 更优 |
|---|---:|---:|---:|---|
| R@3 ↑ | 100 | 100 | 100 | 平（饱和） |
| FID (gen vs gt) ↓ | 0.489 | **1.061** | 1.114 | lane=0 |
| Foot contact ↑ | 0.973 | 0.787 | **0.793** | ≈平 |
| Foot skate (cm/s) ↓ | 4.1 | **17.2** | 18.3 | lane=0 |

#### Constraints（paper table，cm）

| 指标 | Official | lane=0 | lane=0.25 | 更优 |
|---|---:|---:|---:|---|
| 2D Root Pos（with text）↓ | 5.5 | **41.8** | 43.1 | lane=0 |
| End-Effector Pos（with text）↓ | 3.8 | 51.9 | **45.3** | **lane=0.25** |
| Full-Body Pos（with text）↓ | 5.4 | **37.4** | 46.1 | lane=0 |
| 2D Root Pos（no text）↓ | — | 29.6 | **24.7** | **lane=0.25** |
| End-Effector Pos（no text）↓ | — | 58.0 | **50.5** | **lane=0.25** |
| Full-Body Pos（no text）↓ | — | 58.2 | **47.1** | **lane=0.25** |

#### Constraints（内部 metric，content/constraints）

| 指标 | lane=0 | lane=0.25 | 更优 |
|---|---:|---:|---|
| root2d_acc ↑ | 0.376 | **0.392** | lane=0.25 |
| root2d_err ↓ | 0.357 | **0.339** | lane=0.25 |
| end_effector err ↓ | 0.550 | **0.479** | **lane=0.25** |
| fullbody keyframe err ↓ | 0.478 | **0.466** | ≈平 / lane=0.25 |

### 2.4 效果解读

#### （1）lane=0.25 对 EE 约束有帮助，但不是全面胜利

- **End-effector**（with/without text + internal）一致改善 → 与「训练中多喂 benchmark EE / mixture 形状」的假设一致。
- **without-text 约束**（root / EE / FB）整体变好。
- **with-text Full-Body** 变差（37.4 → 46.1 cm）→ 把 22.5% paper-single 换成 13-leaf 后，full-body 与 text 联合上可能挤占了原 paper sparse full-body 覆盖。

#### （2）text / foot 质量基本持平或略损

FID、skate 略差；contact 几乎不变。lane 改的是 **约束形状分布**，不是 text 数据，因此 text 指标无大幅改善是预期内的。

#### （3）相对 Official 的 gap 仍在

两 arm 约束 cm 仍约 **8–10×** Official（~45 cm vs ~5 cm）。**单靠开 lane=0.25 不能关闭 Core10 40k 的约束差距**；更可能还需要更长 Phase 2、V2 数据、或更大训练规模。

### 2.5 结论

| 若优先考虑… | 推荐 | 理由 |
|---|---|---|
| **EE / mixture 约束贴合** | **lane=0.25** | EE cm 与 internal EE 明显更好 |
| **with-text Full-Body / FID** | **lane=0** | FB with text 与 FID 更好 |
| **PPT / 生产默认** | **lane=0.25（V2 默认）** | 与 V2 production recipe 一致；EE 收益明确，text 侧代价有限 |

**一句话结论：**

> 在 Core10 40k、normalized loss 下，**开启 benchmark lane=0.25 主要改善 EE（及 without-text）约束，对 text 跟随帮助不大，with-text FB 可能略损**。
> 它是值得保留的 V2 工程开关，但 **不是** Core10→Official 约束 gap 的主解法。

### 2.6 可信度边界

- 当前结论基于 **128 proxy**；R@3 饱和，绝对值不可对标 NVIDIA 全量。
- 待办：在 **stratified 10%** 上重评 lane=0.25，与 A1 stratified 主表对齐后再定最终表述。
- 本实验未换 V2 数据集；lane 与 V2 manifest 是正交轴。

---


## 附录：复现命令模板

```bash
# Physical 40k（resume 示例，需先 patch checkpoint provenance）
export CUDA_VISIBLE_DEVICES=0,2
export KIMODO_PATHS_CONFIG=configs/paths/core10_physical_resume.local.yaml
scripts/train_two_gpu_seed.sh \
  --config configs/training/kimodo_soma_seed_public.yaml \
  --paths configs/paths/core10_physical_resume.local.yaml \
  --overlay configs/overlays/two_h200_gb512.yaml \
  --overlay configs/experiments/validation_core10_from_scratch.yaml \
  --set loss.direct_feature_domain=physical \
  --set runtime.max_steps_override=40000

# Normalized：将 direct_feature_domain=normalized，paths 换 core10_normalized_resume.local.yaml

# Benchmark
python -m kimodo.evaluation.eval_monitor_cli \
  --run-dir /path/to/run \
  --benchmark /path/to/kimodo-benchmark-proxy-128 \
  --output-root /path/to/output \
  --minimum-step 40000 --once --paper-protocol --text-encoder-fp32
```

Pipeline 脚本：`scripts/core10_loss_domain_40k_pipeline.sh`（physical → normalized 顺序训练 + benchmark +
`comparison-40k.json`）。
