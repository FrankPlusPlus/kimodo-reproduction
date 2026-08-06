# Benchmark-oriented V2 数据配方

V2 的目标是提升公开 Kimodo Motion Generation Benchmark 所覆盖的文本组合与约束能力，同时保持训练源严格来自官方 BONES-SEED train whitelist。它是可审计的工程配方，不冒充 NVIDIA 未公开的完整训练数据配方。

## 文本数据变化

- 保留 V1 的 898,205 条 full-motion 描述和 318,647 条 single-event 描述。
- 删除 190,332 条机械拼接的 `A Then, B` combined rows。
- 从 V1 train event annotations 重建连续 2–5 event spans：整数 30fps 帧边界、最长 300 帧、相邻未标注 gap 不超过 45 帧。
- 按公开 benchmark 的 multi-event 结构确定性选择 223,889 个 semantic spans：
  - 2 events：175,999
  - 3 events：40,681
  - 4 events：6,246
  - 5 events：963
- 先覆盖不同 ordered source-text tuple，再 round-robin 填充重复 tuple；最终只需 70,169 个 Qwen 请求。
- Qwen 输出会由同一锁定模型执行第二次 self-judge，逐项检查动作顺序、方向、身体部位、物体、交互和次数；self-judge 不是独立模型评审。被拒绝或 JSON 失败的极少数请求使用完整保留源事件文本的确定性 fallback，并在逐行 provenance 和汇总计数中显式标记；最终仍需人工分层抽检。

预期 V2 raw manifest 为 1,440,741 rows。文本阶段没有生成新 motion，也没有实现论文所述但未公开配方的 cross-motion diffusion transitions。因此 `paper_parity_gate.eligible` 必须保持 `false`。

## Phase-2 约束变化

V2 保留论文公开的原五类 constraint curriculum，同时让 constrained samples 中 25% 走公开 benchmark 的 13-leaf coverage lane：

- strict endpoint inbetweening：只约束 `[0, T-1]`；
- full-sequence root paths：约束 `0..T-1`；
- 固定双脚、双手、双手双脚 EE sets；
- 四种公开 mixture，包括 `RightHand + LeftFoot` 的特殊组合；
- benchmark sparse counts 最大 9，经验幂次 0.45；原 paper lane 仍保持 1→20 curriculum、arbitrary EE 和 foot contacts。

覆盖概率和 sparse power 是工程假设，位于 `configs/overlays/benchmark_v2_constraints.yaml`。当前版本对齐 constraint shape，但尚未实现 3–10 秒均匀 duration-aware raw crop；该项需要单独 ablation，不能在 normalized constraint tensor 上事后切片。

## 构建顺序

```bash
kimodo_prepare_timeline_v2 \
  --source-manifest /pvc/v1/train.raw.jsonl \
  --train-split artifacts/benchmark-metadata/splits/train_split_paths.txt \
  --output-plan /pvc/v2/provenance/timeline.selected.v2.2.jsonl \
  --output-requests /pvc/v2/provenance/qwen.requests.v2.2.jsonl

kimodo_generate_qwen_v2 \
  --requests /pvc/v2/provenance/qwen.requests.v2.2.jsonl \
  --model /models/Qwen3-32B \
  --model-identity Qwen/Qwen3-32B \
  --revision 9216db5781bf21249d130ec9da846c4624c16137 \
  --shard-count 2 --shard-index 0 --device cuda:0 \
  --output /pvc/v2/provenance/qwen.responses.0.jsonl
```

另一张 H200 使用 `--shard-index 1 --device cuda:2`。两片全部通过后，依次运行 `kimodo_build_manifest_v2`、`kimodo_cache_text`、V2 stats 重算、`kimodo_reference_inventory` 全内容验证和真实 batch preflight。只有这些门禁全部通过后，staging 目录才能原子改名为 train-ready bundle。

## PVC 与权限

V2 staging 可用 hardlink 复用同一文件系统上的 immutable V1 motion/text cache，但迁移归档必须解引用 hardlink。最终目录和子目录需要 group traverse，文件需要 group read；Kubernetes Pod 应配置与 PVC 一致的 `runAsUser`/`fsGroup`。训练启动前必须由真实容器 UID 完成 manifest、随机 motion、embedding 和 inventory 的读取预检。
