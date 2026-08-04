# BONES-SEED dancing 小样本增强

## 目的和边界

这个入口用于以后把 BONES-SEED 中少量 dancing 训练样本提高采样占比。它只组合 manifest：不复制
motion、LLM2Vec embedding 或 stats，不修改 Kimodo 网络/loss/curriculum，也不在训练器里维护第二条数据
管线。DataLoader 原有 shuffle 会打散组合后的行，训练日志会分别记录 `data/base_fraction` 和
`data/dance_fraction`。

它属于公开数据上的实验增强，不是论文未公开的 Qwen paraphrase、跨 motion stitching 或 diffusion
transition recipe，因此不能让 public profile 自动变成 exact paper reproduction。

## 输入

需要两个已经完成 motion conversion 和 LLM2Vec 缓存的 JSONL：

- `train.cached.jsonl`：原训练集；
- `dance.cached.jsonl`：只含允许进入训练集的 dancing 行。

`dance.cached.jsonl` 可以只是从 base cached manifest 选出的少量 JSON 行，继续引用同一批 motion、
embedding 及 embedding metadata；无需复制这些大文件。也可以引用按相同 SOMA30/30 fps/text-cache
契约独立准备的新训练数据。

现有 cached 行没有可靠的统一 `action_category=dancing` 字段，且文件名中的 `dancecard` 可能只是动作
采集卡而非舞蹈。不要直接用文本关键词全自动筛选；应从 BONES metadata/源 take ID 建候选列表并人工
抽查，再把确认过的完整 JSON 行写入 dance manifest。

只允许 `split=train`。不要从 benchmark/content/repetition test 或任何 held-out split 选行，否则会产生
数据泄漏。若 dancing 行本来就在 base 中，这个操作的含义是确定性 oversampling，而不是加入新内容。

## 生成 5% 混合 manifest

```bash
base=/shared/kimodo/prepared/train.cached.jsonl
dance=/shared/kimodo/prepared/dance.cached.jsonl
mixed=/shared/kimodo/experiments/dance-05/train.mixed.jsonl
inventory=/shared/kimodo/experiments/dance-05/train.mixed.references.jsonl

.venv/bin/python -m kimodo.training.manifest_overlay_cli \
  --base-manifest "${base}" \
  --overlay-manifest "${dance}" \
  --output "${mixed}" \
  --overlay-fraction 0.05 \
  --base-name base \
  --overlay-name dance \
  --split train \
  --seed 1234

.venv/bin/python -m kimodo.training.reference_inventory_cli build \
  --manifest "${mixed}" \
  --output "${inventory}"

.venv/bin/python -m kimodo.training.reference_inventory_cli verify \
  --manifest "${mixed}" \
  --inventory "${inventory}"
```

`--overlay-fraction 0.05` 指最终 mixed manifest 中约 5% 的行来自 dance，不是“额外添加 base 行数的
5%”。若 base 有 `N` 行，工具选择约 `0.05*N/(1-0.05)` 个 dance 行；dance 较小时按固定 seed 循环、
洗牌后 oversample。输出和 metadata sidecar 都拒绝覆盖，避免悄悄改变既有实验。

生成的 manifest 使用相对路径回指原资产，并为每行写入 `mixture_source`。metadata sidecar 绑定两个
source manifest、可用的 source sidecar、producer hash、seed、目标/实际比例和输出 hash。reference
inventory 会覆盖混合 manifest、source manifests、motion、embedding 和 embedding semantic sidecar。

## 接入训练

不要改 pipeline 自动生成的原始 `repro.paths.yaml`。复制一份专用于该实验，只改 manifest、inventory 和
新 output directory：

```yaml
schema_version: 1
data:
  manifest: /shared/kimodo/experiments/dance-05/train.mixed.jsonl
  reference_inventory: /shared/kimodo/experiments/dance-05/train.mixed.references.jsonl
model:
  stats_path: /shared/kimodo/prepared/stats/repro-soma30-30fps
  checkpoint_dir: null
  checkpoint_weights: null
runtime:
  output_dir: /shared/kimodo/runs/repro-dance-05
  resume: null
```

先做真实数据 preflight，再训练：

```bash
.venv/bin/python -m kimodo.training.cli \
  --config configs/training/kimodo_soma_seed_public.yaml \
  --paths /shared/kimodo/experiments/dance-05/repro.paths.yaml \
  --overlay configs/overlays/two_h200_gb2048.yaml \
  --preflight

CUDA_VISIBLE_DEVICES=0,2 \
KIMODO_PATHS_CONFIG=/shared/kimodo/experiments/dance-05/repro.paths.yaml \
scripts/train_two_gpu_seed.sh
```

若 dancing 是 base 的子集，继续使用 base stats 是合理且最小的默认：它只改变采样权重，不改变表示定义。
若加入分布不同的新数据，应把“沿用 base stats”与“在最终混合集上重拟合 stats”当成显式消融实验，不能
无记录地替换。

## 建议的最小实验

先比较 0%、2%、5%、10% 四个比例，保持 seed、global batch、optimizer steps、模型和 loss 完全一致。
同时看 dancing held-out 指标与原公开 benchmark，避免只提升 dancing 却破坏整体动作分布。小数据重复率
很高时，优先降低比例，而不是叠加新的 sampler、loss 或网络分支。
