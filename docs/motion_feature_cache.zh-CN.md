# Motion 特征预物化（磁盘缓存 + 页缓存）

把每条样本的 `motion_rep` + `translate_2d_to_zero` 离线写成稠密 `[T, D] float16`，训练热路径改为
`mmap →（必要时）时间窗 → rotate → normalize`。目标是去掉每个 `getitem` 的在线 FK，让热点特征进主机
页缓存；**不是**把全集塞进 GPU 显存。

## 与当前 1M 续训的关系

- 正在跑的 hostnet 1M job **不要停**；缓存构建用单独 CPU 任务即可。
- 缓存语义对 `T > max_frames` 的超长片段会改变随机窗位置（窗从原始旋转空间改到特征空间），因此
  **不能**声称与切缓存前 bit-exact。续训时继续设 `KIMODO_RESUME_ALLOW_CODE_MISMATCH=1`。
- 建议在 **≥20k checkpoint** 落盘后再切缓存 resume，避免丢掉未打 ckpt 的进度。

## 默认落盘位置

```text
/home/share/yezitao-kimodo-reproduction/feature-cache/v1/
  meta.json
  index.jsonl
  features/{hh}/{sha256}.f16.npy
```

两台训练节点共用数据 PVC，只构建一份。

## 离线构建

在代码 PVC 上（可与训练并行）：

```bash
export KIMODO_CODE_ROOT=/home/share/yzt/kimodo-reproduction
export KIMODO_STORAGE_ROOT=/home/share/yezitao-kimodo-reproduction
# 可选：覆盖 workers / 输出目录
export KIMODO_FEATURE_CACHE_WORKERS=64
# export KIMODO_FEATURE_CACHE_DIR=/home/share/yezitao-kimodo-reproduction/feature-cache/v1

bash "${KIMODO_CODE_ROOT}/scripts/build_motion_feature_cache.sh"
```

等价 CLI：

```bash
python -m kimodo.training.feature_cache_cli \
  --manifest /home/share/yezitao-kimodo-reproduction/benchmark-v2-soma30-v2.2/manifests/train.cached.jsonl \
  --stats-path /home/share/yezitao-kimodo-reproduction/benchmark-v2-soma30-v2.2/stats \
  --output /home/share/yezitao-kimodo-reproduction/feature-cache/v1 \
  --num-workers 64 \
  --verify-sample 32
```

断点续写：同一 `--output` 再次运行且不加 `--overwrite` 时，已存在的 `.f16.npy` 会跳过重算。

## 训练切换（建议 20k 之后）

1. 确认 `feature-cache/v1/meta.json` 与 `index.jsonl` 齐全，抽样 verify 已通过。
2. 等当前 job 写出 `step-000020000.pt`（或更新）。
3. 在 yaml / paths overlay 里设置：

```yaml
data:
  feature_cache_dir: /home/share/yezitao-kimodo-reproduction/feature-cache/v1
```

或通过环境覆盖训练 config（与现有 `company_start_hostnet_resume.sh` 同级）：

```bash
# 在 resume 启动前 export，由 train_company / paths 合并进 data.feature_cache_dir
export KIMODO_FEATURE_CACHE_DIR=/home/share/yezitao-kimodo-reproduction/feature-cache/v1
```

4. 继续使用：

```bash
bash /home/share/yzt/kimodo-reproduction/scripts/company_start_hostnet_resume.sh
```

并保持 `KIMODO_RESUME_ALLOW_CODE_MISMATCH=1`。`feature_cache_dir` 已在 resume 非关键字段白名单中，
不会因路径从 `null` 变为缓存目录而拒绝 resume。

5. Fail-closed：一旦配置了 `feature_cache_dir`，缺行或缺文件会直接报错，不会静默回退到在线 FK。

## 预期收益

- 去掉每 step FK，稳态有机会从 ~1.35–1.4 steps/s 推向 ~1.6–1.9（仍受 JuiceFS/页缓存命中影响）。
- 第二 epoch 起热点更易留在 DRAM 页缓存。
- 显存占用几乎不变；GB=2048 / 1M step 配方不变。
