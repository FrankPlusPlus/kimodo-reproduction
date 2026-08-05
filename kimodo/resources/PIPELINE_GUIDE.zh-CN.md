# BONES-SEED 资源、目录与 `pipeline.py` 数据处理详解

本文对应同目录的 `pipeline.py`，从“下载到的 SEED 究竟是什么”开始，解释资源配置、动作/机器人领域概念、
每个预处理阶段、每种 manifest/inventory 的内容，以及离线预处理和训练时在线处理的边界。

## 1. 先纠正一个容易误解的说法：paths YAML 不是硬盘

`resources/paths.local.yaml` 是一个很小的文本配置文件。它本身可以放在仓库、用户配置目录或共享盘中的
任意位置；它的内容是一组**文件系统路径字符串**，告诉程序去哪个挂载点读写大文件。

例如：

```text
配置文件：/home/alice/kimodo/resources/paths.local.yaml
配置值：  prepared_root: /nvme0/kimodo/prepared/public-seed-soma30-v1
```

真正存几十到几百 GB 数据的是 `/nvme0` 所在磁盘或挂载点，不是 YAML 文件所在目录。

只有当配置值写成相对路径时，YAML 所在目录才参与解析：

```yaml
# 文件位于 /home/alice/kimodo/resources/paths.local.yaml
pipeline:
  prepared_root: ./managed/prepared/public-seed-soma30-v1
```

实际会解析为：

```text
/home/alice/kimodo/resources/managed/prepared/public-seed-soma30-v1
```

它不会相对于当前 shell 的 `pwd` 解析。这样从不同目录执行同一命令也不会把数据写到不同地方。

在正式服务器上建议使用绝对路径，让“配置文件放哪”和“大数据放哪”完全解耦。

## 2. 项目边界：为什么不再有 `dependencies.lock.yaml`

`kimodo-reproduction` 与 `kimodo-flowmatching` 是两个独立项目。此前把 Flow Matching 仓库锁成离线
converter 依赖，仍然属于跨项目耦合；“只在 prepare 时依赖”并不能把依赖变成独立。因此现在已经删除
`resources/dependencies.lock.yaml`、FM clone/install 参数和所有 `kimodo_flow` 导入。

当前 `kimodo-reproduction` 自己完整负责：

```text
资源下载与校验 → 安全解压 → BVH 读取 → 120 Hz 降采样到 30 Hz
→ SOMA77 映射为 SOMA30 → NPZ → manifest → LLM2Vec → stats → 训练
```

动作转换入口是本仓库的：

```bash
python -m kimodo.resources.bones ...
# 安装后也可用：kimodo_prepare_bones ...
```

它只调用本项目的 `kimodo.exports.motion_io` 和 `kimodo.skeleton.SOMASkeleton30`。环境脚本不会寻找、
克隆或安装 Flow Matching；准备和训练也不要求另一仓库存在。

### 2.1 转换器仍然怎样锁定可复现身份

预处理程序也是数据的“生产者”。如果 converter 改了以下任意细节，得到的 NPZ 就可能不同：

- 120→30 FPS 的重采样方法；
- SOMA77→SOMA30 的关节映射；
- 坐标系、单位、root 定义；
- rotation 转换和数值精度；
- 输出 NPZ 字段和顺序。

现在不再用另一个仓库的 Git commit 充当生产者身份，而是在 conversion metadata 和最终 receipt 中记录：

- 本地 converter 模块名 `kimodo.resources.bones`；
- 显式的 `conversion_revision`；
- `bones.py` 本身的 SHA-256；
- 每个源 BVH 和目标 NPZ 的 SHA-256；
- metadata、split 和 inventory 的 SHA-256。

这使数据仍然可追溯，同时依赖闭包完全留在本仓库内。两个项目若碰巧读写相似的 NPZ，那只是文件格式
层面的数据交换，不构成 Python 包、Git checkout 或环境依赖。

## 3. 三类路径配置，不要混成一个概念

工程中有三种“路径 YAML”。

### 3.1 资源与预处理路径：`resources/paths.local.yaml`

它回答：

```text
远端资源下载/复用到哪里？
原始 archive 解压到哪里？
派生训练 bundle 写到哪里？
训练 run 放哪里？
最终训练 paths YAML 写到哪里？
```

由 `kimodo.resources.config.load_paths()` 读取，允许 `resources` 和 `pipeline` 两大区块。

### 3.2 自动生成的训练路径：`pipeline.repro_paths_yaml`

它回答：

```text
训练器具体读取哪个 cached manifest？
读取哪个 reference inventory？
使用哪套 stats？
输出到哪个 run 目录？
是否 resume？
```

由 `pipeline.py` 最后生成，传给训练 CLI 的 `--paths`。典型内容只有：

```yaml
schema_version: 1
data:
  manifest: /storage/kimodo/prepared/public-seed-soma30-v1/train.cached.jsonl
  reference_inventory: /storage/kimodo/prepared/public-seed-soma30-v1/train.cached.references.jsonl
model:
  stats_path: /storage/kimodo/prepared/public-seed-soma30-v1/stats/repro-soma30-30fps
  checkpoint_dir: null
  checkpoint_weights: null
runtime:
  output_dir: /storage/kimodo/runs/repro-soma30
  resume: null
```

它不是另一份下载配置，也不包含 `dataset_root`、workers 或 text device。

### 3.3 手工训练路径模板：`configs/paths/public_seed.example.yaml`

这是没有运行资源 pipeline 时的示例/备用入口。它用环境变量表达与自动生成 YAML 相同的训练字段：

```yaml
data:
  manifest: ${oc.env:KIMODO_DATA_ROOT}/train.cached.jsonl
```

二者“看起来一样”是有意的，因为它们都必须满足 trainer 的 paths 白名单：

- 自动生成版本：绝对路径，带 prepare 的完整校验链；
- example 版本：让高级用户手工绑定已有 bundle。

正常从零 prepare 时使用自动生成版本，不需要再复制 example。

## 4. `paths.local.yaml` 的每个字段到底声明什么

### 4.1 `resources.<name>.destination`

不是一个全局 destination，而是**每项远端资源各自的下载目录**：

```yaml
resources:
  bones_seed:
    destination: /shared/kimodo/sources/bones_seed
    existing_path: null
  llm2vec_foundation:
    destination: /shared/kimodo/sources/llm2vec_foundation
    existing_path: null
```

资源管理器会在其中放 catalog 指定的文件，并在 `.cache/kimodo/` 写锁和 receipt。它逐文件校验大小与
SHA-256。

### 4.2 `resources.<name>.existing_path`

服务器已经有固定 snapshot 时使用：

```yaml
resources:
  bones_seed:
    destination: null
    existing_path: /shared/readonly/bones-seed
```

程序完整 hash 后零复制读取，永不修改 existing path。建议 `destination` 和 `existing_path` 只填一个。

### 4.3 `pipeline.dataset_root`

`soma_uniform.tar.gz` 的**解压目的地**，预期最终出现：

```text
<dataset_root>/soma_uniform/bvh/...
```

它不是下载目录：archive 仍位于 `resources.bones_seed` 对应目录。dataset root 存放体积更大的展开文件，
可在 prepare 完成并确认保留策略后单独管理。

### 4.4 `pipeline.prepared_root`

所有训练派生资产构成的 bundle 根目录：

```text
motions/
conversion/
text-cache/
stats/
train.raw.jsonl
train.cached.jsonl
train.cached.references.jsonl
resource-state.json
```

这些内容在配置时不需要预先存在。配置字段是在声明“将来由 pipeline 写到哪里”；`prepare` 才根据输入逐步
创建内容。

manifest 内部尽量使用相对 prepared root 的路径，因此完整复制这个目录后可以重新 bind。

### 4.5 `pipeline.run_root`

训练输出的父目录。pipeline 只把它写进生成的 training paths；真正的训练器创建：

```text
<run_root>/repro-soma30/
├── config.resolved.yaml
├── provenance.json
├── train.jsonl
├── checkpoints/
└── exports/
```

### 4.6 `pipeline.repro_paths_yaml`

自动生成 training paths YAML 的**文件路径**，不是目录。通常放在：

```text
<storage_root>/config/repro.paths.yaml
```

若文件已存在且内容不同，pipeline 拒绝覆盖，避免抹掉人工调整或把旧实验悄悄改指向新数据。

### 4.7 `pipeline.text_device`

离线运行 LLM2Vec 8B encoder 的设备，例如 `cuda:0` 或 `cpu`。它只影响 text cache 阶段，不决定训练用
哪张卡。文本缓存完成后训练不加载 8B 模型。

### 4.8 `pipeline.motion_workers`

并行转换 BVH 的进程数。每个 worker 读取一个动作、重采样、做 SOMA 映射并写一个 NPZ。增加它会提高
CPU/磁盘并行，也会增加内存和随机 I/O 压力。

### 4.9 `pipeline.threads_per_worker`

每个 motion worker 内允许 PyTorch/BLAS 使用的 CPU 线程数。总计算线程上限大致是：

```text
motion_workers × threads_per_worker
```

不限制时，每个进程可能各自启动整机线程池，造成严重过量订阅。

### 4.10 `pipeline.stats_workers`

拟合 normalization stats 时的进程数。它与 motion conversion worker 分开，因为 stats 阶段读取的是
canonical NPZ，还要执行 FK 和 369D 特征计算。

### 4.11 legacy adoption 字段

当前代码还支持：

```yaml
pipeline:
  legacy_bundle_root: /path/to/old/train-ready-data
  legacy_conversion_inventory: /path/to/old/conversion.inventory.jsonl
  adoption_asset_mode: hardlink  # 或 copy
```

这不是 fresh-download pipeline。它用于验证旧 motion/text cache 并收编成当前 portable schema，避免重新
运行 8B 文本 encoder。

## 5. 建议的磁盘布局

一个 storage root 可以这样组织：

```text
/storage/kimodo/
├── sources/                         # 远端原始资源；适合共享只读
│   ├── bones_seed/
│   ├── kimodo_benchmark/
│   ├── llm2vec_foundation/
│   ├── llm2vec_mntp_adapter/
│   └── llm2vec_supervised_adapter/
├── expanded/                        # archive 展开内容
│   └── bones-seed/soma_uniform/bvh/
├── prepared/                        # 本用户可写，训练高频读取
│   └── public-seed-soma30-v1/
├── config/
│   ├── resources.paths.yaml         # 人工/CLI 生成的资源部署配置
│   └── repro.paths.yaml             # pipeline 自动生成的训练路径配置
└── runs/                            # checkpoint、log、export
```

可以放在不同挂载点，不要求有共同父目录。例如 raw archive 放 NFS，prepared 放本机 NVMe，run 放大容量
checkpoint 盘。只需要在 YAML 中分别填写绝对路径。

## 6. BONES-SEED 下载物里到底有哪些东西

catalog 为训练只下载 SEED 仓库中的五个固定文件：

```text
bones_seed/
├── LICENSE.md
├── README.md
├── metadata/
│   ├── seed_metadata_v004.csv
│   └── seed_metadata_v002_temporal_labels.jsonl
└── soma_uniform.tar.gz
```

### 6.1 `LICENSE.md` 和 `README.md`

- license：数据使用许可；BONES-SEED 是 gated 数据，必须先接受许可。
- README：数据集发布方的格式和使用说明。

它们不进入 tensor，但会被 catalog 固定 revision、size 和 SHA-256，确保下载 snapshot 身份明确。

### 6.2 `soma_uniform.tar.gz`

这是主要动作数据 archive，展开后代码要求存在：

```text
soma_uniform/bvh/<date>/<clip>.bvh
```

#### BVH 是什么

BVH（Biovision Hierarchy）是经典骨骼动画格式，通常由两部分组成：

1. `HIERARCHY`：关节名称、父子关系、骨骼静态 offset、每个关节有哪些运动 channel；
2. `MOTION`：总帧数、frame time，以及每帧 root 平移和各关节旋转 channel。

BVH 更接近“骨架关节随时间运动”，不是 mesh、点云、RGB 视频或机器人电机电流。

#### SOMA Uniform 是什么

SOMA 是一套人体骨架/动作语义。本项目选择 uniform 版本，是因为所有 clip 使用一致的骨架定义和比例合同，
便于一个模型共享关节顺序和骨长语义。

原始 BVH loader 可能得到 SOMA77 或已经是 SOMA30：

- SOMA77：更细的 77 关节表示；
- SOMA30：训练模型使用的 30 个主要关节子集。

转换器只接受这两种，并把 77 投影到固定 SOMA30。这里的“投影”不是让神经网络猜 30 个关节，而是按
固定 skeleton mapping 选择/转换对应关节旋转。

### 6.3 `seed_metadata_v004.csv`

一行描述一个录制 take/动作文件及其属性。生产 pipeline 实际依赖的核心列包括：

| 列 | 用途 |
|---|---|
| `move_soma_uniform_path` | 指向 archive 解压后的 SOMA Uniform BVH |
| `filename` | 与 temporal labels 对接，构造 sample ID |
| `take_date` | 在必要时参与形成稳定路径 key |
| `content_natural_desc_1..4` | 自然语言动作描述候选 |
| `content_technical_description` | 技术性动作描述 |
| `content_short_description`、`_2` | 短描述 |

manifest builder 还知道其他骨架路径列，如 `move_soma_proportional_path` 和 `move_g1_mujoco_path`，但当前
fresh production pipeline 明确选择 `soma_uniform`，不会把 G1 CSV 混进 SOMA30 训练。

同一个 motion 可以有多条非空描述，因此一个动作文件会产生多条 full sample row。

### 6.4 `seed_metadata_v002_temporal_labels.jsonl`

这是细粒度时间标注。每行大致表示：

```json
{
  "filename": "clip_name",
  "events": [
    {
      "start_time": 1.2,
      "end_time": 3.8,
      "description": "walks forward and turns left"
    }
  ]
}
```

它回答的不只是“整个 clip 做什么”，还回答“第几秒到第几秒在做什么”。因此 pipeline 可以从一个长动作
构造 atomic event 样本。

时间用秒存储；manifest 指向 30 FPS canonical motion 后，训练器按 `round(time × 30)` 转成帧区间。

### 6.5 SEED 中没有、但 pipeline 还需要的官方 split

`train_split_paths.txt` 来自独立的：

```text
nvidia/Kimodo-Motion-Gen-Benchmark
```

每行是一个允许进入训练集的 motion key，例如概念上的：

```text
230101/some_clip_name
```

pipeline 取 metadata 和 split 的交集，只转换 train split。这样不是把整个 SEED 无差别塞进训练，而是遵循
官方公开的 leakage-safe 划分。

锁定 split 有 128,351 个 key；锁定 metadata 可解析 128,315 个，固定缺 36 个。代码检查缺失 key 集合
本身的 SHA-256，避免上游更新后训练集悄悄变化。

### 6.6 LLM2Vec 模型也不是 SEED 的一部分

三个文字模型资源来自独立 model repos：

```text
NousResearch/Meta-Llama-3-8B-Instruct
McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp
McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised
```

它们负责把 SEED 的文字描述编码为 4096D 向量，不包含 motion 数据。

## 7. 动作学和机器人学基础：为什么不能直接拿 BVH 训练

### 7.1 坐标系和单位必须固定

同样的数值 `(1,0,0)`，若一个数据集表示米、另一个表示厘米，或者一个以 Y 为上、另一个以 Z 为上，
物理含义完全不同。

canonical semantic contract 固定为：

```text
单位：meter
上方向：Y-up
前方向：Z-forward
手性：right-handed
骨架：固定 SOMA30 顺序/父子关系/rest offsets
```

### 7.2 root position 和 joint rotation 承担不同职责

动作通常拆成：

- `root_positions [T,3]`：骨盆在世界中的平移轨迹；
- `local_rot_mats [T,J,3,3]`：每个关节相对父关节的旋转。

只存旋转而不存 root translation，人物可以原地摆腿，却无法表示向前走了几米。只存关节全局位置又容易
丢失关节朝向和严格骨架结构。

### 7.3 local rotation、global rotation 和 FK

对膝关节：

- local rotation 是膝相对大腿的弯曲；
- global rotation 还包含 pelvis、大腿等所有祖先的旋转；
- FK 从 pelvis 开始沿骨长和局部旋转计算膝、踝、脚的位置。

canonical NPZ 保存 local rotations + root positions，是紧凑且可重建全身姿态的骨骼运动表示。

### 7.4 为什么 120 FPS 转成 30 FPS

原始动作一秒约 120 帧，10 秒就是 1200 帧；transformer attention 的时间和显存开销会很大。发布模型
合同使用 30 FPS：

- 保留人体动作常用时间分辨率；
- 10 秒最多 300 tokens；
- 与官方模型、stats 和速度计算匹配。

转换必须使用动作/rotation 合适的重采样逻辑，不能简单每四帧随便丢三帧后还假设所有派生速度都正确。
因此 converter 版本进入 provenance。

## 8. `pipeline.py` 的总控制流程

```text
resource catalog + paths.local.yaml
  │
  ├─ verify required train-minimal sources
  │
  ├─ 1. safe extract BONES archive
  │
  ├─ 2. canonical motion conversion
  │      BVH SOMA77/120 Hz → NPZ SOMA30/30 Hz
  │
  ├─ 3. raw manifest build
  │      full + event + adjacent combined event
  │
  ├─ 4. text cache
  │      text → sanitized text → LLM2Vec [1,4096]
  │
  ├─ 5. cached manifest
  │      raw row + embedding identity/reference
  │
  ├─ 6. normalization stats
  │      unique spans → 369D → mean/std
  │
  ├─ 7. reference inventory
  │      all referenced file size/hash
  │
  ├─ 8. training paths YAML
  │
  ├─ 9. full data preflight
  │
  └─ 10. resource-state.json = repro_train_ready
```

`prepare_pipeline()` 先调用 `plan_pipeline()` 判断每个阶段是 `build` 还是 `reuse`。阶段输出及其 metadata
必须成对存在；只有一个文件存在会被视为中断/孤立状态并 fail closed，而不是静默覆盖。

同一个 `prepared_root` 有独占 prepare lock，防止两个人同时生产同一 bundle。

## 9. 阶段 1：安全解压

输入：

```text
<bones resource root>/soma_uniform.tar.gz
```

输出：

```text
<dataset_root>/soma_uniform/bvh/...
```

`_safe_extract()`：

1. 扫描 tar member；
2. 拒绝绝对路径、`..`、symlink、hardlink 和特殊文件；
3. 累加展开字节数，要求可用空间至少为展开量加 1 GiB；
4. 在 destination 同一文件系统创建唯一 staging 目录；
5. 解压并确认 `soma_uniform/bvh`；
6. 原子 rename 发布。

为什么要 staging：若进程中断，正式 destination 不会留下一个“目录存在但只解压了一半”的假成功状态。

## 10. 阶段 2：canonical motion cache

### 10.1 输入怎样选择

converter 读取 `seed_metadata_v004.csv` 的 `move_soma_uniform_path`，把路径归一化成 key，再与官方 split
求交集。缺文件直接失败，不使用生产路径中的 `allow_missing`。

### 10.2 每个 BVH 做什么

对每个选中动作：

1. 用公开 Kimodo motion loader 解析 BVH；
2. 从 120 FPS 重采样到 30 FPS；
3. 若是 SOMA77，按固定映射转 SOMA30；
4. 提取 float32 local rotation matrices 和 root positions；
5. 验证 joint 数、shape、FPS 和 semantic contract；
6. 临时文件写完后原子替换成 NPZ。

### 10.3 canonical NPZ 存什么

```text
motions/soma30-30fps/<date>/<clip>.npz
```

核心字段：

| 字段 | shape | 含义 |
|---|---:|---|
| `local_rot_mats` | `[T,30,3,3]` | 每帧 30 关节相对父关节的旋转矩阵 |
| `root_positions` | `[T,3]` | 每帧 root/pelvis 的世界位置 |
| `fps` | scalar | 30 Hz |
| `source_provenance_json` | scalar JSON string | source hash、source/target FPS、converter identity |

两个数值数组固定保存为 `float32`。本项目的 SOMA30 关节顺序、米制/Y-up 语义由训练代码和 converter
revision 共同固定，不再嵌入另一个项目定义的 `semantic_contract_json`。来源身份则直接放进
`source_provenance_json`，复用已有 NPZ 时会逐字段核对。

这里故意不缓存：

- 随机 10 秒 crop；
- root 平移归零；
- 随机 heading；
- 369D 派生速度/contact；
- normalization。

这些保留到训练时在线执行，避免把每个 epoch 应变化的增强冻结到 cache。

### 10.4 conversion inventory 是什么

输出：

```text
conversion/soma30-30fps.inventory.jsonl
conversion/soma30-30fps.inventory.jsonl.metadata.json
```

每行对应一个源 motion 到一个缓存 motion 的转换，而不是训练样本：

```json
{
  "source": "soma_uniform/bvh/230101/clip.bvh",
  "source_sha256": "...",
  "cached": "230101/clip.npz",
  "cached_sha256": "...",
  "frames": 180,
  "fps": 30.0,
  "status": "converted"
}
```

metadata 绑定输入 metadata/split hash、转换 revision、官方/effective count 和 inventory hash。

## 11. 阶段 3：raw manifest

### 11.1 manifest 到底是什么

manifest 是**训练样本索引表**，不是动作数据本身，也不是下载文件列表。

使用 JSONL 而不是一个巨大 JSON array 的原因：

- 可流式逐行生成和读取；
- 一行损坏容易定位；
- 不必把完整 array 一次性序列化；
- 每个样本字段可扩展。

一个 canonical motion 文件可以被很多 manifest rows 引用：

- 多个 full descriptions；
- 多个 temporal events；
- 相邻 event 组合。

因此 “motion 数”远小于“训练 row 数”。

### 11.2 raw row 的字段

典型 event row：

```json
{
  "id": "clip:event:3:0",
  "motion": "motions/soma30-30fps/230101/clip.npz",
  "text": "A person walks forward and turns left.",
  "split": "train",
  "source_fps": 30.0,
  "frame_count": 180,
  "start_time": 1.2,
  "end_time": 3.8,
  "sample_kind": "event",
  "text_source": "bones_seed_temporal_label",
  "augmentation_provenance": "single_action_subclip"
}
```

字段含义：

| 字段 | 含义 |
|---|---|
| `id` | 全 manifest 唯一样本 ID；重复会失败 |
| `motion` | 相对 manifest 所在目录的 canonical NPZ 路径 |
| `text` | 原始训练文字，仍可读、可审计 |
| `split` | train/eval 等数据划分 |
| `source_fps` | 被引用 motion 当前 FPS；canonical 是 30 |
| `frame_count` | 源 NPZ 总帧数；用于启动时检查 temporal span 和 min_frames |
| `start_time/end_time` | 可选秒区间；没有则使用 full motion |
| `sample_kind` | full、event、combined_events 等样本类别 |
| `text_source` | 文字来自 metadata 哪列或 temporal labels |
| `augmentation_provenance` | 该 row 如何构造 |

### 11.3 三类公开 row 怎样生成

#### `full`

对 metadata 的每个非空描述生成一行，引用整个动作。一个动作最多可因多个描述列产生多行。

#### `event`

对 temporal labels 中每个合法 event 生成一行，引用同一 NPZ 但带 `start_time/end_time`。

#### `combined_events`

取同一 motion 内相邻两个 event：

```text
时间区间：第一个 start → 第二个 end
文字：first description + " Then, " + second description
```

它不是跨两个动作的 stitching，也没有 diffusion-generated transition。

### 11.4 当前公开 builder 没生成什么

- Qwen3-32B paraphrase；
- 两个不同 source motions 的随机 stitching；
- transition diffusion 生成的过渡帧；
- NVIDIA 未公开的各 sample_kind mixture probability。

raw manifest sidecar 明确把 paper parity 标为 blocked，避免把公开行误说成论文私有 recipe。

## 12. 阶段 4/5：LLM2Vec text cache 和 cached manifest

### 12.1 为什么要离线缓存文字

LLM2Vec foundation 是约 8B 参数。若每个训练 step 都在线编码文字：

- 占用大量 GPU/CPU 内存；
- 训练吞吐被文本模型拖慢；
- 不同软件版本或精度可能造成 embedding 漂移；
- 分布式 rank 会重复相同工作。

所以 prepare 一次性把所有唯一文字编码为 deterministic float32 vectors。

### 12.2 编码链

```text
raw text
  → kimodo.sanitize.sanitize_texts
  → Llama-3 8B foundation
  → MNTP adapter merge
  → supervised adapter（eval/frozen）
  → sentence embedding float32 [1,4096]
```

内部 batch 固定为 1，避免 batch composition 改变数值。

### 12.3 cache 文件

```text
text-cache/<64-hex-key>.npy
text-cache/<64-hex-key>.npy.metadata.json
```

`.npy` 是一个 `[1,4096]` float32 数组。sidecar 记录：

- cache key；
- sanitized text hash；
- encoder identity hash；
- dtype/shape/size；
- embedding content SHA-256。

cache key 绑定文字和 encoder 功能身份，所以相同文字可安全去重；encoder 变化时不会误用旧向量。

### 12.4 cached manifest 比 raw 多什么

cached manifest 保留 raw row 的可读文字和 motion 引用，增加：

```text
text_embedding
text_cache_key
text_embedding_metadata
text_embedding_sha256
```

示意：

```json
{
  "id": "clip:event:3:0",
  "motion": "motions/soma30-30fps/230101/clip.npz",
  "text": "A person walks forward.",
  "text_embedding": "text-cache/abc....npy",
  "text_embedding_metadata": "text-cache/abc....npy.metadata.json",
  "text_cache_key": "abc...",
  "text_embedding_sha256": "...",
  "split": "train"
}
```

训练正式读取 cached manifest。raw manifest 主要保留为可审计中间产物和 text cache 的 source identity。

## 13. 阶段 6：normalization stats

### 13.1 为什么需要 normalization

369D 中不同特征量纲差异很大：

- 位置以米计；
- heading 约在 `[-1,1]`；
- rotation 6D 也有自己的数值分布；
- 速度受 FPS 和动作快慢影响；
- contact 接近 0/1。

直接混合会让大方差特征主导优化。z-score normalization 对每一维执行：

```text
normalized = (value - mean) / std
```

### 13.2 统计时如何避免文字重复改变 motion 分布

多个 caption 可能引用同一个 `(motion,start,end)`。stats 先按这个三元组去重，因此文字多的动作不会仅因
caption 多就被重复计入。

当前 bundle 中 full span、event span、combined span 仍是不同时间范围，会分别计入。

### 13.3 stats 预处理

对每个唯一合法 span：

1. 读取 canonical local rotations/root positions；
2. temporal crop；
3. 用不重叠、最长 10 秒窗口覆盖所有帧；
4. FK 并构建 369D；
5. 第一帧 smooth-root XZ 平移到原点；
6. 用由 span key 派生的稳定 seed 抽一个 heading 并旋转；
7. float64 累积 sum/sum-square；
8. 最终保存 float32 mean/std。

### 13.4 369D 动作表示

| 特征 | 维度 | 专业含义 |
|---|---:|---|
| `smooth_root_pos` | 3 | 平滑 root 世界 XYZ 轨迹 |
| `global_root_heading` | 2 | 水平朝向 `[cos θ,sin θ]` |
| `local_joints_positions` | 90 | 30×3；XZ 相对 smooth root，Y 为高度 |
| `global_rot_data` | 180 | 30×6；全局 rotation 的连续 6D 表示 |
| `velocities` | 90 | 30×3；全局关节速度 |
| `foot_contacts` | 4 | 左/右 heel/toe 接触标签 |
| 总计 | 369 | global root 5 + body 364 |

#### smooth root

真实 pelvis trajectory 会随走路上下/左右摆动。smooth root 是对 root 轨迹做受约束平滑，提供更稳定的
整体移动控制参考；身体相对它的偏移仍保留在 joint position 特征里。

#### heading

人物绕竖直 Y 轴面向的方向。用 cos/sin 而不是角度标量，可避免 `π` 和 `-π` 数值上相距很远但物理方向
几乎相同的问题。

#### continuous 6D rotation

rotation matrix 有 9 个数但受正交约束；欧拉角有跳变和万向节锁；四元数存在 `q` 与 `-q` 表示同一旋转
的双覆盖。6D 表示用矩阵前两列样式的连续参数，再正交化回合法旋转，常用于神经网络回归。

#### foot contact

根据脚部关节速度和高度阈值判断 heel/toe 是否落地。它不是力传感器实测值，而是从运动学轨迹派生的
接触标签，用来帮助模型学习站立支撑和减少 foot skating。

### 13.5 为什么还有 local-root 4D stats

两阶段模型先预测 global root 5D，再转换成 body stage 更适合的 local root 4D：

```text
local_root_rot_vel   1  # heading 的局部角速度
local_root_vel       2  # root 在自身朝向坐标中的平面速度
global_root_y        1  # root 世界高度
```

因此 stats 分为：

```text
global_root [5]
local_root  [4]
body        [364]
```

### 13.6 输出和完整性

```text
stats/repro-soma30-30fps/
├── global_root/{mean,std}.npy
├── local_root/{mean,std}.npy
├── body/{mean,std}.npy
└── stats.metadata.json
```

metadata 绑定六个数组的 shape、dtype、size、finite 和 SHA-256，以及 source manifest hash、统计窗口策略、
seed 和排除的短 span 数。

论文没有公开 stats 拟合的精确 window/reduction，因此这些是明确记录、可重现的工程选择。

## 14. 阶段 7：reference inventory

这里最容易和 manifest、conversion inventory 混淆。

| 文件 | 一行代表什么 | 主要目的 |
|---|---|---|
| conversion inventory | 一个 BVH→NPZ 转换 | 证明 canonical motion 来源和 converter 结果 |
| raw/cached manifest | 一个训练样本 | 告诉 Dataset 训练时取哪个动作区间和文字 |
| reference inventory | 一个被训练 bundle 引用的文件 | 对所有 motion/embedding/sidecar 做内容完整性闭包 |

reference inventory 为每个唯一文件记录：

```json
{"path":"motions/.../clip.npz","size":123456,"sha256":"..."}
```

它覆盖唯一 motion、embedding、embedding sidecar、raw/cached manifest 及 sidecar。stats 使用独立
metadata/hash 验证。

prepare 第一次和 reuse 都会做 full-content verification。trainer 启动时使用 inventory 摘要验证，以免每次
重扫几十 GB。

## 15. 阶段 8/9/10：训练绑定、preflight 和 receipt

### 15.1 生成 `repro.paths.yaml`

pipeline 把最终 cached manifest、reference inventory、stats 和 run output 路径写入
`pipeline.repro_paths_yaml`。

这个文件是训练器的部署 overlay，不是方法配置。它的字段白名单不能包含 batch、learning rate、loss 等。

### 15.2 data preflight

pipeline 调用 public training config 做 `--preflight`：

1. 解析完整 manifest；
2. 检查 ID、split leakage、frame_count、路径和 embedding identity；
3. 加载 inventory summary；
4. 创建 SOMA30 motion representation；
5. 真正读取代表性 motion/text；
6. 做 crop、FK、369D、normalize 和 batch padding；
7. 检查 text 最后一维等于 4096。

它不分配完整 denoiser，重点验证数据合同。

### 15.3 `resource-state.json`

全部通过后写：

```json
{
  "schema_version": 1,
  "status": "repro_train_ready",
  "catalog_sha256": "...",
  "motion_converter_producer": {
    "module": "kimodo.resources.bones",
    "conversion_revision": "...",
    "source_sha256": "..."
  },
  "data_preflight": "full_manifest_contract_passed",
  "outputs": {...}
}
```

这是“该 prepared root 已通过完整准备”的 receipt。训练前首先检查 status，而不是只看目录是否存在。

## 16. 离线 pipeline 到哪里结束，训练时在线处理从哪里开始

### 离线保存的内容

```text
canonical local rotations/root positions
sample 的文字/时间范围索引
4096D text embeddings
normalization stats
内容 hash 和 provenance
```

### 每次 Dataset 取样在线执行

```text
读取 NPZ
→ 按 start/end 秒裁剪
→ 长于 10 秒时随机连续 crop
→ FK：local rotations + root → global joints
→ smooth root、heading、position、rotation6D、velocity、contact
→ 第一帧 root XZ 移到原点
→ 随机第一 heading
→ stats normalization
→ 读取 cached text embedding
```

### collate 在线执行

一个 batch 内动作长度不同：

```text
最长样本长度 = T_max
短样本右侧补零
valid_frames[B,T] 标记真实帧
```

padding 帧不会参与 attention/loss 统计。

### trainer 在线执行

```text
10% text dropout
→ Phase 2 在线 constraint mask
→ 随机 diffusion timestep
→ 随机 Gaussian noise
→ q_sample 得到 noisy motion
→ mask-imputation
→ root/body 两阶段 denoiser
→ 七项 loss
```

因此 canonical NPZ 不是“最终喂模型的 369D 文件”；它是一个稳定的几何中间表示。

## 17. 路径字段与各阶段输入/输出对照

| 配置字段 | 阶段开始前是什么 | pipeline 使用方式 | 阶段结束后有什么 |
|---|---|---|---|
| `resources.bones_seed.*` | 下载目录或已有 snapshot | 读 archive、CSV、temporal labels | 不修改 existing；managed 有 receipt |
| `resources.kimodo_benchmark.*` | split snapshot | 读 train key 列表 | 无派生写入 |
| `resources.llm2vec_*.*` | 固定模型 snapshots | text cache 阶段只读 | 无训练时依赖 |
| `dataset_root` | 可不存在 | 安全解压目的地 | `soma_uniform/bvh/...` |
| `prepared_root` | 可不存在 | 写所有派生阶段 | train-ready portable bundle |
| `run_root` | 可不存在 | 只写入 training paths | 训练器运行时创建 run |
| `repro_paths_yaml` | 可不存在 | 最后原子生成 | 传给 `--paths` 的 YAML |
| `text_device` | 字符串 | text encoder device | 不形成目录 |
| `motion_workers` | 正整数 | BVH conversion 并发 | 不形成目录 |
| `threads_per_worker` | 正整数 | 限制每转换进程线程 | 不形成目录 |
| `stats_workers` | 正整数 | stats 并发 | 不形成目录 |

这正是为什么 paths YAML 可以在理解所有内部 tensor 之前先填写：路径字段首先声明**所有权和目的地**；
pipeline 的固定代码合同决定往目的地写什么。理解数据内容有助于正确规划磁盘、性能和迁移，但不要求用户
手工创建每个中间文件。

## 18. 从空服务器运行的推荐顺序

### 18.1 环境和 Hugging Face 权限

```bash
cd /work/kimodo-reproduction
scripts/resources/setup_env.sh
.venv/bin/hf auth login
```

### 18.2 生成路径配置

```bash
scripts/resources/resources.sh init \
  --output /storage/kimodo/config/resources.paths.yaml \
  --storage-root /storage/kimodo
```

然后根据实际挂载点修改：

- sources 是否放共享只读盘；
- expanded/prepared 是否放本地 NVMe；
- runs 是否有足够 checkpoint 空间；
- text device 和 CPU 并发。

### 18.3 只规划，不改数据

```bash
scripts/resources/resources.sh \
  --paths /storage/kimodo/config/resources.paths.yaml \
  plan
```

`plan` 只做 presence/size 快速检查，不做 full hash。

### 18.4 下载并完整校验

```bash
scripts/resources/resources.sh \
  --paths /storage/kimodo/config/resources.paths.yaml \
  fetch

scripts/resources/resources.sh \
  --paths /storage/kimodo/config/resources.paths.yaml \
  verify
```

### 18.5 先看 prepare 计划

```bash
scripts/resources/resources.sh \
  --paths /storage/kimodo/config/resources.paths.yaml \
  prepare --dry-run
```

### 18.6 正式 prepare

```bash
scripts/resources/resources.sh \
  --paths /storage/kimodo/config/resources.paths.yaml \
  prepare
```

或者 `fetch + prepare` 一步执行：

```bash
scripts/resources/resources.sh \
  --paths /storage/kimodo/config/resources.paths.yaml \
  all
```

最终使用配置中 `pipeline.repro_paths_yaml` 指定的自动生成文件启动训练。

## 19. 当前工作区为什么看不到 fresh-download 的 sources/expanded

当前工作区走的是 legacy adoption：

```text
kimodo-training-data
  → full verification
  → hardlink motion/text assets
  → kimodo-portable-runtime/prepared/adopted-legacy-soma30-v1
```

当前自动生成的训练路径是：

```text
kimodo-portable-runtime/config/repro.paths.yaml
```

当前 receipt 表明：

```text
status            repro_train_ready
unique motions    128315
unique embeddings 132972
manifest rows     1407184
```

`resources/paths.local.yaml` 中虽然保留了 fresh-download 各 source 的 destination，但 adoption 命令不会先下载
这些资源；它验证并收编已经存在的 legacy bundle。

hardlink 模式下旧路径和新 prepared 路径的文件名不同，但可能指向同一 inode：

- 优点：不复制几十 GB；
- 限制：修改任一链接的文件内容会影响另一边；
- 删除某一个目录项不会删除仍有其他 hardlink 的 inode，但资产管理时仍应把两边视为不可变。

跨文件系统无法 hardlink，应选择 `adoption_asset_mode: copy`。

## 20. 迁移 prepared bundle

复制完整 prepared root 后，不要逐行改 manifest 的相对路径。使用：

```bash
python -m kimodo.resources.cli \
  --catalog resources/catalog.public.yaml \
  bind-prepared \
  --prepared-root /new/storage/prepared/public-seed-soma30-v1 \
  --run-root /new/storage/runs \
  --output /new/storage/config/repro.paths.yaml
```

`bind-prepared` 会：

1. 检查 `resource-state.json`；
2. 完整验证 reference inventory；
3. 验证 stats；
4. 生成适用于新机器绝对路径的 training paths YAML。

prepared bundle 内部功能引用是相对路径；历史 receipt/provenance 可能保留旧绝对路径，应作为历史审计信息
保留，而不是伪造修改。

## 21. 常见错误应该怎样理解

### `dataset_root exists but has no soma_uniform/bvh`

目标目录存在但不是完整展开结果。pipeline 不知道它属于谁，也不会删除/覆盖；应人工检查后选择新目录或
处理残留。

### `cached motion provenance is stale`

已有 NPZ 的源文件 hash、FPS 或本仓 converter revision 与当前转换请求不同。pipeline 不会把旧数据冒充为
新数据；应保留旧 bundle 做审计，并在新的 `prepared_root` 重新生成。

### `orphaned ... output requires review`

主文件和 metadata sidecar 只存在一个，通常表示中断或人工移动。程序拒绝猜测；检查内容后重新选择干净
prepared root 或按恢复流程处理。

### `conversion inventory provenance is stale`

已有 conversion inventory 对应的 metadata 或 split hash 与当前 source 不同。不能只复用旧 NPZ 同时假装
它来自新 snapshot。

### `stats were fitted from a different cached manifest`

stats 与当前训练样本集合不匹配。必须从当前 cached manifest 重建，不能仅复制另一个实验的 mean/std。

### `refusing to overwrite ... paths YAML`

自动生成文件已经存在但内容不同。选择新 output 文件名，或在人工确认旧文件不再需要后自行处理；pipeline
不会替用户覆盖配置。

### 大量小文件导致启动/遍历慢

约 13 万 embedding 加 sidecar 会产生约 26 万小文件。即使总字节不大，NFS metadata 和 inode 压力也可能
很高。prepared root 应优先选择低延迟、inode 充足的文件系统。

## 22. 一句话记住各文件的职责

```text
CSV/temporal labels：描述“有哪些动作、各动作说什么、几秒做什么”
BVH：原始骨架动画
canonical NPZ：统一到 SOMA30/30 FPS 的几何动作
conversion inventory：证明 BVH 怎样变成 NPZ
raw manifest：定义训练样本与可读文字
text-cache：把文字预先变成 4096D 数值条件
cached manifest：正式训练样本索引
stats：把 369D 不同量纲标准化
reference inventory：证明 bundle 引用文件的完整性
repro.paths.yaml：告诉训练器在本机从哪里读这些最终资产
resource-state.json：证明整套 prepared bundle 已完成全链路验收
```
