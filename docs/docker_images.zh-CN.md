# Kimodo 公司镜像目录（Hanhai / 金山）

给训练平台、Notebook、以及**任意机器上的 AI 助手**阅读的镜像说明书。  
仓库路径：`docs/docker_images.zh-CN.md`  
PVC 同步路径：`/home/share/yzt/kimodo-reproduction/docs/docker_images.zh-CN.md`

## 1. 仓库与平台用法（先读这段）

| 项 | 值 |
|---|---|
| Registry | `hub.inner.ai.kingsoft.com/hh-678395/kimodo-reproduction` |
| 平台架构 | `linux/amd64` |
| **当前训练推荐 tag** | **`:v8`**（多机 RDMA 修复；旧任务可用 `:v7`） |
| 镜像职责 | CUDA / PyTorch / NCCL / RDMA 工具 / 依赖 / 可选 sshd |
| 代码职责 | **PVC** `/home/share/yzt/kimodo-reproduction`（热更新，不靠重打镜像） |
| 数据职责 | **PVC** `/home/share/yezitao-kimodo-reproduction`（含 V2 bundle、`runs/`） |
| 默认 CMD | `/workspace/scripts/container_start.sh`（无命令时 `idle`） |
| 公司训练入口 | PVC 上 `scripts/train_company.sh`（环境变量控制拓扑，默认 2×8=16） |

**重要契约：**

1. 镜像是环境壳；改训练逻辑/脚本 → 更新 PVC 代码即可，**不必**重打镜像。
2. 改系统包 / CUDA / pip 依赖 → 才需要新 tag（v8…）并 push。
3. 训练 Job 必须挂载 share PVC 到 `/home/share`。
4. 启动命令应 `exec` PVC 上的 `train_company.sh`，并设置 `PYTHONPATH` 指向 PVC 代码。
5. 密钥走 PVC `.env`（`KIMODO_CODE_ROOT/.env`），由 `load_kimodo_env.sh` 加载；**不要**打进镜像。
6. 卡数不绑死在镜像里；`:v8` 可跑 6/16/32 卡，但要用环境变量改 `KIMODO_EXPECTED_WORLD_SIZE` / batch 等。

### 正式 2×8 启动脚本（平台「训练启动脚本」粘贴）

```bash
# 平台用 /bin/sh 执行本框，不要写 set -o pipefail（dash 不支持）。
# 真正训练入口由下面 exec bash 进入 bash。
export KIMODO_CODE_ROOT=/home/share/yzt/kimodo-reproduction
export KIMODO_STORAGE_ROOT=/home/share/yezitao-kimodo-reproduction
export PYTHONPATH="${KIMODO_CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export KIMODO_NNODES="${PET_NNODES:-${NNODES:-2}}"
export KIMODO_NPROC_PER_NODE="${PET_NPROC_PER_NODE:-${NPROC_PER_NODE:-8}}"
export KIMODO_NODE_RANK="${PET_NODE_RANK:-${NODE_RANK:-${JOB_COMPLETION_INDEX:-0}}}"
export MASTER_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR:-}}"
export MASTER_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-29500}}"
cd "${KIMODO_CODE_ROOT}"
exec bash "${KIMODO_CODE_ROOT}/scripts/train_company.sh"
```

表单其它项：代码目录 `/home/share/yzt/kimodo-reproduction`；PVC → `/home/share`；实例 2 × GPU 8。

### 2×3 连通性冒烟（同镜像，改环境变量）

在上面脚本中增加：

```bash
export KIMODO_NNODES=2
export KIMODO_NPROC_PER_NODE=3
export KIMODO_EXPECTED_WORLD_SIZE=6
export KIMODO_BATCH_SIZE=8
export KIMODO_MAX_STEPS=5
export KIMODO_RUN_DIR=/home/share/yezitao-kimodo-reproduction/runs/v2-2x3-smoke
```

平台填 2 实例 × 3 GPU。

---

## 2. Tag 一览（历史推送）

| Tag | 状态 | Digest（已知） | 说明 |
|---|---|---|---|
| **v8** | **推荐（训练）** | 已推送（2026-08-10；平台拉 `:v8`） | CUDA 13.0.2 + Torch 2.11 cu130 + RDMA/compat + 默认 `NCCL_IB_HCA=mlx5`，针对 `ibv_modify_qp errno 19` |
| **v7** | 可用（旧） | `sha256:7afa9c63634e772e7a2ed46ac9f9d5c8489a67f5d3167fbb85bac19ec7ac136e` | PVC-first；CUDA12.6 栈，多机 NCCL 可能卡在 IB |
| **pvc-train** | 别名 | **与 v8 同内容（已重推）** | 语义别名；平台填 `:v8` 或 `:pvc-train` 均可 |
| v6 | 可用（开发机 SSH 修复版） | `sha256:73a81106142a950aa3954900aba4492634ffd936fc922414f80837d08532696b` | 修好 jovyan 解锁 + `authorized_keys` 安全路径；本地验证 jovyan/root 公钥登录 |
| v5 | 过渡 | （未记录 digest） | 尝试解锁 jovyan；仍有 authorized_keys 路径问题 → 被 v6 取代 |
| v4 | 过时 | `sha256:917f9876b891dcd4dc40036a09c9fcc852d058153ae2ee0833615afb5cf179ce` | 增加 jovyan/sshd 约定；仍有账号锁定与 `/tmp` keys 问题，**勿用于登录** |
| v3 | 过时 | （早期） | SSH/entrypoint 不完整，瀚海第二跳 `Permission denied` |
| v1 / v2 | 历史 | （早期 bootstrap） | 依赖/基础环境迭代；勿用于当前训练 |

> Harbor/网页上的「描述」字段若为空，以**本文档 + PVC 副本**为准。其他机器上的 AI 应优先读本文件。

---

## 3. 各版本变更摘要

### v8（多机 NCCL/RDMA 修复，2026-08-10 推送）

- **构建方式：** `FROM nvidia/cuda:13.0.2-devel-ubuntu24.04`（linux/amd64），`KIMODO_REUSE_BASE_ENV=0`；Torch **2.11 + cu130**；`LD_LIBRARY_PATH` 优先 `/usr/local/cuda/compat`（对齐同事 H200 R575 栈）。
- **相对 v7：**
  - 离开 NGC CUDA12.6，改为 CUDA13 / NCCL cu130
  - 安装 `libibverbs-dev` / `librdmacm-dev` / `libnuma-dev` 等 RDMA 用户态
  - `scripts/nccl_rdma_env.sh`：默认 `NCCL_IB_HCA=mlx5`（平台已设置则不覆盖）
  - 缓解 NCCL `ibv_modify_qp ... No such device`
- **用途：** 瀚海多机训练推荐；平台镜像填 `:v8` 或 `:pvc-train`。
- **若仍失败：** 核对 Pod 内 `ibv_devices` / 平台 RDMA 挂载；可试 `NCCL_IB_GID_INDEX` 或临时 `NCCL_IB_DISABLE=1` 分界。

### v7 / pvc-train（2026-08-08 推送）

- **构建方式：** `KIMODO_REUSE_BASE_ENV=1`，base=`...:v6`，platform=`linux/amd64`。
- **相对 v6 的关键变化：**
  - Dockerfile 默认 `KIMODO_CODE_ROOT=/home/share/yzt/kimodo-reproduction`
  - Dockerfile 默认 `KIMODO_STORAGE_ROOT=/home/share/yezitao-kimodo-reproduction`
  - `container_start.sh` 优先解析 PVC 代码根，设置 `PYTHONPATH`，加载 `.env`
  - 训练契约面向公司 share PVC + `train_company` 入口（后续脚本在 PVC 上继续演进）
- **用途：** 瀚海**训练 Job**（多机多卡）。不依赖镜像内 `/workspace` 热更新代码。
- **已知问题：** 多机 NCCL 可能卡在 IB QP；请改用 `:v8`。
- **不解决：** 自定义 Notebook 在 Istio 下 SSH（需平台 `excludeInboundPorts: "22"`）；开发请用官方 Notebook + PVC。

### v6（SSH 可用修复）

- 解锁 `jovyan`（避免 OpenSSH “account is locked”）
- `authorized_keys` 放到 sshd 可信任路径（非 `/tmp`）
- 本地验证：`jovyan` / `root` 公钥登录成功
- **现场限制：** 自定义 Notebook 若注入 Istio 且未排除 22，网关仍可能无法完成第二跳（与镜像内容无关）

### v5

- 部分账号解锁改动；keys 路径未完全修好 → 用 v6 替代

### v4

- 引入 Kubeflow 风格 `jovyan` + runtime 安装 `KIMODO_SSH_PUBLIC_KEY`
- 仍有锁定账号与 keys 路径问题 → **不要再用**

### v1–v3

- 早期 PyTorch/依赖/sshd 迭代；训练与 SSH 约定未定型

---

## 4. 镜像内 vs PVC（读代码时不要搞混）

| 路径 | 在哪 | 用途 |
|---|---|---|
| `/workspace` | 镜像内 | 构建时安装的包与 smoke；idle/fallback |
| `/home/share/yzt/kimodo-reproduction` | PVC | **真实训练代码**（`train_company.sh`、configs、kimodo） |
| `/home/share/yezitao-kimodo-reproduction/benchmark-v2-soma30-v2.2` | PVC | V2 train-ready bundle |
| `/home/share/yezitao-kimodo-reproduction/runs/` | PVC | checkpoint / 日志 |
| `/home/share/yzt/kimodo-reproduction/.env` | PVC | WANDB 等（chmod 600） |

入口脚本：`scripts/train_company.sh` → `scripts/train_distributed.sh` → `torch.distributed.run` + `kimodo.training.cli`。  
配置默认：`configs/training/kimodo_soma_seed_v2_1m_16h200.yaml` + bundle 内 `repro.paths.yaml`。

---

## 5. 给其他机器 AI 的最短指令

若你在另一台机器 / 另一个 Cursor 会话，请先读：

1. `/home/share/yzt/kimodo-reproduction/docs/docker_images.zh-CN.md`（本文）
2. `/home/share/yzt/kimodo-reproduction/docs/multinode_k8s_training.zh-CN.md`
3. `/home/share/yzt/kimodo-reproduction/scripts/train_company.sh`

并记住：

- 训练镜像用 **`hub.inner.ai.kingsoft.com/hh-678395/kimodo-reproduction:v8`**
- `:pvc-train` ≡ `:v8`
- 代码和数据在 PVC，不在镜像层里“最新”

---

## 6. 维护约定

推送新 tag 时必须：

1. 更新本文件的 Tag 表与 digest
2. 写清相对上一版的变更与推荐用途
3. `scp`/同步到 PVC 同路径，便于集群内 AI/人工阅读
4. 勿用可变 tag 覆盖已验证 digest（需要修复就打 `v8`）
