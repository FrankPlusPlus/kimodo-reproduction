# 开发机网络与代理（瀚海 Notebook / 新 Pod）

给换机、重建 Notebook、以及 AI 助手阅读的网络合同。  
仓库路径：`docs/dev_notebook_network.zh-CN.md`  
PVC：`/home/share/yzt/kimodo-reproduction/docs/dev_notebook_network.zh-CN.md`

用途：checkpoint 推理、benchmark、洗数据等开发机日常工作。  
**不覆盖**：多机训练 NCCL/RDMA（见 `multinode_k8s_training.zh-CN.md`）；镜像推送仍在本机 Docker 完成。

## 1. 推荐开发机形态

| 项 | 值 |
|---|---|
| 平台类型 | **官方 Notebook**（不要用自定义训练镜像硬当开发机） |
| 镜像 | `hub.inner.ai.kingsoft.com/default/jupyter-pytorch-cuda-full:v3.9.0` |
| GPU | 1× H200（推理/benchmark）；纯洗数据可用 CPU 规格 |
| PVC | 挂到 `/home/share` |
| 代码 | `/home/share/yzt/kimodo-reproduction` |
| 数据 / `runs/` | `/home/share/yezitao-kimodo-reproduction` |

自定义镜像（如 `kimodo-reproduction:v7`）当 Notebook 时，Istio 常不排除 22，**SSH 第二跳会失败**；开发请用官方 Notebook + PVC。

新 Pod 里家目录（`~`）是空的，**代理/SSH 转发不会跟着 PVC 自动过来**，需要按下文重配一次。

## 2. 网络拓扑（一张图说清）

```text
[你的实体机]
  Clash/VPN 监听 127.0.0.1:<本地端口>     例：7890 或 7993
        ▲
        │ SSH RemoteForward（-R）
        │
[瀚海 Notebook Pod]
  127.0.0.1:7993  ──►  回你本机代理  ──►  外网（Google / GitHub / pip 外源等）
  直连（NO_PROXY）──►  公司内网（Harbor / *.inner.ai.kingsoft.com / 172.20.0.0/16）
```

要点：

1. **外网**走 SSH 转到本机的代理（Pod 内默认端口 **7993**）。
2. **公司内网必须绕过代理**。若 Harbor 也走 7993，会出现 TLS 握手失败（`unexpected eof`）。
3. 本机代理断开或 SSH 断线后，外网会挂；内网直连一般仍可用。

## 3. 本机（实体机）必配：SSH RemoteForward

在你**自己电脑**的 `~/.ssh/config` 里，给连 Notebook 的 Host 加上（端口按你本机 Clash 实际监听改）：

```sshconfig
Host hanhai-dev
    HostName <平台给的 Notebook SSH 地址>
    User jovyan
    # Pod 内 7993 -> 本机代理（示例本机 Clash 在 7890）
    RemoteForward 7993 127.0.0.1:7890
```

或命令行：

```bash
ssh -R 7993:127.0.0.1:7890 jovyan@<notebook-host>
```

Cursor / VS Code Remote-SSH 用同一 Host，每次连上会自动带上转发。

本机代理需允许本机回环访问（监听 `127.0.0.1` 即可）。

## 4. Pod 内环境变量合同

| 变量 | 值 |
|---|---|
| `HTTP_PROXY` / `HTTPS_PROXY` | `http://127.0.0.1:7993` |
| `http_proxy` / `https_proxy` | 同上 |
| `ALL_PROXY`（可选） | `http://127.0.0.1:7993` |
| `NO_PROXY` / `no_proxy` | 见下一行 |

**`NO_PROXY` 必须包含：**

```text
localhost,127.0.0.1,::1,.inner.ai.kingsoft.com,hub.inner.ai.kingsoft.com,172.20.0.0/16
```

写到两处 shell 配置：

- `~/.profile`：登录 shell / Cursor 远程宿主启动时读取（**Codex 子进程主要靠这个**）
- `~/.bashrc`：交互终端的 `proxy_on` / `proxy_off` / `proxy_status`

改完 `~/.profile` 后，需要 **Kill VS Code/Cursor Server on Host 再重连**，扩展进程才会吃到新环境变量。

### Cursor 远程 Machine `settings.json`（Codex / 扩展必备）

Remote-SSH 连上后，代理还要写进 **远程机** 的 Machine settings（不是你本机 Cursor 的 User settings）。  
PVC 模板：`scripts/resources/cursor-machine-settings.proxy.json`

| 客户端 | 远程路径 |
|---|---|
| Cursor | `~/.cursor-server/data/Machine/settings.json` |
| VS Code | `~/.vscode-server/data/Machine/settings.json` |

内容：

```json
{
    "http.proxy": "http://127.0.0.1:7993",
    "http.proxySupport": "on",
    "http.proxyStrictSSL": false
}
```

含义：

| 键 | 作用 |
|---|---|
| `http.proxy` | 远程扩展宿主走 `127.0.0.1:7993`（SSH 转回本机 Clash） |
| `http.proxySupport` | `on`：扩展请求也走代理 |
| `http.proxyStrictSSL` | `false`：避免部分自签/中间盒证书把扩展卡死 |

**Codex 要两层都配齐：**

1. **Machine `settings.json`**：走编辑器网络栈的请求（Reload Window 即可）  
2. **`~/.profile` 的 `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY`**：Codex CLI/插件 spawn 的进程只认环境变量；必须 Kill remote server 后重连才进宿主进程  

只配 json、不配 `.profile` → 终端/`codex` 子进程仍可能出不去。  
只配 `.profile`、不配 json → 部分扩展 UI 请求仍可能不走代理。

手动安装模板：

```bash
mkdir -p ~/.cursor-server/data/Machine
cp /home/share/yzt/kimodo-reproduction/scripts/resources/cursor-machine-settings.proxy.json \
  ~/.cursor-server/data/Machine/settings.json
# 若已有其它 Machine 设置，用 setup_dev_network.sh 合并写入，避免整文件覆盖
```

## 5. 新 Pod 一键安装（PVC 脚本）

PVC 上脚本（随代码热更新）：

```bash
bash /home/share/yzt/kimodo-reproduction/scripts/setup_dev_network.sh
```

作用：

- 把代理 export 写入 `~/.profile`
- 把 `proxy_on` / `proxy_off` / `proxy_status` 与 codex alias 写入 `~/.bashrc`
- **合并写入** Cursor / VS Code 远程 Machine `settings.json`（`http.proxy` 三件套；默认开启）
- **不**把 Harbor 密码或密钥写进任何文件

然后：

1. 确认本机 SSH `RemoteForward 7993` 已连上  
2. **Kill Cursor Server on Host 并重连**（让 `.profile` 进远程宿主；再 Reload 一次也可）  
3. 跑自检（见下节）；再开 Codex

可用环境变量覆盖默认端口：

```bash
KIMODO_DEV_PROXY_PORT=7993 bash /home/share/yzt/kimodo-reproduction/scripts/setup_dev_network.sh
```

## 6. 自检清单

```bash
# 隧道是否在听
proxy_status   # 或: (echo >/dev/tcp/127.0.0.1/7993) && echo ok

# 外网应走 7993
curl -sS -o /dev/null -w 'google=%{http_code}\n' https://www.google.com

# 公司 Harbor 必须直连（401 未登录也算 TLS 通；勿再出现 handshake eof）
curl -sk -o /dev/null -w 'harbor=%{http_code}\n' https://hub.inner.ai.kingsoft.com/v2/

# 详细看是否误走代理：verbose 里应是 Connected to hub...101.41，而不是 127.0.0.1:7993
curl -sk -v https://hub.inner.ai.kingsoft.com/v2/ -o /dev/null 2>&1 | grep -E 'Connected to|proxy'
```

| 结果 | 含义 |
|---|---|
| Google 通 + Harbor 401/200 | 网络合同正确 |
| Harbor TLS eof / decode error | `NO_PROXY` 没生效，内网被拐进 7993 |
| Google 失败、隧道 closed | 本机 Clash 或 SSH `-R` 断了 |

Harbor 证书是公司自签 `harbor-ca`；系统未信任时用 `curl -sk`，或把 CA 导入系统信任库。  
**镜像 push/pull 仍在实体机 Docker 完成**；Notebook 一般没有 Docker 守护进程。

## 7. 和项目密钥 / 训练入口的关系

| 内容 | 位置 | 谁加载 |
|---|---|---|
| `WANDB_API_KEY` 等 | PVC：`KIMODO_CODE_ROOT/.env`（chmod 600） | `scripts/load_kimodo_env.sh`（训练/container 入口） |
| 开发机代理 | 家目录 `~/.profile` / `~/.bashrc` | 本机登录 shell；**不进 Git、不进镜像** |
| 训练 Job 网络 | 平台 CNI + RDMA | 与本文 SSH 代理无关 |

开发机终端若要手动跑训练脚本并带上 `.env`：

```bash
export KIMODO_CODE_ROOT=/home/share/yzt/kimodo-reproduction
export KIMODO_STORAGE_ROOT=/home/share/yezitao-kimodo-reproduction
source "${KIMODO_CODE_ROOT}/scripts/load_kimodo_env.sh" && kimodo_load_env_files
```

WandB 等若公司出口可直连，不一定依赖 7993；需要翻墙的工具（部分模型下载、Codex、外网 pip）才依赖本文代理。

## 8. 常见误区

1. **只在当前终端 `proxy_on`**：Cursor/Codex 插件读不到 → 必须写 `~/.profile` 并重启远程 Server。  
2. **只改本机 Cursor settings、不改远程 Machine json**：Remote-SSH 下扩展跑在 Pod 里，本机 User settings 管不到 Codex。  
3. **`NO_PROXY` 只有 localhost**：Harbor TLS 必炸。  
4. **指望新 Pod 继承旧家目录配置**：`~` 通常是空的；用 PVC 上的 `setup_dev_network.sh` 重装。  
5. **在 Notebook 里 docker pull**：平台 Pod 默认无 daemon；推镜像回实体机。  
6. **用 `kimodo-reproduction:v7` 当 SSH 开发机**：镜像能跑训练，但平台 Istio/SSH 合同不保证；开发用官方 Notebook。
