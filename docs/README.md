# 当前文档

本目录只维护当前训练复现、V2 数据和公司部署合同。NVIDIA 的推理、Demo、API 和公开 benchmark
基础说明直接以[官方在线文档](https://research.nvidia.com/labs/sil/projects/kimodo/docs/index.html)为准，
不再在本仓库复制一套 Sphinx 源文件。

建议按以下顺序阅读：

1. [`benchmark_v2_data_recipe.zh-CN.md`](benchmark_v2_data_recipe.zh-CN.md)：V2 数据组成、MiMo
   增强、质量门禁、bundle 构建和发布合同。
2. [`training_phase2_dataflow.zh-CN.md`](training_phase2_dataflow.zh-CN.md)：从 bundle 读取、约束采样、
   Transformer 输入直到 Phase 1/2 loss 的完整张量流。
3. [`multinode_k8s_training.zh-CN.md`](multinode_k8s_training.zh-CN.md)：PVC、镜像、两机 16×H200、
   NCCL/RDMA 和启动脚本。
4. [`training_benchmark_monitor.zh-CN.md`](training_benchmark_monitor.zh-CN.md)：训练旁路评测和公开
   benchmark 指标监控。
5. [`code_surface.zh-CN.md`](code_surface.zh-CN.md)：公共入口、核心模块和内部审计工具边界。

历史验收快照、V1 设计草案、重复流程图以及上游网站生成源已经删除，避免旧结论与当前 V2 链路并存。
